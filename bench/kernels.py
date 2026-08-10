"""Which kernel did each op actually get?

This answers the question "are we really running OpenVINO's optimised kernels?"
with evidence rather than assurance.

After a plugin compiles a model, `CompiledModel.get_runtime_model()` returns the
*executable* graph: the post-optimisation node list, where each node carries an
`execType` naming the concrete kernel the plugin selected. That is the ground
truth. Examples:

    CPU   jit_avx2_FP32, brgemm_avx512_..., acl_...   <- optimised
          ref_any, ref_FP32                            <- REFERENCE FALLBACK
    GPU   fully_connected_gpu_bf_tiled, gemm_tiled_opt <- optimised
          *_ref, reference_*                           <- REFERENCE FALLBACK

A reference fallback is not an error and raises no warning. It just runs ~10x
slower. On a bandwidth-bound LLM a single fallback on a hot matmul silently
destroys the result, and no amount of downstream tuning recovers it -- so this
check belongs *before* any benchmarking, not after a confusing number appears.

Also reported: the precision each node actually executes at. Asking for FP16 on
a device whose OPTIMIZATION_CAPABILITIES lack FP16 (the laptop's CPU, for one)
means the plugin upconverts to FP32 at runtime -- doubling the bytes streamed per
token on a workload whose whole cost is bytes streamed per token.

    python -m bench.kernels --model models/qwen2.5-1.5b-fp16 --device CPU
"""

from __future__ import annotations

import argparse
import collections
import gc
import os
import sys

# primitiveType substrings meaning a genuine unoptimised reference kernel.
#
# Deliberately NOT included:
#   "unknown"  graph plumbing (Const/Parameter/Result/Reshape) has no kernel.
#   "undef"    the implementation type was not REPORTED. On the CPU plugin the
#              modern FullyConnected and SDPA executors surface as `undef_*`
#              precisely because they bypass the classic impl_desc_type tagging.
#              Counting those as fallbacks says "197 FullyConnected on reference
#              kernels", which is alarming, prominent and wrong.
#
# Only `ref_*` is a real reference implementation.
REF_MARKERS = ("ref_", "_ref", "reference")

# Reported separately: real kernel unknown to us, not a known-bad one.
UNREPORTED_MARKERS = ("undef", "unknown")

# Layer types that carry the decode workload. A fallback on one of these is
# fatal; a fallback on a stray Reshape is cosmetic. Only these are called HOT.
HOT_TYPES = {
    "fullyconnected", "matmul", "convolution", "gemm",
    "scaleddotproductattention", "pagedattention", "sdpa",
    "softmax", "rms", "rope", "mvn",
}

# Node types with no kernel of their own -- excluded from the "unclassified"
# count so the summary reflects compute nodes rather than graph structure.
STRUCTURAL = {"const", "parameter", "result", "input", "output"}


# The executable graph exposes these rt_info keys (verified on OpenVINO 2026.3):
#   execOrder, execTimeMcs, layerType, originalLayersNames,
#   outputLayouts, outputPrecisions, primitiveType, runtimePrecision
# The kernel name lives in `primitiveType`. There is no `execType` key -- reading
# one silently yields nothing and makes every node look unclassified.
KERNEL_KEYS = ("primitiveType", "execType")
PRECISION_KEYS = ("runtimePrecision", "outputPrecisions")


def _rt(node, key: str, default: str = "") -> str:
    """Read one rt_info field as a plain string.

    Values come back as OVAny, whose str() is the useless '<OVAny class>'.
    The real payload is behind .value (equivalently .get()).
    """
    try:
        info = node.get_rt_info()
        if key in info:            # RTMap has no .get(); membership test only
            v = info[key]
            v = getattr(v, "value", v)
            return str(v)
    except Exception:
        pass
    return default


def _first(node, keys, default: str = "?") -> str:
    for k in keys:
        v = _rt(node, k)
        if v:
            return v
    return default


def make_static(model, seq: int = 1, past: int = 1024, batch: int = 1) -> str:
    """Bind every dynamic dimension to a concrete size.

    The NPU compiler cannot consume unbounded dynamic shapes -- an exported LLM
    IR has `?` for batch and sequence, which surface as INT64_MAX bounds and
    fail with "to_shape was called on a dynamic shape".

    GenAI's LLMPipeline does this reshaping internally from MAX_PROMPT_LEN /
    MIN_RESPONSE_LEN, which is why the pipeline can target the NPU while a raw
    Core.compile_model() on the same IR cannot. This reproduces that step so the
    executable graph can be inspected the same way it can on CPU and GPU.

    Dimension roles are inferred from input names, since an LLM IR follows a
    stable convention:
        input_ids / position_ids   [batch, seq]
        attention_mask             [batch, past + seq]
        beam_idx                   [batch]
        past_key_values.N.key|value [batch, kv_heads, past, head_dim]
    """
    from openvino import PartialShape

    shapes = {}
    for inp in model.inputs:
        try:
            name = inp.get_any_name()
        except Exception:
            name = ""
        ps = inp.get_partial_shape()
        low = name.lower()
        dims = []
        dyn_seen = 0
        for d in ps:
            if d.is_static:
                dims.append(d.get_length())
                continue
            if "beam_idx" in low:
                dims.append(batch)
            elif "past_key_value" in low or "present" in low:
                # first dynamic is batch, second is the accumulated KV length
                dims.append(batch if dyn_seen == 0 else past)
            elif "attention_mask" in low:
                dims.append(batch if dyn_seen == 0 else past + seq)
            else:                       # input_ids, position_ids, ...
                dims.append(batch if dyn_seen == 0 else seq)
            dyn_seen += 1
        shapes[name] = PartialShape(dims)

    model.reshape(shapes)
    return ", ".join(f"{k}{list(v.to_shape())}" for k, v in list(shapes.items())[:4])


def inspect(model_dir: str, device: str, props: dict | None = None,
            static: bool | None = None, seq: int = 1, past: int = 1024) -> dict:
    from bench.devices import core  # shared singleton; see devices.core()

    c = core()
    xml = os.path.join(model_dir, "openvino_model.xml")
    if not os.path.isfile(xml):
        raise SystemExit(f"no openvino_model.xml in {model_dir}")

    print(f"reading  : {xml}", flush=True)
    model = c.read_model(xml)

    # NPU cannot compile unbounded dynamic shapes; default to reshaping there.
    if static is None:
        static = device.upper().startswith("NPU")
    if static:
        shown = make_static(model, seq=seq, past=past)
        print(f"reshaped : static seq={seq} past={past}  [{shown} ...]", flush=True)

    # A stateful IR keeps its KV cache in ReadValue/Assign variables rather than
    # in Parameters. model.reshape() only rebinds Parameters, so the cache stays
    # dynamic and the NPU compiler still fails with "to_shape was called on a
    # dynamic shape" -- the reshape above cannot fix it.
    #
    # GenAI's NPU path solves this with NPUW transformations driven by
    # MAX_PROMPT_LEN / MIN_RESPONSE_LEN, which is not reproducible with a few
    # public API calls. So: detect it and say so, rather than emit a traceback
    # that looks like a hardware verdict when it is a tooling limit.
    # `sinks` is a property in some builds and `get_sinks()` a method in others.
    n_state = 0
    for accessor in ("get_sinks", "sinks"):
        try:
            v = getattr(model, accessor)
            v = v() if callable(v) else v
            n_state = len(v or [])
            break
        except Exception:
            continue
    if n_state and device.upper().startswith("NPU"):
        raise SystemExit(
            f"\nCannot inspect NPU kernels for this model: the IR is STATEFUL\n"
            f"  ({n_state} ReadValue/Assign pairs hold the KV cache as internal\n"
            f"  state with dynamic shape, and model.reshape() only rebinds inputs).\n"
            f"\n"
            f"  This is a limitation of raw Core.compile_model(), NOT a statement\n"
            f"  about whether the NPU can run the model. GenAI's LLMPipeline applies\n"
            f"  NPUW static transformations that this path does not.\n"
            f"\n"
            f"  To get the actual NPU decode measurement:\n"
            f"    python -u -m bench.run --strategy single --reps 3 "
            f"--out results\\cloud-single.csv\n"
            f"\n"
            f"  To inspect NPU kernels specifically, re-export without state:\n"
            f"    optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
            f"        --task text-generation-with-past --disable-stateful \\\n"
            f"        --weight-format fp32 models/qwen2.5-1.5b-fp32-nostate\n"
            f"  then run this tool against that directory. Note the non-stateful\n"
            f"  graph is not what you would deploy -- it is a diagnostic only.\n")

    print(f"compiling: device={device} props={props or {}}", flush=True)
    compiled = c.compile_model(model, device, props or {})
    print("inspecting executable graph ...", flush=True)

    rt = compiled.get_runtime_model()
    by_exec: collections.Counter = collections.Counter()
    by_prec: collections.Counter = collections.Counter()
    fallbacks: list[tuple[str, str, str]] = []
    unreported: list[tuple[str, str]] = []
    total = 0

    for node in rt.get_ordered_ops():
        total += 1
        exec_type = _first(node, KERNEL_KEYS)
        layer_type = _rt(node, "layerType", node.get_type_name())
        prec = _first(node, PRECISION_KEYS)
        by_exec[exec_type] += 1
        by_prec[prec] += 1

        low = exec_type.lower()
        lt = layer_type.lower()
        if lt in STRUCTURAL:
            continue
        hot = any(h in lt for h in HOT_TYPES)
        if any(m in low for m in REF_MARKERS):
            fallbacks.append((layer_type, exec_type, "HOT" if hot else "cold"))
        elif hot and any(m in low for m in UNREPORTED_MARKERS):
            unreported.append((layer_type, exec_type))

    out = {"total": total, "by_exec": by_exec, "by_prec": by_prec,
           "fallbacks": fallbacks, "unreported": unreported,
           "device": device, "model": model_dir}

    # Drop the runtime model and the compiled model explicitly, in that order.
    # Letting the interpreter tear them down at exit triggers an access
    # violation (0xC0000005) -- the runtime graph holds references into plugin
    # memory that must not outlive the CompiledModel.
    del rt
    del compiled
    del model
    gc.collect()
    return out


def report(r: dict) -> int:
    print()
    print(f"executable graph: {r['total']} nodes on {r['device']}")
    print()
    print("kernels selected (execType x count):")
    for name, n in r["by_exec"].most_common(30):
        mark = "  <-- REFERENCE" if any(m in name.lower() for m in REF_MARKERS) else ""
        print(f"  {n:6d}  {name}{mark}")

    print()
    print("runtime precision x count:")
    for name, n in r["by_prec"].most_common(10):
        print(f"  {n:6d}  {name}")

    unrep = r.get("unreported", [])
    if unrep:
        seen = collections.Counter(unrep)
        print()
        print(f"hot ops with UNREPORTED implementation type ({len(unrep)} nodes):")
        for (t, e), n in seen.most_common(10):
            print(f"  {n:4d}x  {t:32s} -> {e}")
        print("  These are NOT reference fallbacks. The CPU plugin's modern")
        print("  FullyConnected/SDPA executors bypass impl_desc_type tagging and")
        print("  report 'undef'. Confirm via timing, not via this field.")

    hot = [f for f in r["fallbacks"] if f[2] == "HOT"]
    cold = [f for f in r["fallbacks"] if f[2] == "cold"]

    print()
    if not r["fallbacks"]:
        print("RESULT: no reference (ref_*) kernels. Nothing is on a known-slow path.")
        return 0

    print(f"RESULT: {len(r['fallbacks'])} node(s) on reference kernels "
          f"({len(hot)} hot, {len(cold)} cold)")
    if hot:
        print()
        print("  HOT fallbacks -- these dominate decode time and must be fixed:")
        seen = collections.Counter((t, e) for t, e, _ in hot)
        for (t, e), n in seen.most_common(20):
            print(f"    {n:4d}x  {t:32s} -> {e}")
        print()
        print("  Typical causes: an unsupported precision for this device, a")
        print("  layout the optimised kernel rejects, or a dynamic shape the")
        print("  tiled kernel cannot handle. Check the plugin's supported")
        print("  precisions before assuming this is a bug.")
    if cold and not hot:
        print("  All fallbacks are on cold ops; performance impact is negligible.")
    return 1 if hot else 0


def main() -> int:
    from bench import devices as devmod

    ap = argparse.ArgumentParser(description="Show which kernels the plugin selected.")
    ap.add_argument("--model", required=True, help="exported model dir")
    ap.add_argument("--device", default="CPU", help="CPU / GPU / NPU (kind or exact name)")
    ap.add_argument("--fp16", action="store_true",
                    help="request INFERENCE_PRECISION_HINT=f16")
    ap.add_argument("--static", action="store_true",
                    help="bind dynamic dims to static sizes (automatic for NPU)")
    ap.add_argument("--dynamic", action="store_true",
                    help="force dynamic shapes even on NPU (will fail; for diagnosis)")
    ap.add_argument("--seq", type=int, default=1,
                    help="sequence length when reshaping (1 = decode step)")
    ap.add_argument("--past", type=int, default=1024,
                    help="KV-cache length when reshaping")
    args = ap.parse_args()

    dev = devmod.pick(args.device.split(".")[0]) or args.device
    props: dict = {}
    if args.fp16:
        props["INFERENCE_PRECISION_HINT"] = "f16"

    kind = dev.split(".")[0]
    if not devmod.supports_fp16(kind):
        print(f"NOTE: {kind} does not advertise FP16 in OPTIMIZATION_CAPABILITIES.")
        print("      FP16 weights will be upconverted to FP32 at runtime, which")
        print("      doubles bytes-per-token on a bandwidth-bound decode.")
        print()

    static = True if args.static else (False if args.dynamic else None)
    return report(inspect(args.model, dev, props, static=static,
                          seq=args.seq, past=args.past))


if __name__ == "__main__":
    sys.exit(main())
