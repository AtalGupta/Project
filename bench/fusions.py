"""Which graph optimizations actually fired?

OpenVINO ships ~150 transformation passes. The catalog tells you what exists; it
does not tell you what ran on YOUR model on YOUR device. This diffs the IR graph
against the compiled executable graph and attributes the difference to named
passes.

The evidence is real: node types that exist only after compilation (FullyConnected,
ScaledDotProductAttentionWithKVCache, Subgraph) were manufactured by a pass, and
types that vanish were consumed by one.

IMPORTANT ON ATTRIBUTION: the compiler does not report which passes it ran. The
pass names below are INFERRED from node-type signatures -- each pass has a known
input type it consumes and output type it produces, so a disappearance/appearance
pair identifies it. That inference is strong for the distinctive LLM fusions
(nothing else produces a `ScaledDotProductAttentionWithKVCache`) and weaker for
generic cleanup, where many passes could be responsible for the same collapse.
The report labels confidence accordingly rather than presenting all of it as
equally certain.

    python -u -m bench.fusions --model models/qwen2.5-1.5b-fp16 --device CPU
    python -u -m bench.fusions --model models/qwen2.5-1.5b-fp16 --compare CPU GPU
"""

from __future__ import annotations

import argparse
import collections
import gc
import os
import sys

# The CPU plugin RENAMES ops as it lowers them -- it does not only fuse. A
# renamed type appears "created" and its IR original appears "consumed", which
# would be reported as a fusion when nothing was fused at all. Normalise these
# before diffing, or the report invents optimizations that never happened.
#
# Many-to-one entries (Eltwise) are genuine fusion targets AND renames at once:
# the plugin folds a chain of elementwise ops into one Eltwise node, so the count
# drop is real even though the rename is not.
RENAMES = {
    "Constant": "Const",
    "Parameter": "Input",
    "Result": "Output",
    "ReadValue": "MemoryInput",     # stateful KV-cache read
    "Assign": "MemoryOutput",       # stateful KV-cache write
    "Concat": "Concatenation",
    "ReduceMean": "Reduce",
    "ReduceSum": "Reduce",
}

# IR types the CPU plugin collapses into its generic `Eltwise` node.
ELTWISE_SOURCES = {
    "Multiply", "Add", "Subtract", "Divide", "Power", "Sqrt", "Swish",
    "Relu", "Sigmoid", "Tanh", "Clamp", "PRelu", "Exp", "Negative", "Maximum",
    "Minimum", "SquaredDifference", "Erf",
}

# (node type produced) -> (pass name, layer, confidence, what it did)
#
# "strong"  the type is distinctive; essentially nothing else produces it
# "weak"    several passes could account for it, or the type also exists in IR
PRODUCED_BY = {
    "FullyConnected": (
        "convert_matmul_to_fc (+ fc_bias_fusion)", "CPU/GPU plugin", "strong",
        "MatMul with constant weights -> FullyConnected, bias folded in"),
    "ScaledDotProductAttentionWithKVCache": (
        "stateful_sdpa_fusion", "CPU plugin", "strong",
        "SDPA + KV-cache ReadValue/Assign -> one stateful node"),
    "PagedAttention": (
        "paged attention transformation", "common", "strong",
        "attention + paged KV cache -> one node"),
    "Subgraph": (
        "Snippets tokenization", "Snippets JIT", "strong",
        "fusible region collapsed into a JIT-compiled subgraph"),
    "RoPE": (
        "fuse_rotary_positional_embeddings", "common", "strong",
        "sin/cos/slice/concat chain -> one RoPE node"),
    "RMS": (
        "rms_fusion", "common", "strong",
        "pow/mean/sqrt/div/mul chain -> one RMS node"),
    "SwiGLU": (
        "glu_fusion / swiglu_fusion_with_clamp", "common / GPU plugin", "strong",
        "gate * SiLU(up) -> one SwiGLU node (FFN gate)"),
    "GLU": (
        "glu_fusion", "common", "strong", "gated linear unit fused"),
    "MVN": (
        "mvn_fusion", "common", "strong", "mean-variance normalization fused"),
    "Gelu": (
        "gelu_fusion", "common", "weak", "GELU pattern fused"),
    "KVCache": (
        "kv_cache_fusion", "GPU plugin", "strong",
        "KV-cache read/concat/write -> one node"),
    "IndirectKVCache": (
        "indirect_kv_cache", "GPU plugin", "strong", "indirect KV-cache access"),
    "FullyConnectedCompressed": (
        "convert_fc_to_compressed", "GPU plugin", "strong",
        "FC with compressed weights"),
    "Placeholder": (
        "graph plumbing", "plugin", "weak", "internal placeholder"),
    "Eltwise": (
        "eltwise chain fusion", "CPU plugin", "strong",
        "a chain of elementwise ops collapsed into one node"),
    "Reorder": (
        "layout conversion insertion", "CPU plugin", "strong",
        "INSERTED, not fused - converts between memory layouts a kernel needs"),
}

# Node types whose DISAPPEARANCE is informative.
CONSUMED_BY = {
    "MatMul": "convert_matmul_to_fc - consumed into FullyConnected",
    "ScaledDotProductAttention": "sdpa_fusion / stateful_sdpa_fusion",
    "Multiply": "lin_op_sequence_fusion, glu_fusion, or folded into a GEMM",
    "Add": "fc_bias_fusion (bias into FullyConnected) or eltwise fusion",
    "Convert": "compress_float_constants / precision folding",
    "Reshape": "sink_reshape, optimize_subsequent_reshapes, nop_elimination",
    "Transpose": "transpose_fusion, matmul_const_transposes_extraction",
    "Constant": "folded into the consuming op's weights",
    "Broadcast": "broadcast_elementwise_fusion",
    "Concat": "concat_fusion or kv_cache_fusion",
    "Squeeze": "nop_elimination",
    "Unsqueeze": "eliminate_unsqueeze_gather / nop_elimination",
    "Slice": "optimize_strided_slice",
    "StridedSlice": "optimize_strided_slice",
    "Power": "convert_to_power_static / nop_elimination",
    "Softmax": "sdpa_fusion (absorbed into attention)",
    "ShapeOf": "simplify_shape_of_sub_graph",
}

# Fusions we know OpenVINO offers whose output type we can look for. If the type
# is absent from the executable graph, the pass did not fire on this model.
# Each entry lists ALL node type names that count as evidence the pass fired.
#
# Multiple names are needed because the two OpenVINO data sources disagree:
# get_runtime_model() reports `layerType`, while InferRequest.profiling_info
# reports the internal primitive's own name. Stateful attention shows up as
# `ScaledDotProductAttention` in the first and
# `ScaledDotProductAttentionWithKVCache` in the second -- same node, two names.
# Checking only one produced a false "did not fire" for a pass we had already
# observed running.
EXPECTED_FUSIONS = [
    (["RoPE"], "fuse_rotary_positional_embeddings", "common",
     "rotary position embedding"),
    (["RMS", "RMSNorm"], "rms_fusion", "common", "RMS normalization"),
    (["ScaledDotProductAttentionWithKVCache", "ScaledDotProductAttention", "SDPA"],
     "sdpa_fusion / stateful_sdpa_fusion", "common + CPU plugin",
     "attention fused into one node (KV cache folded in when stateful)"),
    (["SwiGLU", "GLU"], "glu_fusion / swiglu_fusion_with_clamp", "common / GPU",
     "FFN gate (gate * SiLU(up))"),
    (["Subgraph"], "Snippets tokenization", "Snippets", "JIT-compiled subgraph"),
    (["FullyConnected", "FullyConnectedCompressed"], "convert_matmul_to_fc",
     "CPU/GPU plugin", "MatMul -> FullyConnected"),
    (["KVCache", "IndirectKVCache"], "kv_cache_fusion", "GPU plugin",
     "KV cache as one node"),
    (["PagedAttention"], "paged attention", "common", "paged KV cache"),
]


def _rt(node, key: str, default: str = "") -> str:
    """rt_info values are OVAny; str() gives '<OVAny class>', .value gives data."""
    try:
        info = node.get_rt_info()
        if key in info:
            v = info[key]
            return str(getattr(v, "value", v))
    except Exception:
        pass
    return default


def ir_histogram(model) -> tuple[collections.Counter, collections.Counter]:
    """(raw IR histogram, histogram normalised to executable-graph naming).

    The normalised one is what gets diffed, so plugin renames are not mistaken
    for fusions.
    """
    raw = collections.Counter(n.get_type_name() for n in model.get_ordered_ops())
    norm: collections.Counter = collections.Counter()
    for t, n in raw.items():
        if t in RENAMES:
            norm[RENAMES[t]] += n
        elif t in ELTWISE_SOURCES:
            norm["Eltwise"] += n
        else:
            norm[t] += n
    return raw, norm


def exec_histogram(compiled) -> tuple[collections.Counter, collections.Counter]:
    """(layerType histogram, primitiveType histogram) from the executable graph."""
    rt = compiled.get_runtime_model()
    types: collections.Counter = collections.Counter()
    prims: collections.Counter = collections.Counter()
    for node in rt.get_ordered_ops():
        types[_rt(node, "layerType", node.get_type_name())] += 1
        prims[_rt(node, "primitiveType", "?")] += 1
    del rt
    return types, prims


def analyse(model_dir: str, device: str, static: bool = False,
            seq: int = 1, past: int = 256) -> dict:
    from bench.devices import core
    from bench.kernels import make_static

    xml = os.path.join(model_dir, "openvino_model.xml")
    if not os.path.isfile(xml):
        raise SystemExit(f"no openvino_model.xml in {model_dir}")

    c = core()
    model = c.read_model(xml)

    # Stateful IR + NPU cannot be compiled through raw compile_model -- the KV
    # cache lives in ReadValue/Assign variables with dynamic shape that
    # model.reshape() cannot bind. Detect and refuse rather than emit a
    # compiler traceback that looks like a hardware verdict.
    n_state = 0
    for accessor in ("get_sinks", "sinks"):
        try:
            v = getattr(model, accessor)
            v = v() if callable(v) else v
            n_state = len(v or [])
            break
        except Exception:
            continue

    if device.upper().startswith("NPU"):
        if n_state:
            raise SystemExit(
                f"\nCannot inspect fusions on NPU for this model: the IR is "
                f"STATEFUL\n  ({n_state} ReadValue/Assign pairs hold the KV cache "
                f"with dynamic shape).\n\n"
                f"  Raw Core.compile_model() cannot bind those; GenAI's NPUW path\n"
                f"  can, but does not expose the executable graph. This is a\n"
                f"  tooling limit, not a statement about the NPU.\n\n"
                f"  Use --device CPU or GPU. For NPU, an export with\n"
                f"  --disable-stateful would make this inspectable, at the cost\n"
                f"  of measuring a graph nobody would deploy.\n")
        static = True

    # Some fusions are shape-dependent: a pass that needs to know a dimension
    # cannot match against a dynamic one. Binding shapes therefore changes WHICH
    # optimizations fire, not just how fast they run.
    if static:
        make_static(model, seq=seq, past=past)
        print(f"  shapes bound static: seq={seq} past={past}", flush=True)

    ir_raw, ir_norm = ir_histogram(model)
    ir_total = sum(ir_raw.values())

    print(f"  compiling on {device} ...", flush=True)
    compiled = c.compile_model(model, device, {})
    ex, prims = exec_histogram(compiled)
    ex_total = sum(ex.values())

    del compiled, model
    gc.collect()
    return {"device": device, "ir_raw": ir_raw, "ir": ir_norm,
            "ir_total": ir_total, "exec": ex, "exec_total": ex_total,
            "prims": prims, "stateful": n_state}


def report(r: dict) -> None:
    ir, ex = r["ir"], r["exec"]
    it, et = r["ir_total"], r["exec_total"]

    print()
    print("=" * 78)
    print(f"GRAPH COLLAPSE on {r['device']}")
    print("=" * 78)
    pct = 100.0 * (it - et) / it if it else 0
    print(f"  IR graph          {it:6d} nodes")
    print(f"  executable graph  {et:6d} nodes   ({it - et} removed, -{pct:.1f}%)")
    if r["stateful"]:
        print(f"  stateful          {r['stateful']} ReadValue/Assign pairs (KV cache)")

    # ---- types that only exist AFTER compilation ----
    created = {t: n for t, n in ex.items() if t not in ir and n > 0}
    print()
    print("FIRED - node types created by compilation (did not exist in the IR)")
    print("-" * 78)
    if not created:
        print("  none")
    for t, n in sorted(created.items(), key=lambda kv: -kv[1]):
        info = PRODUCED_BY.get(t)
        if info:
            name, layer, conf, what = info
            print(f"  {n:5d}x  {t}")
            print(f"          <- {name}  [{layer}]  ({conf} attribution)")
            print(f"             {what}")
        else:
            print(f"  {n:5d}x  {t}")
            print(f"          <- unrecognised type; not in the PRODUCED_BY table")

    # ---- types that vanished ----
    removed = {t: n for t, n in ir.items() if ex.get(t, 0) == 0 and n > 0}
    print()
    print("CONSUMED - node types present in the IR, absent after compilation")
    print("-" * 78)
    if not removed:
        print("  none")
    for t, n in sorted(removed.items(), key=lambda kv: -kv[1])[:14]:
        why = CONSUMED_BY.get(t, "fused or eliminated (multiple passes possible)")
        print(f"  {n:5d}x  {t:<28} {why}")

    # ---- types that shrank a lot but survived ----
    shrank = [(t, ir[t], ex.get(t, 0)) for t in ir
              if ex.get(t, 0) and ir[t] - ex.get(t, 0) > 20]
    if shrank:
        print()
        print("REDUCED - survived, but far fewer")
        print("-" * 78)
        for t, a, b in sorted(shrank, key=lambda x: -(x[1] - x[2]))[:10]:
            why = CONSUMED_BY.get(t, "partially fused")
            print(f"  {t:<28} {a:5d} -> {b:<5d} ({b - a:+d})  {why}")

    # ---- fusions that did NOT fire ----
    print()
    print("DID NOT FIRE - offered by OpenVINO, but no output node present here")
    print("-" * 78)
    fired, missing = [], []
    for names, p, layer, what in EXPECTED_FUSIONS:
        hit = [(t, ex[t]) for t in names if ex.get(t, 0) > 0]
        (fired if hit else missing).append((names, p, layer, what, hit))

    if not missing:
        print("  every tracked fusion produced output")
    for names, p, layer, what, _ in missing:
        print(f"  {p:<42} [{layer}]")
        print(f"    would produce `{names[0]}` - {what}")

    if fired:
        print()
        print("CONFIRMED FIRED - tracked fusions with output present")
        print("-" * 78)
        for names, p, layer, what, hit in fired:
            got = ", ".join(f"{t} x{n}" for t, n in hit)
            print(f"  {p:<42} [{layer}]")
            print(f"    {got}")

    print()
    print("Attribution is INFERRED from node-type signatures, not reported by the")
    print("compiler. 'strong' means the type is distinctive; 'weak' means several")
    print("passes could account for it.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Show which graph optimizations fired on this model/device.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--compare", nargs="*", metavar="DEV",
                    help="compare fusion depth across devices, e.g. --compare CPU GPU")
    ap.add_argument("--static", action="store_true",
                    help="bind shapes before compiling (some fusions need this)")
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--past", type=int, default=256)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import kinds_present, pick

    if args.compare:
        results = []
        for d in args.compare:
            if d.upper() not in kinds_present():
                print(f"skip {d}: not available on this machine")
                continue
            dev = pick(d) or d
            try:
                r = analyse(args.model, dev, args.static, args.seq, args.past)
                results.append(r)
                report(r)
            except SystemExit as e:
                print(e)
        if len(results) >= 2:
            print()
            print("=" * 78)
            print("FUSION DEPTH COMPARISON")
            print("=" * 78)
            base = results[0]["ir_total"]
            print(f"  {'device':<12}{'exec nodes':>12}{'collapse':>11}"
                  f"{'vs IR':>10}")
            for r in results:
                pct = 100.0 * (base - r["exec_total"]) / base
                print(f"  {r['device']:<12}{r['exec_total']:>12}"
                      f"{pct:>10.1f}%{base:>10}")
            deepest = min(results, key=lambda r: r["exec_total"])
            shallow = max(results, key=lambda r: r["exec_total"])
            if deepest is not shallow:
                print()
                print(f"  {deepest['device']} fuses more aggressively than "
                      f"{shallow['device']}: "
                      f"{shallow['exec_total'] - deepest['exec_total']} fewer nodes.")
                only = {t for t in deepest["exec"] if t not in shallow["exec"]}
                if only:
                    print(f"  Types only on {deepest['device']}: "
                          f"{', '.join(sorted(only)[:8])}")
        return 0

    dev = pick(args.device.split(".")[0]) or args.device
    try:
        report(analyse(args.model, dev, args.static, args.seq, args.past))
    except SystemExit as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
