"""Dump graphs for visual inspection: IR, and the compiled graph per device.

Three graphs matter and they are genuinely different objects:

  1. THE IR              what optimum-cli exported. 4248 nodes of MatMul,
                         Multiply, Add, ScaledDotProductAttention. This is the
                         model as ONNX/PyTorch describes it.
  2. THE EXECUTABLE      what the plugin actually runs after ~150 transformation
     GRAPH (per device)  passes. 1334 nodes on CPU. FullyConnected and RMS and
                         RoPE live here and exist nowhere in the IR -- they were
                         manufactured by fusion.
  3. THE SPLIT           the IR with HETERO affinity tags, so you can see where
                         the attention/FFN boundary falls.

Comparing 1 and 2 side by side is the clearest way to see what OpenVINO's graph
compiler did for you.

Output is OpenVINO IR XML, which Netron renders directly -- open
https://netron.app and drag the .xml in (the .bin beside it is picked up
automatically). Netron is the practical choice because it needs no install and
handles graphs this size; Graphviz .dot is also emitted when available but a
1300-node dot file is rarely readable.

    python -u -m bench.visualize --model models/qwen2.5-1.5b-fp16 --devices CPU NPU
    python -u -m bench.visualize --model models/qwen2.5-1.5b-fp16-split --devices CPU
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save(model, path: str, label: str) -> None:
    from openvino import save_model

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_model(model, path)
    size = os.path.getsize(path) / 1e6
    nodes = len(model.get_ordered_ops())
    print(f"  {label:<22} {nodes:5d} nodes  ->  {path}  ({size:.1f} MB xml)")


def dot(model, path: str) -> bool:
    """Graphviz .dot via VisualizeTree, if this build exposes the pass."""
    try:
        from openvino.passes import Manager, VisualizeTree
    except Exception:
        try:
            from openvino.runtime.passes import Manager, VisualizeTree
        except Exception:
            return False
    try:
        m = Manager()
        m.register_pass(VisualizeTree(path))
        m.run_passes(model)
        return os.path.isfile(path)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump IR and per-device graphs.")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-fp16")
    ap.add_argument("--devices", nargs="*", default=["CPU"])
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs"))
    ap.add_argument("--static", action="store_true",
                    help="bind shapes first (needed for NPU; changes CPU fusions)")
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--past", type=int, default=256)
    ap.add_argument("--dot", action="store_true", help="also emit Graphviz .dot")
    ap.add_argument("--dump-ir", action="store_true",
                    help="write a copy of the IR too (includes multi-GB weights)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import core, kinds_present, pick
    from bench.kernels import make_static

    xml = os.path.join(args.model, "openvino_model.xml")
    if not os.path.isfile(xml):
        print(f"no openvino_model.xml in {args.model}")
        return 1

    tag = os.path.basename(args.model.rstrip("/\\"))
    os.makedirs(args.out, exist_ok=True)
    c = core()

    print(f"model : {args.model}")
    print(f"out   : {args.out}")
    print()

    # ---- 1. the IR as exported ----
    # Do NOT copy it by default: save_model writes the weights too, which for
    # this model is a needless 2.9 GB duplicate. Netron opens the original file
    # perfectly well. Only re-save when the graph has actually been changed
    # (--static) or the user explicitly asks.
    if args.dump_ir or args.static:
        model = c.read_model(xml)
        if args.static:
            make_static(model, seq=args.seq, past=args.past)
        ir_path = os.path.join(args.out, f"{tag}__1_ir.xml")
        print("  (writing a full IR copy including weights - several GB)")
        save(model, ir_path, "IR (reshaped)" if args.static else "IR (copy)")
        if args.dot:
            d = os.path.join(args.out, f"{tag}__1_ir.dot")
            print(f"  {'dot':<22} "
                  f"{'written' if dot(model, d) else 'unavailable'}: {d}")
        del model
        gc.collect()
    else:
        n = len(c.read_model(xml).get_ordered_ops())
        print(f"  {'IR (as exported)':<22} {n:5d} nodes  ->  {xml}")
        print(f"  {'':<22} (open this file directly; not copied)")

    # ---- 2. the executable graph, per device ----
    available = kinds_present()
    for kind in args.devices:
        k = kind.split(".")[0].upper()
        if k not in available:
            print(f"\n  skip {kind}: not available (have {sorted(available)})")
            continue
        dev = pick(k) or kind

        m2 = c.read_model(xml)
        needs_static = args.static or k == "NPU"
        if needs_static:
            make_static(m2, seq=args.seq, past=args.past)

        print(f"\n  compiling on {dev}{' (static shapes)' if needs_static else ''} ...")
        try:
            compiled = c.compile_model(m2, dev, {})
            rt = compiled.get_runtime_model()
            p = os.path.join(args.out, f"{tag}__2_exec_{k.lower()}.xml")
            save(rt, p, f"executable ({dev})")
            if args.dot:
                d = os.path.join(args.out, f"{tag}__2_exec_{k.lower()}.dot")
                print(f"  {'dot':<22} "
                      f"{'written' if dot(rt, d) else 'unavailable'}: {d}")
            del rt, compiled
        except Exception as e:
            # NPU + stateful IR is the known one: the KV cache lives in
            # ReadValue/Assign variables that model.reshape() cannot bind, so raw
            # compile_model refuses. That is a tooling limit, not a device fault.
            print(f"  FAILED on {dev}: {type(e).__name__}: {str(e)[:140]}")
        del m2
        gc.collect()

    print()
    print("=" * 74)
    print("HOW TO VIEW")
    print("=" * 74)
    print("  1. open https://netron.app  (nothing to install)")
    print("  2. drag in any .xml from the folder above")
    print("  3. the matching .bin beside it is loaded automatically")
    print()
    print("  What to look for, comparing __1_ir against __2_exec_*:")
    print("    - MatMul in the IR becomes FullyConnected in the executable graph")
    print("    - long pow/mean/sqrt/div/mul chains collapse into a single RMS node")
    print("    - sin/cos/slice/concat chains collapse into a single RoPE node")
    print("    - with static shapes, Subgraph nodes appear (Snippets JIT)")
    print()
    print("  Netron on a 1300-node graph: use the search box (top-left) for a")
    print("  node type rather than scrolling; click a node to see its precision,")
    print("  layout and the original layer names it was fused from.")
    print()
    print("  For the split model, open the IR and click any node: rt_info carries")
    print("  an `affinity` field naming the device that node was assigned to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
