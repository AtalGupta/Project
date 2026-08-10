"""Raw decode-step measurement on CPU. Numbers only.

Times a single decode step (seq=1) directly through compile_model + InferRequest,
with no text generation and no GenAI pipeline in the way. That isolates the thing
we can actually control -- how the graph was compiled -- from tokenizer, sampling
and pipeline overhead.

Configurations measured:
    dynamic   shapes left dynamic       (Snippets does NOT tokenize)
    static    shapes bound              (Snippets tokenizes -> 84 Subgraphs)
    +any extra model dirs passed with --also (e.g. the split model)

    python -u -m bench.measure --model models/qwen2.5-1.5b-fp16
    python -u -m bench.measure --model models/qwen2.5-1.5b-fp16 \
        --also models/qwen2.5-1.5b-fp16-split --iters 30
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time


def _feed(compiled, seq: int, past: int):
    """Concrete inputs for one decode step.

    get_shape() raises on a dynamically-shaped port, which is exactly the case
    we want to measure, so build from the partial shape and fill dynamic dims
    with the same role-based values make_static() uses.
    """
    import numpy as np

    out = {}
    for port in compiled.inputs:
        name = port.get_any_name()
        low = name.lower()
        ps = port.get_partial_shape()
        dims, dyn = [], 0
        for d in ps:
            if d.is_static:
                dims.append(d.get_length())
                continue
            if "beam_idx" in low:
                dims.append(1)
            elif "past_key_value" in low or "present" in low:
                dims.append(1 if dyn == 0 else past)
            elif "attention_mask" in low:
                dims.append(1 if dyn == 0 else past + seq)
            else:
                dims.append(1 if dyn == 0 else seq)
            dyn += 1
        arr = np.zeros(dims, dtype=port.get_element_type().to_dtype())
        if "attention_mask" in low:
            arr[:] = 1
        out[name] = arr
    return out


def run(model_dir: str, device: str, static: bool, seq: int, past: int,
        iters: int, warmup: int, props: dict | None = None) -> dict:
    from bench.devices import core
    from bench.kernels import make_static

    xml = os.path.join(model_dir, "openvino_model.xml")
    if not os.path.isfile(xml):
        return {"error": f"no openvino_model.xml in {model_dir}"}

    c = core()
    model = c.read_model(xml)
    if static:
        make_static(model, seq=seq, past=past)

    t0 = time.perf_counter()
    compiled = c.compile_model(model, device, props or {})
    compile_s = time.perf_counter() - t0

    # Executable-graph node count: the direct measure of how much fusion happened.
    try:
        rt = compiled.get_runtime_model()
        nodes = len(rt.get_ordered_ops())
        subgraphs = sum(1 for n in rt.get_ordered_ops()
                        if "Subgraph" in n.get_rt_info()["layerType"].value
                        if "layerType" in n.get_rt_info())
        del rt
    except Exception:
        nodes, subgraphs = -1, -1

    req = compiled.create_infer_request()
    feed = _feed(compiled, seq, past)

    for _ in range(warmup):
        req.infer(feed)

    times = []
    for _ in range(iters):
        t = time.perf_counter()
        req.infer(feed)
        times.append((time.perf_counter() - t) * 1000.0)

    del req, compiled, model
    gc.collect()

    med = statistics.median(times)
    return {"ms_median": med, "ms_mean": statistics.fmean(times),
            "ms_min": min(times), "ms_p90": sorted(times)[int(len(times) * 0.9) - 1],
            "sd": statistics.stdev(times) if len(times) > 1 else 0.0,
            "tok_s": 1000.0 / med if med else 0.0,
            "compile_s": compile_s, "nodes": nodes, "subgraphs": subgraphs}


def main() -> int:
    ap = argparse.ArgumentParser(description="Raw decode-step timing on CPU.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--also", nargs="*", default=[], help="extra model dirs")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--past", type=int, default=256)
    ap.add_argument("--csv", help="append results here")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import pick

    dev = pick(args.device.split(".")[0]) or args.device

    jobs = []
    for m in [args.model] + list(args.also):
        name = os.path.basename(m.rstrip("/\\"))
        jobs.append((f"{name} dynamic", m, False))
        jobs.append((f"{name} static", m, True))

    print(f"device {dev}   seq={args.seq} past={args.past}   "
          f"{args.warmup} warmup + {args.iters} iters, median reported")
    print()
    hdr = (f"{'config':<38}{'ms':>9}{'sd':>7}{'min':>9}{'p90':>9}"
           f"{'tok/s':>9}{'nodes':>8}{'subgr':>7}{'cmp s':>8}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for label, path, static in jobs:
        r = run(path, dev, static, args.seq, args.past, args.iters, args.warmup)
        if "error" in r:
            print(f"{label:<38} {r['error']}")
            continue
        rows.append((label, r))
        print(f"{label:<38}{r['ms_median']:>9.2f}{r['sd']:>7.2f}"
              f"{r['ms_min']:>9.2f}{r['ms_p90']:>9.2f}{r['tok_s']:>9.2f}"
              f"{r['nodes']:>8}{r['subgraphs']:>7}{r['compile_s']:>8.1f}")

    if len(rows) >= 2:
        best = min(rows, key=lambda kv: kv[1]["ms_median"])
        worst = max(rows, key=lambda kv: kv[1]["ms_median"])
        print()
        print(f"fastest: {best[0]}  {best[1]['ms_median']:.2f} ms "
              f"({best[1]['tok_s']:.2f} tok/s)")
        if worst[1]["ms_median"] > 0:
            print(f"slowest: {worst[0]}  {worst[1]['ms_median']:.2f} ms "
                  f"({worst[1]['ms_median'] / best[1]['ms_median']:.2f}x slower)")

    if args.csv:
        import csv as _csv
        new = not os.path.isfile(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if new:
                w.writerow(["config", "device", "ms_median", "sd", "ms_min",
                            "ms_p90", "tok_s", "nodes", "subgraphs", "compile_s"])
            for label, r in rows:
                w.writerow([label, dev, r["ms_median"], r["sd"], r["ms_min"],
                            r["ms_p90"], r["tok_s"], r["nodes"], r["subgraphs"],
                            r["compile_s"]])
        print(f"\nappended {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

