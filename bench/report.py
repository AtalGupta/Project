"""Turn a sweep CSV into a readable table.

The column that matters most is `%roof`: measured tok/s as a fraction of the
memory-bandwidth ceiling for this model on this machine. Raw tok/s tells you what
happened; %roof tells you whether anything can be done about it.

  %roof near 100   you are bandwidth-bound. No kernel change will help. Shrink
                   the model (quantise) or reduce bytes-per-token (speculative
                   decoding) -- those are the only levers left.
  %roof low        something is leaving performance on the table: a reference
                   kernel (check bench/kernels.py), a bad layout, sync overhead,
                   or a device that simply cannot reach its own bandwidth.

Usage:
    python -m bench.report results/sweep-*.csv [--roofline results/roofline.json]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os


def _f(row: dict, key: str):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def load_peak_bandwidth(path: str | None):
    """Achieved GB/s from a roofline JSON, or None."""
    if path and os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f).get("peak_gb_per_s")
        except Exception:
            pass
    return None


def model_sizes(root: str) -> dict[str, int]:
    """Bytes per exported variant, keyed by directory name.

    Needed because fp32 and fp16 have DIFFERENT ceilings on the same machine --
    the bandwidth is shared but the bytes-per-token are not. Applying one
    ceiling to both would understate fp16 and overstate fp32, which is exactly
    the confusion this column exists to prevent.
    """
    out = {}
    mdir = os.path.join(root, "models")
    if not os.path.isdir(mdir):
        return out
    for name in os.listdir(mdir):
        p = os.path.join(mdir, name)
        if os.path.isfile(os.path.join(p, "openvino_model.xml")):
            total = 0
            for r, _, files in os.walk(p):
                for f in files:
                    if f.endswith((".bin", ".xml")):
                        total += os.path.getsize(os.path.join(r, f))
            out[name] = total
    return out


def row_ceiling(row: dict, peak_gb_s, sizes: dict[str, int]):
    """Per-row ceiling, derived from the variant that row actually ran."""
    if not peak_gb_s:
        return None
    try:
        cfg = json.loads(row.get("config") or "{}")
    except Exception:
        return None
    key = cfg.get("model_key")
    b = sizes.get(key)
    if not b:
        return None
    return peak_gb_s * 1e9 / b


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a sweep CSV.")
    ap.add_argument("csv", nargs="?", help="sweep CSV (default: newest in results/)")
    ap.add_argument("--roofline", help="roofline JSON from bench.roofline")
    ap.add_argument("--model", help="model dir, to derive the ceiling if needed")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.csv
    if not path:
        found = sorted(glob.glob(os.path.join(root, "results", "sweep-*.csv")))
        if not found:
            print("no sweep CSV found; run: python -m bench.run")
            return 1
        path = found[-1]

    rf_path = args.roofline or os.path.join(root, "results", "roofline.json")
    peak = load_peak_bandwidth(rf_path)
    sizes = model_sizes(root)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"source    : {path}")
    if peak:
        print(f"bandwidth : {peak:.1f} GB/s measured")
        for k in sorted(sizes):
            print(f"  {k:28s} {sizes[k] / 1e9:5.2f} GB "
                  f"-> ceiling {peak * 1e9 / sizes[k]:6.1f} tok/s")
    else:
        print("bandwidth : not measured "
              "(run bench/roofline.py --json results/roofline.json)")
    print()

    ok = [r for r in rows if str(r.get("ok", "")).lower() == "true"]
    bad = [r for r in rows if str(r.get("ok", "")).lower() != "true"]

    def sort_key(r):
        v = _f(r, "aggregate_tok_s_mean") or _f(r, "throughput_tok_s_mean") or 0.0
        return -v

    ok.sort(key=sort_key)

    hdr = (f"{'strategy':<14} {'label':<34} {'tok/s':>9} {'+/-':>7} "
           f"{'%roof':>6} {'TTFT ms':>9} {'TPOT ms':>8} {'accept':>7} {'n':>3}")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        tp = _f(r, "throughput_tok_s_mean")
        agg = _f(r, "aggregate_tok_s_mean")
        shown = agg if agg else tp
        sd = _f(r, "aggregate_tok_s_stdev") if agg else _f(r, "throughput_tok_s_stdev")
        ceiling = row_ceiling(r, peak, sizes)
        # Prefill rows are compute-bound, so a memory-bandwidth ceiling does not
        # apply to them -- showing one would invite a meaningless comparison.
        is_prefill = r.get("strategy") == "prefill"
        pct = (100.0 * shown / ceiling) if (ceiling and shown and not is_prefill) else None
        acc = _f(r, "acceptance_rate_mean")
        ttft = _f(r, "ttft_ms_mean")
        tpot = _f(r, "tpot_ms_mean")
        tag = "*" if agg else " "
        print(f"{r.get('strategy', ''):<14} {r.get('label', '')[:34]:<34} "
              f"{shown:>8.2f}{tag} {sd if sd is not None else 0:>7.2f} "
              f"{(f'{pct:.0f}%' if pct is not None else '-'):>6} "
              f"{(f'{ttft:.0f}' if ttft else '-'):>9} "
              f"{(f'{tpot:.1f}' if tpot else '-'):>8} "
              f"{(f'{acc:.2f}' if acc is not None else '-'):>7} "
              f"{r.get('n', ''):>3}")

    if any(_f(r, "aggregate_tok_s_mean") for r in ok):
        print()
        print("* aggregate across concurrent streams, not single-stream latency.")
        print("  Higher aggregate with lower per-stream tok/s is the expected shape:")
        print("  total system throughput rises while each conversation gets slower.")

    if bad:
        print()
        print(f"failed / skipped ({len(bad)}):")
        for r in bad:
            print(f"  {r.get('strategy', ''):<14} {r.get('config', '')[:60]}")
            print(f"      {str(r.get('error', ''))[:200]}")
        print()
        print("  Failures here are data, not necessarily bugs: HETERO and NPU-FP16")
        print("  are expected to fail on some configurations, and knowing exactly")
        print("  how they fail is part of the result.")

    decode = [r for r in ok if r.get("strategy") in ("single", "cpu_threads")]
    if peak and decode:
        best = decode[0]
        v = _f(best, "throughput_tok_s_mean")
        c = row_ceiling(best, peak, sizes)
        if v and c:
            pct = 100.0 * v / c
            print()
            print(f"Best decode config: {best.get('label', '')} at {pct:.0f}% of its "
                  f"bandwidth ceiling.")
            if pct > 80:
                print("That is memory-bound. The silicon is delivering essentially all the")
                print("bandwidth it has; kernel tuning cannot move it. The only remaining")
                print("levers change the workload rather than the machine.")
            elif pct < 40:
                print("There is real headroom here -- the machine is not bandwidth-limited")
                print("at this point. Run bench/kernels.py to check for reference-kernel")
                print("fallbacks before concluding anything about the hardware.")

    prefill = [r for r in ok if r.get("strategy") == "prefill"]
    if prefill:
        print()
        print("Prefill rows measure the COMPUTE-bound regime (prompt tokens/s) and are")
        print("not subject to the decode bandwidth ceiling. This is where a high-TOPS")
        print("device separates from a low-TOPS one; on decode they look alike because")
        print("both are waiting on DRAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
