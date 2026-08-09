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


def load_roofline(path: str | None, model_dir: str | None):
    """Return (tok_s_ceiling, description) or (None, why-not)."""
    if path and os.path.isfile(path):
        with open(path) as f:
            d = json.load(f)
        if "tok_s_ceiling" in d:
            return d["tok_s_ceiling"], (
                f"{d.get('peak_gb_per_s', 0):.0f} GB/s over "
                f"{d.get('model_bytes', 0) / 1e9:.2f} GB")
        if "peak_gb_per_s" in d and model_dir:
            from bench.roofline import dir_bytes, token_ceiling
            mb = dir_bytes(model_dir)
            if mb:
                return token_ceiling(mb, d["peak_gb_per_s"]), (
                    f"{d['peak_gb_per_s']:.0f} GB/s over {mb / 1e9:.2f} GB")
    return None, "no roofline measured (run bench/roofline.py --model ... --json ...)"


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
    ceiling, ceil_desc = load_roofline(rf_path, args.model)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"source  : {path}")
    print(f"ceiling : "
          + (f"{ceiling:.1f} tok/s  ({ceil_desc})" if ceiling else ceil_desc))
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
        pct = (100.0 * shown / ceiling) if (ceiling and shown) else None
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

    if ceiling and ok:
        best = ok[0]
        v = _f(best, "aggregate_tok_s_mean") or _f(best, "throughput_tok_s_mean")
        if v:
            pct = 100.0 * v / ceiling
            print()
            if pct > 80:
                print(f"Best config is at {pct:.0f}% of the bandwidth ceiling: this workload")
                print("is memory-bound. Kernel tuning will not move it meaningfully --")
                print("reduce bytes per token instead (quantisation, speculative decoding).")
            elif pct < 40:
                print(f"Best config is only at {pct:.0f}% of the bandwidth ceiling: there is")
                print("real headroom. Run bench/kernels.py to check for reference-kernel")
                print("fallbacks before doing anything more elaborate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
