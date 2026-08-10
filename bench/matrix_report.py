"""Render the CPU vs NPU vs SPLIT matrix and declare a winner per cell.

The headline question is not "which has the lowest TTFT" or "the lowest ITL" --
those pick different winners. It is end-to-end latency:

    e2e = TTFT + n_out * ITL

The NPU trades worse TTFT for better ITL, so the winner flips with output
length, and the crossover point is the actual finding. This prints all three
views and then the one that decides it.

    python -m bench.matrix_report results\\matrix.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(r: dict, k: str):
    v = r.get(k, "")
    try:
        return float(v) if v != "" else None
    except (TypeError, ValueError):
        return None


def table(rows, metric: str, inputs, outputs, configs, unit="ms", scale=1.0):
    idx = {(r["config"], int(r["input_len"]), int(r["output_len"])): r
           for r in rows if not r.get("error")}
    w = 11
    hdr = f"{'input':>7}{'output':>8}" + "".join(f"{c:>{w}}" for c in configs)
    print(hdr)
    print("-" * len(hdr))
    for n_in in inputs:
        for n_out in outputs:
            line = f"{n_in:>7}{n_out:>8}"
            vals = {}
            for c in configs:
                r = idx.get((c, n_in, n_out))
                v = _f(r, metric) if r else None
                vals[c] = v
                line += f"{v/scale:>{w}.1f}" if v is not None else f"{'-':>{w}}"
            good = {c: v for c, v in vals.items() if v is not None}
            if good:
                best = min(good, key=good.get)
                line += f"   <- {best}"
            print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the CPU/NPU/SPLIT matrix.")
    ap.add_argument("csv", nargs="?", default="results/matrix.csv")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not os.path.isfile(args.csv):
        print(f"no such file: {args.csv}")
        return 1
    rows = load(args.csv)
    ok = [r for r in rows if not r.get("error")]
    bad = [r for r in rows if r.get("error")]
    if not ok:
        print("no successful rows")
        return 1

    inputs = sorted({int(r["input_len"]) for r in ok})
    outputs = sorted({int(r["output_len"]) for r in ok})
    order = ["cpu", "npu", "split", "gpu", "split_gpu"]
    configs = [c for c in order if any(r["config"] == c for r in ok)]
    configs += sorted({r["config"] for r in ok} - set(configs))

    print(f"source: {args.csv}   {len(ok)} cells, {len(bad)} failed")
    print()
    print("=" * 74)
    print("TTFT (ms) - prompt processing. Lower is better.")
    print("=" * 74)
    table(ok, "ttft_ms", inputs, outputs, configs)

    print()
    print("=" * 74)
    print("ITL (ms/token) - generation. Lower is better.")
    print("=" * 74)
    table(ok, "itl_ms", inputs, outputs, configs)

    print()
    print("=" * 74)
    print("END-TO-END (seconds) = TTFT + output x ITL. THIS DECIDES THE WINNER.")
    print("=" * 74)
    table(ok, "e2e_ms", inputs, outputs, configs, scale=1000.0)

    # ---- scoreboard ----
    idx = {(r["config"], int(r["input_len"]), int(r["output_len"])): r for r in ok}
    wins: collections.Counter = collections.Counter()
    margins = []
    for n_in in inputs:
        for n_out in outputs:
            vals = {}
            for c in configs:
                v = _f(idx.get((c, n_in, n_out), {}), "e2e_ms")
                if v is not None:
                    vals[c] = v
            if len(vals) < 2:
                continue
            best = min(vals, key=vals.get)
            wins[best] += 1
            rest = sorted(v for c, v in vals.items() if c != best)
            if rest:
                margins.append((best, n_in, n_out, rest[0] / vals[best]))

    print()
    print("=" * 74)
    print("SCOREBOARD - end-to-end wins")
    print("=" * 74)
    for c, n in wins.most_common():
        print(f"  {c:<12} {n:3d} of {sum(wins.values())} cells")

    if "split" in wins:
        sw = [m for m in margins if m[0] == "split"]
        if sw:
            print()
            print(f"  split wins {len(sw)} cells; margin over the next best config:")
            print(f"    best  {max(sw, key=lambda m: m[3])[3]:.3f}x  "
                  f"(in={max(sw, key=lambda m: m[3])[1]}, "
                  f"out={max(sw, key=lambda m: m[3])[2]})")
            print(f"    worst {min(sw, key=lambda m: m[3])[3]:.3f}x")

    # ---- crossover: where does the winner change with output length? ----
    print()
    print("CROSSOVER - winner by output length, per input length")
    print("-" * 74)
    for n_in in inputs:
        line = f"  input {n_in:>5}: "
        for n_out in outputs:
            vals = {c: _f(idx.get((c, n_in, n_out), {}), "e2e_ms")
                    for c in configs}
            vals = {c: v for c, v in vals.items() if v is not None}
            line += f"out={n_out}:{min(vals, key=vals.get) if vals else '-':<8}"
        print(line)
    print()
    print("  A winner that changes with output length is the expected shape:")
    print("  TTFT is paid once, ITL is paid per token, so a config with worse")
    print("  TTFT and better ITL takes over as output grows.")

    if bad:
        print()
        print(f"FAILED CELLS ({len(bad)})")
        print("-" * 74)
        seen = collections.Counter()
        for r in bad:
            key = (r["config"], str(r.get("error", ""))[:60])
            seen[key] += 1
        for (c, e), n in seen.most_common(12):
            print(f"  {n:3d}x  {c:<10} {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
