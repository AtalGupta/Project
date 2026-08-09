"""Sweep driver.

Measurement hygiene, which is the whole reason this file is more than a for-loop:

  * every repetition runs in a SEPARATE PROCESS. Compiled models, kernel caches
    and allocator state persist inside a process and make the second run of
    anything look faster than the first. A fresh process per rep is the only
    reliable way to stop that leaking into the numbers.
  * one warmup rep is discarded.
  * >= 3 measured reps, reported as mean +/- stdev.
  * compile time is captured separately, never folded into throughput.
  * strategies whose devices are absent are SKIPPED and recorded as skipped,
    not silently dropped -- an empty row and a missing row mean different things.

Usage
    python -m bench.run --list
    python -m bench.run --reps 3
    python -m bench.run --strategy single spec
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")

MARKER = "@@RESULT@@"


def discover_models() -> dict[str, str]:
    """Map variant key -> path, for every variant actually exported."""
    out = {}
    if not os.path.isdir(MODELS):
        return out
    for name in sorted(os.listdir(MODELS)):
        p = os.path.join(MODELS, name)
        if os.path.isfile(os.path.join(p, "openvino_model.xml")):
            out[name] = p
    return out


# --------------------------------------------------------------------------
# worker: runs exactly one (strategy, config) and prints one JSON line
# --------------------------------------------------------------------------

def worker_main(strategy_name: str, cfg_json: str) -> int:
    from bench import strategies

    cfg = json.loads(cfg_json)
    models = discover_models()
    strat = strategies.BY_NAME[strategy_name]
    try:
        res = strat.run(cfg, models)
    except Exception as e:  # a crash is a result too
        res = strategies.RunResult(strategy_name, str(cfg), "?", ok=False,
                                   error=f"UNCAUGHT {type(e).__name__}: {e}")
    print(MARKER + json.dumps(res.as_row(), default=str))
    return 0


def _run_worker(strategy_name: str, cfg: dict, timeout: int) -> dict:
    cmd = [sys.executable, "-m", "bench.run", "--worker", strategy_name,
           json.dumps(cfg)]
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"TIMEOUT after {timeout}s",
                "strategy": strategy_name, "label": str(cfg)}
    for line in p.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    tail = (p.stderr or p.stdout or "").strip().splitlines()
    return {"ok": False, "strategy": strategy_name, "label": str(cfg),
            "error": f"no result (exit {p.returncode}): " + " | ".join(tail[-3:])}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    from bench import devices as devmod
    from bench import strategies

    ap = argparse.ArgumentParser(description="Sweep stitching strategies.")
    ap.add_argument("--worker", nargs=2, metavar=("STRATEGY", "CFG_JSON"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--strategy", nargs="*", choices=sorted(strategies.BY_NAME),
                    help="subset to run (default: all applicable)")
    ap.add_argument("--reps", type=int, default=3, help="measured reps (default 3)")
    ap.add_argument("--warmup", type=int, default=1, help="discarded reps (default 1)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-rep seconds")
    ap.add_argument("--list", action="store_true", help="show planned runs and exit")
    ap.add_argument("--out", default=None, help="CSV path")
    args = ap.parse_args()

    if args.worker:
        return worker_main(args.worker[0], args.worker[1])

    models = discover_models()
    available = devmod.kinds_present()

    print(devmod.summary())
    print()
    print(f"models exported: {sorted(models) or 'NONE - run bench/export.py first'}")
    print()

    if not models:
        print("No exported models. Run:  python -m bench.export")
        return 1

    wanted = args.strategy or list(strategies.BY_NAME)
    plan: list[tuple[str, dict]] = []
    skipped: list[tuple[str, str]] = []
    for name in wanted:
        strat = strategies.BY_NAME[name]
        cfgs = strat.configs(available, models)
        if not cfgs:
            skipped.append((name, f"no viable config for devices {sorted(available)}"))
            continue
        for c in cfgs:
            plan.append((name, c))

    print(f"planned runs: {len(plan)}  "
          f"({args.warmup} warmup + {args.reps} measured each)")
    for name, c in plan:
        print(f"  {name:14s} {c}")
    if skipped:
        print("\nskipped:")
        for name, why in skipped:
            print(f"  {name:14s} {why}")
    if args.list:
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    out_path = args.out or os.path.join(
        RESULTS, f"sweep-{os.environ.get('COMPUTERNAME', 'host')}-"
                 f"{time.strftime('%Y%m%d-%H%M%S')}.csv")

    rows = []
    t_start = time.time()
    for i, (name, cfg) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {name} {cfg}")
        for w in range(args.warmup):
            print(f"    warmup {w + 1}/{args.warmup} ...", end="", flush=True)
            r = _run_worker(name, cfg, args.timeout)
            print(" ok" if r.get("ok") else f" FAILED: {r.get('error', '')[:120]}")
            if not r.get("ok"):
                break  # a config that fails warmup will fail every rep

        samples = []
        last = None
        for k in range(args.reps):
            print(f"    rep {k + 1}/{args.reps} ...", end="", flush=True)
            r = _run_worker(name, cfg, args.timeout)
            last = r
            if r.get("ok"):
                samples.append(r)
                tp = r.get("throughput_tok_s")
                agg = r.get("aggregate_tok_s")
                shown = agg if (tp is None or (isinstance(tp, float) and tp != tp)) else tp
                print(f" {shown:.2f} tok/s" if isinstance(shown, (int, float))
                      and shown == shown else " ok")
            else:
                print(f" FAILED: {r.get('error', '')[:120]}")

        if not samples:
            rows.append({"strategy": name, "config": json.dumps(cfg), "ok": False,
                         "n": 0, "error": (last or {}).get("error", "no samples")})
            continue

        def agg(field):
            vals = [s[field] for s in samples
                    if isinstance(s.get(field), (int, float)) and s[field] == s[field]]
            if not vals:
                return None, None
            return (statistics.fmean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0)

        row = {"strategy": name, "config": json.dumps(cfg), "ok": True,
               "n": len(samples), "label": samples[0].get("label", ""),
               "devices": samples[0].get("devices", ""), "error": ""}
        for f in ("throughput_tok_s", "aggregate_tok_s", "ttft_ms", "tpot_ms",
                  "compile_s", "acceptance_rate"):
            m, sd = agg(f)
            row[f + "_mean"] = round(m, 4) if m is not None else ""
            row[f + "_stdev"] = round(sd, 4) if sd is not None else ""
        row["generated_tokens"] = samples[0].get("generated_tokens", 0)
        row["output_head"] = samples[0].get("output_head", "")
        rows.append(row)

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nelapsed: {time.time() - t_start:.0f}s")
    print(f"wrote {out_path}")
    print(f"\nnow:  python -m bench.report {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
