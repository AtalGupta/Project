"""Per-operation profiling, and an empirical compute-vs-memory classification.

The question this answers: *which operations are compute-bound and which are
memory-bound, on this machine, measured rather than assumed?*

The method is differential rather than analytical. Modelling FLOPs and bytes per
node would need per-op shape maths and would still be an estimate. Instead we
time every op at several sequence lengths and look at how its cost SCALES:

    memory-bound op   cost is dominated by streaming weights, which does not
                      depend on how many positions are processed
                      -> time roughly FLAT in S          -> slope ~ 0

    compute-bound op  cost is dominated by arithmetic, which is proportional to
                      the number of positions
                      -> time roughly LINEAR in S        -> slope ~ 1

We fit slope = dlog(time)/dlog(S) per node type. That single number places each
op on the spectrum without needing to know its FLOP count.

The expected -- and important -- result is that FullyConnected has a slope near 0
at small S and near 1 at large S. It is the SAME operation being memory-bound
during decode and compute-bound during prefill, which is why "put the
compute-bound ops on the NPU and the memory-bound ops on the CPU" has nothing to
divide: the bottleneck belongs to the phase, not to the operation.

    python -m bench.profile --model models/qwen2.5-1.5b-fp16 --device CPU
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys

# Sequence lengths to profile at. 1 is a decode step; the larger values are
# prefill of increasing size. The span has to cover both regimes for the slope
# to mean anything.
DEFAULT_SEQ = (1, 8, 64, 256)


def _dummy_inputs(compiled, seq: int, past: int):
    """Build zero-filled inputs matching the compiled model's actual shapes."""
    import numpy as np

    feed = {}
    for port in compiled.inputs:
        shape = list(port.get_shape())
        dtype = port.get_element_type().to_dtype()
        name = port.get_any_name()
        arr = np.zeros(shape, dtype=dtype)
        if "attention_mask" in name:
            arr[:] = 1          # all positions visible; zeros can trip masking
        feed[name] = arr
    return feed


def profile_at(core, xml: str, device: str, seq: int, past: int,
               iters: int = 5) -> tuple[dict, float]:
    """Return (time_us by node_type, total_us) for one sequence length."""
    from bench.kernels import make_static

    model = core.read_model(xml)
    make_static(model, seq=seq, past=past)
    compiled = core.compile_model(model, device, {"PERF_COUNT": True})
    req = compiled.create_infer_request()
    feed = _dummy_inputs(compiled, seq, past)

    req.infer(feed)                     # warmup, discarded
    by_type: collections.Counter = collections.Counter()
    total = 0.0
    for _ in range(iters):
        req.infer(feed)
        for p in req.profiling_info:
            # NOT_RUN nodes were folded away; counting them would dilute the
            # averages with zeros for ops that do not exist at this shape.
            if str(p.status) == "Status.NOT_RUN":
                continue
            us = p.real_time.total_seconds() * 1e6
            by_type[p.node_type] += us
            total += us
    for k in by_type:
        by_type[k] /= iters
    del req, compiled, model
    return dict(by_type), total / iters


def slope(seqs: list[int], times: list[float]) -> float:
    """Least-squares dlog(time)/dlog(S). ~0 = flat = memory-bound,
    ~1 = linear = compute-bound."""
    pts = [(math.log(s), math.log(t)) for s, t in zip(seqs, times) if s > 0 and t > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den else float("nan")


def classify(s: float) -> str:
    if math.isnan(s):
        return "?"
    if s < 0.25:
        return "MEMORY-bound"
    if s > 0.75:
        return "COMPUTE-bound"
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Profile per-op cost and classify compute vs memory bound.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--past", type=int, default=256, help="KV cache length")
    ap.add_argument("--seq", type=int, nargs="*", default=list(DEFAULT_SEQ))
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import core, pick

    dev = pick(args.device.split(".")[0]) or args.device
    xml = os.path.join(args.model, "openvino_model.xml")
    if not os.path.isfile(xml):
        print(f"no openvino_model.xml in {args.model}")
        return 1

    c = core()
    seqs = sorted(args.seq)
    per_seq: dict[int, dict] = {}
    totals: dict[int, float] = {}

    for s in seqs:
        print(f"profiling seq={s} past={args.past} on {dev} ...", flush=True)
        try:
            by_type, total = profile_at(c, xml, dev, s, args.past, args.iters)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        per_seq[s] = by_type
        totals[s] = total
        print(f"  total {total/1000:.2f} ms across {len(by_type)} op types")

    if len(per_seq) < 2:
        print("\nNeed at least two sequence lengths to compute scaling.")
        return 1

    got = sorted(per_seq)
    all_types = set()
    for d in per_seq.values():
        all_types |= set(d)

    rows = []
    for t in all_types:
        times = [per_seq[s].get(t, 0.0) for s in got]
        if max(times) <= 0:
            continue
        sl = slope(got, times)
        share_lo = 100.0 * times[0] / totals[got[0]] if totals[got[0]] else 0
        share_hi = 100.0 * times[-1] / totals[got[-1]] if totals[got[-1]] else 0
        rows.append((times[-1], t, times, sl, share_lo, share_hi))
    rows.sort(reverse=True)

    print()
    print(f"Per-op scaling on {dev}   (S = sequence length)")
    hdr = (f"{'operation':<30}" + "".join(f"{'S='+str(s):>11}" for s in got)
           + f"{'slope':>8}  {'%@S='+str(got[0]):>9} {'%@S='+str(got[-1]):>9}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for _, t, times, sl, lo, hi in rows[:18]:
        cells = "".join(f"{v/1000:>10.2f}m" for v in times)
        print(f"{t:<30}{cells}{sl:>8.2f}  {lo:>8.1f}% {hi:>8.1f}%  {classify(sl)}")

    print()
    print(f"Total inference time: " +
          "   ".join(f"S={s}: {totals[s]/1000:.2f} ms" for s in got))
    print()
    print("How to read `slope`:")
    print("  ~0.0  cost does not grow with sequence length -> dominated by")
    print("        streaming weights from memory        -> MEMORY-bound")
    print("  ~1.0  cost grows in proportion to positions -> dominated by")
    print("        arithmetic                           -> COMPUTE-bound")
    print()
    print("If the dominant op (usually FullyConnected) sits near 0 at small S and")
    print("climbs toward 1 at large S, that is the same operation changing regime")
    print("with phase -- decode is memory-bound, prefill is compute-bound. In that")
    print("case there is no fixed set of 'compute ops' to place on one device and")
    print("'memory ops' on another; the split has to be made per PHASE, not per op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
