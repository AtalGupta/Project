"""Memory-bandwidth roofline.

Why this exists: Qwen2.5-1.5B at INT4 is ~1 GB of weights, and LLM *decode*
streams the whole weight set once per token. Decode is therefore bandwidth-bound,
not compute-bound, and the achievable token rate is capped at

    tok/s_ceiling  =  achieved_bytes_per_sec / model_bytes

CPU, GPU and NPU on these client parts all sit behind the *same* memory
controller with no dedicated VRAM, so that ceiling is shared, not per-device.
Every measured tok/s in this project is reported as a percentage of it. Without
that column there is no way to distinguish "we optimised it" from "we hit physics".

Implementation is a STREAM-style triad in numpy (a = b + s*c) over arrays far
larger than last-level cache, threaded across slices. numpy releases the GIL for
elementwise ops, so threads give real parallelism here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

# Byte accounting per kernel. This has to match what numpy ACTUALLY does, not
# what the classic STREAM C code does -- numpy has no loop fusion, so each ufunc
# call is its own full pass over memory.
#
#   copy   np.copyto(a, b)                  read b, write a            = 16 B/elem
#   scale  np.multiply(b, s, out=a)         read b, write a            = 16 B/elem
#   triad  np.multiply(c, s, out=a)         read c, write a      16 B
#          np.add(a, b, out=a)              read a, read b, write a 24 B  = 40 B/elem
#
# Getting this wrong understates bandwidth and produces a "ceiling" below
# measured throughput, which is how the error announces itself.
KERNEL_BYTES = {"copy": 16, "scale": 16, "triad": 40}


@dataclass
class BandwidthResult:
    kernel: str
    threads: int
    elems: int
    array_mb: float
    best_seconds: float
    gb_per_s: float


def _copy_slice(a, b, c, s, lo, hi):
    np.copyto(a[lo:hi], b[lo:hi])


def _scale_slice(a, b, c, s, lo, hi):
    np.multiply(b[lo:hi], s, out=a[lo:hi])


def _triad_slice(a, b, c, s, lo, hi):
    np.multiply(c[lo:hi], s, out=a[lo:hi])
    np.add(a[lo:hi], b[lo:hi], out=a[lo:hi])


KERNELS = {"copy": _copy_slice, "scale": _scale_slice, "triad": _triad_slice}


def measure_bandwidth(elems: int = 24_000_000, threads: int | None = None,
                      reps: int = 7, kernel: str = "copy") -> BandwidthResult:
    """Best-of-`reps` bandwidth. Best-of, not mean: we want the machine's
    capability, and noise on a shared/remote box is one-sided (only ever slower).

    Default kernel is `copy` -- it is a single unambiguous pass, so its byte
    count cannot be wrong the way a multi-pass triad's can.
    """
    if threads is None:
        threads = os.cpu_count() or 1

    a = np.ones(elems, dtype=np.float64)
    b = np.full(elems, 2.0, dtype=np.float64)
    c = np.full(elems, 3.0, dtype=np.float64)
    s = 3.0
    fn = KERNELS[kernel]

    bounds = []
    step = (elems + threads - 1) // threads
    for t in range(threads):
        lo, hi = t * step, min((t + 1) * step, elems)
        if lo < hi:
            bounds.append((lo, hi))

    best = float("inf")
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for _ in range(reps):
            t0 = time.perf_counter()
            list(ex.map(lambda lh: fn(a, b, c, s, *lh), bounds))
            dt = time.perf_counter() - t0
            best = min(best, dt)

    return BandwidthResult(
        kernel=kernel,
        threads=threads,
        elems=elems,
        array_mb=elems * 8 / 1e6,
        best_seconds=best,
        gb_per_s=KERNEL_BYTES[kernel] * elems / best / 1e9,
    )


def sweep(elems: int = 24_000_000) -> list[BandwidthResult]:
    """Thread sweep across all three kernels.

    The shape of the curve is the interesting part: if a small thread count
    already saturates the controller, that is direct evidence that adding
    *another engine* to the same token stream cannot add throughput.

    Peak is taken across kernels because they stress the controller differently
    (pure streaming reads vs. read-modify-write) and the highest one is the best
    lower bound on what the memory system can actually deliver.
    """
    n = os.cpu_count() or 1
    counts = sorted({1, 2, 4, max(1, n // 2), n})
    out = []
    for k in ("copy", "scale", "triad"):
        for t in counts:
            out.append(measure_bandwidth(elems=elems, threads=t, kernel=k))
    return out


def dir_bytes(path: str, patterns=(".bin", ".xml")) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(patterns):
                total += os.path.getsize(os.path.join(root, f))
    return total


def token_ceiling(model_bytes: int, gb_per_s: float) -> float:
    """Upper bound on decode tok/s for a model of this size at this bandwidth."""
    if model_bytes <= 0:
        return float("nan")
    return gb_per_s * 1e9 / model_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the memory-bandwidth roofline.")
    ap.add_argument("--elems", type=int, default=24_000_000,
                    help="elements per array (default 24M = 192 MB/array, 576 MB total)")
    ap.add_argument("--model", help="optional exported-model dir, to print its tok/s ceiling")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    print(f"triad arrays: 3 x {args.elems * 8 / 1e6:.0f} MB "
          f"= {3 * args.elems * 8 / 1e6:.0f} MB total (>> last-level cache)")
    print()
    results = sweep(args.elems)
    print(f"{'kernel':>7}  {'threads':>8}  {'GB/s':>8}  {'best (ms)':>10}")
    for r in results:
        print(f"{r.kernel:>7}  {r.threads:>8}  {r.gb_per_s:>8.1f}  "
              f"{r.best_seconds * 1e3:>10.1f}")

    peak = max(r.gb_per_s for r in results)
    best_r = max(results, key=lambda r: r.gb_per_s)
    print()
    print(f"achieved peak: {peak:.1f} GB/s  ({best_r.kernel}, {best_r.threads} threads)")
    print("NOTE: numpy has no loop fusion, so this is a LOWER BOUND on what the")
    print("      memory system can do. A real inference engine issuing wide")
    print("      streaming loads will reach higher than numpy does.")

    payload = {"sweep": [asdict(r) for r in results], "peak_gb_per_s": peak}

    if args.model:
        mb = dir_bytes(args.model)
        payload["model_path"] = args.model
        payload["model_bytes"] = mb
        if mb:
            ceil = token_ceiling(mb, peak)
            payload["tok_s_ceiling"] = ceil
            print(f"model        : {args.model}")
            print(f"model size   : {mb / 1e9:.2f} GB")
            print(f"decode ceiling: {ceil:.0f} tok/s  "
                  f"(one full weight sweep per token at {peak:.0f} GB/s)")
        else:
            print(f"WARNING: no .bin/.xml found under {args.model}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
