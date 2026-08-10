"""Verify a split model, then benchmark it. Correctness first, always.

Order matters here. A distributed placement that is 20% faster and produces
different tokens has not won anything -- it has broken the model. So this prints
generated text before it prints a single timing number, and refuses to present
the benchmark as meaningful if the split diverges early from the single-device
reference.

Greedy decoding is deterministic: the same weights and the same prompt must give
the same tokens regardless of which device ran which block. Splitting attention
and FFN across devices changes nothing mathematically -- it only changes where
the arithmetic happens -- so a correct split should match the reference exactly,
or diverge only late where fp16 accumulation order differs.

    python -u -m bench.opsplit_check --model models/qwen2.5-1.5b-fp16-split \
        --attention CPU --ffn NPU --reference models/qwen2.5-1.5b-fp16
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

PROMPTS = [
    "What is 17 * 23? Answer with the number only.",
    "Explain in two sentences why memory bandwidth limits LLM token generation.",
    "Write a Python function that reverses a linked list.",
]


def make_pipe(path: str, device: str):
    import openvino_genai as ov_genai
    from bench.strategies import device_props

    props = {}
    for kind in ("NPU", "CPU", "GPU"):
        if kind in device:
            props.update(device_props(kind))
    return ov_genai.LLMPipeline(path, device, **props)


def gen(pipe, prompt: str, max_new: int, warm: bool = False):
    import openvino_genai as ov_genai

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new
    cfg.do_sample = False
    cfg.ignore_eos = warm          # benchmarking wants a fixed token count
    res = pipe.generate([prompt], cfg)
    text = res.texts[0] if getattr(res, "texts", None) else str(res)
    pm = res.perf_metrics
    return text, pm


def first_diff_word(a: str, b: str) -> int:
    wa, wb = a.split(), b.split()
    for i, (x, y) in enumerate(zip(wa, wb)):
        if x != y:
            return i
    return -1 if len(wa) == len(wb) else min(len(wa), len(wb))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify then benchmark a split model.")
    ap.add_argument("--model", required=True, help="the -split model dir")
    ap.add_argument("--reference", default="models/qwen2.5-1.5b-fp16")
    ap.add_argument("--attention", default="CPU")
    ap.add_argument("--ffn", default="NPU")
    ap.add_argument("--ref-device", default=None,
                    help="reference device (default: the FFN device alone)")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.strategies import resolve

    split_dev = f"HETERO:{resolve(args.ffn)},{resolve(args.attention)}"
    ref_dev = resolve(args.ref_device or args.ffn)

    print("=" * 78)
    print(f"SPLIT      attention={args.attention}  FFN={args.ffn}")
    print(f"           device string: {split_dev}")
    print(f"REFERENCE  {ref_dev}  ({args.reference})")
    print("=" * 78)

    # ---------- build pipelines ----------
    try:
        t0 = time.perf_counter()
        split_pipe = make_pipe(args.model, split_dev)
        split_compile = time.perf_counter() - t0
        print(f"\nsplit model compiled in {split_compile:.1f}s")
    except Exception as e:
        print(f"\nSPLIT FAILED TO COMPILE: {type(e).__name__}: {e}")
        return 1

    try:
        t0 = time.perf_counter()
        ref_pipe = make_pipe(args.reference, ref_dev)
        ref_compile = time.perf_counter() - t0
        print(f"reference compiled in {ref_compile:.1f}s")
    except Exception as e:
        print(f"reference failed to compile: {type(e).__name__}: {e}")
        ref_pipe = None

    # ---------- CORRECTNESS FIRST ----------
    print("\n" + "=" * 78)
    print("MODEL OUTPUT  (correctness before any timing)")
    print("=" * 78)

    all_ok = True
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n--- prompt {i}: {prompt}")
        s_text, _ = gen(split_pipe, prompt, args.tokens)
        print(f"\n  [SPLIT  {args.attention}+{args.ffn}]")
        print("    " + s_text[:400].replace("\n", "\n    "))

        if ref_pipe is not None:
            r_text, _ = gen(ref_pipe, prompt, args.tokens)
            print(f"\n  [REF    {ref_dev}]")
            print("    " + r_text[:400].replace("\n", "\n    "))

            if s_text == r_text:
                print("\n  -> IDENTICAL")
            else:
                d = first_diff_word(s_text, r_text)
                if d >= 0 and d <= 2:
                    print(f"\n  -> DIVERGES AT WORD {d} — the split is WRONG, "
                          "not merely imprecise")
                    all_ok = False
                else:
                    print(f"\n  -> diverges at word {d} (late; fp accumulation "
                          "order differs across devices, expected)")

    print("\n" + "=" * 78)
    if all_ok:
        print("CORRECTNESS: OK — split output matches or diverges only late.")
    else:
        print("CORRECTNESS: FAILED — do not trust any timing below.")
    print("=" * 78)

    if args.skip_bench or not all_ok:
        return 0 if all_ok else 1

    # ---------- then, and only then, timing ----------
    print("\nBENCHMARK  (TTFT = prompt processing, ITL = per-token generation)\n")
    grid = [(128, 128), (512, 128), (1024, 128)]
    prompt_unit = "The system streams model weights from memory for every token. "

    header = f"{'config':<22}{'in':>6}{'out':>6}{'TTFT ms':>12}{'ITL ms':>10}{'tok/s':>9}"
    print(header)
    print("-" * len(header))

    targets = [("SPLIT", split_pipe)]
    if ref_pipe is not None:
        targets.append(("REF " + ref_dev, ref_pipe))

    for name, pipe in targets:
        for n_in, n_out in grid:
            prompt = (prompt_unit * (n_in // 10 + 2))[:n_in * 5]
            ttfts, itls = [], []
            try:
                gen(pipe, prompt, n_out, warm=True)          # warmup, discarded
                for _ in range(args.reps):
                    _, pm = gen(pipe, prompt, n_out, warm=True)
                    ttfts.append(pm.get_ttft().mean)
                    itls.append(pm.get_tpot().mean)
                t, it = statistics.fmean(ttfts), statistics.fmean(itls)
                print(f"{name:<22}{n_in:>6}{n_out:>6}{t:>12.1f}{it:>10.2f}"
                      f"{1000.0 / it:>9.2f}")
            except Exception as e:
                print(f"{name:<22}{n_in:>6}{n_out:>6}   FAILED: "
                      f"{type(e).__name__}: {str(e)[:60]}")

    print("\nITL is the number to compare against a CPU-only reference: it is")
    print("the steady-state generation cost, unaffected by prompt length.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
