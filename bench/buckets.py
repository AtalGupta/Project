"""How much TTFT does NPU shape-padding cost, and what does bucketing recover?

The NPU compiles to static shapes, so LLMPipeline is given a single
MAX_PROMPT_LEN and processes that many positions for EVERY request regardless of
the real prompt length. Measured consequence on Panther Lake: TTFT is flat at
~625 ms whether the prompt is 64 tokens or 960. A 128-token prompt pays for 896
tokens of padding.

This sweeps MAX_PROMPT_LEN against real prompt length and reports the waste, so
"bucket the shapes" stops being advice and becomes a number.

Read the diagonal (bucket == prompt) as the achievable TTFT if you compiled a
bucket per size. Read the last column as what you pay today with a single
1024 bucket.

    python -u -m bench.buckets --model models\\qwen2.5-1.5b-fp16 --device NPU
    python -u -m bench.buckets --model models\\qwen2.5-1.5b-fp16 --device CPU \\
        --buckets 128 256 512 1024
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

DEFAULT_BUCKETS = (128, 256, 512, 1024)
DEFAULT_PROMPTS = (64, 128, 256, 512, 960)
UNIT = "The system streams model weights from memory for every token. "


def prompt_of(n_tokens: int) -> str:
    """Character-estimate fallback, only used if no tokenizer is available.

    Deliberately UNDER-shoots: the NPU hard-fails when a prompt exceeds
    MAX_PROMPT_LEN, so overshooting kills the run. 3 chars/token is below the
    ~3.7 this repetitive text actually achieves.
    """
    return (UNIT * (n_tokens // 5 + 2))[:max(1, n_tokens * 3)]


def prompt_exact(pipe, n_tokens: int) -> tuple[str, int]:
    """Text that encodes to AT MOST n_tokens, plus the real count.

    Estimating tokens from character length was the original bug: at a target of
    64 it produced 86 real tokens, and at 128 it produced enough to breach a
    128-token bucket and abort the sweep. Ask the tokenizer instead of guessing,
    then trim until it genuinely fits.
    """
    try:
        tok = pipe.get_tokenizer()
    except Exception:
        return prompt_of(n_tokens), -1

    def count(txt: str) -> int:
        try:
            return int(tok.encode(txt).input_ids.get_shape()[-1])
        except Exception:
            return -1

    text = UNIT * (n_tokens // 5 + 4)
    n = count(text)
    if n < 0:
        return prompt_of(n_tokens), -1

    # Shrink by character ratio, then walk down. Converges in 2-3 steps.
    for _ in range(12):
        n = count(text)
        if n <= n_tokens:
            break
        text = text[:max(1, int(len(text) * (n_tokens / n) * 0.97))]
    while count(text) > n_tokens and len(text) > 8:
        text = text[:int(len(text) * 0.9)]
    return text, count(text)


def measure(model_dir: str, device: str, bucket: int, prompt_tokens: int,
            reps: int, min_response: int) -> dict:
    import openvino_genai as ov_genai

    props = {}
    if device.upper().startswith("NPU"):
        props["MAX_PROMPT_LEN"] = bucket
        props["MIN_RESPONSE_LEN"] = min_response

    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(model_dir, device, **props)
    compile_s = time.perf_counter() - t0

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1          # TTFT only: one token means prefill dominates
    cfg.do_sample = False
    cfg.ignore_eos = True

    # LLMPipeline wraps the prompt in a chat template before encoding, so the
    # tokenizer's count of the raw text UNDERSTATES what the model actually
    # receives -- measured at ~21-26 extra tokens here. Ignoring that produced
    # 90 real tokens for a 64-token target and blew past a 128 bucket.
    # Probe the overhead once, then size the prompt against the real budget.
    overhead = 0
    try:
        probe_text, probe_raw = prompt_exact(pipe, 8)
        probe_cfg = ov_genai.GenerationConfig()
        probe_cfg.max_new_tokens = 1
        probe_cfg.do_sample = False
        pr = pipe.generate([probe_text], probe_cfg)
        overhead = max(0, pr.perf_metrics.get_num_input_tokens() - max(probe_raw, 0))
    except Exception:
        overhead = 0

    text, planned = prompt_exact(pipe, max(1, prompt_tokens - overhead))
    effective = planned + overhead
    if effective > bucket:
        del pipe
        gc.collect()
        return {"error": f"prompt is {effective} tokens after chat template "
                         f"(+{overhead}), exceeds bucket {bucket}",
                "compile_s": compile_s}

    ttfts, ntok = [], 0
    try:
        pipe.generate([text], cfg)          # warmup, discarded
        for _ in range(reps):
            res = pipe.generate([text], cfg)
            pm = res.perf_metrics
            ttfts.append(pm.get_ttft().mean)
            ntok = pm.get_num_input_tokens()
    except Exception as e:
        del pipe
        gc.collect()
        return {"error": f"{type(e).__name__}: {str(e)[:90]}", "compile_s": compile_s}

    del pipe
    gc.collect()
    med = statistics.median(ttfts)
    return {"ttft_ms": med,
            "sd": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            "input_tokens": ntok,
            "compile_s": compile_s,
            "pad_ratio": bucket / max(ntok, 1),
            "tok_per_s": (ntok / (med / 1000.0)) if med else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the TTFT cost of static-shape padding.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--buckets", type=int, nargs="*", default=list(DEFAULT_BUCKETS))
    ap.add_argument("--prompts", type=int, nargs="*", default=list(DEFAULT_PROMPTS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-response", type=int, default=64)
    ap.add_argument("--csv", help="write results here")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import pick

    dev = pick(args.device.split(".")[0]) or args.device
    buckets = sorted(args.buckets)
    prompts = sorted(args.prompts)

    print(f"device {dev}   model {os.path.basename(args.model)}")
    print(f"buckets (MAX_PROMPT_LEN): {buckets}")
    print(f"prompt lengths: {prompts}")
    print(f"{args.reps} reps, median TTFT, 1 warmup discarded, max_new_tokens=1")
    if not dev.upper().startswith("NPU"):
        print("\nNOTE: MAX_PROMPT_LEN only applies to the NPU plugin. On CPU/GPU")
        print("      every bucket compiles the same dynamic graph, so the rows")
        print("      should come out identical -- that is the control case.")
    print()

    results: dict[tuple[int, int], dict] = {}
    for b in buckets:
        for p in prompts:
            if p > b:
                continue        # prompt does not fit this bucket
            print(f"  bucket={b:<5} prompt={p:<5} ...", end="", flush=True)
            r = measure(args.model, dev, b, p, args.reps, args.min_response)
            results[(b, p)] = r
            if "error" in r:
                print(f" FAILED: {r['error']}")
            else:
                print(f" TTFT {r['ttft_ms']:7.1f} ms   "
                      f"({r['input_tokens']} real tokens, "
                      f"{r['pad_ratio']:.1f}x padded)")

    ok = {k: v for k, v in results.items() if "error" not in v}
    if not ok:
        print("\nno successful runs")
        return 1

    print()
    print("TTFT (ms) - rows are prompt length, columns are compiled bucket")
    hdr = f"{'prompt':>8}" + "".join(f"{'b=' + str(b):>12}" for b in buckets)
    print(hdr)
    print("-" * len(hdr))
    for p in prompts:
        row = f"{p:>8}"
        for b in buckets:
            v = ok.get((b, p))
            row += f"{v['ttft_ms']:>12.1f}" if v else f"{'-':>12}"
        print(row)

    # The payoff: smallest fitting bucket vs the largest one.
    biggest = buckets[-1]
    print()
    print("SAVING from bucketing (smallest fitting bucket vs single b="
          f"{biggest} compile)")
    print(f"{'prompt':>8}{'best bucket':>14}{'TTFT best':>12}"
          f"{'TTFT b=' + str(biggest):>14}{'saved':>10}{'speedup':>10}")
    print("-" * 68)
    total_saved = 0.0
    for p in prompts:
        fits = [b for b in buckets if b >= p and (b, p) in ok]
        base = ok.get((biggest, p))
        if not fits or not base:
            continue
        best_b = min(fits)
        best = ok[(best_b, p)]
        saved = base["ttft_ms"] - best["ttft_ms"]
        total_saved += saved
        sp = base["ttft_ms"] / best["ttft_ms"] if best["ttft_ms"] else 0
        print(f"{p:>8}{best_b:>14}{best['ttft_ms']:>12.1f}"
              f"{base['ttft_ms']:>14.1f}{saved:>10.1f}{sp:>9.2f}x")

    print()
    print(f"compile cost: {len(buckets)} buckets x "
          f"{statistics.fmean(v['compile_s'] for v in ok.values()):.0f}s "
          f"= one-time. Use cache_dir so it is paid once, not per process.")

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["device", "bucket", "prompt_tokens", "input_tokens",
                        "ttft_ms", "sd", "pad_ratio", "tok_per_s", "compile_s",
                        "error"])
            for (b, p), r in sorted(results.items()):
                w.writerow([dev, b, p, r.get("input_tokens", ""),
                            r.get("ttft_ms", ""), r.get("sd", ""),
                            r.get("pad_ratio", ""), r.get("tok_per_s", ""),
                            r.get("compile_s", ""), r.get("error", "")])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
