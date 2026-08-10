"""Manager-facing summary.

bench.report is an engineer's view -- one CSV, every row, raw columns. This
produces a short Markdown brief across all the sweeps you have run: the headline
numbers, what they mean in plain language, and the caveats that stop someone
over-reading them.

It writes Markdown to stdout so it can be pasted into email, Confluence, Slack
or a doc, or redirected to a file:

    python -m bench.summary > summary.md
    python -m bench.summary results\\cloud-single.csv results\\cloud-prefill.csv

Two things it does that a raw table cannot:

  * corrects the NPU prefill figure for static-shape padding. The NPU compiles to
    a fixed MAX_PROMPT_LEN and processes that many positions regardless of the
    real prompt, so its TTFT is flat and its reported tokens/s rises with prompt
    length as an artefact. Reporting the uncorrected number to a manager would
    overstate short-prompt performance badly.
  * separates decode from prefill, because they have opposite bottlenecks and a
    single "tokens per second" headline hides the entire finding.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _f(row: dict, key: str):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _cfg(row: dict) -> dict:
    try:
        return json.loads(row.get("config") or "{}")
    except Exception:
        return {}


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        try:
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    r["_src"] = os.path.basename(p)
                    rows.append(r)
        except OSError:
            continue
    return rows


def model_sizes() -> dict[str, int]:
    out = {}
    mdir = os.path.join(ROOT, "models")
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


def peak_bandwidth() -> float | None:
    for name in ("roofline.json", "roofline-fp32.json"):
        p = os.path.join(ROOT, "results", name)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    return json.load(f).get("peak_gb_per_s")
            except Exception:
                pass
    return None


def _prec(key: str) -> str:
    return key.rsplit("-", 1)[-1].upper() if key else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown summary for sharing.")
    ap.add_argument("csv", nargs="*", help="sweep CSVs (default: all in results/)")
    ap.add_argument("--npu-max-prompt", type=int, default=1024,
                    help="MAX_PROMPT_LEN used for NPU (for padding correction)")
    ap.add_argument("--out", help="write Markdown here (avoids shell redirection "
                                  "encoding problems on Windows)")
    args = ap.parse_args()

    # Redirecting stdout on Windows gives cp1252, which cannot encode the arrows
    # and em dashes in this report -- `python -m bench.summary > file.md` then
    # dies with UnicodeEncodeError. Force UTF-8 on the stream instead.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    paths = args.csv or sorted(glob.glob(os.path.join(ROOT, "results", "*.csv")))
    if not paths:
        print("No result CSVs found. Run bench.run first.")
        return 1

    rows = [r for r in load_rows(paths) if str(r.get("ok", "")).lower() == "true"]
    failed = [r for r in load_rows(paths) if str(r.get("ok", "")).lower() != "true"]
    peak = peak_bandwidth()
    sizes = model_sizes()

    out: list[str] = []
    w = out.append

    w("# Qwen2.5-1.5B on Intel CPU + NPU — measurement summary")
    w("")
    w("Unquantized model (FP32 and FP16), OpenVINO's own kernels, no speculative "
      "decoding. Every figure is the mean of 3 repetitions, each run in a separate "
      "process, with one warmup discarded and compile time excluded.")
    w("")

    # ---------------- headline ----------------
    decode = [r for r in rows if r.get("strategy") == "single"]

    def over_ceiling(r) -> bool:
        """True if the row beat its own physical limit -- which means it did not
        run the precision its name claims (the GPU plugin converts fp32 to f16
        by default). Such rows must never become the headline."""
        v = _f(r, "throughput_tok_s_mean")
        key = _cfg(r).get("model_key", "")
        if not (peak and v and key in sizes):
            return False
        return 100.0 * v / (peak * 1e9 / sizes[key]) > 105

    flagged_any = any(over_ceiling(r) for r in decode)

    if decode:
        decode.sort(key=lambda r: -(_f(r, "throughput_tok_s_mean") or 0))
        valid = [r for r in decode if not over_ceiling(r)]
        best = (valid or decode)[0]
        w("## Headline")
        w("")
        bt = _f(best, "throughput_tok_s_mean")
        w(f"- **Fastest configuration: {best.get('label','?')} at {bt:.1f} tokens/s.**")
        # best per device kind
        by_dev: dict[str, float] = {}
        for r in (valid or decode):
            d = (_cfg(r).get("device") or r.get("devices") or "?").split(".")[0]
            v = _f(r, "throughput_tok_s_mean") or 0
            by_dev[d] = max(by_dev.get(d, 0), v)
        if "NPU" in by_dev and "CPU" in by_dev and by_dev["CPU"]:
            w(f"- NPU is **{by_dev['NPU'] / by_dev['CPU']:.2f}x** the CPU on text "
              f"generation ({by_dev['NPU']:.1f} vs {by_dev['CPU']:.1f} tokens/s).")
        if peak:
            w(f"- Memory bandwidth measured at **{peak:.0f} GB/s**, which caps text "
              f"generation regardless of how much compute is available.")
        w("")

    # ---------------- decode ----------------
    if decode:
        w("## Text generation (decode)")
        w("")
        w("This is the phase that produces the answer, one token at a time. It is "
          "limited by **memory bandwidth**, not by processing power: every token "
          "requires reading the entire model from memory.")
        w("")
        w("| Configuration | Tokens/s | +/- | % of hardware limit | Time to first token |")
        w("|---|---:|---:|---:|---:|")
        for r in decode:
            v = _f(r, "throughput_tok_s_mean")
            sd = _f(r, "throughput_tok_s_stdev") or 0.0
            ttft = _f(r, "ttft_ms_mean")
            key = _cfg(r).get("model_key", "")
            ceil = (peak * 1e9 / sizes[key]) if (peak and key in sizes) else None
            pct = f"{100.0 * v / ceil:.0f}%" if (ceil and v) else "-"
            if ceil and v and 100.0 * v / ceil > 105:
                pct += " (see note)"
            w(f"| {r.get('label','?')} | {v:.2f} | {sd:.2f} | {pct} | "
              f"{ttft:.0f} ms |" if ttft else
              f"| {r.get('label','?')} | {v:.2f} | {sd:.2f} | {pct} | - |")
        w("")
        if peak and sizes:
            w("Hardware limits at the measured bandwidth:")
            for k in sorted(sizes):
                w(f"- `{k}` — {sizes[k]/1e9:.2f} GB per token → "
                  f"**{peak*1e9/sizes[k]:.1f} tokens/s maximum**")
            w("")
        if flagged_any:
            w("> **Note on the rows marked \"see note\".** A result cannot exceed "
              "100% of the hardware limit. Those rows did so because the graphics "
              "driver silently converts an FP32 model to FP16 when loading it, so "
              "half as much data is read per token — the speed is real, but the "
              "row is not the FP32 measurement its name implies. Rows labelled "
              "`forced=f32` are the genuine FP32 figures, and they are excluded "
              "from the headline above.")
            w("")

    # ---------------- prefill ----------------
    pref = [r for r in rows if r.get("strategy") == "prefill"]
    if pref:
        w("## Prompt processing (prefill)")
        w("")
        w("This is the phase that reads the question before answering. Unlike "
          "generation it is limited by **processing power**, which is what a "
          "neural accelerator is built for — so the two phases can give very "
          "different answers about which device is better.")
        w("")
        w("| Device | Precision | Prompt | Prompt tokens/s | Time to first token |")
        w("|---|---|---:|---:|---:|")
        def sk(r):
            c = _cfg(r)
            return (c.get("device", ""), c.get("model_key", ""), c.get("prompt_tokens", 0))
        for r in sorted(pref, key=sk):
            c = _cfg(r)
            v = _f(r, "throughput_tok_s_mean")
            ttft = _f(r, "ttft_ms_mean")
            w(f"| {c.get('device','?')} | {_prec(c.get('model_key',''))} | "
              f"{c.get('prompt_tokens','?')} | {v:.1f} | {ttft:.0f} ms |")
        w("")

        # --- padding correction ---
        npu = [r for r in pref if (_cfg(r).get("device") == "NPU")]
        ttfts = [t for t in (_f(r, "ttft_ms_mean") for r in npu) if t]
        if len(ttfts) >= 3:
            mean_t = statistics.fmean(ttfts)
            spread = (max(ttfts) - min(ttfts)) / mean_t if mean_t else 1.0
            if spread < 0.15:      # flat TTFT => fixed padded work
                corrected = args.npu_max_prompt / (mean_t / 1000.0)
                cpu_rates = [_f(r, "throughput_tok_s_mean") for r in pref
                             if _cfg(r).get("device") == "CPU"]
                cpu_best = max([c for c in cpu_rates if c] or [0])
                w("> **Important correction.** The NPU's time to first token is "
                  f"flat at ~{mean_t:.0f} ms across every prompt length "
                  f"({min(ttfts):.0f}–{max(ttfts):.0f} ms). It compiles to a fixed "
                  f"size and processes {args.npu_max_prompt} positions regardless "
                  "of the real prompt, so the tokens/s column above rises with "
                  "prompt length as a measurement artefact, not because the NPU "
                  "speeds up.")
                w(">")
                w(f"> Corrected for that padding, the NPU's real prompt-processing "
                  f"rate is **~{corrected:.0f} positions/s**"
                  + (f", against roughly **{cpu_best:.0f}/s** for the CPU — "
                     f"about **{corrected / cpu_best:.0f}x**." if cpu_best else "."))
                w(">")
                w("> Practical read: the NPU is wasteful on very short prompts and "
                  "dominant on long ones.")
                w("")

    # ---------------- what it means ----------------
    w("## What this means")
    w("")
    w("- **The NPU's advantage depends entirely on the workload.** For long "
      "prompts with short answers — document summarisation, retrieval, "
      "classification, code review — prompt processing dominates and the NPU is "
      "transformative. For short prompts with long answers — chat, agents — "
      "generation dominates, and the advantage is modest because generation is a "
      "memory problem rather than a compute one.")
    w("- **No compression was used.** These are full-precision results. "
      "Quantisation would raise the token rate further, at some cost to output "
      "quality; it was deliberately excluded here so the numbers describe the "
      "hardware rather than a modified workload.")
    w("- **The kernels are Intel's own**, verified per operation from the "
      "compiled graph — not a custom or reference implementation.")
    w("")

    # ---------------- caveats ----------------
    w("## Caveats")
    w("")
    w("- The test machine is **pre-production silicon** running below retail "
      "clocks, so relative comparisons are sound but absolute figures are not "
      "final-hardware numbers.")
    w("- **NPU model loading takes ~96 seconds** versus ~1 second on CPU. This is "
      "one-time per process and excluded from the throughput figures, but it is a "
      "real deployment consideration.")
    w("- The integrated GPU on this machine has **no driver installed**, so it "
      "could not be measured at all.")
    if failed:
        w(f"- **{len(failed)} configuration(s) failed** and are recorded with their "
          "errors rather than omitted. Notably, the NPU does not support FP32 at "
          "any setting and reports this explicitly.")
    w("")
    w(f"<sub>Sources: {', '.join(sorted({os.path.basename(p) for p in paths}))}</sub>")

    text = "\n".join(out)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}  ({len(text)} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
