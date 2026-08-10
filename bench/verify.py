"""Output correctness across devices and precisions.

Every number measured so far is throughput. None of it is worth anything if the
model is producing wrong tokens -- a device that is 2x faster and subtly wrong is
not faster, it is broken. This checks that before any optimisation work starts,
and again after any kernel change.

Method: greedy decoding (do_sample=False) is deterministic. Given the same
prompt and the same weights, every device MUST emit the identical token
sequence. Any divergence is either a numerical-precision effect or a bug, and
the position of the first divergence tells you which:

  * identical for all N tokens        -> device is numerically exact here
  * diverges late (token 40 of 64)    -> normal fp accumulation drift; the
                                         argmax flipped on a near-tie
  * diverges at token 0-2             -> something is actually wrong: a bad
                                         kernel, a mis-set precision hint, a
                                         broken layout

FP32 on CPU is the reference, because it is the highest-precision path available
and the one with no silent conversion (verified: 1000 nodes at f32 runtime
precision).

    python -m bench.verify --devices CPU NPU --tokens 64
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

PROMPTS = [
    "Explain in two sentences why memory bandwidth limits LLM token generation.",
    "Write a Python function that reverses a linked list.",
    "What is 17 * 23? Answer with the number only.",
]


def run_one(model_path: str, device: str, prompt: str, tokens: int,
            precision: str | None = None) -> dict:
    import openvino_genai as ov_genai
    from bench.strategies import device_props, resolve

    props = device_props(device)
    if precision:
        props["INFERENCE_PRECISION_HINT"] = precision
    pipe = ov_genai.LLMPipeline(model_path, resolve(device), **props)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = tokens
    cfg.do_sample = False          # greedy -> deterministic
    cfg.ignore_eos = False         # let it stop naturally; we compare real text
    res = pipe.generate([prompt], cfg)
    text = res.texts[0] if getattr(res, "texts", None) else str(res)

    # Token ids are the strict comparison; text is what a human reads.
    ids = []
    try:
        tok = pipe.get_tokenizer()
        ids = list(tok.encode(text).input_ids.data.flatten())
    except Exception:
        pass
    del pipe
    return {"text": text, "ids": ids}


def first_divergence(a: list, b: list) -> int:
    """Index of first differing element, or -1 if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify output correctness.")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-fp32")
    ap.add_argument("--fp16-model", default="models/qwen2.5-1.5b-fp16")
    ap.add_argument("--devices", nargs="*", default=None,
                    help="default: everything available")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", help="write JSON report here")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import kinds_present

    available = kinds_present()
    devices = args.devices or sorted(available)

    # (label, model_path, device, forced precision)
    configs = []
    for d in devices:
        if d not in available:
            print(f"skip {d}: not available")
            continue
        if os.path.isfile(os.path.join(args.model, "openvino_model.xml")):
            configs.append((f"{d}/fp32", args.model, d, None))
        if os.path.isfile(os.path.join(args.fp16_model, "openvino_model.xml")):
            configs.append((f"{d}/fp16", args.fp16_model, d, None))

    if not configs:
        print("no runnable configs")
        return 1

    report = {"tokens": args.tokens, "prompts": PROMPTS, "results": {}}
    all_ok = True

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'=' * 78}")
        print(f"PROMPT {pi + 1}: {prompt}")
        print("=" * 78)

        outputs = {}
        for label, path, dev, prec in configs:
            try:
                outputs[label] = run_one(path, dev, prompt, args.tokens, prec)
                print(f"\n[{label}]")
                print("  " + outputs[label]["text"][:300].replace("\n", "\n  "))
            except Exception as e:
                print(f"\n[{label}] FAILED: {type(e).__name__}: {e}")
                all_ok = False

        if len(outputs) < 2:
            continue

        # Reference is the highest-precision CPU path if present.
        ref_label = next((l for l in outputs if l.startswith("CPU/fp32")),
                         sorted(outputs)[0])
        ref = outputs[ref_label]

        print(f"\n  --- compared against {ref_label} ---")
        for label, o in sorted(outputs.items()):
            if label == ref_label:
                continue
            exact_text = o["text"] == ref["text"]
            div = first_divergence(o["ids"], ref["ids"]) if o["ids"] and ref["ids"] else None
            n = min(len(o["ids"]), len(ref["ids"])) if o["ids"] and ref["ids"] else 0

            if exact_text:
                verdict = "IDENTICAL"
            elif div is None:
                verdict = "text differs (token ids unavailable)"
            elif div < 0:
                verdict = f"prefix match, lengths differ ({len(o['ids'])} vs {len(ref['ids'])})"
            elif div <= 2:
                verdict = f"DIVERGES AT TOKEN {div} -- investigate, this is early"
                all_ok = False
            else:
                verdict = f"diverges at token {div}/{n} -- likely fp drift"

            print(f"  {label:<14} {verdict}")
            report["results"].setdefault(label, []).append(
                {"prompt": pi, "identical": exact_text, "divergence": div})

            if not exact_text and o["text"] and ref["text"]:
                diff = list(difflib.unified_diff(
                    ref["text"].split(), o["text"].split(),
                    lineterm="", n=0))[2:8]
                if diff:
                    print("      " + " ".join(diff)[:220])

    print(f"\n{'=' * 78}")
    print("VERDICT:", "all configs produce sane output" if all_ok
          else "SOMETHING DIVERGED EARLY -- do not trust throughput numbers "
               "from the affected config until resolved")
    print("=" * 78)
    print("Late divergence is expected and fine: fp16 and fp32 accumulate")
    print("differently, so an argmax over near-tied logits can flip. Early")
    print("divergence is not -- it means the computation itself is wrong.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
