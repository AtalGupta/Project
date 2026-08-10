"""Full CPU vs NPU vs SPLIT matrix, with per-input-length bucketing.

The question: across realistic input and output lengths, which of
    CPU native   /   NPU native   /   SPLIT (attention=CPU, FFN=NPU)
is fastest end to end -- and does the split beat BOTH natives?

Three things this does that earlier tools did not:

  * BUCKETS THE NPU AND THE SPLIT per input length. Measured on this hardware,
    NPU TTFT tracks MAX_PROMPT_LEN, not the real prompt: bucket 128 -> ~92 ms,
    bucket 256 -> ~140 ms, bucket 1024 -> ~625 ms, for the SAME 64-token prompt.
    A fixed 1024 bucket therefore taxes every short request ~7x. The split
    inherited that tax because device_props() hardcoded 1024; here each input
    length gets its own compile.

  * REPORTS END-TO-END LATENCY, not just TTFT and ITL separately. TTFT + n_out *
    ITL is what a user waits, and it is the only number that can declare a
    winner. The NPU trades better ITL for worse TTFT, so which config wins
    depends on output length -- and that crossover is the actual result.

  * RESUMES. The full sweep is hours, most of it NPU compilation. Every row is
    appended to CSV as it completes and re-runs skip finished cells, so a crash
    or a device hang costs one cell rather than the run.

Compile budgeting: MIN_RESPONSE_LEN is set once to the LARGEST output length, so
one compile per (config, input) serves all output lengths. Compiling per output
length would triple an already long sweep for no measurement benefit -- that
property sizes the KV cache, it does not change per-token work.

    python -u -m bench.matrix --csv results\\matrix.csv
    python -u -m bench.matrix --inputs 128 256 512 --outputs 128 --configs cpu npu
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUTS = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_OUTPUTS = (128, 256, 512)

FIELDS = ["config", "device", "model", "bucket", "input_len", "real_input_tokens",
          "output_len", "ttft_ms", "ttft_sd", "itl_ms", "itl_sd",
          "e2e_ms", "tok_s", "compile_s", "reps", "error"]


def configs_for(available: set[str], model: str, split_model: str) -> list[dict]:
    """Which pipelines can run here. Split needs both devices and a split IR."""
    out = []
    if "CPU" in available:
        out.append({"name": "cpu", "device": "CPU", "model": model})
    if "NPU" in available:
        out.append({"name": "npu", "device": "NPU", "model": model})
    if "NPU" in available and "CPU" in available and \
            os.path.isfile(os.path.join(split_model, "openvino_model.xml")):
        out.append({"name": "split", "device": "HETERO:NPU,CPU",
                    "model": split_model})
    # GPU only appears on the laptop; included so the same script works there.
    if "GPU" in available:
        out.append({"name": "gpu", "device": "GPU", "model": model})
        if os.path.isfile(os.path.join(split_model, "openvino_model.xml")):
            out.append({"name": "split_gpu", "device": "HETERO:GPU,CPU",
                        "model": split_model})
    return out


def build_pipe(model_dir: str, device: str, bucket: int, min_response: int):
    import openvino_genai as ov_genai

    props = {}
    if "NPU" in device.upper():
        # Only the NPU plugin consumes these; passing them elsewhere errors.
        props["MAX_PROMPT_LEN"] = bucket
        props["MIN_RESPONSE_LEN"] = min_response
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(model_dir, device, **props)
    return pipe, time.perf_counter() - t0


def run_cell(pipe, text: str, n_out: int, reps: int, warmup: int = 3) -> dict:
    import openvino_genai as ov_genai

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = n_out
    cfg.do_sample = False
    cfg.ignore_eos = True            # fixed token count, else ITL is unstable

    # TTFT needs several warmups, not one. A 3.1 GB model is still paging into
    # resident memory on the first inferences, which inflates time-to-first-token
    # while leaving ITL roughly correct. Reference runs that sweep warmup counts
    # show TTFT settling by about the fifth iteration; a single warmup measured
    # 534 ms where a warm run gives ~190 ms for the same prompt.
    #
    # But warming does not require GENERATING the full output -- it only has to
    # touch the weights. So do the repeated warmups with a 1-token generation
    # (cheap, and it exercises exactly the prefill path TTFT measures), then one
    # full-length pass so the decode loop is warm too. At output=512 this turns
    # ~2 minutes of warmup per cell into ~25 seconds.
    warm_cfg = ov_genai.GenerationConfig()
    warm_cfg.max_new_tokens = 1
    warm_cfg.do_sample = False
    warm_cfg.ignore_eos = True
    for _ in range(max(1, warmup)):
        pipe.generate([text], warm_cfg)
    pipe.generate([text], cfg)          # one full-length pass, discarded
    ttfts, itls, ntok = [], [], 0
    for _ in range(reps):
        res = pipe.generate([text], cfg)
        pm = res.perf_metrics
        ttfts.append(pm.get_ttft().mean)
        itls.append(pm.get_tpot().mean)
        ntok = pm.get_num_input_tokens()

    ttft = statistics.median(ttfts)
    itl = statistics.median(itls)
    return {
        "real_input_tokens": ntok,
        "ttft_ms": round(ttft, 2),
        "ttft_sd": round(statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0, 2),
        "itl_ms": round(itl, 3),
        "itl_sd": round(statistics.stdev(itls) if len(itls) > 1 else 0.0, 3),
        "e2e_ms": round(ttft + n_out * itl, 1),
        "tok_s": round(1000.0 / itl, 2) if itl else 0.0,
    }


def load_done(path: str) -> set[tuple]:
    done = set()
    if not os.path.isfile(path):
        return done
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("error"):
                    continue          # retry failures on a later run
                done.add((r["config"], int(r["input_len"]), int(r["output_len"])))
    except Exception:
        pass
    return done


def append_row(path: str, row: dict) -> None:
    new = not os.path.isfile(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU vs NPU vs SPLIT full matrix.")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-fp16")
    ap.add_argument("--split-model", default="models/qwen2.5-1.5b-fp16-split")
    ap.add_argument("--inputs", type=int, nargs="*", default=list(DEFAULT_INPUTS))
    ap.add_argument("--outputs", type=int, nargs="*", default=list(DEFAULT_OUTPUTS))
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset: cpu npu split gpu split_gpu")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3,
                    help="discarded iterations before timing; TTFT needs >=3")
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "matrix.csv"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from bench.devices import kinds_present
    from bench.buckets import prompt_exact

    available = kinds_present()
    cfgs = configs_for(available, args.model, args.split_model)
    if args.configs:
        cfgs = [c for c in cfgs if c["name"] in args.configs]
    if not cfgs:
        print(f"no runnable configs. devices={sorted(available)}")
        return 1

    inputs = sorted(args.inputs)
    outputs = sorted(args.outputs)
    max_out = max(outputs)
    done = set() if args.no_resume else load_done(args.csv)

    total = len(cfgs) * len(inputs) * len(outputs)
    print(f"devices    : {sorted(available)}")
    print(f"configs    : {[c['name'] for c in cfgs]}")
    print(f"inputs     : {inputs}")
    print(f"outputs    : {outputs}")
    print(f"cells      : {total}  ({len(done)} already done, resuming)"
          if done else f"cells      : {total}")
    print(f"buckets    : MAX_PROMPT_LEN set per input length; "
          f"MIN_RESPONSE_LEN={max_out} for all")
    print(f"csv        : {args.csv}")
    print()

    started = time.time()
    n = 0
    for cfg in cfgs:
        for n_in in inputs:
            pending = [o for o in outputs
                       if (cfg["name"], n_in, o) not in done]
            if not pending:
                n += len(outputs)
                continue

            # One compile per (config, input): the bucket must be >= the real
            # token count, and the chat template adds ~25 tokens on top.
            bucket = n_in
            print(f"[{cfg['name']}] input={n_in} bucket={bucket} -> compiling ...",
                  end="", flush=True)
            try:
                pipe, compile_s = build_pipe(cfg["model"], cfg["device"],
                                             bucket, max_out)
                print(f" {compile_s:.1f}s")
            except Exception as e:
                print(f" FAILED: {type(e).__name__}: {str(e)[:80]}")
                for o in pending:
                    append_row(args.csv, {
                        "config": cfg["name"], "device": cfg["device"],
                        "model": os.path.basename(cfg["model"]), "bucket": bucket,
                        "input_len": n_in, "output_len": o, "reps": args.reps,
                        "error": f"compile: {type(e).__name__}: {str(e)[:120]}"})
                    n += 1
                continue

            # Size the prompt against the real budget, allowing for the template.
            try:
                text, planned = prompt_exact(pipe, max(1, n_in - 30))
            except Exception:
                text, planned = "hello " * (n_in // 2), -1

            for o in pending:
                n += 1
                el = time.time() - started
                print(f"  ({n}/{total}, {el/60:.0f}m) out={o} ...",
                      end="", flush=True)
                row = {"config": cfg["name"], "device": cfg["device"],
                       "model": os.path.basename(cfg["model"]), "bucket": bucket,
                       "input_len": n_in, "output_len": o, "reps": args.reps,
                       "compile_s": round(compile_s, 1)}
                try:
                    row.update(run_cell(pipe, text, o, args.reps, args.warmup))
                    print(f" TTFT {row['ttft_ms']:8.1f}  ITL {row['itl_ms']:6.2f}"
                          f"  e2e {row['e2e_ms']/1000:7.2f}s")
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                    print(f" FAILED: {row['error'][:70]}")
                append_row(args.csv, row)

            del pipe
            gc.collect()

    print(f"\ndone in {(time.time() - started)/60:.0f} min -> {args.csv}")
    print(f"render with:  python -m bench.matrix_report {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
