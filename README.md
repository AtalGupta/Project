# Qwen2.5-1.5B on Intel CPU / GPU / NPU with OpenVINO

A harness for running Qwen2.5-1.5B across every compute engine an Intel client
machine has, and for finding out which combination is actually fastest — rather
than assuming that using more engines is better.

Primary model format is **FP16**. Quantised variants are available for
comparison and because the NPU requires one.

## The one thing to understand first

LLM *decode* is **memory-bandwidth bound**, not compute bound. Generating one
token requires streaming the entire weight set from DRAM:

| Format | Qwen2.5-1.5B weights | Bytes per token |
|---|---|---|
| FP16 | ~3.1 GB | ~3.1 GB |
| INT8 | ~1.6 GB | ~1.6 GB |
| INT4 | ~1.0 GB | ~1.0 GB |

So the token-rate ceiling is `achieved_bandwidth / model_bytes`. At ~100 GB/s
that is ~32 tok/s for FP16 and ~100 tok/s for INT4, regardless of how much
compute is available.

On Intel client parts the CPU, integrated GPU and NPU **share one memory
controller** and have no dedicated VRAM. Two consequences follow, and they shape
everything in this repo:

1. **Running two engines on one token stream does not make it faster.** They
   contend for the same bytes. Only strategies that reduce *bytes per token*
   (quantisation, speculative decoding) beat this wall.
2. **Running independent streams on separate engines does raise total
   throughput**, until the controller saturates.

`bench/roofline.py` measures the ceiling so every result can be reported as a
percentage of it. Without that number there is no way to distinguish "we
optimised it" from "we hit physics".

## Are these OpenVINO's optimised kernels?

Yes. Everything runs through `openvino_genai.LLMPipeline` → the plugin compiler
→ that plugin's kernels (oneDNN JIT on CPU, the `kernel_selector` OpenCL kernels
on GPU, a driver-compiled blob on NPU). No tensor math happens in Python.

The prebuilt `pip install openvino` wheel contains the same optimised kernels as
a source build; building from source is only needed to *modify* them.

But an op can silently fall back to a **reference kernel** and cost ~10× with no
error. Verify with:

```bash
python -m bench.kernels --model models/qwen2.5-1.5b-fp16 --device GPU
```

This prints the actual `execType` the plugin chose per node and flags reference
fallbacks on hot ops. Run it before trusting any benchmark.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install openvino openvino-genai openvino-tokenizers \
    nncf "optimum-intel[openvino]" transformers torch

python -m bench.devices        # what hardware is actually usable here
```

`bench/devices.py` filters out non-Intel GPUs — the `intel_gpu` plugin
enumerates discrete NVIDIA/AMD cards over OpenCL but cannot execute on them.

## Workflow

```bash
# 1. export (FP16 pair by default; --all adds INT8/INT4 variants)
python -m bench.export
python -m bench.export --all

# 2. measure the bandwidth ceiling for this machine + model
python -m bench.roofline --model models/qwen2.5-1.5b-fp16 --json results/roofline.json

# 3. confirm optimised kernels, no reference fallbacks
python -m bench.kernels --model models/qwen2.5-1.5b-fp16 --device GPU

# 4. sweep the stitching strategies
python -m bench.run --list      # what would run on this machine
python -m bench.run --reps 3

# 5. read the results
python -m bench.report
```

## Strategies

| Name | What it does | Expectation |
|---|---|---|
| `single` | Whole model on one engine | Baseline for everything else |
| `spec` | 0.5B draft + 1.5B target, across device pairs | **The key experiment** — the only mechanism that beats the bandwidth wall on a single stream |
| `prefill_split` | Best prefill device + best decode device | **Upper bound probe, not an implementation** — GenAI cannot split one pipeline across devices |
| `concurrent` | One pipeline per device, simultaneously | Best aggregate throughput; saturates at the roofline |
| `cpu_threads` | CPU thread-count sweep | Finds where CPU threads start *stealing* bandwidth from other engines |
| `hetero` | Op-level graph split via HETERO | Expected to **lose**; measured so the claim has a number behind it |

On `spec`: prior art ([openvino#36484](https://github.com/openvinotoolkit/openvino/discussions/36484))
found NPU-drafting was a net loss (0.55–0.74×) while CPU-drafting won (1.35×) —
but that was NPU 3 at 13 TOPS. On newer NPUs this may invert. Acceptance rate and
throughput are reported separately because they diagnose different failures: low
acceptance means a bad draft, high acceptance with low throughput means the draft
is too slow to amortise.

## Measurement hygiene

Non-negotiable, and the reason `bench/run.py` is more than a for-loop:

- every repetition runs in a **separate process** — compiled models and kernel
  caches persist within a process and make later runs look faster
- one warmup discarded, ≥3 measured reps, reported as mean ± stdev
- compile time captured separately, never folded into throughput
- `ignore_eos=True` and greedy sampling so every run generates the same token
  count
- strategies with absent devices are recorded as **skipped**, not dropped

## Layout

```
bench/
  devices.py     device enumeration, Intel-GPU filtering, FP16 capability
  export.py      model export, idempotent, tokenizer-identity assertion
  roofline.py    STREAM triad -> bandwidth -> tok/s ceiling
  kernels.py     which kernel each op got; reference-fallback detection
  strategies.py  the six stitching strategies
  run.py         sweep driver, process isolation, CSV output
  report.py      results table with % of roofline
models/          exported OpenVINO IR
results/         roofline.json, sweep CSVs
```
