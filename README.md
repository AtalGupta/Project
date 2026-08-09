# Qwen2.5-1.5B on Intel CPU / GPU / NPU with OpenVINO

A harness for running Qwen2.5-1.5B across every compute engine an Intel client
machine has, and for finding out which combination is actually fastest — rather
than assuming that using more engines is better.

**Unquantized only: FP32 and FP16.** No INT8, no INT4, no speculative decoding.

That exclusion is deliberate. Both techniques raise tok/s by *changing the
workload* — quantisation makes the model smaller, speculative decoding produces
several tokens per weight sweep — without the silicon having become any faster.
They are legitimate deployment techniques and useless as measurements. This
harness answers "what does this hardware do", so every strategy runs the same
unmodified model.

## The two regimes, and why both are measured

An LLM has two phases with opposite bottlenecks. Measuring only one
characterises half the machine.

**Decode is memory-bandwidth bound.** Each token streams the entire weight set
from DRAM:

| Format | Qwen2.5-1.5B weights | Bytes per token |
|---|---|---|
| FP32 | ~6.2 GB | ~6.2 GB |
| FP16 | ~3.1 GB | ~3.1 GB |

The ceiling is `achieved_bandwidth / model_bytes` — so FP32 costs roughly half
the token rate of FP16 on identical hardware. Compute sits idle here; a 50 TOPS
device and a 5 TOPS device both just wait on DRAM.

**Prefill is compute bound.** The whole prompt is processed at once and weights
are reused across every prompt position, so arithmetic throughput dominates.
This is the regime where TOPS actually shows up, and where devices that look
identical on decode separate widely. Measured by `bench/prefill` as prompt
tokens/second across a prompt-length sweep.

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

| Name | What it does | Regime |
|---|---|---|
| `single` | Whole model on one engine, FP32 **and** FP16 | Decode — the baseline everything else is judged against |
| `prefill` | Prompt-length sweep, prompt tokens/s | **Compute** — where TOPS shows up |
| `cpu_threads` | CPU thread-count sweep | Decode — finds where threads stop buying bandwidth |
| `concurrent` | One pipeline per device, simultaneously | Aggregate — the honest "use all the hardware" |
| `prefill_split` | Best prefill device + best decode device | Upper-bound probe, **not** an implementation |
| `hetero` | Op-level split via `HETERO:A,B`, both orders | Expected to lose — measured anyway |
| `distributed` | `PIPELINE_PARALLEL` / `TENSOR_PARALLEL` | Expected to lose — measured anyway |

### Why splitting one model across devices does not speed up decode

A transformer forward pass is a **serial dependency chain**: layer N+1 consumes
layer N's output. Splitting that chain across devices adds transfers without
creating concurrency.

- `HETERO:A,B` assigns ops by *affinity* and runs subgraphs **sequentially**. It
  is a fallback mechanism, not a parallelism one.
- `PIPELINE_PARALLEL` puts stages on different devices but executes them **one
  by one**; overlap requires multiple requests in flight.
- `TENSOR_PARALLEL` is a genuine intra-op split, but is documented for CPU
  sockets/NUMA and multi-GPU — CPU+NPU is not a documented combination.

`hetero` and `distributed` are measured regardless, in both device orders, so
the claim rests on numbers from the machine rather than on documentation.

The one strategy that genuinely uses two engines at once is `concurrent`, and it
raises *aggregate* throughput across independent requests — never single-stream
latency.

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
