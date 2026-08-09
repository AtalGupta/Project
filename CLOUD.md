# Running on the cloud node (`BLR01-C01-DPTL8`, Panther Lake, CPU + NPU)

Target: **Qwen2.5-1.5B in FP32, unquantized**, using OpenVINO's own kernels,
across CPU and NPU.

## Three things that constrain this, up front

### 1. The NPU has no FP32 compute

The NPU's hardware computation precision is **FP16**. An FP32 IR does not
execute as FP32 there — it is converted down. "FP32 on NPU" is not a thing the
silicon offers, regardless of what the IR file contains. Only the CPU executes
FP32 natively.

Worse, for LLMs the NPU path expects INT4-symmetric, and there is an open bug
where [`LLMPipeline(ir, "NPU")` silently accepts a non-INT4 IR and then dies
with an uncatchable `0xC0000005` at `generate()`](https://github.com/openvinotoolkit/openvino/issues/35641).
So an unquantized model on the NPU may **hard-crash the process** rather than
raise a Python exception.

The harness runs each repetition in a separate process precisely so this cannot
take the sweep down with it — a crashed rep is recorded as a failed row and the
sweep continues.

### 2. FP32 costs about half the token rate of FP16

Decode is memory-bandwidth bound: every token streams the whole weight set.

| Format | Weights | Ceiling at ~120 GB/s |
|---|---|---|
| FP32 | ~6.2 GB | ~19 tok/s |
| FP16 | ~3.1 GB | ~38 tok/s |

This is arithmetic, not tuning. Choosing FP32 is choosing to halve throughput in
exchange for full precision. That is a legitimate trade — just make it knowingly.
Measured on the laptop for reference: FP16 hit 15.1 tok/s at **93% of its
measured ceiling**, so there was no headroom being wasted.

### 3. Op-level CPU+NPU parallelism will not speed up decode

A transformer forward pass is a **serial dependency chain**: layer N+1 consumes
layer N's output. Splitting that chain across two devices adds transfers without
creating concurrency.

What the two available mechanisms actually do:

| Mechanism | Behaviour | Helps single-stream decode? |
|---|---|---|
| `HETERO:A,B` | assigns ops by *affinity*, runs subgraphs **sequentially** | no — it is a fallback mechanism, not a parallelism one |
| `PIPELINE_PARALLEL` | stages on different devices, executed **one by one** | no — overlap requires multiple requests in flight |
| `TENSOR_PARALLEL` | genuine intra-op split | documented for CPU sockets/NUMA and multi-GPU; **CPU+NPU is not a documented combination** |

Both are still measured (`hetero` and `distributed` strategies, in both device
orders) because a number from this machine beats an argument from documentation.

**What genuinely uses both devices at once:**

- `concurrent` — separate requests on CPU and NPU simultaneously. Raises total
  throughput, capped by shared bandwidth. This is the honest version of "use all
  the hardware".
- `spec` — draft on one device, target on the other. The only mechanism that
  beats the bandwidth wall on a single stream, by reducing target-model weight
  sweeps per accepted token. **No quantization required** — the FP32/FP16 draft
  works fine.
- `prefill_split` — prefill is compute-bound and the NPU is ~50 TOPS, so NPU
  prefill should win TTFT decisively even when decode does not improve.

## Expected numbers (predictions)

The node has LPDDR5X-7467 across 8 subchannels ≈ **~120 GB/s theoretical**,
roughly 2× the laptop's measured 50.5 GB/s.

| Config | Bytes/token | Predicted ceiling | Realistic |
|---|---|---|---|
| CPU FP32 | 6.2 GB | ~19 tok/s | 8–14 |
| CPU FP16 | 3.1 GB | ~38 tok/s | 15–25 |
| NPU (FP16 after conversion) | 3.1 GB | ~38 tok/s | may not compile at all |

Two reasons these could come in low: it is **ES silicon** (`Genuine Intel(R)
0000`, 2.0 GHz nominal), and the CPU is **12C/12T with no AVX-512 and no AMX**
versus the laptop's 20 threads. CPU throughput may not scale with bandwidth.

## Step by step

### 1. Code onto the node

```
cd %USERPROFILE%\Desktop
git clone <your-repo-url> ovbench
cd ovbench
```

The harness is 8 files under `bench\` with no build step — copying that folder
is equally fine.

### 2. Power plan (before any measurement)

The node is on **Balanced**, which makes a 2.0 GHz ES part drift between runs.

```
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
powercfg /getactivescheme
```

### 3. Install the one missing package

The node already has openvino 2026.3.0, nncf 3.3.0, optimum-intel 2.1.0,
torch 2.13.0, transformers 5.5.4 — identical to the laptop. Reuse that
environment rather than building a venv, so any difference in results is
hardware rather than software.

```
python -m pip install openvino-genai==2026.3.0.0
python -m bench.devices
```

Expect `devices : ['CPU', 'NPU']`.

### 4. Export FP32 (the default — no quantized model is produced)

```
python -u -m bench.export
python -m bench.export --check-only
```

That exports `qwen2.5-1.5b-fp32` (~6.2 GB) and `qwen2.5-0.5b-fp32` (~2.0 GB) for
speculative decoding. Nothing compressed is created unless you explicitly ask.

### 5. Bandwidth ceiling

```
python -m bench.roofline --model models\qwen2.5-1.5b-fp32 --json results\roofline.json
```

Run it with the node idle — it is shared. If this returns near 50 GB/s rather
than the predicted ~90+, that alone explains any disappointing tok/s, and is
worth knowing before tuning anything.

### 6. Verify OpenVINO kernels

```
python -u -m bench.kernels --model models\qwen2.5-1.5b-fp32 --device CPU
python -u -m bench.kernels --model models\qwen2.5-1.5b-fp32 --device NPU
```

Reading the output: `ref_*` is a genuine reference fallback; `undef_*` only means
the implementation type was not reported (the modern FullyConnected/SDPA
executors do this) and is **not** a problem. On the laptop's CPU this found
**RoPE on `ref_any_f32`**, 56 nodes — check whether Panther Lake does the same.

The NPU run is the interesting one, and may crash per constraint 1. That is
itself a result worth recording.

### 7. Sweep

```
python -u -m bench.run --list
python -u -m bench.run --reps 3 --out results\sweep-cloud.csv
python -m bench.report results\sweep-cloud.csv
```

Always `--list` first — it shows exactly what will run and what was skipped for
missing hardware before you commit an hour.

On this node that is 20 runs: `single` (CPU, NPU), `spec` both directions ×3
draft lengths, `prefill_split`, `concurrent`, `cpu_threads` ×4, `hetero` ×2
(NPU-first and CPU-first), `distributed` ×4 (both orders × both policies).

## What the laptop already tells us

Measured on Iris Xe + i7-12700H, FP16, 3 reps each:

| Config | tok/s |
|---|---|
| GPU alone | **15.1** |
| CPU alone | 12.7 |
| GPU target ← CPU draft, nat=1 | 12.9 |
| GPU target ← CPU draft, nat=3 | 12.0 |
| CPU target ← GPU draft, nat=3 | 9.9 |

Speculative decoding **loses**, and loses more as draft length grows. The reason
is scale: the [prior characterization](https://github.com/openvinotoolkit/openvino/discussions/36484)
that found a 1.35× win used a **14B** target, whose verification pass is
expensive enough to hide draft latency. A **1.5B** target verifies too cheaply
for there to be anything to amortize.

Do not expect the cloud node to overturn this. The NPU being ~50 TOPS helps
draft *latency*, but the structural problem is that the target is small, and
that does not change.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| no NPU in `available_devices` | driver/permissions | `Get-PnpDevice -Class ComputeAccelerator` should show `Intel(R) NPU`, Status OK |
| NPU rep dies, exit `-1073741819` | [#35641](https://github.com/openvinotoolkit/openvino/issues/35641), unquantized IR on NPU | expected; recorded as a failed row. Only INT4-sym avoids it |
| NPU shape error | prompt exceeds static bounds | raise `NPU_MAX_PROMPT_LEN` in `bench/strategies.py` |
| numbers drift run to run | Balanced plan, or shared node | step 2; confirm the box is idle |
| tok/s far below prediction | bandwidth lower than assumed | compare step 5 against the prediction table |

Failures are written to the CSV rather than aborting the sweep — a config that
cannot run is data, not a crash.
