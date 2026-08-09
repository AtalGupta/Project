# Running on the cloud node (`BLR01-C01-DPTL8`, Panther Lake, CPU + NPU)

## Read this first: FP16 does not run on the NPU

The NPU compiles LLMs to **static shapes with symmetric channel-wise INT4
weights**. It will not take an FP16 LLM through `LLMPipeline`. So on that node:

| Engine | Format | Status |
|---|---|---|
| CPU | **FP16** | works — same as the laptop |
| NPU | **INT4 symmetric channel-wise** | required; FP16 will fail to compile |
| GPU | — | unavailable: no Intel graphics driver installed |

This is not a limitation of the harness — it is what the NPU plugin accepts.
The harness handles it automatically: `PREFERENCE` in `bench/strategies.py`
gives the NPU `int4-sym-cw` first and FP16 last, so the NPU **still attempts**
FP16 and records the real error rather than us assuming it fails.

If you want NPU numbers at all, you must export the INT4 variant. Step 4 below
does that.

## What to expect (predictions, not measurements)

Laptop baseline for comparison: 50.5 GB/s measured, FP16 ceiling 16 tok/s,
GPU achieved 15.1 tok/s (93% of ceiling).

The cloud node has **LPDDR5X at 7467 MT/s across 8 subchannels** — roughly
**~120 GB/s theoretical**, about 2× the laptop. Predicted:

| Config | Model bytes | Predicted ceiling | Realistic |
|---|---|---|---|
| CPU FP16 | 3.10 GB | ~25–29 tok/s | 15–25 tok/s |
| NPU INT4 | ~1.0 GB | ~60–90 tok/s | 25–45 tok/s |
| CPU INT4 | ~1.0 GB | ~60–90 tok/s | 25–40 tok/s |

Two things that could pull these down, and the reason they are predictions:

- **ES silicon.** `Genuine Intel(R) 0000` at 2.0 GHz nominal is pre-production
  and may be clocked below retail Panther Lake.
- **12C/12T, no HT, no AVX-512, no AMX.** The CPU has fewer threads than the
  laptop's 20 and no AMX-INT8, so CPU numbers may not scale with bandwidth.

Where the NPU should clearly win is **TTFT**: prefill is compute-bound and NPU 5
is ~50 TOPS. Expect a large TTFT improvement over CPU even if decode tok/s is
comparable.

## Step by step

Run these in `cmd.exe` or PowerShell on the node.

### 1. Get the code onto the node

The node already has git. From your laptop, push this repo somewhere you can
reach, then on the node:

```
cd %USERPROFILE%\Desktop
git clone <your-repo-url> ovbench
cd ovbench
```

If there is no shared remote, copying the `bench\` folder and `README.md` is
enough — that is the entire harness (7 files, no build step).

### 2. Fix the power plan (do this before any measurement)

The node is on **Balanced**. On a 2.0 GHz ES part that makes every number drift
between runs.

```
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
powercfg /getactivescheme
```

### 3. Install the one missing package

The node already has openvino 2026.3.0, nncf 3.3.0, optimum-intel 2.1.0,
torch 2.13.0 and transformers 5.5.4 — matching the laptop exactly. Only
`openvino-genai` is missing:

```
python -m pip install openvino-genai==2026.3.0.0
python -m bench.devices
```

Expected output: `devices : ['CPU', 'NPU']`. If NPU is absent, stop — check
`Get-PnpDevice -Class ComputeAccelerator` and the driver before continuing.

> Using the existing environment rather than a fresh venv is deliberate: it
> avoids re-downloading ~500 MB of torch and keeps package versions identical
> to the laptop, so any difference in results is hardware, not software.

### 4. Export the models

FP16 for CPU, INT4-sym for the NPU. This downloads ~3 GB and takes a while.

```
python -m bench.export --only qwen2.5-1.5b-fp16 qwen2.5-0.5b-fp16
python -m bench.export --only qwen2.5-1.5b-int4-sym-cw qwen2.5-0.5b-int4-sym-cw
python -m bench.export --check-only
```

The second command is what makes NPU results possible. Skipping it means every
NPU run fails.

### 5. Measure the bandwidth ceiling

This is the number every result gets judged against. Do it with the machine
otherwise idle — it is a shared node.

```
python -m bench.roofline --model models\qwen2.5-1.5b-fp16 --json results\roofline.json
```

Record the achieved GB/s. If it comes back near the laptop's 50 GB/s rather
than the predicted ~90+, that alone explains any disappointing tok/s and is
worth knowing before anything else is tuned.

### 6. Verify optimised kernels (and see what the NPU actually does)

```
python -u -m bench.kernels --model models\qwen2.5-1.5b-fp16       --device CPU
python -u -m bench.kernels --model models\qwen2.5-1.5b-int4-sym-cw --device NPU
```

On the laptop's CPU this found **RoPE running on `ref_any_f32`** — 56 nodes on a
reference kernel. Check whether Panther Lake's CPU does the same. The NPU output
is genuinely unknown territory; whatever it prints is new information.

Remember the classification rule this tool encodes: `ref_*` is a real reference
fallback, `undef_*` only means the implementation type was not reported (the
modern FullyConnected/SDPA executors report that way) and is **not** a problem.

### 7. Sweep the strategies

```
python -u -m bench.run --list
python -u -m bench.run --reps 3 --out results\sweep-cloud.csv
python -m bench.report results\sweep-cloud.csv
```

`--list` first, always: it shows exactly what will run and what got skipped for
missing hardware, before you commit an hour to it.

On this node the sweep will cover `single` (CPU, NPU), `spec` in both
directions, `concurrent` (CPU+NPU together), `cpu_threads`, and `hetero`.

### 8. The interesting result

The one genuinely open question is **speculative decoding with an NPU draft**.
Prior art ([openvino#36484](https://github.com/openvinotoolkit/openvino/discussions/36484))
measured NPU-drafting as a net **loss** (0.55–0.74×) against a GPU target on
Lunar Lake — but that was **NPU 3 at 13 TOPS**. This node is **NPU 5 at ~50
TOPS**, roughly 4× the compute. The conclusion may invert.

Read acceptance rate and tok/s as separate signals:

- low acceptance → the draft model is proposing badly
- high acceptance but low tok/s → the draft is fine, it is just too slow to
  amortise, which is exactly the failure mode #36484 documented

Caveat on scale: that study used a **14B** target, where verification is
expensive enough to hide draft latency. A 1.5B target verifies cheaply, so the
margin for speculative decoding is much thinner. On the laptop it is already
**losing** (13.2 tok/s vs 15.1 GPU-only). Do not expect a large win; expect a
clean measurement.

## If something fails

| Symptom | Cause | Fix |
|---|---|---|
| `available_devices` shows no NPU | driver or permissions | check `Get-PnpDevice -Class ComputeAccelerator` shows `Intel(R) NPU` / Status OK |
| NPU run fails to compile | FP16 model on NPU | export `int4-sym-cw` (step 4) |
| NPU fails with shape error | prompt exceeds static bounds | raise `NPU_MAX_PROMPT_LEN` in `bench/strategies.py` |
| Numbers drift run to run | Balanced power plan, or node is shared | step 2; confirm the box is idle |
| tok/s far below prediction | bandwidth lower than assumed | compare step 5's GB/s against the prediction table |

Every failure is recorded in the CSV rather than aborting the sweep — a config
that cannot run is data, not a crash.
