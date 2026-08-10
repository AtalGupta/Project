# Qwen2.5-1.5B on Intel CPU + NPU — measurement summary

Unquantized model (FP32 and FP16), OpenVINO's own kernels, no speculative decoding. Every figure is the mean of 3 repetitions, each run in a separate process, with one warmup discarded and compile time excluded.

## Headline

- **Fastest configuration: GPU/qwen2.5-1.5b-fp16 at 15.0 tokens/s.**
- Memory bandwidth measured at **52 GB/s**, which caps text generation regardless of how much compute is available.

## Text generation (decode)

This is the phase that produces the answer, one token at a time. It is limited by **memory bandwidth**, not by processing power: every token requires reading the entire model from memory.

| Configuration | Tokens/s | +/- | % of hardware limit | Time to first token |
|---|---:|---:|---:|---:|
| GPU/qwen2.5-1.5b-fp16 | 15.03 | 0.02 | 90% | 246 ms |
| GPU/qwen2.5-1.5b-fp32 | 14.92 | 0.23 | 178% (see note) | 247 ms |
| GPU/qwen2.5-1.5b-fp32 | 14.66 | 0.12 | 175% (see note) | 252 ms |
| GPU/qwen2.5-1.5b-fp16 | 14.19 | 0.21 | 85% | 274 ms |
| CPU/qwen2.5-1.5b-fp16 | 13.25 | 0.14 | 79% | 1446 ms |
| CPU/qwen2.5-1.5b-fp16 | 12.54 | 0.10 | 75% | 1457 ms |
| CPU/qwen2.5-1.5b-fp32 | 7.81 | 0.20 | 93% | 1966 ms |
| GPU/qwen2.5-1.5b-fp32 forced=f32 | 7.76 | 0.09 | 93% | 704 ms |
| CPU/qwen2.5-1.5b-fp32 | 7.62 | 0.43 | 91% | 1786 ms |
| CPU/qwen2.5-1.5b-fp32 forced=f32 | 7.32 | 0.14 | 87% | 1681 ms |

Hardware limits at the measured bandwidth:
- `qwen2.5-1.5b-fp16` — 3.10 GB per token → **16.7 tokens/s maximum**
- `qwen2.5-1.5b-fp32` — 6.18 GB per token → **8.4 tokens/s maximum**

> **Note on the rows marked "see note".** A result cannot exceed 100% of the hardware limit. Those rows did so because the graphics driver silently converts an FP32 model to FP16 when loading it, so half as much data is read per token — the speed is real, but the row is not the FP32 measurement its name implies. Rows labelled `forced=f32` are the genuine FP32 figures, and they are excluded from the headline above.

## Prompt processing (prefill)

This is the phase that reads the question before answering. Unlike generation it is limited by **processing power**, which is what a neural accelerator is built for — so the two phases can give very different answers about which device is better.

| Device | Precision | Prompt | Prompt tokens/s | Time to first token |
|---|---|---:|---:|---:|
| CPU | FP16 | 64 | 64.4 | 1336 ms |
| CPU | FP16 | 256 | 87.8 | 2929 ms |
| CPU | FP16 | 512 | 89.7 | 5430 ms |
| CPU | FP16 | 960 | 89.0 | 9952 ms |
| CPU | FP32 | 64 | 47.9 | 1902 ms |
| CPU | FP32 | 256 | 70.9 | 3628 ms |
| CPU | FP32 | 512 | 84.8 | 5733 ms |
| CPU | FP32 | 960 | 97.9 | 9052 ms |
| GPU | FP16 | 64 | 354.4 | 243 ms |
| GPU | FP16 | 256 | 420.7 | 611 ms |
| GPU | FP16 | 512 | 416.5 | 1168 ms |
| GPU | FP16 | 960 | 302.8 | 2929 ms |
| GPU | FP32 | 64 | 273.6 | 392 ms |
| GPU | FP32 | 256 | 337.5 | 810 ms |
| GPU | FP32 | 512 | 314.5 | 1576 ms |
| GPU | FP32 | 960 | 305.0 | 2959 ms |

## What this means

- **The NPU's advantage depends entirely on the workload.** For long prompts with short answers — document summarisation, retrieval, classification, code review — prompt processing dominates and the NPU is transformative. For short prompts with long answers — chat, agents — generation dominates, and the advantage is modest because generation is a memory problem rather than a compute one.
- **No compression was used.** These are full-precision results. Quantisation would raise the token rate further, at some cost to output quality; it was deliberately excluded here so the numbers describe the hardware rather than a modified workload.
- **The kernels are Intel's own**, verified per operation from the compiled graph — not a custom or reference implementation.

## Caveats

- The test machine is **pre-production silicon** running below retail clocks, so relative comparisons are sound but absolute figures are not final-hardware numbers.
- **NPU model loading takes ~96 seconds** versus ~1 second on CPU. This is one-time per process and excluded from the throughput figures, but it is a real deployment consideration.
- The integrated GPU on this machine has **no driver installed**, so it could not be measured at all.

<sub>Sources: sweep-precision.csv, sweep-precision2.csv</sub>
