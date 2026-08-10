# Qwen2.5-1.5B on Intel CPU + NPU — measurement summary

Unquantized model (FP32 and FP16), OpenVINO's own kernels, no speculative decoding. Every figure is the mean of 3 repetitions, each run in a separate process, with one warmup discarded and compile time excluded.

## Headline

- **Fastest configuration: NPU/qwen2.5-1.5b-fp16 at 20.7 tokens/s.**
- NPU is **1.38x** the CPU on text generation (20.7 vs 15.0 tokens/s).
- Memory bandwidth measured at **93 GB/s**, which caps text generation regardless of how much compute is available.

## Text generation (decode)

This is the phase that produces the answer, one token at a time. It is limited by **memory bandwidth**, not by processing power: every token requires reading the entire model from memory.

| Configuration | Tokens/s | +/- | % of hardware limit | Time to first token |
|---|---:|---:|---:|---:|
| NPU/qwen2.5-1.5b-fp16 | 20.68 | 0.09 | 69% | 631 ms |
| CPU/qwen2.5-1.5b-fp16 | 14.97 | 0.05 | 50% | 1919 ms |
| CPU/qwen2.5-1.5b-fp32 | 8.62 | 0.15 | 57% | 1560 ms |
| CPU/qwen2.5-1.5b-fp32 forced=f32 | 8.55 | 0.07 | 57% | 1575 ms |
| NPU/qwen2.5-1.5b-fp32 | 6.65 | 0.02 | 44% | 757 ms |

Hardware limits at the measured bandwidth:
- `qwen2.5-1.5b-fp16` — 3.10 GB per token → **30.0 tokens/s maximum**
- `qwen2.5-1.5b-fp32` — 6.18 GB per token → **15.0 tokens/s maximum**

## Prompt processing (prefill)

This is the phase that reads the question before answering. Unlike generation it is limited by **processing power**, which is what a neural accelerator is built for — so the two phases can give very different answers about which device is better.

| Device | Precision | Prompt | Prompt tokens/s | Time to first token |
|---|---|---:|---:|---:|
| CPU | FP16 | 64 | 44.8 | 1920 ms |
| CPU | FP16 | 256 | 54.5 | 4714 ms |
| CPU | FP16 | 512 | 57.5 | 8448 ms |
| CPU | FP16 | 960 | 58.8 | 15059 ms |
| CPU | FP32 | 64 | 52.9 | 1626 ms |
| CPU | FP32 | 256 | 63.1 | 4076 ms |
| CPU | FP32 | 512 | 65.8 | 7388 ms |
| CPU | FP32 | 960 | 66.1 | 13413 ms |
| NPU | FP16 | 64 | 137.2 | 627 ms |
| NPU | FP16 | 256 | 395.0 | 653 ms |
| NPU | FP16 | 512 | 774.7 | 627 ms |
| NPU | FP16 | 960 | 1420.0 | 624 ms |
| NPU | FP32 | 64 | 112.9 | 762 ms |
| NPU | FP32 | 256 | 341.9 | 752 ms |
| NPU | FP32 | 512 | 645.4 | 753 ms |
| NPU | FP32 | 960 | 1172.8 | 755 ms |

## What this means

- **The NPU's advantage depends entirely on the workload.** For long prompts with short answers — document summarisation, retrieval, classification, code review — prompt processing dominates and the NPU is transformative. For short prompts with long answers — chat, agents — generation dominates, and the advantage is modest because generation is a memory problem rather than a compute one.
- **No compression was used.** These are full-precision results. Quantisation would raise the token rate further, at some cost to output quality; it was deliberately excluded here so the numbers describe the hardware rather than a modified workload.
- **The kernels are Intel's own**, verified per operation from the compiled graph — not a custom or reference implementation.

## Caveats

- The test machine is **pre-production silicon** running below retail clocks, so relative comparisons are sound but absolute figures are not final-hardware numbers.
- **NPU model loading takes ~96 seconds** versus ~1 second on CPU. This is one-time per process and excluded from the throughput figures, but it is a real deployment consideration.
- The integrated GPU on this machine has **no driver installed**, so it could not be measured at all.
- **1 configuration(s) failed** and are recorded with their errors rather than omitted. Notably, the NPU does not support FP32 at any setting and reports this explicitly.

<sub>Sources: cloud-prefill.csv, cloud-single.csv</sub>
