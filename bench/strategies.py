"""The stitching strategies.

Each strategy is one way of mapping Qwen2.5-1.5B onto the machine's engines.
They share a `run()` contract so `run.py` can sweep them uniformly, and each
declares the device kinds it needs so absent hardware is skipped rather than
crashed on.

Ordering below is the order they should be trusted in. Read the class docstrings
before interpreting any number -- several of these are expected to LOSE, and
knowing which is the point of measuring them.
"""

from __future__ import annotations

import math
import os
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict

NAN = float("nan")

# One fixed prompt everywhere, so numbers are comparable across strategies.
# Long enough that prefill is a real cost, short enough to stay inside the NPU's
# static MAX_PROMPT_LEN without inflating compile time.
PROMPT = (
    "You are a systems engineer. Explain, step by step and in concrete terms, "
    "how memory bandwidth limits the token generation rate of a quantised large "
    "language model running on an integrated GPU, and what a developer can "
    "actually do about it. Be specific and technical."
)

DEFAULT_MAX_NEW_TOKENS = 128
NPU_MAX_PROMPT_LEN = 1024
NPU_MIN_RESPONSE_LEN = 256

# Which exported variant each device should use, in preference order.
#
# FP16 first: that is the requested target. The fallbacks matter because the NPU
# cannot take FP16 LLM weights through the GenAI pipeline -- it compiles to
# static shapes and requires symmetric channel-wise INT4 -- so forcing FP16
# everywhere would simply delete the NPU from the results. Listing FP16 last for
# the NPU rather than omitting it means we still ATTEMPT it and record the real
# error, instead of assuming.
# Unquantized only, and the SAME model on every device.
#
# Giving different devices different models would make the comparison
# meaningless: a device running a smaller model looks faster without being
# faster. Every device gets fp32 if it is exported, else fp16 -- never a
# compressed variant, and never a different one per device.
PREFERENCE = ["qwen2.5-1.5b-fp32", "qwen2.5-1.5b-fp16"]


def pick_model(kind: str, models: dict) -> str | None:
    """First exported variant available, identical across devices.

    Used by the multi-device strategies, where sweeping precision too would
    multiply an already large config count for little extra insight.
    """
    for key in PREFERENCE:
        if key in models:
            return key
    return None


def all_models(models: dict) -> list[str]:
    """Every exported unquantized variant, in PREFERENCE order.

    The core per-device strategies sweep ALL of these, because fp32-vs-fp16 is
    itself one of the measurements: it separates "what precision does this
    silicon execute natively" from "how many bytes must cross the memory bus".
    Those two effects are easy to confuse and only a side-by-side separates them.
    """
    return [k for k in PREFERENCE if k in models]


def resolve(kind: str) -> str:
    """Kind -> concrete OpenVINO device string, e.g. GPU -> 'GPU.0'.

    Never pass a bare kind straight to LLMPipeline: on a machine with both an
    Intel iGPU and a discrete non-Intel card, "GPU" is ambiguous and the plugin's
    default is not guaranteed to be the one we filtered for.
    """
    from bench import devices as _d

    return _d.pick(kind) or kind


@dataclass
class RunResult:
    strategy: str
    label: str
    devices: str
    ok: bool = False
    error: str = ""
    ttft_ms: float = NAN
    tpot_ms: float = NAN
    throughput_tok_s: float = NAN
    generated_tokens: int = 0
    compile_s: float = NAN
    acceptance_rate: float = NAN
    aggregate_tok_s: float = NAN
    output_head: str = ""
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d["extra"] = repr(d["extra"]) if d["extra"] else ""
        return d


def _gen_config(max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, **kw):
    """Greedy + ignore_eos so every run generates exactly the same token count.

    Without ignore_eos a run that stops early looks artificially fast, and TPOT
    gets computed over too few samples to be stable.
    """
    import openvino_genai as ov_genai

    c = ov_genai.GenerationConfig()
    c.max_new_tokens = max_new_tokens
    c.do_sample = False
    c.ignore_eos = True
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def device_props(kind: str, **overrides) -> dict:
    """Per-device properties. The NPU ones are mandatory, not tuning:
    it compiles to static shapes and refuses to run without bounds."""
    p: dict = {}
    if kind.upper().startswith("NPU"):
        p["MAX_PROMPT_LEN"] = NPU_MAX_PROMPT_LEN
        p["MIN_RESPONSE_LEN"] = NPU_MIN_RESPONSE_LEN
    p.update(overrides)
    return p


def _collect(res, compile_s: float, strategy: str, label: str, devices: str) -> RunResult:
    """Pull metrics off a DecodedResults.

    Note: LLMPipeline.generate() returns a bare `str` when handed a bare `str`,
    and a DecodedResults (which carries perf_metrics) only when handed a list.
    Everything here passes [PROMPT] for that reason -- see the call sites.
    """
    out = RunResult(strategy=strategy, label=label, devices=devices, ok=True,
                    compile_s=compile_s)
    pm = getattr(res, "perf_metrics", None)
    if pm is None:
        out.ok = False
        out.error = (f"generate() returned {type(res).__name__} without perf_metrics "
                     "(pass a list of prompts, not a bare string)")
        return out
    try:
        out.ttft_ms = pm.get_ttft().mean
        out.tpot_ms = pm.get_tpot().mean
        out.throughput_tok_s = pm.get_throughput().mean
        out.generated_tokens = pm.get_num_generated_tokens()
    except Exception as e:
        out.extra["perf_metrics_partial"] = str(e)

    texts = getattr(res, "texts", None)
    text = texts[0] if texts else str(res)
    out.output_head = text[:160].replace("\n", " ")
    return out


# --------------------------------------------------------------------------
# 1. Single device
# --------------------------------------------------------------------------

class SingleDevice:
    """Baseline: the whole model on one engine.

    Everything else is judged against this. On a bandwidth-bound decode the best
    single-device number is usually close to the machine's ceiling, which is
    exactly why 'use all three at once' does not automatically help.
    """

    name = "single"

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        out = []
        for kind in ("CPU", "GPU", "NPU"):
            if kind not in available:
                continue
            for key in all_models(models):     # fp32 AND fp16, both measured
                out.append({"device": kind, "model_key": key})
                # An fp32 IR does NOT imply fp32 execution. The GPU plugin
                # defaults INFERENCE_PRECISION_HINT to f16 and silently converts
                # the weights down at compile time -- measured on Iris Xe, an
                # fp32 IR ran with 625 nodes at f16 and only 10 at f32, giving a
                # throughput identical to the fp16 model. Forcing the hint is the
                # only way to obtain a genuine fp32 number, so measure both and
                # let the difference be visible.
                if key.endswith("fp32"):
                    out.append({"device": kind, "model_key": key,
                                "inference_precision": "f32"})
        return out

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        dev = cfg["device"]
        dev_str = resolve(dev)
        path = models[cfg["model_key"]]
        forced = cfg.get("inference_precision")
        label = f"{dev}/{cfg['model_key']}" + (f" forced={forced}" if forced else "")
        props = device_props(dev)
        if forced:
            props["INFERENCE_PRECISION_HINT"] = forced
        try:
            t0 = time.perf_counter()
            pipe = ov_genai.LLMPipeline(path, dev_str, **props)
            compile_s = time.perf_counter() - t0
            res = pipe.generate([PROMPT], _gen_config())
            out = _collect(res, compile_s, "single", label, dev_str)
            out.extra["inference_precision"] = forced or "device default"
            return out
        except Exception as e:
            return RunResult("single", label, dev, ok=False, error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 2. Speculative decoding  -- the key experiment
# --------------------------------------------------------------------------

class PrefillScaling:
    """Prompt-length sweep: measures the COMPUTE-bound regime.

    Decode and prefill stress completely different parts of the machine, and
    measuring only decode characterises only half the hardware:

      decode   one token at a time, streams all weights per token
               -> MEMORY-BANDWIDTH bound. Compute sits idle.
      prefill  whole prompt at once, weights reused across all prompt positions
               -> COMPUTE bound. This is where TOPS actually shows up.

    So this is the strategy that can reveal an NPU's arithmetic throughput. A
    device with 50 TOPS and a device with 5 TOPS look nearly identical on decode
    -- both are waiting on DRAM -- and separate by a wide margin here.

    Reported as prompt tokens/second (prompt_len / TTFT), swept across prompt
    lengths so the scaling curve is visible rather than a single point. Flat
    scaling with length means compute-bound; linear degradation means something
    else dominates.
    """

    name = "prefill"

    # Kept inside the NPU's default static MAX_PROMPT_LEN of 1024.
    PROMPT_TOKENS = (64, 256, 512, 960)

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        out = []
        for kind in ("CPU", "GPU", "NPU"):
            if kind not in available:
                continue
            for key in all_models(models):
                for n in PrefillScaling.PROMPT_TOKENS:
                    out.append({"device": kind, "model_key": key,
                                "prompt_tokens": n})
        return out

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        dev, n = cfg["device"], cfg["prompt_tokens"]
        dev_str = resolve(dev)
        prec = cfg["model_key"].rsplit("-", 1)[-1]      # fp32 / fp16
        label = f"{dev} {prec} prefill {n} tok"
        try:
            # Build a prompt of approximately n tokens. Exact length is measured
            # afterwards from perf_metrics rather than assumed -- the tokenizer,
            # not us, decides how many tokens a string becomes.
            unit = "The system streams weights from memory for every token. "
            prompt = (unit * ((n // 10) + 2))[:n * 5]

            t0 = time.perf_counter()
            pipe = ov_genai.LLMPipeline(models[cfg["model_key"]], dev_str,
                                        **device_props(dev))
            compile_s = time.perf_counter() - t0

            # Generate a single token: TTFT is then essentially pure prefill.
            res = pipe.generate([prompt], _gen_config(max_new_tokens=1))
            out = _collect(res, compile_s, "prefill", label, dev_str)
            pm = res.perf_metrics
            try:
                in_tok = pm.get_num_input_tokens()
                ttft_s = pm.get_ttft().mean / 1000.0
                out.extra["input_tokens"] = in_tok
                out.extra["prefill_tok_per_s"] = (in_tok / ttft_s) if ttft_s else None
                # Report prefill rate in the throughput column so it ranks in
                # the summary table alongside everything else.
                out.throughput_tok_s = (in_tok / ttft_s) if ttft_s else NAN
            except Exception as e:
                out.extra["prefill_metrics_error"] = str(e)
            return out
        except Exception as e:
            return RunResult("prefill", label, dev_str, ok=False,
                             error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 3. Prefill / decode split  -- feasibility probe
# --------------------------------------------------------------------------

class PrefillDecodeSplit:
    """Prefill is compute-bound; decode is bandwidth-bound. In principle the NPU
    should do prefill and the GPU/CPU should do decode.

    IMPORTANT: this is a PROBE, not an implementation. OpenVINO GenAI does not
    expose a cross-device prefill/decode split for a single pipeline (the KV cache
    lives with the compiled model). What this measures is the *upper bound on the
    benefit*: TTFT from the best prefill device plus TPOT from the best decode
    device, as if the handoff were free. It is not achievable as reported -- it
    tells you whether the handoff would be worth engineering at all.

    Read the `bound` label on these rows as "no implementation can beat this".
    """

    name = "prefill_split"

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        if len(available) < 2:
            return []
        return [{"devices": sorted(available)}]

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        devs = cfg["devices"]
        label = "bound: best-prefill + best-decode"
        per_dev: dict[str, dict] = {}
        try:
            for d in devs:
                key = pick_model(d, models)
                if key is None:
                    continue
                try:
                    pipe = ov_genai.LLMPipeline(models[key], resolve(d), **device_props(d))
                    res = pipe.generate([PROMPT], _gen_config())
                    pm = res.perf_metrics
                    per_dev[d] = {"ttft_ms": pm.get_ttft().mean,
                                  "tpot_ms": pm.get_tpot().mean}
                    del pipe
                except Exception as e:
                    per_dev[d] = {"error": f"{type(e).__name__}: {e}"}

            good = {d: v for d, v in per_dev.items() if "ttft_ms" in v}
            if not good:
                return RunResult("prefill_split", label, ",".join(devs), ok=False,
                                 error="no device produced metrics")

            best_prefill = min(good, key=lambda d: good[d]["ttft_ms"])
            best_decode = min(good, key=lambda d: good[d]["tpot_ms"])
            ttft = good[best_prefill]["ttft_ms"]
            tpot = good[best_decode]["tpot_ms"]

            out = RunResult("prefill_split",
                            f"{label} ({best_prefill}->{best_decode})",
                            f"prefill={best_prefill},decode={best_decode}", ok=True)
            out.ttft_ms = ttft
            out.tpot_ms = tpot
            out.throughput_tok_s = 1000.0 / tpot if tpot else NAN
            out.extra = {"per_device": per_dev, "note": "UPPER BOUND, not implemented"}
            return out
        except Exception as e:
            return RunResult("prefill_split", label, ",".join(devs), ok=False,
                             error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 4. Concurrent independent streams
# --------------------------------------------------------------------------

class ConcurrentStreams:
    """One pipeline per device, all generating at once on separate requests.

    This is the honest version of 'use all the hardware at once'. It cannot speed
    up a single conversation, but it does raise total system throughput -- until
    the shared memory controller saturates, which is precisely what the roofline
    number predicts. If aggregate tok/s lands well below the sum of the
    individual devices, you are looking at bandwidth contention, not a bug.
    """

    name = "concurrent"

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        usable = [d for d in ("CPU", "GPU", "NPU") if d in available]
        if len(usable) < 2:
            return []
        return [{"devices": usable}]

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        devs = cfg["devices"]
        label = "+".join(devs)
        pipes: dict[str, object] = {}
        try:
            for d in devs:
                key = pick_model(d, models)
                if key is not None:
                    pipes[d] = ov_genai.LLMPipeline(models[key], resolve(d),
                                                    **device_props(d))
            if len(pipes) < 2:
                return RunResult("concurrent", label, label, ok=False,
                                 error="fewer than 2 pipelines compiled")

            results: dict[str, dict] = {}
            barrier = threading.Barrier(len(pipes))

            def work(dev, pipe):
                barrier.wait()          # start together, else early finishers
                t0 = time.perf_counter()  # get uncontended bandwidth
                r = pipe.generate([PROMPT], _gen_config())
                dt = time.perf_counter() - t0
                pm = r.perf_metrics
                results[dev] = {"tok": pm.get_num_generated_tokens(),
                                "wall_s": dt,
                                "tok_s": pm.get_throughput().mean}

            threads = [threading.Thread(target=work, args=(d, p))
                       for d, p in pipes.items()]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0

            total_tok = sum(v["tok"] for v in results.values())
            out = RunResult("concurrent", label, ",".join(results), ok=True)
            out.aggregate_tok_s = total_tok / wall if wall else NAN
            out.throughput_tok_s = statistics.fmean(
                [v["tok_s"] for v in results.values()]) if results else NAN
            out.generated_tokens = total_tok
            out.extra = {"per_device": results, "wall_s": wall}
            return out
        except Exception as e:
            return RunResult("concurrent", label, label, ok=False,
                             error=f"{type(e).__name__}: {e}")
        finally:
            pipes.clear()


# --------------------------------------------------------------------------
# 5. CPU thread partition
# --------------------------------------------------------------------------

class CpuThreadPartition:
    """How many CPU threads should the CPU pipeline get?

    More is not better. CPU threads consume the same memory bandwidth the GPU and
    NPU need, so on a decode-bound workload the CPU can actively slow the rest of
    the machine down. This sweep finds where the CPU stops buying throughput and
    starts stealing it -- which is the number strategy 4 needs to be tuned well.
    """

    name = "cpu_threads"

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        if "CPU" not in available:
            return []
        key = pick_model("CPU", models)
        if key is None:
            return []
        n = os.cpu_count() or 8
        counts = sorted({2, 4, max(1, n // 2), n})
        return [{"threads": t, "model_key": key} for t in counts]

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        n = cfg["threads"]
        label = f"CPU threads={n}"
        try:
            t0 = time.perf_counter()
            pipe = ov_genai.LLMPipeline(models[cfg["model_key"]], "CPU",
                                        INFERENCE_NUM_THREADS=n)
            compile_s = time.perf_counter() - t0
            res = pipe.generate([PROMPT], _gen_config())
            out = _collect(res, compile_s, "cpu_threads", label, "CPU")
            out.extra["threads"] = n
            return out
        except Exception as e:
            return RunResult("cpu_threads", label, "CPU", ok=False,
                             error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 6. HETERO op-level split
# --------------------------------------------------------------------------

class Hetero:
    """Split the graph across devices at op level via the HETERO plugin.

    Expected to LOSE for autoregressive decode: every token pays cross-device
    transfer and synchronisation for activations, and on a shared memory
    controller there is no compute headroom to win it back. Measured once so the
    claim is backed by a number on this machine rather than by assertion.

    A failure here is a legitimate result, not a bug to fix.
    """

    name = "hetero"

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        # Try every ordered pair that exists, including NPU-first. Device order
        # matters to HETERO: the first device gets every op it can claim.
        out = []
        for a, b in (("GPU", "CPU"), ("NPU", "CPU"), ("CPU", "NPU"), ("GPU", "NPU")):
            if a in available and b in available:
                key = pick_model(a, models) or pick_model(b, models)
                if key:
                    out.append({"order": [a, b], "model_key": key})
        return out

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        dev = "HETERO:" + ",".join(resolve(d) for d in cfg["order"])
        label = dev
        try:
            t0 = time.perf_counter()
            pipe = ov_genai.LLMPipeline(models[cfg["model_key"]], dev)
            compile_s = time.perf_counter() - t0
            res = pipe.generate([PROMPT], _gen_config())
            return _collect(res, compile_s, "hetero", label, dev)
        except Exception as e:
            return RunResult("hetero", label, dev, ok=False,
                             error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 7. Model distribution policy  (PIPELINE_PARALLEL / TENSOR_PARALLEL)
# --------------------------------------------------------------------------

class Distributed:
    """Split ONE model across two devices via ov::hint::model_distribution_policy.

    This is OpenVINO's actual facility for distributing a single model, and it is
    the closest thing to "run ops on CPU and NPU together". Two policies exist
    and they are not equivalent:

      PIPELINE_PARALLEL  stages assigned to different devices, executed ONE BY
                         ONE. Devices only overlap when several requests are in
                         flight. For single-stream autoregressive decode there
                         is nothing to overlap, so no speedup is expected -- its
                         real purpose is fitting a model that does not fit on
                         one device.
      TENSOR_PARALLEL    genuine intra-op split, devices work on one tensor
                         simultaneously. Documented for CPU sockets/NUMA and
                         multiple GPUs; CPU+NPU is not a documented combination.

    Why single-stream decode cannot win here regardless of policy: a transformer
    forward pass is a serial dependency chain -- layer N+1 consumes layer N's
    output. Splitting a chain across devices adds transfers without creating
    concurrency. Speedup would require either independent work (see
    ConcurrentStreams) or reduced bytes per token (see Speculative).

    Measured anyway, because "the docs imply it won't help" is weaker evidence
    than a number from this machine.
    """

    name = "distributed"

    @staticmethod
    def configs(available: set[str], models: dict) -> list[dict]:
        out = []
        pairs = [(a, b) for a, b in (("NPU", "CPU"), ("GPU", "CPU"), ("CPU", "NPU"))
                 if a in available and b in available]
        for a, b in pairs:
            key = pick_model(a, models) or pick_model(b, models)
            if not key:
                continue
            for policy in ("PIPELINE_PARALLEL", "TENSOR_PARALLEL"):
                out.append({"order": [a, b], "policy": policy, "model_key": key})
        return out

    @staticmethod
    def run(cfg: dict, models: dict) -> RunResult:
        import openvino_genai as ov_genai

        order, policy = cfg["order"], cfg["policy"]
        dev = "HETERO:" + ",".join(resolve(d) for d in order)
        label = f"{policy} {'+'.join(order)}"
        try:
            t0 = time.perf_counter()
            pipe = ov_genai.LLMPipeline(models[cfg["model_key"]], dev,
                                        MODEL_DISTRIBUTION_POLICY={policy})
            compile_s = time.perf_counter() - t0
            res = pipe.generate([PROMPT], _gen_config())
            out = _collect(res, compile_s, "distributed", label, dev)
            out.extra["policy"] = policy
            return out
        except Exception as e:
            return RunResult("distributed", label, dev, ok=False,
                             error=f"{type(e).__name__}: {e}")


# No Speculative, no quantised variants anywhere. Every strategy here runs the
# SAME unmodified fp32 model, so differences between rows are differences in the
# hardware and the runtime -- not in the workload.
ALL = [SingleDevice, PrefillScaling, PrefillDecodeSplit,
       ConcurrentStreams, CpuThreadPartition, Hetero, Distributed]
BY_NAME = {s.name: s for s in ALL}
