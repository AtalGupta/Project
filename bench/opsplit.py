"""Module-level device placement: attention on one device, FFN on the other.

We are NOT writing attention or FFN kernels. OpenVINO's IR compiler produces
those. All this does is tag each node with an affinity so HETERO runs the
attention block on one device and the feed-forward block on the other, then
lets the plugin pick its own optimized kernels on each side.

Why this boundary and not an op-type boundary:

  * The transformer's own structure is attention-block / FFN-block, and the IR
    preserves it in node names -- `...layers.N.self_attn...` and
    `...layers.N.mlp...`. Splitting there cuts the graph at 2 points per layer
    instead of ~6, because it follows the model's real seam.
  * The two blocks have genuinely different memory behaviour. Attention reads
    the KV cache, which GROWS with context length, so at long context it is
    increasingly bandwidth-bound. FFN reads a fixed weight set every token
    regardless of context.
  * So the split has a rationale that holds at long context specifically, which
    is where the reference numbers we are trying to beat were measured.

Nodes with no module in their name -- Constants, Convert wrappers, graph glue --
are assigned to follow their CONSUMER rather than to a fixed device. Pinning a
weight Constant away from the operation that reads it would invent a device
crossing for every weight in the model.

    python -m bench.opsplit --model models/qwen2.5-1.5b-fp16 \
        --attention CPU --ffn NPU
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# Substrings that identify each transformer block in the exported IR.
ATTENTION_MARKERS = ("self_attn", "input_layernorm")
FFN_MARKERS = ("mlp", "post_attention_layernorm")

SUPPORT_FILES = (
    "openvino_tokenizer.xml", "openvino_tokenizer.bin",
    "openvino_detokenizer.xml", "openvino_detokenizer.bin",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "config.json", "generation_config.json", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja",
)


def block_of(name: str) -> str | None:
    """'attention', 'ffn', or None if the node names no module."""
    for m in ATTENTION_MARKERS:
        if m in name:
            return "attention"
    for m in FFN_MARKERS:
        if m in name:
            return "ffn"
    return None


def annotate(model, attn_dev: str, ffn_dev: str, default_dev: str):
    """Assign an affinity to every node. Two passes.

    Pass 1 places everything that names a module. Pass 2 places the remainder --
    Constants and glue -- on whichever device their consumers use, so weights
    live next to the operation that reads them.
    """
    aff: dict[str, str] = {}
    named = 0

    for node in model.get_ordered_ops():
        nm = node.get_friendly_name()
        b = block_of(nm)
        if b == "attention":
            aff[nm] = attn_dev
            named += 1
        elif b == "ffn":
            aff[nm] = ffn_dev
            named += 1

    # Pass 2: unassigned nodes follow their consumer. Iterate in reverse
    # topological order so a chain of Constant -> Convert -> MatMul resolves in
    # one sweep rather than needing repeated passes.
    ops = list(model.get_ordered_ops())
    for node in reversed(ops):
        nm = node.get_friendly_name()
        if nm in aff:
            continue
        votes: dict[str, int] = {}
        for out in node.outputs():
            for consumer in out.get_target_inputs():
                cn = consumer.get_node().get_friendly_name()
                d = aff.get(cn)
                if d:
                    votes[d] = votes.get(d, 0) + 1
        aff[nm] = max(votes, key=votes.get) if votes else default_dev

    for node in model.get_ordered_ops():
        node.get_rt_info()["affinity"] = aff[node.get_friendly_name()]

    counts: dict[str, int] = {}
    for node in model.get_ordered_ops():
        nm = node.get_friendly_name()
        b = block_of(nm) or "glue"
        counts[f"{b} -> {aff[nm]}"] = counts.get(f"{b} -> {aff[nm]}", 0) + 1
    return counts, aff, named


def count_crossings(model, aff: dict[str, str]) -> tuple[int, dict]:
    """Device boundaries, and which block pairs they occur between.

    Reads the affinity dict rather than rt_info: rt_info values come back as
    OVAny whose str() is '<OVAny class>', so every node would compare equal and
    the count would be a flat zero.
    """
    crossings = 0
    where: dict[str, int] = {}
    for node in model.get_ordered_ops():
        nm = node.get_friendly_name()
        mine = aff.get(nm)
        if mine is None:
            continue
        for inp in node.inputs():
            src = inp.get_source_output().get_node()
            sn = src.get_friendly_name()
            theirs = aff.get(sn)
            if theirs is not None and theirs != mine:
                crossings += 1
                k = f"{block_of(sn) or 'glue'} -> {block_of(nm) or 'glue'}"
                where[k] = where.get(k, 0) + 1
    return crossings, where


def build(model_dir: str, out_dir: str, attn_dev: str, ffn_dev: str) -> str:
    from openvino import save_model
    from bench.devices import core

    xml = os.path.join(model_dir, "openvino_model.xml")
    if not os.path.isfile(xml):
        raise SystemExit(f"no openvino_model.xml in {model_dir}")

    c = core()
    model = c.read_model(xml)
    counts, aff, named = annotate(model, attn_dev, ffn_dev, attn_dev)
    crossings, where = count_crossings(model, aff)

    os.makedirs(out_dir, exist_ok=True)
    save_model(model, os.path.join(out_dir, "openvino_model.xml"))
    copied = 0
    for f in SUPPORT_FILES:
        src = os.path.join(model_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, f))
            copied += 1

    total = sum(counts.values())
    print(f"attention -> {attn_dev}    FFN -> {ffn_dev}")
    print(f"{total} nodes ({named} named a module, {total - named} glue "
          f"placed by consumer)")
    print()
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {counts[k]:5d}  {k}")
    print()
    print(f"DEVICE CROSSINGS: {crossings}")
    for k in sorted(where, key=lambda k: -where[k])[:6]:
        print(f"  {where[k]:5d}  {k}")
    print()
    print("Each crossing is a tensor transfer plus a sync, paid every token.")
    print("HETERO runs subgraphs in dependency order, so they are added latency")
    print("rather than overlapped work -- the split has to save more than this")
    print("costs to be worth it.")
    print()
    print(f"wrote {out_dir}  ({copied} support files)")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Split attention and FFN across devices.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--attention", default="CPU", help="device for attention blocks")
    ap.add_argument("--ffn", default="NPU", help="device for feed-forward blocks")
    ap.add_argument("--out", help="output dir (default: <model>-split)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = args.out or (args.model.rstrip("/\\") + "-split")
    build(args.model, out, args.attention, args.ffn)
    print()
    print("Verify correctness FIRST, then benchmark:")
    print(f"  python -u -m bench.opsplit_check --model {out} "
          f"--attention {args.attention} --ffn {args.ffn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
