"""Model export to OpenVINO IR.

FP16 is the primary target. The quantised variants are kept as comparison points
and because the NPU needs one.

  fp16            PRIMARY. Full-precision weights, ~3.1 GB for the 1.5B model.
                  Highest quality, and the reference the others are judged
                  against. Note the cost: decode streams every weight byte once
                  per token, so 3.1 GB/token versus INT4's ~1.0 GB means roughly
                  3x lower token rate on the same memory bandwidth. That is
                  physics, not a defect -- see bench/roofline.py.
  int8            ~1.6 GB. The usual quality/bandwidth compromise.
  int4-asym-g128  ~1.0 GB. Fastest on CPU/GPU; asymmetric per-group.
  int4-sym-cw     ~1.0 GB. NPU-legal form -- the NPU rejects asymmetric
                  per-group INT4, so it needs symmetric channel-wise (-1).
  0.5B drafts     Speculative-decoding drafts, in fp16 and in NPU-legal int4.

Exports are idempotent: an existing directory containing openvino_model.xml is
left alone unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Variant:
    key: str
    model_id: str
    args: list[str]
    note: str

    @property
    def path(self) -> str:
        return os.path.join(MODELS, self.key)


VARIANTS: dict[str, Variant] = {
    v.key: v for v in [
        # --- primary ---
        Variant("qwen2.5-1.5b-fp16", TARGET_ID,
                ["--weight-format", "fp16"],
                "PRIMARY target, full precision (~3.1 GB)"),
        Variant("qwen2.5-0.5b-fp16", DRAFT_ID,
                ["--weight-format", "fp16"],
                "speculative draft, full precision (~1.0 GB)"),
        # --- comparison / NPU-required ---
        Variant("qwen2.5-1.5b-int8", TARGET_ID,
                ["--weight-format", "int8"],
                "comparison + accuracy anchor"),
        Variant("qwen2.5-1.5b-int4-asym-g128", TARGET_ID,
                ["--weight-format", "int4", "--group-size", "128", "--ratio", "1.0"],
                "comparison, fastest on CPU/GPU"),
        Variant("qwen2.5-1.5b-int4-sym-cw", TARGET_ID,
                ["--weight-format", "int4", "--sym", "--group-size", "-1", "--ratio", "1.0"],
                "NPU-legal target (sym channel-wise)"),
        Variant("qwen2.5-0.5b-int4-sym-cw", DRAFT_ID,
                ["--weight-format", "int4", "--sym", "--group-size", "-1", "--ratio", "1.0"],
                "NPU-legal draft"),
    ]
}

# Exported by default: the FP16 pair, which is what was asked for. The quantised
# variants are opt-in via --only or --all.
DEFAULT_KEYS = ["qwen2.5-1.5b-fp16", "qwen2.5-0.5b-fp16"]


def is_exported(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "openvino_model.xml"))


def export_one(v: Variant, force: bool = False) -> bool:
    if is_exported(v.path) and not force:
        print(f"[skip]   {v.key}  (already exported)")
        return True

    os.makedirs(MODELS, exist_ok=True)
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino",
        "--model", v.model_id,
        "--task", "text-generation-with-past",
        *v.args,
        v.path,
    ]
    print(f"[export] {v.key}  ({v.note})")
    print(f"         {' '.join(cmd[3:])}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[FAIL]   {v.key}  exit={r.returncode}")
        return False
    if not is_exported(v.path):
        print(f"[FAIL]   {v.key}  no openvino_model.xml produced")
        return False
    print(f"[ok]     {v.key}  -> {v.path}")
    return True


def size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def check_tokenizers_identical(a: str, b: str) -> tuple[bool, str]:
    """Speculative decoding is only valid if draft and target share a token space.

    A mismatch does not crash -- it silently produces a near-zero acceptance rate
    and plausible-looking but wrong output, which is far more expensive to debug
    later than to assert now.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return False, "transformers not installed; cannot verify"

    try:
        ta = AutoTokenizer.from_pretrained(a)
        tb = AutoTokenizer.from_pretrained(b)
    except Exception as e:
        return False, f"failed to load tokenizers: {e}"

    va, vb = ta.get_vocab(), tb.get_vocab()
    if va != vb:
        only_a = set(va) - set(vb)
        only_b = set(vb) - set(va)
        return False, (f"vocab differs: {len(va)} vs {len(vb)} entries, "
                       f"{len(only_a)} only-in-target, {len(only_b)} only-in-draft")

    probe = ("The quick brown fox jumps over the lazy dog. "
             "def f(x):\n    return x**2  # 1234567890 你好世界")
    ia, ib = ta.encode(probe), tb.encode(probe)
    if ia != ib:
        return False, "identical vocab but different encoding of probe string"
    return True, f"identical ({len(va)} tokens, probe -> {len(ia)} ids)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Qwen2.5 variants to OpenVINO IR.")
    ap.add_argument("--only", nargs="*", choices=sorted(VARIANTS), help="subset to export")
    ap.add_argument("--all", action="store_true",
                    help="export every variant, not just the FP16 pair")
    ap.add_argument("--force", action="store_true", help="re-export even if present")
    ap.add_argument("--check-only", action="store_true", help="report status, export nothing")
    args = ap.parse_args()

    if args.only:
        keys = args.only
    elif args.all:
        keys = list(VARIANTS)
    else:
        keys = list(DEFAULT_KEYS)

    if args.check_only:
        for k in keys:
            v = VARIANTS[k]
            state = "present" if is_exported(v.path) else "MISSING"
            sz = f"{size_bytes(v.path) / 1e9:.2f} GB" if is_exported(v.path) else "-"
            print(f"{k:32s} {state:8s} {sz:>9s}  {v.note}")
        return 0

    # Sequentially, one at a time: export peaks well above the final artefact size
    # and the laptop has 15.7 GB total.
    ok = True
    for k in keys:
        ok &= export_one(VARIANTS[k], force=args.force)

    print()
    manifest = {}
    for k in keys:
        v = VARIANTS[k]
        if is_exported(v.path):
            b = size_bytes(v.path)
            manifest[k] = {"path": v.path, "bytes": b, "gb": round(b / 1e9, 3),
                           "model_id": v.model_id, "note": v.note}
            print(f"{k:32s} {b / 1e9:6.2f} GB")

    # Check the tokenizers of whichever target/draft pair actually got exported.
    pairs = [("qwen2.5-1.5b-fp16", "qwen2.5-0.5b-fp16"),
             ("qwen2.5-1.5b-int4-sym-cw", "qwen2.5-0.5b-int4-sym-cw")]
    tgt = drf = None
    for tk, dk in pairs:
        if is_exported(VARIANTS[tk].path) and is_exported(VARIANTS[dk].path):
            tgt, drf = VARIANTS[tk].path, VARIANTS[dk].path
            break
    if tgt and drf:
        same, why = check_tokenizers_identical(tgt, drf)
        print(f"\ntokenizer identity (target vs draft): {'OK' if same else 'MISMATCH'} - {why}")
        manifest["_tokenizer_check"] = {"identical": same, "detail": why}
        if not same:
            print("  -> speculative-decoding results would be meaningless. Fix before Phase 2.")
            ok = False

    if manifest:
        os.makedirs(MODELS, exist_ok=True)
        with open(os.path.join(MODELS, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
