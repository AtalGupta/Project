"""Model export to OpenVINO IR.

UNQUANTIZED ONLY, by design.

This harness measures what the hardware does on a fixed, unmodified workload.
Quantisation changes the workload -- it makes the model smaller, which raises
tok/s on a bandwidth-bound decode without the silicon having become any faster.
That is a real deployment technique, but it is not a measurement of the machine,
so it has no place here. Same reasoning excludes speculative decoding.

  fp32   ~6.2 GB for the 1.5B model. Full precision. The CPU executes this
         natively; the NPU's hardware compute precision is fp16, so on that
         device an fp32 IR is converted down regardless of the file's contents.
  fp16   ~3.1 GB. Half precision, still unquantized -- no weight compression,
         no scales, no zero points. Included because it is what the GPU and NPU
         execute in natively, so it isolates "precision" from "compression".

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
        Variant("qwen2.5-1.5b-fp32", TARGET_ID,
                ["--weight-format", "fp32"],
                "full precision, no compression (~6.2 GB)"),
        Variant("qwen2.5-1.5b-fp16", TARGET_ID,
                ["--weight-format", "fp16"],
                "half precision, no compression (~3.1 GB)"),
    ]
}

TARGET_ID_ONLY = TARGET_ID  # no draft model: speculative decoding is out of scope

# FP32 is the default. FP16 is opt-in via --all, for the precision-vs-compression
# comparison; neither is quantised.
DEFAULT_KEYS = ["qwen2.5-1.5b-fp32"]


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

    if manifest:
        os.makedirs(MODELS, exist_ok=True)
        with open(os.path.join(MODELS, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
