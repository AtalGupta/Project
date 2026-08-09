"""Device enumeration.

The whole point of this module: the local laptop has CPU+GPU, the cloud node has
CPU+NPU, and the cloud node may gain a GPU later if its Intel graphics driver is
installed. Nothing downstream may hardcode a device list -- strategies declare
which devices they need, and the sweep skips the ones that cannot run here.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    name: str                       # as OpenVINO reports it, e.g. "GPU.0"
    kind: str                       # normalised: CPU / GPU / NPU
    full_name: str = "?"
    props: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name:8s} {self.full_name}"


# Properties worth capturing per device. Missing ones are skipped silently --
# support varies by plugin and by OpenVINO version.
_PROPS = (
    "FULL_DEVICE_NAME",
    "DEVICE_TYPE",
    "DEVICE_ARCHITECTURE",
    "OPTIMIZATION_CAPABILITIES",
    "GPU_DEVICE_TOTAL_MEM_SIZE",
    "GPU_EXECUTION_UNITS_COUNT",
    "NPU_DEVICE_TOTAL_MEM_SIZE",
    "NPU_DRIVER_VERSION",
    "NPU_MAX_TILES",
    "AVAILABLE_DEVICES",
    "RANGE_FOR_STREAMS",
)


_CORE = None
_CACHE: list[DeviceInfo] | None = None


def core():
    """One process-wide Core, created once and never released.

    Creating and dropping multiple Core instances destabilises the plugins --
    repeatedly loading and unloading the OpenCL/Level-Zero stacks (especially
    with a non-Intel OpenCL device also present) produces access violations at
    teardown. A single long-lived Core avoids the whole class of problem.
    """
    global _CORE
    if _CORE is None:
        from openvino import Core

        _CORE = Core()
    return _CORE


def _kind(name: str) -> str:
    """'GPU.1' -> 'GPU'. OpenVINO suffixes multi-instance devices."""
    return name.split(".", 1)[0].upper()


def _is_intel_gpu(info: DeviceInfo) -> bool:
    """The GPU plugin is OpenCL-based and will happily *enumerate* a discrete
    NVIDIA/AMD card, but `intel_gpu` cannot execute on one. Running the sweep
    against it produces a pile of confusing compile failures, so filter by PCI
    vendor id: 0x8086 is Intel.

    The laptop hits this exactly -- it reports GPU.0 (Iris Xe) and GPU.1
    (RTX 3060), and only the first is usable.
    """
    arch = str(info.props.get("DEVICE_ARCHITECTURE", ""))
    if "vendor=" in arch:
        return "0x8086" in arch
    return "intel" in info.full_name.lower()


def enumerate_devices(intel_only: bool = True) -> list[DeviceInfo]:
    """Enumerate once and cache -- querying plugin properties is not free, and
    every caller wants the same answer."""
    global _CACHE
    if _CACHE is None:
        c = core()
        found: list[DeviceInfo] = []
        for name in c.available_devices:
            info = DeviceInfo(name=name, kind=_kind(name))
            for p in _PROPS:
                try:
                    info.props[p] = c.get_property(name, p)
                except Exception:
                    pass  # plugin doesn't expose it; not an error
            info.full_name = str(info.props.get("FULL_DEVICE_NAME", "?"))
            info.props["_usable"] = (info.kind != "GPU") or _is_intel_gpu(info)
            found.append(info)
        _CACHE = found
    return [d for d in _CACHE if d.props.get("_usable", True) or not intel_only]


def kinds_present() -> set[str]:
    return {d.kind for d in enumerate_devices()}


def supports_fp16(kind: str) -> bool:
    for d in enumerate_devices():
        if d.kind == kind.upper():
            return "FP16" in (d.props.get("OPTIMIZATION_CAPABILITIES") or [])
    return False


def pick(kind: str) -> str | None:
    """Return the OpenVINO device string for a kind, or None if absent.

    Returns an explicit indexed name ("GPU.0") when the bare kind is ambiguous,
    so we never rely on the plugin's default landing on the Intel device.
    """
    names = [d.name for d in enumerate_devices() if d.kind == kind.upper()]
    if not names:
        return None
    if kind.upper() in names:
        return kind.upper()
    return sorted(names)[0]


def summary() -> str:
    lines = [
        f"host        : {platform.node()}",
        f"platform    : {platform.platform()}",
        f"processor   : {platform.processor()}",
    ]
    try:
        from openvino import get_version

        lines.append(f"openvino    : {get_version()}")
    except Exception as e:
        lines.append(f"openvino    : IMPORT FAILED ({e})")
        return "\n".join(lines)

    try:
        import openvino_genai

        lines.append(f"openvino_genai: {getattr(openvino_genai, '__version__', 'installed')}")
    except Exception:
        lines.append("openvino_genai: NOT INSTALLED")

    all_devs = enumerate_devices(intel_only=False)
    excluded = [d for d in all_devs if not d.props.get("_usable", True)]
    devs = [d for d in all_devs if d.props.get("_usable", True)]
    lines.append(f"devices     : {[d.name for d in devs]}")
    for d in excluded:
        lines.append(f"  EXCLUDED {d.name}: {d.full_name} "
                     f"(not an Intel GPU; intel_gpu plugin cannot execute on it)")
    for d in devs:
        lines.append(f"  {d.name}")
        lines.append(f"      full name : {d.full_name}")
        for k in ("DEVICE_ARCHITECTURE", "OPTIMIZATION_CAPABILITIES",
                  "GPU_EXECUTION_UNITS_COUNT", "GPU_DEVICE_TOTAL_MEM_SIZE",
                  "NPU_DRIVER_VERSION", "NPU_MAX_TILES"):
            if k in d.props:
                lines.append(f"      {k.lower():25s}: {d.props[k]}")

    missing = {"CPU", "GPU", "NPU"} - {d.kind for d in devs}
    if missing:
        lines.append("")
        lines.append(f"NOT AVAILABLE: {sorted(missing)}")
        if "GPU" in missing:
            lines.append("  GPU absent -> Intel graphics driver not installed, or no Intel GPU.")
            lines.append("  On the cloud node this is the known blocker: the Xe3 iGPU shows as")
            lines.append("  'Microsoft Basic Display Adapter', so no OpenCL/Level-Zero GPU runtime.")
        if "NPU" in missing:
            lines.append("  NPU absent -> no Intel AI Boost NPU on this machine (expected on the laptop).")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
