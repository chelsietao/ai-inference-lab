"""Hardware detection and device selection.

Design rule: never crash because a GPU is missing. Report what is available,
and explain what is not.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

SUPPORTED_PRECISIONS = ("fp32", "fp16")


class UnsupportedConfigurationError(ValueError):
    """Raised when a device/precision/batch combination cannot run."""


@dataclass
class HardwareInfo:
    """Snapshot of the execution environment, stored with every benchmark."""

    device_type: str  # "cpu" | "cuda" | "mps"
    device_name: str
    cuda_available: bool
    cuda_version: Optional[str]
    pytorch_version: str
    torchvision_version: Optional[str]
    python_version: str
    platform: str
    runtime_environment: str  # "colab" | "local"
    cpu_count: Optional[int]
    gpu_total_memory_mb: Optional[float]
    supported_precisions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Device type        : {self.device_type}",
            f"Device name        : {self.device_name}",
            f"CUDA available     : {self.cuda_available}",
            f"CUDA version       : {self.cuda_version or 'n/a'}",
            f"PyTorch version    : {self.pytorch_version}",
            f"GPU total memory   : "
            + (f"{self.gpu_total_memory_mb:.0f} MB" if self.gpu_total_memory_mb else "n/a"),
            f"Supported precision: {', '.join(self.supported_precisions)}",
            f"Runtime            : {self.runtime_environment} ({self.platform})",
        ]
        lines += [f"Note               : {n}" for n in self.notes]
        return "\n".join(lines)


def _is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def _cpu_name() -> str:
    """Best-effort human-readable CPU name (platform.processor() is often just 'arm')."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if out:
                return out
        elif os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover - purely cosmetic
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def supported_precisions(device_type: str) -> List[str]:
    """fp32 runs everywhere; fp16 is only enabled on CUDA in this project.

    fp16 on CPU is slow/poorly supported in many PyTorch ops, so we mark it
    unsupported instead of producing misleading numbers.
    """
    if device_type == "cuda":
        return ["fp32", "fp16"]
    return ["fp32"]


def detect_hardware(prefer: str = "auto") -> HardwareInfo:
    """Describe the hardware that `select_device(prefer)` would use."""
    device = select_device(prefer)
    notes: List[str] = []
    cuda_ok = torch.cuda.is_available()
    name = _cpu_name()
    total_mb: Optional[float] = None

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        name = props.name
        total_mb = props.total_memory / 1024**2
    elif device.type == "mps":
        name = "Apple Silicon (MPS)"
        notes.append("MPS memory statistics are not collected by this project.")
    if not cuda_ok:
        notes.append(
            "CUDA is not available, so no NVIDIA GPU experiments can run. In Colab, use "
            "Runtime > Change runtime type > T4 GPU."
        )
    try:
        import torchvision

        tv_version: Optional[str] = torchvision.__version__
    except Exception:  # pragma: no cover - torchvision is a hard dependency
        tv_version = None

    return HardwareInfo(
        device_type=device.type,
        device_name=name,
        cuda_available=cuda_ok,
        cuda_version=torch.version.cuda if cuda_ok else None,
        pytorch_version=torch.__version__,
        torchvision_version=tv_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        runtime_environment="colab" if _is_colab() else "local",
        cpu_count=os.cpu_count(),
        gpu_total_memory_mb=total_mb,
        supported_precisions=supported_precisions(device.type),
        notes=notes,
    )


def select_device(prefer: str = "auto") -> torch.device:
    """Pick a device.

    `prefer` is "auto", "cpu", "cuda" or "mps". "auto" picks cuda > mps > cpu.
    An explicit but unavailable request falls back to CPU with a warning
    (it does not crash, and it is never silently reported as the requested device).
    """
    prefer = prefer.lower()
    if prefer not in {"auto", "cpu", "cuda", "mps"}:
        raise UnsupportedConfigurationError(f"Unknown device '{prefer}'. Use auto/cpu/cuda/mps.")
    if prefer in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in {"auto", "mps"} and _mps_available():
        return torch.device("mps")
    if prefer in {"cuda", "mps"}:
        logger.warning("Requested device '%s' is unavailable; falling back to CPU.", prefer)
    return torch.device("cpu")


def validate_precision(device: torch.device, precision: str) -> None:
    """Raise UnsupportedConfigurationError if precision cannot run on device."""
    if precision not in SUPPORTED_PRECISIONS:
        raise UnsupportedConfigurationError(
            f"Unknown precision '{precision}'. Supported: {SUPPORTED_PRECISIONS}."
        )
    if precision not in supported_precisions(device.type):
        raise UnsupportedConfigurationError(
            f"Precision '{precision}' is not supported on device '{device.type}' in this project."
        )
