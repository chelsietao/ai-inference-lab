"""Backend interface so ONNX Runtime / TensorRT can be added in later phases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class InferenceBackend(ABC):
    """Minimal contract a benchmarkable inference backend must satisfy."""

    name: str
    device_type: str
    precision: str

    @abstractmethod
    def make_input(self, batch_size: int, seed: int) -> Any:
        """Build a deterministic input already placed on the target device."""

    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Run one forward pass (asynchronous on GPU; do not sync here)."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all queued device work finished (no-op on CPU)."""

    def reset_peak_memory(self) -> None:
        """Start peak-memory tracking, if the device supports it."""

    def memory_stats_mb(self) -> Optional[Tuple[float, float]]:
        """Return (baseline_allocated_mb, peak_allocated_mb) or None if unsupported."""
        return None
