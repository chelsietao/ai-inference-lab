"""Stable, versioned result schema (see docs/benchmarking.md)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

METHODOLOGY = (
    "Per-iteration wall-clock latency via time.perf_counter() around one forward pass "
    "under torch.inference_mode() with model.eval(). On CUDA, torch.cuda.synchronize() is "
    "called before starting and after finishing every timed iteration so asynchronous kernel "
    "execution is fully counted. Model loading, weight download, input creation and "
    "preprocessing are excluded. The first forward call (cold start) is timed separately and "
    "warm-up iterations are discarded; reported statistics describe steady-state latency of one "
    "batch. Throughput = batch_size / mean batch latency. Inputs are seeded random tensors of a "
    "fixed shape. Latency units: milliseconds; throughput units: images/second."
)

MEMORY_NOTE_CUDA = (
    "gpu_peak_allocated_mb = torch.cuda.max_memory_allocated(): peak bytes held by PyTorch "
    "tensors (weights + input + activations) during the timed run. It excludes the CUDA context, "
    "the caching allocator's reserved-but-unused pool, and other processes on the GPU."
)
MEMORY_NOTE_CPU = "Peak memory is not measured on CPU/MPS in this version (value is null)."


@dataclass
class BenchmarkRecord:
    """One experiment: a single (model, device, precision, batch size) measurement."""

    model: Dict[str, Any]
    framework: Dict[str, Any]
    hardware: Dict[str, Any]
    backend: str
    precision: str
    input_shape: List[int]
    batch_size: int
    config: Dict[str, Any]
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    methodology: str = METHODOLOGY
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkRecord":
        return cls(**data)
