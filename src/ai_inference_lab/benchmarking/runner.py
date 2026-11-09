"""The benchmark engine: warm-up, synchronized timing, statistics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from ai_inference_lab.backends.base import InferenceBackend
from ai_inference_lab.backends.pytorch import PyTorchBackend
from ai_inference_lab.benchmarking.schema import (
    MEMORY_NOTE_CPU,
    MEMORY_NOTE_CUDA,
    BenchmarkRecord,
)
from ai_inference_lab.metrics.stats import (
    latency_stats,
    per_image_latency_ms,
    throughput_images_per_sec,
)
from ai_inference_lab.models.classifier import LoadedModel, load_classifier
from ai_inference_lab.utils.device import (
    UnsupportedConfigurationError,
    detect_hardware,
    select_device,
    validate_precision,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    precision: str = "fp32"
    batch_size: int = 1
    warmup_iters: int = 10
    measure_iters: int = 50
    seed: int = 0

    def validate(self) -> None:
        if self.batch_size < 1:
            raise UnsupportedConfigurationError("batch_size must be >= 1")
        if self.warmup_iters < 0 or self.measure_iters < 1:
            raise UnsupportedConfigurationError("need warmup_iters >= 0 and measure_iters >= 1")


def is_oom_error(exc: BaseException) -> bool:
    """True for CUDA/CPU out-of-memory failures raised by PyTorch."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, (RuntimeError, MemoryError)) and "out of memory" in str(exc).lower()


def resolve_device_strict(requested: str) -> torch.device:
    """Return the requested device or raise (no silent substitution).

    A benchmark labelled 'cuda' must really run on CUDA.
    """
    device = select_device(requested)
    if requested != "auto" and device.type != requested:
        raise UnsupportedConfigurationError(
            f"Requested device '{requested}' is not available (got '{device.type}')."
        )
    return device


def measure_latencies(
    backend: InferenceBackend,
    inputs,
    warmup_iters: int,
    measure_iters: int,
) -> tuple:
    """Return (cold_start_ms, [per-iteration latency ms]).

    Synchronization: GPU kernels launch asynchronously, so without a sync the
    timer would stop when the kernel is *queued*, not when it *finishes*.
    """
    backend.synchronize()
    t0 = time.perf_counter()
    backend.run(inputs)
    backend.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup_iters):
        backend.run(inputs)
    backend.synchronize()

    latencies: List[float] = []
    for _ in range(measure_iters):
        backend.synchronize()
        start = time.perf_counter()
        backend.run(inputs)
        backend.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return cold_ms, latencies


def run_benchmark(
    config: BenchmarkConfig,
    loaded: Optional[LoadedModel] = None,
    pretrained: bool = True,
    model_name: str = "mobilenet_v3_small",
) -> BenchmarkRecord:
    """Benchmark one configuration and return a BenchmarkRecord.

    Raises UnsupportedConfigurationError for impossible configs (e.g. fp16 on CPU,
    or cuda requested without CUDA). CUDA OOM is *recorded* in `record.errors`
    rather than raised, so a batch-size sweep can continue.
    """
    config.validate()
    device = resolve_device_strict(config.device)
    validate_precision(device, config.precision)
    loaded = loaded or load_classifier(model_name, pretrained=pretrained)  # load time excluded

    hw = detect_hardware(config.device)
    backend = PyTorchBackend(loaded.model, device, config.precision, loaded.input_size)
    record = BenchmarkRecord(
        model={
            "name": loaded.name,
            "pretrained": loaded.pretrained,
            "load_time_s": round(loaded.load_time_s, 4),
        },
        framework={"pytorch": hw.pytorch_version, "torchvision": hw.torchvision_version},
        hardware=hw.to_dict(),
        backend=backend.name,
        precision=config.precision,
        input_shape=[config.batch_size, *loaded.input_size],
        batch_size=config.batch_size,
        config={
            "device": config.device,
            "warmup_iters": config.warmup_iters,
            "measure_iters": config.measure_iters,
            "seed": config.seed,
        },
    )
    if not loaded.pretrained:
        record.warnings.append("Random weights: valid for speed, meaningless for accuracy.")

    try:
        inputs = backend.make_input(config.batch_size, config.seed)
        backend.reset_peak_memory()
        cold_ms, latencies = measure_latencies(
            backend, inputs, config.warmup_iters, config.measure_iters
        )
        mem = backend.memory_stats_mb()
    except Exception as exc:
        if is_oom_error(exc):
            record.errors.append(f"OutOfMemory at batch_size={config.batch_size}: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return record
        raise

    stats = latency_stats(latencies)
    record.metrics = {
        **stats,
        "cold_start_ms": cold_ms,
        "per_image_mean_ms": per_image_latency_ms(config.batch_size, stats["mean_ms"]),
        "per_image_median_ms": per_image_latency_ms(config.batch_size, stats["median_ms"]),
        "throughput_images_per_sec": throughput_images_per_sec(config.batch_size, stats["mean_ms"]),
        "gpu_baseline_allocated_mb": mem[0] if mem else None,
        "gpu_peak_allocated_mb": mem[1] if mem else None,
    }
    record.config["memory_note"] = MEMORY_NOTE_CUDA if mem else MEMORY_NOTE_CPU
    if stats["std_ms"] > 0.25 * stats["mean_ms"]:
        record.warnings.append(
            "High variability (std > 25% of mean): other processes or thermal/clock changes "
            "may affect this result. Re-run or increase measure_iters."
        )
    return record
