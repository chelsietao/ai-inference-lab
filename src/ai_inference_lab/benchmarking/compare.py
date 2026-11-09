"""Higher-level experiments built on run_benchmark."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from ai_inference_lab.benchmarking.runner import BenchmarkConfig, run_benchmark
from ai_inference_lab.benchmarking.schema import BenchmarkRecord
from ai_inference_lab.metrics.stats import speedup
from ai_inference_lab.models.classifier import LoadedModel, load_classifier
from ai_inference_lab.utils.device import UnsupportedConfigurationError

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = (1, 4, 8, 16, 32)


def compare_devices(
    loaded: Optional[LoadedModel] = None,
    batch_size: int = 1,
    warmup_iters: int = 10,
    measure_iters: int = 50,
    precision: str = "fp32",
) -> Dict[str, Any]:
    """Run CPU, then CUDA if available. Never fabricates GPU numbers.

    Returns {"records": [...], "speedup": float|None, "gpu_available": bool, "message": str}.
    Speedup = CPU mean latency / GPU mean latency, only for identical workloads.
    """
    loaded = loaded or load_classifier()
    records: List[BenchmarkRecord] = []
    cpu = run_benchmark(
        BenchmarkConfig("cpu", "fp32", batch_size, warmup_iters, measure_iters), loaded
    )
    records.append(cpu)
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        return {
            "records": records,
            "speedup": None,
            "gpu_available": False,
            "message": "CUDA GPU not available: only the CPU experiment was run. "
            "No GPU numbers are shown.",
        }
    gpu = run_benchmark(
        BenchmarkConfig("cuda", precision, batch_size, warmup_iters, measure_iters), loaded
    )
    records.append(gpu)
    ratio = None
    if cpu.ok and gpu.ok:
        ratio = speedup(cpu.metrics["mean_ms"], gpu.metrics["mean_ms"])
    note = "fp32 CPU vs fp32 GPU" if precision == "fp32" else f"fp32 CPU vs {precision} GPU"
    return {
        "records": records,
        "speedup": ratio,
        "gpu_available": True,
        "message": f"Speedup = CPU mean latency / GPU mean latency ({note}, batch={batch_size}).",
    }


def sweep_batch_sizes(
    loaded: Optional[LoadedModel] = None,
    device: str = "cpu",
    precision: str = "fp32",
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    warmup_iters: int = 5,
    measure_iters: int = 20,
) -> List[BenchmarkRecord]:
    """Benchmark several batch sizes. OOM / unsupported sizes are recorded, not fatal."""
    loaded = loaded or load_classifier()
    out: List[BenchmarkRecord] = []
    for bs in batch_sizes:
        try:
            rec = run_benchmark(
                BenchmarkConfig(device, precision, bs, warmup_iters, measure_iters), loaded
            )
        except UnsupportedConfigurationError:
            raise  # same device/precision for every size: fail fast with a clear message
        out.append(rec)
        if rec.errors:
            logger.warning("batch_size=%s skipped: %s", bs, rec.errors[0])
            break  # larger sizes will not fit either
    return out
