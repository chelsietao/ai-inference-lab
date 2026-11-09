"""Pure, hardware-independent metric calculations (easy to unit-test)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Summarize per-iteration latencies (milliseconds, per batch)."""
    if len(latencies_ms) == 0:
        raise ValueError("latencies_ms must not be empty")
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def throughput_images_per_sec(batch_size: int, mean_batch_latency_ms: float) -> float:
    """images/sec = batch_size / seconds-per-batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mean_batch_latency_ms <= 0:
        raise ValueError("latency must be positive")
    return batch_size / (mean_batch_latency_ms / 1000.0)


def per_image_latency_ms(batch_size: int, batch_latency_ms: float) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return batch_latency_ms / batch_size


def speedup(baseline_ms: float, candidate_ms: float) -> Optional[float]:
    """baseline / candidate. >1 means the candidate is faster."""
    if baseline_ms <= 0 or candidate_ms <= 0:
        return None
    return baseline_ms / candidate_ms
