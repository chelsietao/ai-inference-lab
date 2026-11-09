"""JSON / CSV export of benchmark records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import pandas as pd

from ai_inference_lab.benchmarking.schema import SCHEMA_VERSION, BenchmarkRecord

PathLike = Union[str, Path]


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, list):
        out[prefix] = "; ".join(str(v) for v in value)
    else:
        out[prefix] = value


def record_to_flat_dict(record: BenchmarkRecord) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    _flatten("", record.to_dict(), flat)
    return flat


def records_to_dataframe(records: Iterable[BenchmarkRecord]) -> pd.DataFrame:
    """Flat table: nested keys become 'metrics.mean_ms', 'hardware.device_name', ..."""
    return pd.DataFrame([record_to_flat_dict(r) for r in records])


def save_json(records: Iterable[BenchmarkRecord], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "records": [r.to_dict() for r in records]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_json(path: PathLike) -> List[BenchmarkRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchmarkRecord.from_dict(r) for r in payload["records"]]


def save_csv(records: Iterable[BenchmarkRecord], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_dataframe(records).to_csv(path, index=False)
    return path


SUMMARY_COLUMNS = {
    "hardware.device_type": "device",
    "precision": "precision",
    "batch_size": "batch",
    "metrics.mean_ms": "mean_ms",
    "metrics.median_ms": "median_ms",
    "metrics.p90_ms": "p90_ms",
    "metrics.p95_ms": "p95_ms",
    "metrics.std_ms": "std_ms",
    "metrics.per_image_mean_ms": "per_image_ms",
    "metrics.throughput_images_per_sec": "images/s",
    "metrics.gpu_peak_allocated_mb": "gpu_peak_mb",
    "errors": "errors",
}


def summary_table(records: Iterable[BenchmarkRecord]) -> pd.DataFrame:
    """Compact, human-readable view of the most important columns."""
    df = records_to_dataframe(records)
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    return df[cols].rename(columns=SUMMARY_COLUMNS).round(3)
