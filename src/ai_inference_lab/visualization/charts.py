"""Charts built from real BenchmarkRecords only (no hardcoded numbers)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union


import matplotlib.pyplot as plt

from ai_inference_lab.benchmarking.schema import BenchmarkRecord

PALETTE = {"cpu": "#6b7280", "cuda": "#16a34a", "mps": "#2563eb"}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _ok(records: Iterable[BenchmarkRecord]) -> List[BenchmarkRecord]:
    return [r for r in records if r.ok]


def _label(r: BenchmarkRecord) -> str:
    return f"{r.hardware['device_type']} ({r.precision})"


def _save(fig, path: Optional[Union[str, Path]]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def _bar(records, metric, title, ylabel, path, err_metric=None):
    recs = _ok(records)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labels = [_label(r) for r in recs]
    vals = [r.metrics[metric] for r in recs]
    colors = [PALETTE.get(r.hardware["device_type"], "#6b7280") for r in recs]
    errs = [r.metrics[err_metric] for r in recs] if err_metric else None
    bars = ax.bar(labels, vals, color=colors, width=0.5, yerr=errs, capsize=4)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    bs = recs[0].batch_size if recs else "?"
    _style(ax, f"{title} (batch={bs})", "Device (precision)", ylabel)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_device_latency(records, path=None):
    """Mean batch latency per device; error bars = 1 std."""
    return _bar(records, "mean_ms", "CPU vs GPU latency", "Mean latency per batch (ms)", path, "std_ms")


def plot_device_throughput(records, path=None):
    return _bar(records, "throughput_images_per_sec", "CPU vs GPU throughput", "Throughput (images/s)", path)


def _sweep_plot(records, metrics, title, ylabel, path):
    recs = sorted(_ok(records), key=lambda r: r.batch_size)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xs = [r.batch_size for r in recs]
    for metric, label, style in metrics:
        ax.plot(xs, [r.metrics[metric] for r in recs], style, label=label, color="#111827")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    dev = _label(recs[0]) if recs else ""
    _style(ax, f"{title} [{dev}]", "Batch size", ylabel)
    if len(metrics) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_batch_latency(records, path=None):
    """Per-batch latency (solid) vs per-image latency (dashed): they are different quantities."""
    return _sweep_plot(
        records,
        [("mean_ms", "per batch", "o-"), ("per_image_mean_ms", "per image", "s--")],
        "Batch size vs latency",
        "Latency (ms)",
        path,
    )


def plot_batch_throughput(records, path=None):
    return _sweep_plot(
        records,
        [("throughput_images_per_sec", "throughput", "o-")],
        "Batch size vs throughput",
        "Throughput (images/s)",
        path,
    )


def plot_memory(records, path=None):
    """GPU peak allocated memory; only records that measured it are drawn."""
    recs = [r for r in _ok(records) if r.metrics.get("gpu_peak_allocated_mb") is not None]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if not recs:
        ax.text(0.5, 0.5, "No memory data: CUDA GPU not used.", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [f"bs={r.batch_size}\n{r.precision}" for r in recs]
        vals = [r.metrics["gpu_peak_allocated_mb"] for r in recs]
        bars = ax.bar(labels, vals, color=PALETTE["cuda"], width=0.5)
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        _style(ax, "GPU peak allocated memory", "Configuration", "Peak allocated (MB)")
    fig.tight_layout()
    _save(fig, path)
    return fig
