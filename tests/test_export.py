import json

import pandas as pd

from ai_inference_lab.benchmarking.runner import BenchmarkConfig, run_benchmark
from ai_inference_lab.benchmarking.schema import SCHEMA_VERSION
from ai_inference_lab.utils.export import load_json, records_to_dataframe, save_csv, save_json
from ai_inference_lab.visualization import charts

REQUIRED_KEYS = {
    "experiment_id", "timestamp", "schema_version", "model", "framework", "hardware",
    "backend", "precision", "input_shape", "batch_size", "config", "metrics",
    "methodology", "errors", "warnings",
}


def _records(loaded):
    return [
        run_benchmark(BenchmarkConfig("cpu", "fp32", bs, 1, 3), loaded) for bs in (1, 2)
    ]


def test_schema_fields_and_json_roundtrip(tiny_loaded, tmp_path):
    recs = _records(tiny_loaded)
    path = save_json(recs, tmp_path / "out" / "r.json")
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert REQUIRED_KEYS <= set(raw["records"][0])
    loaded = load_json(path)
    assert loaded[0].to_dict() == recs[0].to_dict()


def test_csv_export_flat_columns(tiny_loaded, tmp_path):
    recs = _records(tiny_loaded)
    path = save_csv(recs, tmp_path / "r.csv")
    df = pd.read_csv(path)
    assert len(df) == 2
    assert {"metrics.mean_ms", "hardware.device_type", "batch_size"} <= set(df.columns)
    assert records_to_dataframe(recs).shape[0] == 2


def test_charts_render_from_records(tiny_loaded, tmp_path):
    recs = _records(tiny_loaded)
    for fn in (
        charts.plot_device_latency,
        charts.plot_device_throughput,
        charts.plot_batch_latency,
        charts.plot_batch_throughput,
        charts.plot_memory,  # no CUDA data -> placeholder message, must not crash
    ):
        fig = fn(recs, tmp_path / f"{fn.__name__}.png")
        assert fig is not None
    assert len(list(tmp_path.glob("*.png"))) == 5
