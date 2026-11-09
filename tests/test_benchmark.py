import pytest
import torch

from ai_inference_lab.benchmarking.compare import compare_devices, sweep_batch_sizes
from ai_inference_lab.benchmarking.runner import (
    BenchmarkConfig,
    is_oom_error,
    measure_latencies,
    run_benchmark,
)
from ai_inference_lab.backends.base import InferenceBackend
from ai_inference_lab.utils.device import UnsupportedConfigurationError
from conftest import requires_cuda

FAST = dict(warmup_iters=1, measure_iters=3)


def test_run_benchmark_cpu_record(tiny_loaded):
    rec = run_benchmark(BenchmarkConfig("cpu", "fp32", 2, **FAST), tiny_loaded)
    assert rec.ok and rec.errors == []
    m = rec.metrics
    assert rec.input_shape == [2, 3, 224, 224]
    assert m["median_ms"] > 0 and m["p95_ms"] >= m["p90_ms"] >= m["median_ms"]
    assert m["throughput_images_per_sec"] == pytest.approx(2 / (m["mean_ms"] / 1000))
    assert m["per_image_mean_ms"] == pytest.approx(m["mean_ms"] / 2)
    assert m["gpu_peak_allocated_mb"] is None  # CPU: not measured
    assert rec.config["warmup_iters"] == 1 and rec.config["measure_iters"] == 3
    assert any("Random weights" in w for w in rec.warnings)


def test_fp16_on_cpu_raises(tiny_loaded):
    with pytest.raises(UnsupportedConfigurationError):
        run_benchmark(BenchmarkConfig("cpu", "fp16", 1, **FAST), tiny_loaded)


def test_cuda_requested_without_cuda_raises(tiny_loaded, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(UnsupportedConfigurationError):
        run_benchmark(BenchmarkConfig("cuda", "fp32", 1, **FAST), tiny_loaded)


def test_invalid_config_raises(tiny_loaded):
    with pytest.raises(UnsupportedConfigurationError):
        run_benchmark(BenchmarkConfig("cpu", "fp32", 0, **FAST), tiny_loaded)
    with pytest.raises(UnsupportedConfigurationError):
        run_benchmark(BenchmarkConfig("cpu", "fp32", 1, 1, 0), tiny_loaded)


def test_warmup_and_measure_counts_and_sync():
    class FakeBackend(InferenceBackend):
        name, device_type, precision = "fake", "cpu", "fp32"

        def __init__(self):
            self.runs = 0
            self.syncs = 0

        def make_input(self, batch_size, seed):
            return None

        def run(self, inputs):
            self.runs += 1

        def synchronize(self):
            self.syncs += 1

    b = FakeBackend()
    cold, lats = measure_latencies(b, None, warmup_iters=4, measure_iters=6)
    assert len(lats) == 6 and cold >= 0
    assert b.runs == 1 + 4 + 6  # cold + warm-up + measured
    assert b.syncs >= 2 * 6  # synchronized before and after every timed iteration


def test_oom_is_recorded_not_raised(tiny_loaded, monkeypatch):
    from ai_inference_lab.backends import pytorch as pb

    def boom(self, inputs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    monkeypatch.setattr(pb.PyTorchBackend, "run", boom)
    rec = run_benchmark(BenchmarkConfig("cpu", "fp32", 4, **FAST), tiny_loaded)
    assert not rec.ok and "OutOfMemory" in rec.errors[0]
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


def test_sweep_batch_sizes_cpu(tiny_loaded):
    recs = sweep_batch_sizes(tiny_loaded, "cpu", batch_sizes=(1, 2), warmup_iters=1, measure_iters=2)
    assert [r.batch_size for r in recs] == [1, 2] and all(r.ok for r in recs)


def test_sweep_stops_after_oom(tiny_loaded, monkeypatch):
    from ai_inference_lab.backends import pytorch as pb

    real_run = pb.PyTorchBackend.run

    def flaky(self, inputs):
        if inputs.shape[0] >= 2:
            raise RuntimeError("out of memory")
        return real_run(self, inputs)

    monkeypatch.setattr(pb.PyTorchBackend, "run", flaky)
    recs = sweep_batch_sizes(tiny_loaded, "cpu", batch_sizes=(1, 2, 4), warmup_iters=1, measure_iters=2)
    assert [r.batch_size for r in recs] == [1, 2] and recs[1].errors


def test_compare_devices_cpu_fallback(tiny_loaded, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = compare_devices(tiny_loaded, batch_size=1, **FAST)
    assert out["gpu_available"] is False and out["speedup"] is None
    assert len(out["records"]) == 1 and out["records"][0].hardware["device_type"] == "cpu"
    assert "not available" in out["message"]


@requires_cuda
def test_compare_devices_cuda(tiny_loaded):
    out = compare_devices(tiny_loaded, batch_size=1, **FAST)
    assert out["gpu_available"] and out["speedup"] > 0
    assert out["records"][1].metrics["gpu_peak_allocated_mb"] > 0


@requires_cuda
def test_fp16_cuda(tiny_loaded):
    rec = run_benchmark(BenchmarkConfig("cuda", "fp16", 2, **FAST), tiny_loaded)
    assert rec.ok
