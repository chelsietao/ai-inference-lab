import pytest

from ai_inference_lab.metrics.stats import (
    latency_stats,
    per_image_latency_ms,
    speedup,
    throughput_images_per_sec,
)


def test_latency_stats_known_values():
    s = latency_stats(list(range(1, 101)))  # 1..100
    assert s["mean_ms"] == pytest.approx(50.5)
    assert s["median_ms"] == pytest.approx(50.5)
    assert s["p90_ms"] == pytest.approx(90.1)
    assert s["p95_ms"] == pytest.approx(95.05)
    assert s["min_ms"] == 1 and s["max_ms"] == 100


def test_latency_stats_single_sample_has_zero_std():
    assert latency_stats([5.0])["std_ms"] == 0.0


def test_latency_stats_empty_raises():
    with pytest.raises(ValueError):
        latency_stats([])


def test_throughput():
    assert throughput_images_per_sec(8, 100.0) == pytest.approx(80.0)
    with pytest.raises(ValueError):
        throughput_images_per_sec(0, 10.0)
    with pytest.raises(ValueError):
        throughput_images_per_sec(1, 0.0)


def test_per_image_latency():
    assert per_image_latency_ms(4, 20.0) == pytest.approx(5.0)


def test_speedup():
    assert speedup(100.0, 25.0) == pytest.approx(4.0)
    assert speedup(10.0, 20.0) == pytest.approx(0.5)  # candidate slower
    assert speedup(0.0, 5.0) is None
