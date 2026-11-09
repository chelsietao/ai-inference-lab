import pytest
import torch

from ai_inference_lab.utils.device import (
    UnsupportedConfigurationError,
    detect_hardware,
    select_device,
    supported_precisions,
    validate_precision,
)
from conftest import requires_cuda


def test_cpu_always_selectable():
    assert select_device("cpu").type == "cpu"


def test_unknown_device_raises():
    with pytest.raises(UnsupportedConfigurationError):
        select_device("tpu")


def test_cuda_request_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("cuda").type in {"cpu"}


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("auto").type == "cuda"


def test_precision_support_table():
    assert supported_precisions("cpu") == ["fp32"]
    assert "fp16" in supported_precisions("cuda")


def test_fp16_on_cpu_is_unsupported():
    with pytest.raises(UnsupportedConfigurationError):
        validate_precision(torch.device("cpu"), "fp16")
    with pytest.raises(UnsupportedConfigurationError):
        validate_precision(torch.device("cpu"), "int3")


def test_detect_hardware_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    hw = detect_hardware("cpu")
    assert hw.device_type == "cpu" and not hw.cuda_available
    assert hw.cuda_version is None and hw.gpu_total_memory_mb is None
    assert any("CUDA is not available" in n for n in hw.notes)
    assert "PyTorch version" in hw.summary()


@requires_cuda
def test_detect_hardware_cuda():
    hw = detect_hardware("cuda")
    assert hw.device_type == "cuda" and hw.gpu_total_memory_mb > 0
