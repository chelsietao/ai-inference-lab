import pytest
import torch
from PIL import Image

from ai_inference_lab.models.classifier import classify, load_classifier, synthetic_image


def test_load_classifier_interface(tiny_loaded):
    assert tiny_loaded.name == "mobilenet_v3_small"
    assert not tiny_loaded.model.training  # eval mode
    assert len(tiny_loaded.categories) == 1000
    assert tiny_loaded.load_time_s >= 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        load_classifier("not_a_model", pretrained=False)


def test_classify_cpu_smoke(tiny_loaded):
    preds = classify(tiny_loaded, synthetic_image(), torch.device("cpu"), top_k=5)
    assert len(preds) == 5
    probs = [p for _, p in preds]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_synthetic_image_is_deterministic():
    a, b = synthetic_image(), synthetic_image()
    assert isinstance(a, Image.Image) and list(a.getdata()) == list(b.getdata())
