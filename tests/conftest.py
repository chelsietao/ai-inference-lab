import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch

from ai_inference_lab.models.classifier import load_classifier

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")


@pytest.fixture(scope="session")
def tiny_loaded():
    """Random-weight MobileNetV3-Small: no network needed."""
    return load_classifier(pretrained=False)
