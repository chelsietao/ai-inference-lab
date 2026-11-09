"""Image classification with a pretrained torchvision model (MobileNetV3-Small)."""

from __future__ import annotations

import io
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch import nn
from torchvision import models

from ai_inference_lab.utils.device import select_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mobilenet_v3_small"
SAMPLE_IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"


@dataclass
class LoadedModel:
    """A model plus the official transform that goes with its weights."""

    name: str
    model: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    categories: List[str]
    pretrained: bool
    load_time_s: float
    input_size: Tuple[int, int, int]  # (C, H, W)


def load_classifier(name: str = DEFAULT_MODEL, pretrained: bool = True) -> LoadedModel:
    """Load a torchvision classifier in eval mode on CPU.

    Weights download once at runtime (cached by torch, not stored in git).
    Load time is measured here so it never leaks into inference latency.
    `pretrained=False` builds a randomly-initialised model: useful for
    offline tests and pure speed benchmarks, but its predictions are meaningless.
    """
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision model '{name}'")
    start = time.perf_counter()
    builder = getattr(models, name)
    if pretrained:
        weights_enum = models.get_model_weights(name).DEFAULT
        model = builder(weights=weights_enum)
        transform = weights_enum.transforms()
        categories = list(weights_enum.meta["categories"])
    else:
        model = builder(weights=None)
        transform = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
        categories = [f"class_{i}" for i in range(1000)]
    model.eval()
    return LoadedModel(
        name=name,
        model=model,
        transform=transform,
        categories=categories,
        pretrained=pretrained,
        load_time_s=time.perf_counter() - start,
        input_size=(3, 224, 224),
    )


def load_image(source: Union[str, "io.BytesIO", None] = None) -> Image.Image:
    """Load an image from a path, URL, or (default) the PyTorch Hub sample dog.

    If the sample cannot be downloaded, a deterministic synthetic image is
    returned and a warning is logged, so offline runs still work (the
    predictions will then not be meaningful).
    """
    src = source if source is not None else SAMPLE_IMAGE_URL
    try:
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "ai-inference-lab"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        return Image.open(src).convert("RGB")
    except Exception as exc:
        if source is not None:
            raise
        logger.warning("Could not download sample image (%s); using a synthetic image.", exc)
        return synthetic_image()


def synthetic_image(size: int = 256) -> Image.Image:
    """Deterministic gradient image used as an offline fallback."""
    gen = torch.Generator().manual_seed(0)
    arr = (torch.rand(size, size, 3, generator=gen) * 255).byte().numpy()
    return Image.fromarray(arr)


def classify(
    loaded: LoadedModel,
    image: Image.Image,
    device: Optional[torch.device] = None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return Top-k (label, probability) pairs for one image."""
    device = device or select_device("auto")
    model = loaded.model.to(device).eval()
    batch = loaded.transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]
    values, indices = probs.topk(top_k)
    return [(loaded.categories[i], float(v)) for v, i in zip(values.cpu(), indices.cpu())]
