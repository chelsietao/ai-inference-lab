"""PyTorch eager-mode backend (CPU / CUDA / MPS)."""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
from torch import nn

from ai_inference_lab.backends.base import InferenceBackend
from ai_inference_lab.utils.device import validate_precision


class PyTorchBackend(InferenceBackend):
    name = "pytorch"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        precision: str = "fp32",
        input_chw: Tuple[int, int, int] = (3, 224, 224),
    ) -> None:
        validate_precision(device, precision)
        self.device = device
        self.device_type = device.type
        self.precision = precision
        self.input_chw = input_chw
        # Copy so that moving/casting never mutates the caller's model.
        self.model = copy.deepcopy(model).eval().to(device)
        if precision == "fp16":
            self.model = self.model.half()
        self._dtype = torch.float16 if precision == "fp16" else torch.float32

    def make_input(self, batch_size: int, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(batch_size, *self.input_chw, generator=gen)
        return x.to(device=self.device, dtype=self._dtype)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(inputs)

    def synchronize(self) -> None:
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device_type == "mps":
            torch.mps.synchronize()

    def reset_peak_memory(self) -> None:
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats_mb(self) -> Optional[Tuple[float, float]]:
        if self.device_type != "cuda":
            return None
        mb = 1024**2
        return (
            torch.cuda.memory_allocated(self.device) / mb,
            torch.cuda.max_memory_allocated(self.device) / mb,
        )
