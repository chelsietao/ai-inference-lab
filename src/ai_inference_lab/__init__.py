"""AI Inference Lab: benchmark and understand AI inference on CPU and GPU."""

from ai_inference_lab.backends.pytorch import PyTorchBackend
from ai_inference_lab.benchmarking.compare import compare_devices, sweep_batch_sizes
from ai_inference_lab.benchmarking.runner import BenchmarkConfig, run_benchmark
from ai_inference_lab.benchmarking.schema import SCHEMA_VERSION, BenchmarkRecord
from ai_inference_lab.models.classifier import classify, load_image, load_classifier
from ai_inference_lab.utils.device import (
    HardwareInfo,
    UnsupportedConfigurationError,
    detect_hardware,
    select_device,
)
from ai_inference_lab.utils.export import records_to_dataframe, save_csv, save_json, load_json, summary_table

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRecord",
    "HardwareInfo",
    "PyTorchBackend",
    "SCHEMA_VERSION",
    "UnsupportedConfigurationError",
    "classify",
    "compare_devices",
    "detect_hardware",
    "load_classifier",
    "load_image",
    "load_json",
    "records_to_dataframe",
    "run_benchmark",
    "save_csv",
    "save_json",
    "summary_table",
    "select_device",
    "sweep_batch_sizes",
]
