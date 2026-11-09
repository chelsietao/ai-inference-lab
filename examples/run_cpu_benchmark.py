"""Run the benchmark suite from the command line (no notebook needed).

    python examples/run_cpu_benchmark.py --device cpu --out results/my_run.json

Runs a batch-size sweep (and CPU-vs-GPU comparison when CUDA exists) and saves
all records to one JSON file following the documented schema.
"""

from __future__ import annotations

import argparse

import torch

import ai_inference_lab as lab


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    p.add_argument("--out", default="results/run.json")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--random-weights", action="store_true", help="skip weight download")
    args = p.parse_args()

    loaded = lab.load_classifier(pretrained=not args.random_weights)
    device = lab.select_device(args.device).type
    print(lab.detect_hardware(device).summary(), "\n")

    records = lab.sweep_batch_sizes(
        loaded, device, "fp32", (1, 4, 8, 16, 32), args.warmup, args.iters
    )
    if device == "cpu" and torch.cuda.is_available():
        records += lab.compare_devices(loaded, 8, args.warmup, args.iters)["records"][1:]
    print(lab.summary_table(records).to_string(index=False))
    print("saved", lab.save_json(records, args.out))


if __name__ == "__main__":
    main()
