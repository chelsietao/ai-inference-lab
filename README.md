# AI Inference Lab

**An Interactive Playground for GPU Computing and AI Inference Optimization**

在 Google Colab Free 上執行真實的 pretrained model,量測 CPU / GPU inference,視覺化 latency 與 throughput,並理解背後的 AI Systems 原理。零成本、不需要 API key、沒有 GPU 也能學。

## Project Motivation

Inference 的速度與成本決定 AI 產品能不能上線。本專案用可實際執行的實驗回答:GPU 為什麼有時比 CPU 慢?Batch size 如何影響 latency 與 throughput?怎樣才是正確的量測?它是 [AI Engineering Atlas](https://chelsietao.github.io/ai-rookie-skill-tree-lab/)(概念知識庫,獨立專案)的實作搭檔。

## Key Features

- 真實的 **MobileNetV3-Small** 影像分類(torchvision pretrained weights + official transforms)
- 自動 **Hardware 偵測**(CUDA / MPS / CPU),CPU 永遠可用
- 可重用的 **Benchmark 引擎**:warm-up、CUDA synchronize、mean / median / P90 / P95、throughput、GPU peak memory、OOM 處理
- **CPU vs GPU** 比較與 **Batch size** sweep(1 / 4 / 8 / 16 / 32)
- 結果匯出 **JSON / CSV**(有版本的 schema)與 matplotlib 圖表
- 靜態 **Dashboard**(GitHub Pages 相容),明確標示「預錄結果、非 live」
- 繁體中文教學 notebooks

## System Architecture

```
notebooks (Colab)  ─┐
examples/ (CLI)    ─┼─▶ ai_inference_lab ─▶ results JSON/CSV ─▶ charts / dashboard
tests (pytest)     ─┘     ├─ utils.device      device 偵測、precision 檢查
                          ├─ models.classifier 載入 model、preprocess、Top-K
                          ├─ backends          InferenceBackend 介面 + PyTorch 實作
                          ├─ benchmarking      runner / compare / schema
                          ├─ metrics.stats     純函式統計(latency、throughput、speedup)
                          └─ visualization     charts
```

詳見 [docs/architecture.md](docs/architecture.md)。

## Supported Hardware

| Hardware | 狀態 |
|---|---|
| CPU | 必備,已在本機驗證 |
| NVIDIA CUDA GPU | 已實作並自動偵測;**尚未在 CUDA 硬體上驗證**(見 Limitations) |
| Apple Silicon MPS | 選用,本機可執行(不收集記憶體統計) |
| TPU | 不支援 |

## Google Colab Quick Start

1. 點下方任一 **Open in Colab**。
2. (選用)`Runtime` → `Change runtime type` → **T4 GPU**。Colab Free 不保證隨時有 GPU,沒有也能跑。
3. `Runtime` → `Run all`。第一個 cell 會自動 `pip install` 本套件。

套件由 GitHub 安裝(`git+https://github.com/chelsietao/ai-inference-lab.git@main`),因此 notebooks 需要本 repo 的 `main` 分支包含此程式碼才能在 Colab 安裝成功(合併 PR 後生效)。更多說明:[docs/getting-started.md](docs/getting-started.md)。

## Available Experiments

| Lab | 內容 | Colab |
|---|---|---|
| 01 PyTorch Inference | 影像分類、Top-5、hardware 偵測 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chelsietao/ai-inference-lab/blob/main/notebooks/01_pytorch_inference.ipynb) |
| 02 CPU vs GPU | latency / throughput / speedup、圖表、匯出 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chelsietao/ai-inference-lab/blob/main/notebooks/02_cpu_vs_gpu.ipynb) |
| 03 Batch Size | per-batch vs per-image latency、throughput、OOM 處理 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chelsietao/ai-inference-lab/blob/main/notebooks/03_batch_size_benchmark.ipynb) |

## Benchmark Methodology

- `time.perf_counter()` 量測單次 forward pass;`torch.inference_mode()` + `model.eval()`。
- CUDA:每次計時前後 `torch.cuda.synchronize()`(GPU 為非同步執行)。
- Model 載入 / 權重下載 / 輸入建立 / preprocessing **不計入** latency。
- 第一次呼叫(cold start)單獨記錄;warm-up 次數不納入統計;報告 mean、median、P90、P95、std。
- 記憶體:`torch.cuda.max_memory_allocated()`——只含 PyTorch tensors 的 peak,**不含** CUDA context 與 allocator 保留空間;CPU 不量測。
- 輸入為固定 shape、固定 seed 的隨機 tensor(測速度,不測準確度)。

完整說明:[docs/benchmarking.md](docs/benchmarking.md)。

## Example Result Format

```json
{
  "schema_version": "1.0",
  "records": [{
    "experiment_id": "…", "timestamp": "…", "backend": "pytorch", "precision": "fp32",
    "model": {"name": "mobilenet_v3_small", "pretrained": true, "load_time_s": 0.0},
    "hardware": {"device_type": "cpu", "device_name": "…", "pytorch_version": "…"},
    "input_shape": [8, 3, 224, 224], "batch_size": 8,
    "config": {"warmup_iters": 5, "measure_iters": 50},
    "metrics": {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0,
                "throughput_images_per_sec": 0.0, "gpu_peak_allocated_mb": null},
    "methodology": "…", "errors": [], "warnings": []
  }]
}
```
(數值僅示意格式,皆為 0;真實範例見 [dashboard/data/reference_cpu.json](dashboard/data/reference_cpu.json)。)

## Limitations

- **GPU 路徑尚未在 CUDA 硬體上驗證。** 本專案開發機(Apple Silicon Mac)沒有 NVIDIA GPU;CUDA 相關程式碼(synchronize、memory stats、fp16)已實作且有 mock/skip 測試,但從未在真實 CUDA 上執行。請在 Colab GPU runtime 自行驗證並回報問題。
- Dashboard 內建的參考資料是在一台 **Apple M5 Pro CPU** 上的單次量測,僅供展示格式,不代表任何其他環境的表現。
- 預測正確性僅在 Lab 01 示範;benchmark 使用隨機輸入。
- 目前只有 PyTorch eager backend(ONNX Runtime、TensorRT 等見 Roadmap)。

## Roadmap

Phase 1(本版)影像分類與 benchmark → Phase 2 ONNX Runtime / FP16 / Quantization → Phase 3 GPU computing → Phase 4 Triton kernels → Phase 5 TensorRT → Phase 6 LLM inference。詳見 [docs/roadmap.md](docs/roadmap.md)。

## Project Structure

```
src/ai_inference_lab/   套件(models, backends, benchmarking, metrics, visualization, utils)
notebooks/              三本 Colab notebooks
tests/                  pytest(CPU;CUDA 測試自動 skip)
examples/               CLI 範例
dashboard/              靜態 dashboard(GitHub Pages)與參考資料
docs/                   getting-started, architecture, benchmarking, glossary, roadmap, troubleshooting
results/                本機輸出(預設不提交,見 results/README.md)
.github/workflows/      CI(CPU)與 Pages 部署
```

## Local Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # 或依平台使用 pytorch.org 指令
pip install -e ".[dev]"
python examples/run_cpu_benchmark.py --device cpu --out results/run.json
python -m http.server --directory dashboard 8000   # 預覽 dashboard
```

## Testing Instructions

```bash
pytest -q
```
測試使用隨機 weights,不需要網路;CUDA 專屬測試在無 GPU 時顯示 skipped。CI 在 GitHub 的 CPU runner 上執行。

## Technologies

Python · PyTorch · torchvision · NumPy · pandas · matplotlib · pytest · GitHub Actions · GitHub Pages(vanilla HTML/JS,無外部依賴)

## References

- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [torchvision models](https://pytorch.org/vision/stable/models.html)
- [MobileNetV3 paper](https://arxiv.org/abs/1905.02244)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

## License

[MIT](LICENSE)
