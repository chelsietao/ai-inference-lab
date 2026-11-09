# Architecture

## 設計原則
- Notebook 不重複貼 timing 程式碼:所有量測在 `ai_inference_lab.benchmarking`。
- 純函式(`metrics.stats`)與硬體相關程式碼分離,方便 unit test。
- 結果是有版本的資料(`BenchmarkRecord`, `schema_version`),圖表與 dashboard 只讀資料。
- 沒有 server、資料庫、帳號;零成本。

## 模組
| 模組 | 職責 |
|---|---|
| `utils.device` | `detect_hardware`、`select_device`(CUDA > MPS > CPU)、precision 驗證、`UnsupportedConfigurationError` |
| `models.classifier` | 載入 torchvision model(`eval()`)、official transforms、`load_image`、`classify` |
| `backends.base` | `InferenceBackend` 介面:`make_input / run / synchronize / memory_stats_mb` |
| `backends.pytorch` | PyTorch eager 實作(deep copy model,避免改動呼叫者的 model) |
| `benchmarking.runner` | `measure_latencies`(cold start、warm-up、同步計時)、`run_benchmark` |
| `benchmarking.compare` | `compare_devices`、`sweep_batch_sizes` |
| `benchmarking.schema` | `BenchmarkRecord`、methodology 與 memory 說明文字 |
| `utils.export` | JSON / CSV / DataFrame / summary table |
| `visualization.charts` | 5 種圖表 |

## 資料流
```
load_classifier ─▶ PyTorchBackend(device, precision) ─▶ measure_latencies ─▶ BenchmarkRecord
                                                                                  │
                                           save_json / save_csv ◀────────────────┤
                                           charts.plot_* ◀───────────────────────┘
```

## 擴充新 backend(Phase 2+)
實作 `InferenceBackend`(例如 `OnnxRuntimeBackend`),讓 `run_benchmark` 接受 backend 工廠;`BenchmarkRecord.backend` 已記錄 backend 名稱,schema 不需變動。

## 與 AI Engineering Atlas 的整合點(未修改 Atlas)
| Atlas 概念 | 對應 Lab |
|---|---|
| GPU Computing | Lab 02 CPU vs GPU |
| Inference Optimization | Phase 2 PyTorch vs ONNX Runtime(planned) |
| Quantization | Phase 2 FP32 / FP16 / INT8(planned) |
| Compiler Engineering | Phase 4 Operator fusion、Triton(planned) |
