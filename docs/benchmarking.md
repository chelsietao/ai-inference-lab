# Benchmarking Guide

## 量測方法
1. **載入**:`load_classifier` 計算載入時間,與 latency 分開記錄(`model.load_time_s`)。首次執行含權重下載。
2. **Cold start**:第一次 forward(含 lazy init)單獨計時 → `metrics.cold_start_ms`。
3. **Warm-up**:`warmup_iters` 次,丟棄。
4. **量測**:`measure_iters` 次,每次 `synchronize → perf_counter → run → synchronize → perf_counter`。
5. **統計**:mean / median / P90 / P95 / min / max / std(單位 ms,每個 batch)。
6. **Throughput** = `batch_size / mean_ms × 1000`(images/s)。**Per-image latency** = `mean_ms / batch_size`。

## 為什麼要 synchronize
CUDA kernel 非同步啟動;Python 呼叫返回時 GPU 可能還沒做完。不 synchronize,計時器停在「排入佇列」而非「完成」。本專案每次迭代前後都 synchronize,因此量到的是**單批次的完整 latency**,不是流水線化的最大 throughput。

## Preprocessing
預設 benchmark **不含** preprocessing(輸入已在 device 上)。Lab 01 示範分開量測。End-to-end latency 需另外量測 preprocessing + inference。

## 記憶體
- CUDA:`gpu_peak_allocated_mb` = `torch.cuda.max_memory_allocated()`,含 weights + input + activations;`gpu_baseline_allocated_mb` 為量測開始時已配置量。
- 不含:CUDA context、caching allocator 保留未使用空間、其他行程、系統 RAM。
- CPU / MPS:不量測(`null`)。CPU 與 CUDA 的記憶體數字**意義不同**,不可直接比較。

## Speedup
`speedup = CPU mean / GPU mean`,僅在相同 model、batch size 的 CPU fp32 與 GPU 結果都成功時計算。>1 表示 GPU 較快。這只是單一環境、單一 workload 的比值,不可一般化。

## Precision
支援 `fp32`;`fp16` 僅 CUDA。CPU 要求 fp16 會丟出 `UnsupportedConfigurationError`,不會偷偷改用 fp32。

## 錯誤處理
- OOM:記錄於 `errors`,sweep 於第一個失敗的 batch size 停止。
- 不可用的 device / precision:丟出 `UnsupportedConfigurationError`(不靜默替換,避免把 CPU 結果標成 GPU)。
- 高變異(std > 25% mean):加入 `warnings`。

## 結果 schema(`schema_version = "1.0"`)
頂層 `{schema_version, records[]}`。每筆 record:`experiment_id, timestamp, schema_version, model, framework, hardware, backend, precision, input_shape, batch_size, config, metrics, methodology, errors, warnings`。CSV 為攤平欄位(`metrics.mean_ms`、`hardware.device_type` …)。新增欄位只會往後加;破壞性變更會提高 `schema_version`。

## 解讀注意事項
- 單次 run 不是統計結論;看 std 與重複結果。
- Colab 是共用環境,不同 session 的 GPU 型號與效能可能不同。
- 不要跨硬體、跨 batch size、跨 precision 當成同一種比較。
