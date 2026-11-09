# Glossary

| 名詞 | 說明 |
|---|---|
| Inference | 用訓練好的 model 做預測(只有 forward pass)。 |
| Latency | 單一請求(或單一批次)從開始到完成的時間。 |
| Throughput | 單位時間處理的量(images/sec)。 |
| P90 / P95 latency | 90% / 95% 的請求不超過的延遲;反映尾端(tail)表現。 |
| Batch size | 一次送進 model 的樣本數。 |
| Warm-up | 正式計時前先跑幾次,避開初始化成本。 |
| Cold start | 第一次執行,含初始化(CUDA context、cache 等)。 |
| Steady state | 暖機後穩定的表現。 |
| Asynchronous execution | GPU 工作排入佇列後立即返回,不等完成。 |
| `torch.cuda.synchronize()` | 等 GPU 完成所有已排入的工作。 |
| Kernel | 在 GPU 上執行的函式。 |
| Kernel launch overhead | 啟動 kernel 的固定成本,小 workload 時佔比高。 |
| GPU utilization | GPU 運算單元被使用的程度。 |
| Parallelism | 同時處理多個運算。 |
| Precision (FP32 / FP16) | 數值位元數;FP16 省記憶體且可能更快,但數值結果可能不同。 |
| OOM | Out of Memory,記憶體不足。 |
| Allocated vs reserved memory | allocated = tensors 實際使用;reserved = PyTorch allocator 向 GPU 保留的總量。 |
| Speedup | baseline latency / candidate latency。 |
| Preprocessing | 把原始資料轉成 model 輸入(resize、normalize…)。 |
| Top-K | 機率最高的 K 個預測。 |
| Pretrained weights | 已在大型資料集(如 ImageNet)訓練好的參數。 |
| Bottleneck | 限制整體速度的環節。 |
