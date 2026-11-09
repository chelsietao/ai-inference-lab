# Troubleshooting

| 問題 | 原因 | 處理 |
|---|---|---|
| GPU unavailable | Runtime 未選 GPU,或 Colab Free 暫無配額 | `Runtime > Change runtime type > T4 GPU`;沒有就用 CPU,Lab 會自動只跑 CPU |
| `CUDA out of memory` | batch 過大或前一次實驗佔用 | 縮小 `BATCH_SIZES`;`Runtime > Restart session`。sweep 會把 OOM 記錄在 `errors` 並停止 |
| Package installation failure | 網路、或 `@main` 尚未包含本程式碼 | 重跑安裝 cell;feature branch 請改 `@<branch>`;必要時 Restart session |
| Runtime restart 後變數消失 | Colab 重啟會清空記憶體與暫存 | 從安裝 cell 重新執行;先下載 `results/` |
| Model download failure | 網路或 PyTorch 下載站暫時失敗 | 重試;或 `USE_PRETRAINED=False` 做速度測試(預測無意義) |
| `Expected all tensors to be on the same device` | model 與 input 在不同 device | 兩者都 `.to(device)`;本專案 backend 已處理 |
| `UnsupportedConfigurationError: fp16 ... cpu` | CPU 不支援 fp16 benchmark | 改用 `fp32`,或切換到 CUDA |
| 結果每次差很多 | 共用環境、CPU/GPU 頻率變化、iters 太少 | 增加 `MEASURE_ITERS`、重跑、看 std 與 P95;不要用單次結果下結論 |
| 本機 macOS 的 `device: mps` | `auto` 選到 Apple GPU | 需要 CPU 結果時設 `DEVICE = "cpu"` |
