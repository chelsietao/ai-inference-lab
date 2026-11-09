# Roadmap

狀態:✅ 已實作並驗證(CPU)· 🟡 已實作、未在 CUDA 驗證 · ⬜ 規劃中

## Phase 1 — Image classification & benchmarking(v0.1)
- ✅ Hardware 偵測、classification、benchmark 引擎、JSON/CSV、charts、dashboard、3 notebooks
- 🟡 CUDA 路徑(synchronize、memory、fp16)

## Phase 2 — Inference optimization ⬜
ONNX export、ONNX Runtime、PyTorch vs ORT、FP32 vs FP16、quantization、model size、輸出一致性檢查(精度改變可能影響數值,需驗證而非假設無損)。

## Phase 3 — GPU computing ⬜
GPU 架構、thread/block、memory hierarchy、bandwidth、compute- vs memory-bound、profiling、matmul、kernel launch overhead。

## Phase 4 — Kernel optimization ⬜
Triton 基礎、vector add、matmul、coalescing、tiling、fusion;每個 kernel 先對照 reference 驗證正確性再比較效能。

## Phase 5 — TensorRT ⬜
僅在相容 NVIDIA 硬體上;不強制安裝。

## Phase 6 — LLM inference ⬜
Token generation、prefill vs decode、TTFT、tokens/s、KV cache、batching、小模型 serving(依 Colab Free 可行性選擇工具)。
