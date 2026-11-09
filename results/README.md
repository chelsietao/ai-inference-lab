# results/

Notebooks 與 `examples/run_cpu_benchmark.py` 會把輸出(JSON / CSV / PNG)寫到這裡。預設 **不提交**(見 `.gitignore`),避免把大型或與環境相關的產物放進 repo。

要發布一筆結果:確認它是真實量測,保留其 hardware metadata,然後放到 `dashboard/data/` 並更新 `dashboard/app.js` 的載入路徑。

目前 repo 內唯一的資料是 `dashboard/data/reference_cpu.json`:在 Apple M5 Pro CPU(macOS、PyTorch 2.8.0)上以 `examples/run_cpu_benchmark.py` 實際量測(warm-up 5、measure 50),僅供展示格式。
