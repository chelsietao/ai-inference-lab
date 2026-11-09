# Getting Started

## 在 Google Colab 使用(推薦)
1. 開啟 README 中的 **Open in Colab** 連結。
2. (選用)`Runtime > Change runtime type > T4 GPU`。沒有 GPU 時,實驗自動只跑 CPU。
3. 依序執行 cell。第一個 cell 會執行:
   `pip install git+https://github.com/chelsietao/ai-inference-lab.git@main`
   Colab 內建 torch / torchvision,pip 不會重裝。
4. Model weights 在 runtime 下載到 Colab 暫存空間,不會進入 GitHub。
5. 結果會寫入 Colab 暫存的 `results/`,要保留請從左側檔案面板下載。

> Colab 載入的是 `main` 分支。若你在 feature branch 測試,把安裝 cell 的 `@main` 改成你的分支名稱。

## 在本機使用
見 README 的 Local Development。

## 學習順序
Lab 01(流程)→ Lab 02(正確量測)→ Lab 03(batch size)。每本 notebook 開頭說明目標、先備知識與是否需要 GPU。
