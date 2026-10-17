# 內附教學素材 / Bundled examples

Clone 或下載專案 ZIP 後，素材已包含在專案內，不需另抓完整 COCO，也不需要搬到固定系統目錄。

| 用途 | 位置 | 內容 |
|---|---|---|
| 圖片推論 | `media/images/` | COCO val2017 固定隨機抽樣 10 張原圖 |
| 影片推論 | `media/video/freezy_traffic.webm` | 1080p、約 20 秒的街頭交通影片 |
| 偵測微調 | `finetuning/coco10_yolo/dataset.yaml` | YOLO Detection，train 8 / val 2，71 個框 |
| 原始標註 | `media/annotations/` | 10 張圖的 COCO instances / captions 子集 |

## 使用

1. 啟動工作台，在「推論比較」的素材區可直接選擇 10 張圖或影片。
2. 在「微調訓練」選擇支援 YOLO Detection 的模型，再按 **載入教學資料集**。系統只驗證並選用；設定模型權重及訓練參數後，按開始才會訓練。
3. 亦可透過本機路徑指定這份 YAML，不必重新壓縮上傳。模型權重需另外準備。

這個小資料集只供確認操作流程，不適合模型準確率排名。影片未附行為分類標籤，不能直接當作行為模型的監督微調資料；行為微調需自備 Video CSV 與獨立 train / val 影片。

## 來源與授權

逐張圖片作者／來源／授權、COCO 標註授權和影片署名請見 [media/ATTRIBUTION.md](media/ATTRIBUTION.md)。圖片保留原始 bytes，包含 NC／ND／SA 等不同條件，供非商業教學示範；素材授權與本專案程式 AGPL-3.0 分開。

公開範例影片使用 **Freezy Traffic / LIFEOFVIDS / CC BY 3.0**；先前本機的 Pexels 影片不包含在此套件。原始抽樣 seed 為 `20261017`，從 5000 張 COCO val2017 中抽 10 張。完整來源、雜湊與微調轉換紀錄均隨附。

`checksums.sha256` 列出所有範例檔案（不含其自身），可用來確認下載是否完整。
