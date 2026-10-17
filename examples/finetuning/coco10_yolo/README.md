# COCO 10-image fine-tuning example

在「微調訓練」選擇支援 YOLO Detection 資料格式的偵測模型，再按 **載入教學資料集**，即可驗證並選用本資料集。也可輸入此目錄或 `dataset.yaml` 的本機路徑；不需要 ZIP。

- 8 張訓練圖片、2 張驗證圖片，80 個 COCO 類別定義。
- 60 個訓練框、11 個驗證框；只包含物件偵測標註。
- 影像 ID 排序後以前 8 張訓練、後 2 張驗證，非分層切分；訓練集只出現 23 類、驗證集 5 類，broccoli 與 dining table 未出現在訓練集。
- 用來測試資料驗證、短程微調、存檔及回載流程，不能用來正式評估準確率或模型排名。

`path: .` 相對於本 YAML；圖片和標註均在目錄內，不含原開發電腦的絕對路徑。原圖沒有修改；COCO bbox 轉成 YOLO 正規化中心座標，類別索引為按 COCO category ID 排序的 0–79。轉換規則、逐檔雜湊與切分請見 [provenance.json](provenance.json)。

圖片各自授權與作者見 [媒體署名](../../media/ATTRIBUTION.md)；標註為 COCO Consortium 的 CC BY 4.0。程式的 AGPL-3.0 不會取代資料授權。
