# 洗手工作流：模型、下載與判讀

洗手工作流把手部位置、真實分割遮罩及影片動作分數分開計算，再依 ROI、持續時間與證據有效期產生候選事件。這些輸出用來測試視覺流程；沒有沖洗與流程順序的證據時，不代表已完成洗手，也不是衛生合格評分。

## 模型各自負責什麼

| 模型 | 輸入與輸出 | 在洗手流程中的用途與限制 |
|---|---|---|
| MediaPipe Hand Landmarker Full | RGB 影格 → 手掌偵測、每手 21 個關節、左右手分類 | 適合手部近拍，提供手的位置與提示框。手互相遮擋時可能只找到一手；不能補造另一手的證據。不是洗手動作分類器。 |
| SAM 2.1 Hiera Tiny | RGB 影格＋物件提示框 → 像素遮罩 | 對偵測到的手框產生真實分割，計算遮罩與洗手 ROI 的重疊。SAM 不自行知道物件是手、水流或洗手動作；提示框品質會影響結果。 |
| X3D-L／SlowFast／ViViT（Kinetics-400） | RGB 時序片段 → 動作分數 | 標籤表包含 `washing hands`（零起算類別 382）。分數是封閉類別模型輸出，不是洗手完成率。這些權重沿用原有動作模型安裝。 |
| VideoMAE V2（Kinetics-710） | RGB 時序片段 → 動作分數 | 原有模型的標籤表包含 `washing hands`（零起算類別 211）。資源需求較高，可另外比較。 |
| YOLO26n Pose＋ST-GCN++ | 同一人連續的 COCO17 人體骨架 → NTU120 動作分數 | 本工具使用 HRNet 2D／COCO17 的 ST-GCN++ 權重。類別 33 是 `rub two hands together`，只提供搓手線索；沒有 `washing hands` 類別。手部近拍缺少完整人體骨架時，應顯示證據不足。 |

**MediaPipe 的每手 21 點與 ST-GCN++ 的人體 17 點不能直接互換。** 兩者是不同的關節定義與訓練資料；本工具不把手指點硬套成身體點。MediaPipe 不提供逐關節信心機率，工作流中的畫面內旗標也不是該機率。

模型依據：[Google Hand Landmarker 文件](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)、[Meta SAM 2 文件與權重](https://github.com/facebookresearch/sam2)、[PYSKL ST-GCN++ 模型表](https://github.com/kennymckormick/pyskl/blob/main/configs/stgcn%2B%2B/README.md)、[SlowFast Kinetics-400 標籤表](https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json)。模型的原資料集成績不等同於此洗手場景的準確率。

## 下載兩個新增模型

在專案根目錄執行；`python` 請使用啟動工作檯的同一個 Python 環境。

```powershell
python -m pip install -r requirements-workflows.txt
python download_workflow_models.py --models hand mask
python download_workflow_models.py --models hand mask --download
python download_workflow_models.py --models hand mask --verify
```

第一個下載器命令只列計畫，加入 `--download` 才下載。兩個新增權重合計 **163,827,571 bytes（約 163.83 MB）**，不含 Python 套件、RAM 或 VRAM。這不會下載影片素材，也不會下載大型影片動作模型。

| 別名／模型 ID | 預設目標檔案 | 大小 | 權重授權 |
|---|---|---:|---|
| `hand`／`mediapipe_hand_landmarker` | `models/mediapipe_hand_landmarker/hand_landmarker.task` | 7,819,105 bytes | Apache-2.0 |
| `mask`／`sam2_1_tiny` | `models/sam2_1_tiny/sam2.1_t.pt` | 156,008,466 bytes | Apache-2.0 |

可用 `--output-dir` 指定資產根目錄；下載器會保留上表的模型子目錄。目前這些選配工作流模型搜尋專案的 `models/`、`model_weights/` 與專案上一層的 `model_weights/`。任意外部下載目錄不會自動加入這些節點的搜尋範圍，請將輸出設在上述任一位置；一般偵測／影片模型的外部路徑登錄則維持原有方式。不指定 `--models` 時仍沿用原本的 `pose skeleton relation`，不會自動增加本輪模型。

固定來源、位元組大小與完整 SHA256 記錄在 [模型資產清單](../backend/assets/workflow_model_sources.json)：

- MediaPipe 使用 Google 官方 `float16/1`，不使用會變動的 `latest`；SHA256：`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`。
- SAM 2.1 使用 Meta 官方 Hugging Face 倉庫的固定 commit `de431c4043854a71d8101e17995dfe596bf101a5`；SHA256：`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。原檔 `sam2.1_hiera_tiny.pt` 僅改名為現有 Ultralytics 載入器需要的 `sam2.1_t.pt`，沒有轉換權重張量。

下載器以 `.download.part` 串流寫入並支援續傳，大小與 SHA256 都符合才發布正式檔案。已存在且正確的檔案直接跳過；現有檔案不符合時先報錯。只有明確指定 `--replace-invalid`，才會在新下載通過驗證後替換它。下載或驗證失敗不覆蓋原檔。若部分下載內容錯誤，可檢查來源後使用 `--restart-partial` 重新取得。

## 依賴與裝置

手部模型使用 `mediapipe==0.10.31`，這個版本不會額外安裝 `opencv-contrib-python`，也不要求降版目前的 NumPy 或 protobuf。SAM 2.1 透過現有 Ultralytics SAM 載入器執行，不需再安裝另一套 SAM 2 或編譯 CUDA 擴充；本次驗證環境為 Ultralytics 8.4.0、PyTorch 2.8.0、NumPy 2.2.6、OpenCV 4.12.0.88。

**Windows 套件已知問題：** 官方 MediaPipe 0.10.31 Windows wheel 的內部 `WHEEL` 標籤誤寫為 `cp39-cp39-linux_x86_64`，因此新版本 pip 的 `pip check` 可能報 `mediapipe 0.10.31 is not supported on this platform`。這與模型缺失不同；本次 Windows x64／Python 3.11 環境已實際完成 CPU 手部推論。0.10.30 Windows wheel 也有同樣問題。專案沒有修改套件 metadata 來隱藏警告；若發生 import／DLL 錯誤，仍需檢查當地環境，不能只依這項紀錄假設可用。

手部節點在 CPU 執行；SAM 可依工作檯裝置設定使用 CPU 或 CUDA。提高分割取樣頻率、增加手數或增加同時運行的影片模型都會提高負載。手部缺失、遮罩過期或骨架不足應保留為「無足夠證據」，不沿用舊影格結果冒充新影格觀察。

授權依據：[Google Hand Tracking Model Card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf)、[SAM 2 授權](https://github.com/facebookresearch/sam2/blob/main/LICENSE)。Meta 權重為 Apache-2.0；本工具使用的 Ultralytics runtime 另依 AGPL-3.0。本工具及各依賴維持各自授權。

## 範例素材與驗證方式

參考素材是使用者提供的 [RTI「正確洗手步驟【行政院防疫宣導影片】」](https://www.youtube.com/watch?v=8wiBWkgZ3iI) 22–44 秒片段。原片為 60 秒，指定片段是洗手台與雙手近拍，包含搓手與末尾紙巾擦乾；不適合當作完整人體骨架示範，也不應把它當成完整進入、沖洗、離開流程的標註真值。影片沒有隨原始碼重新散布，請使用自己有權使用的素材並匯入工作檯。

對新的攝影機視角，先校正 ROI，再檢查手部框、21 點骨架與像素遮罩，最後比較 RGB 動作分數與融合事件。評估應包含搓手、沖洗、擦乾及非洗手活動的分段標註與漏檢情況；成功載入模型或畫出遮罩本身，不代表已驗證洗手辨識效果。
