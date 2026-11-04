# Vision Lab Workbench

A local visual workbench for comparing computer-vision models, running live-camera inference, and supervised fine-tuning.

**把圖片、影片、手機鏡頭與微調實驗放進同一個視覺化工作台。**

Vision Lab 提供可並排的推論畫布、每個模型獨立的參數、即時影像比較，以及資料集驗證、訓練監控與微調權重回測。預設在本機運算，服務只監聽 `127.0.0.1`；選用應用工作檯的雲端視覺節點時，才會把取樣影像送往已設定的服務。

[安裝與啟動](#安裝與啟動) · [多模型應用工作檯](#多模型應用工作檯) · [預設模型下載](#預設模型下載) · [支援的模型與微調](#目前支援的模型與微調) · [教學素材](examples/README.md)

需要獨立於介面的研究流程，可參考 [YOLO26 固定原路徑＋新增分支實驗](experiments/bento/README.md)：共用凍結骨幹、分開保存新舊輸出，提供六類預試與全來源粗細混合基線的訓練、輸出一致性驗證與計分指令。階層標註複核另存版本，尚非完整商用驗證，也不代表已完成 RF-DETR／Grounding DINO 的同等實驗。

## 功能

- 圖片／影片推論畫布：新增區塊、拖曳排列、調整參數、比較預訓練與微調權重。
- 多模型應用工作檯：連接 OpenCV 影像處理、偵測、骨架、追蹤、動作、視覺表徵、物件關係與規則節點；提供十五個可修改模板、流程驗證及圖片／影片／即時串流執行。完整安裝會準備 ComfyUI，透過 31 種原生自訂節點使用相同資源。
- 推論區塊依內容自動增高，結果、下載連結與操作按鈕不會被固定高度裁切；預設寬度隨畫布調整，手動尺寸可按區塊標題上的「自動調整區塊尺寸」還原。
- 素材若曾用於本機已完成工作的訓練集，選取框與勾號顯示紅色，並標示「已用於訓練」。以檔案內容比對，可辨識改名或重新上傳的相同檔案；只用於驗證或推論的素材不會因此標紅。
- 推論模型保留在記憶體中，換圖片／影片或調整辨識參數時可直接重用；結果顯示是否重用、載入時間與總處理時間。
- USB／虛擬攝影機與 VDO.Ninja 手機串流；最多四個即時比較區塊。
- 物件偵測與整段影片的行為分類；分別顯示來源 FPS、實際傳送速度與辨識更新速度。
- 自訂執行時間或不設限、停止清理、只保留最新待處理影格。
- 微調支援資料集驗證、最佳化器、損失函數、排程、梯度累積、混合精度及 Early Stopping；只提供該模型實作支援的選項。
- 提供標準微調，以及三種擴充方式：凍結骨幹後擴充、混合新舊資料擴充、固定原模型＋新增分支。只有混合模式必須提供舊資料。
- 「類別學習方式」旁的「方式差異」資訊卡支援懸停、鍵盤與點按，說明各模式的資料需求、參數更新、原能力保留與推論成本；預覽說明不會改動訓練設定。
- 可從原始模型或已登錄的微調 checkpoint 開始新實驗；訓練前預覽類別合併、名稱對照與資料覆蓋。
- 權重可留在原資料夾：掃描常用／自訂位置、確認候選、登錄路徑；也可放入 `models/`。
- 微調成功後，輸出的 checkpoint 自動出現在推論權重選單；也可登錄外部的相容微調模型。偵測雙分支模型可用於圖片／影片比較及應用工作流；動作與視覺表徵的雙分支模型尚未接入工作流。
- 偵測電腦硬體、Python、PyTorch、CUDA 實際可用狀態；隔離環境安裝工具會先顯示計畫。

## 目前支援的模型與微調

下表指本工作台已接入的**特定版本**，不表示同系列所有大小、任務或任意外部 checkpoint 均可載入。

| 模型（模型 ID） | 圖片／影片推論 | 即時輸入 | 微調／擴充類別 | 微調資料格式 |
|---|---|---|---|---|
| YOLOv12x (`yolov12x`) | 物件偵測 | 支援 | 支援 | YOLO Detection、COCO Detection |
| YOLO26x (`yolo26x`) | 物件偵測 | 支援 | 支援 | YOLO Detection、COCO Detection |
| YOLOv12x-Seg (`yolov12x_seg`) | 實例分割 | 支援 | 分割微調尚未接入 | — |
| YOLO26x-Seg (`yolo26x_seg`) | 實例分割 | 支援 | 分割微調尚未接入 | — |
| Grounding DINO Base (`grounding_dino_base`) | 文字指定目標偵測 | 支援 | 支援；以舊資料或指定名稱定義提示範圍 | YOLO Detection、COCO Detection |
| RF-DETR 2XL (`rf_detr_2xl`) | 物件偵測 | 支援 | 支援，需額外套件與原模型授權 | YOLO Detection、COCO Detection |
| RF-DETR-Seg-2XL (`rf_detr_seg_2xl`) | 實例分割 | 支援 | 分割微調尚未接入 | — |
| ViViT Base K400 (`vivit_base_k400`) | 影片分類 | 滾動片段分類 | 支援 | Video CSV |
| X3D-L (`x3d_l`) | 影片分類 | 滾動片段分類 | 支援 | Video CSV |
| SlowFast R101 8×8 (`slowfast_r101_8x8`) | 影片分類 | 滾動片段分類 | 支援 | Video CSV |
| VideoMAE V2 Giant K710 (`videomaev2_giant_k710`) | 影片分類 | 滾動片段分類 | 支援 | Video CSV |
| DINOv3 ViT-7B (`dinov3_7b`) | 原始視覺表徵；微調分類頭的影像／逐幀分類 | 應用工作流的 embedding 節點 | 分類接頭已接入；擴充需既有分類 checkpoint | ImageFolder |

在推論比較或即時攝影機選擇 YOLOv12、YOLO26、RF-DETR 後，可用「任務」切換物件偵測／實例分割。兩種任務使用不同權重，切換會清除不相容的 checkpoint 與舊結果。分割顯示半透明遮罩與輪廓，下載的 JSON／JSONL 包含原圖尺寸的 COCO RLE 遮罩；畫面狀態僅保留面積等摘要。偵測模型不能直接變成分割模型；現有便當 bbox 資料也不能直接作為分割監督標註。RF-DETR 的分割 2XL 使用 Apache-2.0 權重與 `rfdetr` 套件，與偵測 2XL 的 PML／`rfdetr-plus` 不同。

YOLOv12 分割使用原作者 Seg 分支權重；工作台內的相容接頭保留其獨立 qk／v 注意力投影與位置編碼，不修改已安裝套件。此相容接頭目前僅驗證推論，不能視為已支援分割訓練。

DINOv3 原始權重是視覺表徵骨幹，沒有 COCO 或 Kinetics 的分類輸出；推論輸出 pooled embedding 檔案與預覽，監督微調會建立線性分類頭。完整 7B 權重與所需環境都齊全時才允許執行；接頭存在不代表已下載完成，也不代表 12 GB GPU 能負擔完整 7B 參數微調。

推論比較與即時比較中的行為模型預設分析最近 4 秒，再均勻取樣架構所需的影格；它們輸出整段分類，不直接提供逐人追蹤或動作起訖定位。「應用工作檯」可另外連接追蹤與個人裁切分類節點。30 FPS 輸入不代表每秒完成 30 次分類。

### 訓練選項

| 模型群組 | 最佳化器 | 損失 | 微調策略 |
|---|---|---|---|
| YOLO | SGD、Adam、AdamW、NAdam、RAdam、RMSProp、MuSGD | YOLO 原生複合損失 | 完整／凍結主幹 |
| Grounding DINO | SGD、Adam、AdamW、Adagrad、RMSprop | Grounding DINO 原生複合損失 | 完整／凍結主幹 |
| RF-DETR 2XL | AdamW | RF-DETR 原生複合損失 | 完整／凍結主幹 |
| 四款行為模型 | SGD、Adam、AdamW、Adagrad、RMSprop | Cross Entropy、Focal | 分類頭／完整 |
| DINOv3 影像分類 | SGD、Adam、AdamW、Adagrad、RMSprop | Cross Entropy、Focal | 分類頭／完整 |

Early Stopping 的監控指標依模型而定：YOLO／RF-DETR 使用驗證 mAP50–95；Grounding DINO／行為模型／DINOv3 分類使用驗證損失。`patience=0` 停用提前停止。

YOLO 的暖身學習率從 0 增加至設定值，避免原生 `warmup_bias_lr=0.1` 遠高於微調學習率。實際 AMP、暖身與訓練參數會在第一批之前寫入 `effective_training.json`；出現 NaN／Inf 時仍會停止，請檢查學習率或關閉混合精度後重試。

YOLO 訓練資料增強可個別設定旋轉角度 `degrees`、色相 `hsv_h`、彩度 `hsv_s`、亮度 `hsv_v`；旋轉角度表示正負最大角度，HSV 數值是 Ultralytics 的擾動參數。預設維持原生值 `0 / 0.015 / 0.7 / 0.4`。關閉「訓練資料增強」時，這些參數與原有幾何／Mosaic 增強都停用；增強只用於訓練影像，驗證集不增強。比較增強策略時，應維持相同來源權重、資料切分、隨機種子與訓練設定。

Grounding DINO 會保留完整訓練類別提示。若 token 數超過原始 `max_text_len`，接頭會在載入模型前增加長度，上限為本機文字骨幹的位置容量（預設 BERT 為 512）。超過容量會明確拒絕，不截斷類別；完整提示與有效長度會保存至 checkpoint，供推論重載。

### 三種能力擴充方式

在微調頁的「類別學習方式」選擇策略，載入新任務資料集，再按「驗證擴充計畫」。三種模式都會預覽來源類別、追加類別、名稱對照與資料覆蓋；變更來源權重、資料、對照或超參數後，需要重新驗證。

| 模式 | 舊資料 | 訓練時更新什麼 | 推論結果與限制 |
|---|---|---|---|
| **凍結骨幹後擴充** (`expand_frozen`) | 不需要；可選填作驗證 | 保留原分類權重並追加類別，固定骨幹，訓練分類／偵測接頭 | 同一模型輸出新舊類別；原分類頭仍會更新，**可能遺忘** |
| **混合新舊資料擴充** (`expand_classes`) | **必須**提供舊訓練與驗證資料 | 保留原分類權重並追加類別，混合新舊訓練資料；可選完整或凍結策略 | 同一模型輸出新舊類別；用舊任務訓練前後指標檢查是否退步 |
| **固定原模型＋新增分支** (`dual_branch`) | 不需要；可選填作驗證 | 原模型整個分支保持不變；另一個分支依設定學新任務 | 圖片／影片依序執行兩個分支，分別顯示結果；增加模型儲存與處理耗時 |

**標準微調** (`standard`) 仍可選，依新資料集建立輸出類別，未在新資料中的原類別不會自動保留。它適合將模型轉成特定任務；想在原有類別之外增加能力時，請選上列三種方式。

凍結模式固定使用偵測模型的 `freeze_backbone`，或影像／影片分類的 `head_only`；YOLO 的 `freeze=0` 會自動選完整骨幹。固定骨幹不等於固定原分類頭，因此它不保證原能力。單一模型擴充保持原類別順序，將新類別追加在後；YOLO、RF-DETR 與分類模型複製對應原分類權重列，新類別才初始化，RF-DETR 原始 COCO category ID 另作明確映射。

雙分支模式保存**完整未更新的原模型**及新模型。相同輸入、前處理與參數下，原分支沿用原模型的預測；但合併畫面的排序、框重疊和展示可能改變，不能把整體畫面相同當成保留能力的定義。各分支分數獨立，並不是所有類別共同正規化的機率。推論比較依序執行分支；應用工作流的偵測節點會保留兩個分支供後續影格重用，因此需要容納兩者的記憶體。輸出為 `frozen_branch_bundle` 模型目錄，移轉時需保留完整目錄；目前不能拿雙分支 bundle 再作訓練起點。

- **舊資料用途**：混合模式的舊資料參與訓練，並須覆蓋原有類別及獨立驗證切分。另兩種模式的舊資料是選填，僅使用其驗證切分評估舊任務，不混入訓練。原資料不會被改寫。
- **標註完整性**：單一模型擴充時，同一張影像裡所有要學習的新舊類別物件都應標註。只標新類別可能讓未標註的舊物件被當成背景；類別映射無法補出缺少的框。雙分支的新分支使用新任務標註，原分支不更新。
- **名稱對照**：可將新資料的不同名稱指向同一個原類別。只有確認為同義的類別才應合併，例如 `broccoli` 與 `cauliflower` 不會因視覺相似而自動合併。
- **分開驗證**：有舊驗證資料時，訓練前測來源模型的舊任務，結束後測舊任務與新任務；介面及 `retention.json` 顯示 mAP50–95 或 Accuracy 和差值。未提供舊驗證資料會標示**未測量**，不會填 0 或宣稱準確率已證明不變。新驗證切分缺少的類別會明確提示。

Grounding DINO 是開放詞彙模型。混合模式用舊資料類別界定要評估的提示範圍；免舊資料模式可逐行輸入原有英文提示類別，留空預設 COCO 80 類。這只是本次使用及評估的提示範圍，不能代表模型全部開放詞彙。DINOv3 原始骨幹沒有分類頭，先完成一次標準分類微調，再以該單一分類 checkpoint 使用擴充模式。

Grounding DINO 的完整類別提示超過文字骨幹容量時，工作台按實際 tokenizer 分成多組，保存每組與全域類別的對應。每張訓練／驗證影像都處理全部組別，包含沒有該組物件的負例；不截斷類別，也不修改原始類別名稱。推論選取這類 checkpoint 會自動啟用「使用完整微調類別」，以同一份模型依序執行全部組別。取消勾選可自行指定文字目標。多組會增加推論時間，各組分數沒有做跨組機率校準；訓練前後評估固定使用同一組提示、尺寸與閾值。

從微調 checkpoint 開始的是**新實驗的權重初始化**：載入模型與類別，依本次設定重新建立最佳化器和排程。這不等於精確恢復中斷工作的 optimizer／scheduler／隨機狀態。

### 訓練程式的來源

YOLO 使用安裝版本的 Ultralytics 原生 trainer；RF-DETR 使用 `rfdetr-plus` 原生 trainer。工作台負責資料驗證、參數限制、類別擴充、監控及可攜 checkpoint。Grounding DINO 使用 Transformers 模型及原生 criterion，配合工作台訓練迴圈；ViViT、DINOv3 使用 Transformers 模型 API，X3D／SlowFast 使用 PyTorchVideo 架構，VideoMAE V2 使用其模型實作，配合本專案的 PyTorch 監督訓練迴圈與保存／載入介面。

原生或官方模型 API 不會自動保留所有舊類別；類別擴充、舊任務評估及雙分支保存／推論是工作台另外實作的流程。目前提供介面中列出的全參數、凍結主幹、分類頭與上述三種擴充方法，**沒有把 LoRA、知識蒸餾或任意自訂 loss 標示為已支援**。

### 驗證範圍

Windows + RTX 3060 12 GB 上，八款模型已完成真實 GPU 推論。YOLO26x、X3D-L、Grounding DINO 與 RF-DETR 2XL 完成過小型資料集的訓練、存檔及回測。其他訓練接頭有參數與流程測試，但未全部完成大型正式資料集訓練。

類別擴充另有分類列保留、映射、資料覆蓋、訓練前後驗證及存檔重載測試；DINOv3 接頭以小型模型驗證完整 CPU 訓練流程。這些測試不代表各最大模型都完成過實機完整訓練，DINOv3 7B 尤其需要另外準備全部權重與足夠硬體。

免舊資料的凍結模式以小型 YOLO 真實訓練、存檔及重載，核對骨幹參數和 BatchNorm 統計未變。雙分支另以小型 DINOv3 分類模型完成新分支訓練與完整打包；實際 YOLO26x 與既有 BentoCal checkpoint 在 RTX 3060 上完成雙分支推論，核對原分支的類別、分數與框座標和原模型一致，並確認再次推論可重用兩個分支快取。這證明受測輸入的輸出保留，沒有宣稱新分支準確率已改善。

YOLO26x、RF-DETR 2XL 另完成 BentoCal 的單一模型凍結骨幹擴充實驗（保留 COCO 80 類，加上 90 個新類），逐項核對保存後的骨幹參數與 buffers 均未變。Grounding DINO 的 170 類、兩組提示完成真實 GPU 訓練、存檔重載及完整類別推論測試。這些是流程驗證；固定骨幹仍可能伴隨舊任務準確率下降。

測試影片與小型切分用於確認流程，不能作為辨識準確率排名。Linux／macOS 的安裝與 CPU 路徑採可攜式設計；完整模型組合的實機驗證目前以 Windows／NVIDIA 為主。

可攜性驗證包含全新的 `base` 隔離環境安裝、前端從原始碼建置、本機資料集驗證、外部權重路徑登錄，以及 X3D-L 微調模型在缺少預訓練權重時獨立完成 GPU 推論。

## 安裝與啟動

支援 Python **3.10–3.12**，建議 **3.11**；需要 Node.js **20 或以上**與足夠的模型及環境磁碟空間。CPU 可啟動工作台；大型影片模型與微調通常需要較多 RAM／VRAM。macOS 的固定 PyTorch 2.8 安裝路徑限 Apple Silicon，工作台使用 CPU；目前不自動安裝 Intel Mac 環境。

```bash
git clone https://github.com/jpgdesign/vision-lab-workbench.git
cd vision-lab-workbench
python setup.py
```

`python setup.py` 只檢查環境與顯示安裝計畫。執行以下指令會建立工作檯自己的 `.venv`，並準備 ComfyUI：

```bash
python setup.py --install
python launch.py
```

Windows 首次安裝也可雙擊 **`INSTALL_AND_START.cmd`**，自動安裝並開啟；之後雙擊 **`START.cmd`** 即可。仍須先安裝上述 Python 與 Node.js，安裝過程需要網路及下載空間。

完整安裝預設包含 ComfyUI：先尋找已設定或常見位置的既有安裝；找不到時，從 [ComfyUI 官方專案](https://github.com/Comfy-Org/ComfyUI)下載固定版本，核對 SHA256，放入本專案的 `.tools/ComfyUI/`，另建 `.tools/comfyui-venv/`。安裝器會接入本專案節點；`python launch.py` 隨後啟動／沿用兩個本機服務，匯入 15 個範例並開啟 ComfyUI。工作檯仍可從 <http://127.0.0.1:8765/> 使用，ComfyUI 位於 <http://127.0.0.1:8188/>。

GitHub 的 ZIP／Git clone 包含安裝工具與整合程式；**ComfyUI 本體與環境依賴是在執行安裝器時下載**，沒有將大型環境塞進原始碼包。已存在的 ComfyUI 會沿用其 Python 環境，不會更新其核心或替換 PyTorch；可用 `--comfy-root PATH --comfy-python PATH` 明確指定其他安裝。

新建的 ComfyUI 環境使用 CPU 版 PyTorch，負責畫布、排程與預覽；`VisionWB_*` 模型推論仍由工作檯依實際硬體使用 GPU／CPU。這不代表其他 ComfyUI 生圖節點也會自動取得 CUDA；要使用那些節點的 GPU 環境，可接入已配置完成的 ComfyUI。詳細流程見 [ComfyUI 安裝與整合](docs/comfyui-integration.md)。

安裝工具會檢查作業系統、Python、NVIDIA 驅動及 GPU 架構，選用對應的官方 PyTorch wheel 或 CPU 版本。可用的既有 PyTorch 環境會保留；系統 NVIDIA 驅動不會自動更動。PyTorch wheel 所帶的 CUDA runtime 與系統 `nvcc`／CUDA Toolkit 是不同項目，實際狀態以工具中的 PyTorch CUDA 檢查為準。

「保留既有環境」指選定的專案 `.venv`；要檢查其他環境，使用 `python setup.py --venv /path/to/venv`。加入 `--install` 才會安裝，加入 `--repair-torch` 才允許替換既有 PyTorch。啟動其他環境時，可直接用該環境的 Python 執行 `launch.py`，或設定 `VISION_WORKBENCH_VENV`。

只想先開啟介面，或先安裝 API／資料集驗證依賴：

```bash
python setup.py --install --profile base
```

`base` 預設只安裝工作檯介面，若也需要 ComfyUI 可加 `--with-comfyui`。完整安裝可用 `--without-comfyui` 略過；已配置 ComfyUI 的專案可用 `python launch.py --without-comfyui` 只啟動工作檯。`python launch.py --no-browser` 會啟動服務但不開瀏覽器。

RF-DETR 2XL 的 `rfdetr-plus` 為額外選配，不包含在預設安裝中。具備原模型所需使用權限後再使用安裝工具的 `--with-rfdetr-plus` 選項。工作台只記住使用者的確認；模型權利仍依原提供者條款。

首次安裝會下載程式依賴；**模型權重另外準備**，安裝器不會擅自下載數十 GB 權重。完整安裝參數請執行 `python setup.py --help`。

## 多模型應用工作檯

「應用工作檯」把影像、偵測框、同一物件的追蹤、時間片段與事件連成一張有型別的流程圖。模型節點使用原有權重或已登錄的微調 checkpoint；偵測節點也接受原模型＋新分支的雙分支 bundle。骨架與物件關係使用下列額外權重，不需要另建微調資料集才能測試。

### 安裝選配依賴與模型

完整的 ComfyUI 整合安裝已包含工作流執行依賴；以下選項也可用來為未包含它的環境補裝。模型權重仍需另外下載。

```bash
# 先顯示計畫；第二行才安裝至專案隔離環境
python setup.py --with-workflows
python setup.py --install --with-workflows

# 顯示下載計畫；加 --download 才下載
python download_workflow_models.py
python download_workflow_models.py --download --models pose skeleton relation
python download_workflow_models.py --verify
python launch.py
```

`--with-workflows` 加入固定版本 `onnxruntime==1.23.2` 的 CPU provider；`full` 與 ComfyUI 一起安裝時會自動加入，不替換可用的 PyTorch／CUDA。原有偵測器仍需其推論依賴與權重；姿態與骨架需要 PyTorch，姿態另用 Ultralytics。Windows 的 ONNX Runtime 需要相容的 Microsoft Visual C++ 2015–2022 x64 Runtime；若缺少，關係節點會回報錯誤，其他本機節點仍可使用。

| 應用模型 | 下載大小 | 官方來源與用途 | 授權 |
|---|---:|---|---|
| YOLO26n Pose | 7.88 MB | [官方權重](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt)；每人 COCO 17 關節 | Ultralytics AGPL-3.0／原提供者商業方案 |
| ST-GCN++ NTU120 XSub HRNet Joint | 5.92 MB | [PYSKL](https://github.com/kennymckormick/pyskl) · [官方權重](https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth)；固定 NTU120 動作分類 | 程式 Apache-2.0；權重與 NTU 資料條件依上游 |
| RelateAnything ConvNeXt-Large | 978.15 MB | [原作者專案](https://github.com/Maelic/RelateAnything) · [固定模型版本](https://huggingface.co/maelic/relsgg-convnext-large/tree/3f320abdb0c579f685815089a08c472ae6928d4f)；ONNX、243 個預先計算的關係詞與設定 | 程式 AGPL-3.0-only 與署名條款；權重另遵循 DINOv3 License |

三者合計約 **991.94 MB**，不含 Python 依賴、RAM／VRAM。只需要骨架時可使用 `--models pose skeleton`。下載器使用[固定來源、大小與 SHA256 清單](backend/assets/workflow_model_sources.json)，核對已存在的檔案、支援部分下載續傳；發現不同內容不會直接覆蓋，須明確指定 `--replace-invalid`。`--verify` 只核對、不下載。

預設目錄為 `models/yolo26n_pose/`、`models/stgcnpp_ntu120/` 與 `models/relateanything_convnext_large/`。這三個應用模型也會在專案 `model_weights/`、專案上一層的 `model_weights/` 搜尋相同子目錄。下載器的 `--output-dir` 可更改存放根目錄，但任意外部目錄不會自動加入這三個節點的搜尋範圍；一般偵測／影片模型仍使用原有路徑登錄功能。

### 使用 ComfyUI 畫布

可在既有 ComfyUI 使用 31 種 `VisionWB_*` 原生自訂節點，串接已登錄的模型、微調權重、OpenCV、追蹤與規則。ComfyUI 與工作檯服務需同時執行；權重及素材依登錄 ID 引用，不需要複製到 ComfyUI。

一般安裝已自動準備節點與啟動設定：

```console
python setup.py --install
python launch.py
```

要指定既有 ComfyUI，將 `<ComfyUI>` 換成含有 `main.py` 的目錄，`<Python>` 換成該安裝使用的 Python 執行檔：

```console
python setup.py --install --with-comfyui --comfy-root "<ComfyUI>" --comfy-python "<Python>"
python launch.py --with-comfyui
```

若希望自行啟動兩個服務，也保留只安裝橋接／匯出範例的工具：

```console
python install_comfyui.py --comfy-root "<ComfyUI>"
python export_workflow_templates.py --destination "<ComfyUI>/user/default/workflows/VisionWorkbench"
```

15 個範例會出現在 ComfyUI 工作流資料夾，也可從 `Vision Workbench` 側欄載入。已安裝橋接時，可直接開啟[雙模型＋人體骨架攝影機範例](http://127.0.0.1:8188/?vision_workflow=live-multi-model&source=camera)。

來源支援圖片／影片，以及攝影機／VDO.Ninja 即時流程。執行後即時檢視器會嵌入輸出節點；選擇攝影機或觀看連結後按下連接、開始，也能將同一來源接到多個模型。載入模板不會自動啟用攝影機。安裝、範例素材、模型選單與接線方式見 [ComfyUI 整合說明](docs/comfyui-integration.md)。

原生應用工作檯也保留：可直接修改節點上的常用參數、雙擊搜尋新增、整理排版、復原／重做、複製多個節點與連線，以及收合／調整側欄。載入示範時會同時選取對應素材；缺少範例時會提示使用自己的素材或攝影機。

### 十五個可修改模板

| 模板 | 主要節點 | 可測試的結果與必要條件 |
|---|---|---|
| 同畫面：雙模型＋人體骨架 | 同一來源 → YOLO26／YOLOv12／Pose → 輸出 | 在同一畫面疊加兩種模型與骨架結果；可換微調權重，支援攝影機與 VDO.Ninja |
| 區域闖入與停留 | 偵測 → 追蹤 → 區域 | 進入、離開、停留、人數超限；需先設定多邊形區域 |
| 骨架 → 時序動作 → 狀態事件 | Pose → 追蹤 → ST-GCN++ → 條件 | 同一人的 NTU120 動作；至少 30 個有效關節影格 |
| 姿態轉換／跌倒候選 | Pose → 追蹤 → 幾何姿態 → 條件 | 站立快速轉成躺姿並持續的候選事件；不是訓練式跌倒分類器 |
| 檢查區 → 作業區 → 出口 | Pose → 追蹤 → 區域 → 順序 | 同一身份的 `zone_enter:check`、`zone_enter:work`、`zone_enter:exit`；跳步與逾時事件 |
| 影片動作 → 規則事件 | 影片分類 → 條件 | 使用 X3D、SlowFast、ViViT 或 VideoMAE V2；預設場景分類，個人模式須接人體追蹤 |
| 骨架可見性＋PPE 檢查 | Pose／裝備偵測 → 追蹤 → PPE | 頭部或軀幹可見且連續未配對裝備才發候選；可換自行微調的 PPE checkpoint |
| 物件 → 追蹤 → 關係辨識 | 偵測 → 追蹤 → RelateAnything | 有向的主體／關係／客體；可接條件節點建立關係持續事件 |
| 事件＋雲端視覺判讀 | 偵測 → 區域事件／影像 → 雲端視覺 | 傳送取樣影像與上下文至已設定的服務；需有效金鑰及可用視覺模型 |
| OpenCV 裁切與前處理 → 模型 | 裁切 → 縮放 → 模糊 → 偵測 | 模型接收處理後的影像，結果框可映回原圖；需要偵測模型 |
| OpenCV 顏色區域 → 輪廓追蹤 | HSV → 形態學 → 輪廓 → 追蹤 → 區域 | 不需模型權重；自訂色彩區域與輪廓標籤，適合顏色物件計數示範 |
| OpenCV 背景減除 → 移動事件 | MOG2 → 前景比例 → 持續條件 | 不需模型權重；固定鏡頭影片／串流，先暖機，再以遮罩顯示前景 |
| OpenCV 光流 → 移動強度 | Farneback → 像素速度 → 持續條件 | 不需模型權重；至少兩幀，畫面顏色呈現光流方向與相對強度 |
| OpenCV 監控區域 → 追蹤移動與停留 | 多邊形 ROI → 偵測 → 追蹤 → 移動 → 區域 | 區域監控概念示範，輸出同一物件的像素移動速度與停留事件 |
| OpenCV 外觀比較 → 變化事件 | HSV 直方圖 → 相似度 → 持續條件 | 不需模型權重；比較前一幀或第一幀，可另接追蹤物件做逐物件比較 |

在原生畫布左側「模板」選擇「載入流程與範例」或「載入流程」，修改模型與參數，從輸出端拖曳到相容的輸入端，再選素材並驗證執行。區域可在影像預覽上編輯；流程可儲存在本機、重新開啟或匯出／匯入 JSON。即時來源沿用攝影機、虛擬攝影機與 VDO.Ninja 輸入；辨識更新速度由整張流程的模型成本決定，不等於來源 FPS。預檢會檢查連線、參數、模型路徑、依賴與關係詞，不會為了驗證流程載入大型模型。

### OpenCV 節點與串接方式

在「節點」搜尋 `OpenCV`，或用分類選單篩選；也可以直接搜尋 `MOG2`、`gaussian` 等方法名稱。點選或拖曳加入畫布，選中節點後即可修改參數。這些功能使用基礎環境已安裝的 OpenCV，不需要額外下載模型權重。

| 功能 | 可調內容與輸出 |
|---|---|
| 裁切、縮放、旋轉 | 相對 ROI、目標寬高／等比例、旋轉角度與展開畫布；輸出處理後影像 |
| 灰階與對比 | 灰階、直方圖均衡化、CLAHE；輸出 RGB 格式影像，能繼續接模型 |
| 模糊 | Gaussian、median、bilateral 與核大小等參數 |
| 二值化、邊緣、形態學 | 固定閾值／Otsu／adaptive、Canny、開閉運算／侵蝕／膨脹等；輸出可串接的遮罩影像 |
| HSV 範圍與輪廓 | 色彩上下界、面積篩選、最多物件；輪廓輸出外框與多邊形，可接追蹤和區域規則 |
| 背景減除 | MOG2／KNN、學習率、暖機、前景比例；輸出遮罩與 `motion`／`still`／`unknown` 狀態 |
| 光流 | Farneback、處理尺寸、時間間隔與速度門檻；輸出視覺化及移動強度狀態 |
| 多邊形 ROI | 保留區內／保留區外／著色；與原有可繪製的區域規則分開，可真正限制模型輸入影像 |
| HSV 直方圖比較 | 比較前一幀／第一幀，整張圖或已追蹤物件；輸出外觀相似度與變化狀態 |
| 追蹤移動 | 追蹤框中心／底部的像素距離與速度、歷史時間窗、失聯重設；可接持續或區域規則 |

**「影像」連線決定下一節點收到的畫面。** 把前處理節點接到偵測器，模型會實際使用處理後的圖片；把處理後的影像接到輸出，可檢查灰階、邊緣或前景遮罩。輸出也能接原始來源，同時繪製經過裁切／縮放／旋轉分支產生的結果；幾何座標會映射至輸出影像的座標系。區域規則與區域編輯器使用原始來源的相對座標。

背景減除、光流、移動與外觀比較需要連續影格；第一幀、暖機或中斷後會顯示 `unknown`，不會把缺少資料當成靜止。單張圖片可以測前處理與輪廓，時序示範可先選預設的 `freezy_traffic.webm`，再換成攝影機／VDO.Ninja。光流也包含鏡頭移動，像素速度不是實際公尺速度；HSV 與輪廓標籤由使用者指定，並不等於已辨識車、人、食物或違規動作。

原參考專案的影像讀取、區域幾何、外觀比較、繪製與影片輸出也納入現有來源／規則／輸出及新增 OpenCV 節點；詳細對照見 [OpenCV 與參考專案功能對照](docs/workflow-opencv-reference-map.md)。示範流程為獨立實作，沒有複製未授權或專有專案的程式碼。

### 時間、身份與結果的範圍

- **來源時間與追蹤**：停留、持續與順序以影片／串流的來源秒數計算。IoU 配對只匹配同類別，目前是 Hungarian 關聯；缺少 SciPy 時明示使用 deterministic greedy。遮擋超時會產生新 ID，交錯人群仍可能換 ID；這不是跨鏡頭 ReID。沒有本幀觀察就不輸出預測框，也不憑空判定離開區域。
- **未知不是缺少**：關節低信心、遮擋、時間間隔過大會重設相關持續證據。PPE 需要可見性與裝備位置配對；通用 COCO 模型沒有安全帽／反光背心類別，應使用提示偵測或相容微調模型。幾何站立、蹲下、舉手、躺姿及跌倒候選不等於學習式動作分類。
- **模型動作有固定類別**：ST-GCN++ 使用 NTU120 標籤；訓練來源是 HRNet 關節，本工具輸入 YOLO Pose，兩者的誤差分布不同。目前逐人填入第一個骨架位置，第二位置留空，不能視為完整雙人互動分類。RGB 影片分類也只辨識原有或微調的類別；個人裁切不等於精確動作定位。任意操作流程須先定義可觀察事件，再接順序規則，不能只輸入一段流程文字就宣稱已訓練完成。
- **關係不是因果判斷**：RelateAnything 在 CPU 執行，每次最多 32 個物件框，採官方 448×448 輸入及校準。只接受隨附的 243 個關係詞；未知詞會在執行前拒絕，不會偷偷下載文字編碼器。關係持續時間需要主體與客體都具有穩定追蹤身份，不能把陣列索引當同一人。
- **長時間執行**：規則歷史有時間與容量上限；已完成且不自動重設的順序，在該身份停止觀察超過 `max_gap_seconds` 後會過期。串流匯出最多保存 20,000 筆影格、100,000 筆事件與合計 256 MiB；達上限仍繼續即時處理，保留最新影像與最近 1,000 筆事件，結果的 `data_export` 會標記截斷。

這些模板用於建立與測試應用流程。請用現場獨立標註影片評估漏報、誤報、遮擋及 ID 切換，不能把示範事件當作已驗證的安全判定。

### 選配雲端視覺連線

在伺服器電腦建立包含 API key 的本機文字檔，再到工作檯「連接」輸入該檔案的**完整路徑**。支援直接貼金鑰、`API_KEY=...`、`LMSTUDIO_GATEWAY_API_KEY=...` 等環境變數格式，也接受引號與 `export`；同一檔案只能有一個金鑰。後端讀取金鑰，設定檔只保存路徑與選用模型；流程 JSON、公開工作資訊與 API 回應不包含金鑰或該路徑。不要將金鑰檔放進原始碼或公開資料夾。

目前接頭固定使用 `https://api.munolithsai.com/v1`，不接受自訂轉送 URL。設定時驗證模型列表，HTTP 401、連線失敗或沒有選定可用模型都不算連接成功；包含雲端節點的流程會被阻止執行，本機流程仍可使用。列出模型不代表每個模型都支援影像，請選服務提供的視覺模型；實際請求失敗會回報錯誤。只有執行含雲端節點的流程時，才會依取樣設定傳送影像。

### 方法來源與署名

區域與安全應用以 [ZoneGuard](https://github.com/K-saif/ZoneGuard)、[AI Safety Monitoring](https://github.com/nusRying/ai-safety-monitoring) 為**功能概念參照**。前者未提供明確 LICENSE，後者採 Proprietary / All Rights Reserved；本專案沒有複製兩者程式碼。流程模板與規則在本專案獨立實作。

骨架模型保留 [PYSKL Apache-2.0 授權與來源紀錄](backend/vendor/pyskl_stgcnpp/NOTICE.md)。關係模組保留 [RelateAnything 授權與來源紀錄](backend/vendor/relateanything/NOTICE)，並在介面顯示 **[Based on RelateAnything by Maëlic Neau](https://github.com/Maelic/RelateAnything)**；其權重另受 [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) 約束。原始碼的 AGPL 授權不會改寫模型、資料及第三方元件的條件。

## 預設模型下載

以下為本工作台使用的**確切版本**。點選單檔連結即可下載；Grounding DINO、ViViT 和 DINOv3 需要完整模型目錄，請使用下方的 `hf` 指令。下載位置可自行更改，完成後到「模型與任務」掃描或登錄該檔案／資料夾，無須搬移到專案目錄。

| 預設模型 | 下載大小約值 | 官方來源／下載連結 | 取得方式與備註 |
|---|---:|---|---|
| **YOLOv12x** | 119 MB | [官方模型說明](https://docs.ultralytics.com/models/yolo12/) · [下載 yolo12x.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt) | 單檔；Ultralytics v8.4.0 發布權重，對應本工作台原版 YOLOv12 接頭 |
| **YOLO26x** | 119 MB | [官方模型說明](https://docs.ultralytics.com/models/yolo26/) · [下載 yolo26x.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt) | 單檔；物件偵測版本 |
| **YOLOv12x-Seg** | 130 MB | [原作者 Seg 分支](https://github.com/sunsmarterjie/yolov12/tree/Seg) · [下載 yolov12x-seg.pt](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt) | 單檔；實例分割版本，與偵測版分開登錄 |
| **YOLO26x-Seg** | 142 MB | [官方模型說明](https://docs.ultralytics.com/models/yolo26/) · [下載 yolo26x-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt) | 單檔；實例分割版本 |
| **Grounding DINO Base** | 934 MB | [IDEA-Research 官方模型](https://huggingface.co/IDEA-Research/grounding-dino-base) · [完整檔案列表](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb) | 使用下方 `hf` 指令，包含 safetensors、config、processor、tokenizer |
| **RF-DETR 2XL** | 508 MB | [官方專案](https://github.com/roboflow/rf-detr_plus) · [下載 rf-detr-xxlarge.pth](https://repo.roboflow.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth) | 單檔；使用須符合 [PML 1.0](https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE)，並安裝選配 `rfdetr-plus` |
| **RF-DETR-Seg-2XL** | 155 MB | [官方專案](https://github.com/roboflow/rf-detr) · [下載分割權重](https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth) | 另存 `rf-detr-seg-xxlarge.pt`；Apache-2.0，使用 `rfdetr`，不需 `rfdetr-plus` |
| **ViViT Base K400** | 356 MB | [Google 官方模型](https://huggingface.co/google/vivit-b-16x2-kinetics400) · [完整檔案列表](https://huggingface.co/google/vivit-b-16x2-kinetics400/tree/8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42) | 使用下方 `hf` 指令；此處為 **Base**，含影片分類權重與設定 |
| **X3D-L** | 50 MB | [官方 Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md) · [下載 X3D_L.pyth](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth) | 單檔；Kinetics-400 |
| **SlowFast R101 8×8** | 504 MB | [官方 Model Zoo](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md) · [下載 SLOWFAST_8x8_R101.pyth](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R101.pyth) | 單檔；Kinetics-400 |
| **VideoMAE V2 Giant K710** | 2.03 GB | [官方 Model Zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md) · [下載 K710 微調權重](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/mae-g/vit_g_hybrid_pt_1200e_k710_ft.pth) | 單檔 `vit_g_hybrid_pt_1200e_k710_ft.pth`；此為 K710 分類 checkpoint |
| **DINOv3 ViT-7B/16 LVD-1689M（選配）** | **26.86 GB** | [Meta 官方模型／申請存取](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) | 須獲核准並登入 Hugging Face；6 個權重分片與設定。影像分類微調需完整目錄，硬體需求另計 |

大小按十進位 MB／GB 計算，指上列指定權重與必要設定的磁碟下載量，**不是 RAM／VRAM 需求**；不含 Python、PyTorch、CUDA runtime、下載暫存與訓練產物。原有八款（不含分割與 DINOv3）約 **4.62 GB**；三款分割權重另約 **427 MB**，合計約 **5.05 GB**；再加 DINOv3 約 **31.91 GB**。原有模型於 2026-10-18 核對，分割模型於 2026-11-01 核對；權重各自遵循原提供者授權。

<details>
<summary><strong>複製單檔模型下載指令（YOLO、RF-DETR、X3D、SlowFast、VideoMAE）</strong></summary>

在專案根目錄開啟終端，**只執行所需模型的那一行**。以下適用 Linux／macOS；Windows PowerShell 請將行首 `curl` 改成 `curl.exe`，避免舊版 PowerShell 的同名別名。`--create-dirs` 會建立輸出資料夾；也可把 `models/...` 改成自己的權重存放路徑。

```bash
# YOLOv12x
curl --fail --location --retry 3 --create-dirs --output models/yolov12x/yolo12x.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt"

# YOLO26x
curl --fail --location --retry 3 --create-dirs --output models/yolo26x/yolo26x.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt"

# 實例分割：三個獨立權重，不與偵測權重共用
curl --fail --location --retry 3 --create-dirs --output models/yolov12x_seg/yolov12x-seg.pt "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt"
curl --fail --location --retry 3 --create-dirs --output models/yolo26x_seg/yolo26x-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt"
curl --fail --location --retry 3 --create-dirs --output models/rf_detr_seg_2xl/rf-detr-seg-xxlarge.pt "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth"

# RF-DETR 2XL：先確認符合原模型的 PML 1.0 使用條款
curl --fail --location --retry 3 --create-dirs --output models/rf_detr_2xl/rf-detr-xxlarge.pth "https://repo.roboflow.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth"

# X3D-L
curl --fail --location --retry 3 --create-dirs --output models/x3d_l/X3D_L.pyth "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth"

# SlowFast R101 8x8
curl --fail --location --retry 3 --create-dirs --output models/slowfast_r101_8x8/SLOWFAST_8x8_R101.pyth "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R101.pyth"

# VideoMAE V2 Giant K710：下載指定的分類 checkpoint
curl --fail --location --retry 3 --create-dirs --output models/videomaev2_giant_k710/vit_g_hybrid_pt_1200e_k710_ft.pth "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/mae-g/vit_g_hybrid_pt_1200e_k710_ft.pth"
```

下載完成後，登錄各行 `--output` 指定的檔案。RF-DETR 2XL 另需安裝 `python setup.py --install --with-rfdetr-plus`；下載成功本身不會授予模型使用權利。

</details>

<details>
<summary><strong>複製完整模型目錄下載指令（Grounding DINO、ViViT）</strong></summary>

在已安裝工作台的 Python 環境中執行。若找不到 `hf`，Windows 可改用 `.venv\Scripts\hf.exe`，Linux／macOS 可改用 `.venv/bin/hf`。若只安裝了 `base` 環境，先在該環境執行 `python -m pip install "huggingface_hub>=0.34,<1.0"` 安裝下載工具。

```bash
# Grounding DINO：下載 safetensors 與所有必要設定，不重複下載另一份 .bin 權重
hf download IDEA-Research/grounding-dino-base --revision 12bdfa3120f3e7ec7b434d90674b3396eccf88eb --include "*.json" "*.txt" "model.safetensors" "README.md" --local-dir models/grounding_dino_base

# ViViT Base K400：權重、模型設定與影像處理器三者缺一不可
hf download google/vivit-b-16x2-kinetics400 config.json preprocessor_config.json pytorch_model.bin --revision 8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42 --local-dir models/vivit_base_k400
```

這兩款目前可匿名下載。完成後登錄整個 `models/grounding_dino_base` 或 `models/vivit_base_k400` 資料夾。指令固定為本工作台驗證過的模型 revision。

</details>

<details>
<summary><strong>DINOv3 7B 選配下載（26.86 GB）</strong></summary>

先到 [Meta 模型頁](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) 申請這個 **ViT-7B** repository 的存取權；核准後，在終端使用同一帳號的讀取 token 登入。瀏覽器登入與終端登入分開；能否下載以這個 ViT-7B repository 的實際存取狀態為準。

```bash
hf auth login
hf download facebook/dinov3-vit7b16-pretrain-lvd1689m --revision b80367753773648a6793235ab9c65cdbb029506f --include "*.safetensors" "*.json" "LICENSE.md" "README.md" --local-dir models/dinov3_7b
```

登入指令會互動式詢問 token，無須把 token 寫進指令、README 或 Git。須保留全部 6 個分片、權重索引與設定；登錄後會驗證完整性與環境。分類微調輸出會保存整個 backbone、processor、分類頭與類別設定，不會額外自動壓縮一份數十 GB 的 ZIP。

</details>

## 登錄與搜尋模型權重

1. 開啟「模型與任務」的權重管理。
2. 掃描預設位置，或加入你存放權重的資料夾，例如 `D:\AI\models`、`/data/models`。
3. 選擇正確的模型及候選路徑，確認登錄；也可直接貼上完整檔案／資料夾路徑。
4. 重新檢查模型狀態；檔案齊全且依賴可用後，即可推論或訓練。

登錄只建立本機路徑設定，不搬移、不複製檔案。多個候選不會被任意覆蓋。掃描有數量／深度限制，並不遍歷所有磁碟。

| 模型 | 常見權重入口 |
|---|---|
| YOLOv12x | `yolo12x.pt` |
| YOLO26x | `yolo26x.pt` |
| YOLOv12x-Seg | `yolov12x-seg.pt` |
| YOLO26x-Seg | `yolo26x-seg.pt` |
| RF-DETR-Seg-2XL | `rf-detr-seg-xxlarge.pt` |
| RF-DETR 2XL | `rf-detr-xxlarge.pth` |
| Grounding DINO Base | 完整 Hugging Face 目錄，含權重、config、processor、tokenizer |
| ViViT Base | 完整 Hugging Face 目錄，含權重、config、preprocessor |
| X3D-L | `X3D_L.pyth` |
| SlowFast R101 | `SLOWFAST_8x8_R101.pyth` |
| VideoMAE V2 Giant | `vit_g_hybrid_pt_1200e_k710_ft.pth` |

檔名與設定檢查用來尋找候選，不能代替實際載入驗證。請僅使用信任來源的 checkpoint；`.pt`／`.pth` 等格式可能涉及 Python 反序列化。

## 微調完成後直接推論

同一個工作台的成功訓練會自動加入 checkpoint 選單，保留模型類型與訓練類別。回到「推論比較」，選同一模型，再切換「模型權重」即可。支援即時任務的單一模型也可在「即時攝影機」選用；雙分支 bundle 及 DINOv3 目前僅供圖片／影片推論比較。

外部或其他電腦帶來的微調結果，可在模型庫登錄完整路徑。需要相容架構與必要的設定／類別 metadata；請保留完整匯出資料夾，勿只搬走權重檔。工作台會在送出任務前再次確認路徑與模型是否相符。任務入列時會固定使用當時選定的權重路徑，後續修改預設路徑不會切換已入列的實驗。

## 內附圖片、影片與微調範例

專案包含約 **17 MB** 的[教學素材](examples/README.md)，clone 或下載 ZIP 後即可讀取：

- **推論**：10 張 COCO val2017 原圖，以及約 20 秒、1080p 的街頭交通影片，會自動出現在素材區。
- **偵測微調**：YOLO Detection 資料集，8 張訓練／2 張驗證、71 個框。在微調頁選擇相容模型，按 **載入教學資料集** 即可驗證並選用；按開始訓練後才會執行。
- **來源**：附 COCO instances／captions 子集、逐檔雜湊、作者、授權、類別映射與切分紀錄。圖片保持原始 bytes。

這是操作流程範例，不適合正式評估準確率。影片沒有行為分類監督標籤；行為微調仍需自備 Video CSV 資料集。模型權重另外準備。媒體有獨立授權，包含 NC／ND／SA 條件，見 [素材署名與授權](examples/media/ATTRIBUTION.md)。

## 資料集不只接受 ZIP

「微調訓練」提供兩種入口：

- **本機路徑**：直接指定資料集根目錄或 YOLO 的 `data.yaml`，原始資料留在原處。
- **ZIP 上傳**：解壓到工作台的本機 runtime 後驗證，適合攜帶完整的小型資料集。

訓練接頭可能建立自己需要的格式轉換／暫存資料。原始資料必須具備訓練與驗證切分；驗證會檢查缺檔、類別、標註及重複樣本等問題。

### YOLO Detection

```text
my-dataset/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

```yaml
path: .
train: images/train
val: images/val
names: [person, helmet]
```

標註為 `class x_center y_center width height`，座標正規化到 0–1；背景圖片也需要對應的空白 `.txt`。YAML 不接受下載或執行指令。

### COCO Detection

```text
my-dataset/
  train.json
  val.json
  images/train_001.jpg
  images/val_001.jpg
```

兩個 JSON 都包含 `images`、`annotations`、`categories`；`file_name` 相對於資料集根目錄。`bbox` 為像素座標的 `[x, y, width, height]`。

### Video CSV（行為辨識）

```text
my-dataset/
  classes.txt
  train.csv
  val.csv
  videos/train_001.mp4
  videos/val_001.mp4
```

`classes.txt` 每行一個類別；CSV 有 `path,label` 標頭，路徑相對於根目錄。`label` 可使用類別名稱或從 0 開始的索引。訓練與驗證使用不同影片。

```csv
path,label
videos/train_001.mp4,walking
```

## 即時手機與攝影機

「攝影機裝置」可選 USB 或瀏覽器能看到的虛擬攝影機；「VDO.Ninja 連結」貼入手機分享的 `https://vdo.ninja/?view=…` 觀看網址即可直接接收，不需要 OBS。

直接接收使用官方 VDO.Ninja SDK。相容模式保留 iframe 取圖。傳送、轉碼與模型速度分開量測；大型模型、多區塊及較長片段會降低辨識更新率。串流品質取決於網路，跨網路可用性由 WebRTC／TURN 連線決定。

影像來源於離開頁面或停止時關閉；本機服務超過 20 秒收不到影格也會釋放工作。原始暫存影格數量有上限，工作結束後清除。保留最後辨識結果與任務摘要，不錄製完整影片。

## 開發與測試

後端 FastAPI，前端 React + Vite。圖片／影片推論使用常駐工作程序，最多快取四組「模型、權重、運算裝置」，超過時釋放最久未使用的模型。相同權重換素材或調整信心門檻、提示文字等辨識參數時可重用；權重或設定檔更新後會重新載入。記憶體不足時會釋放快取並重試一次。

訓練與即時攝影機仍使用各自的工作程序，開始前先釋放推論快取；單一序列佇列避免任務同時搶占 GPU。取消執行中的推論或關閉服務也會釋放常駐程序。畫布的「釋放模型」可在佇列空閒時手動釋放；下一次推論會重新載入。結果的總處理時間包含模型載入、素材解碼、推論與輸出，不含排隊與程序啟動時間。

```bash
# 在安裝好的 Python 環境中
python -m pip install -r requirements-dev.txt
python -m pytest tests -q

cd frontend
npm ci
node --test tests/*.test.js
npm run build
```

CI 測試不下載模型權重，不進行完整 GPU 訓練。新模型需同時提供推論接頭、資料格式／訓練能力描述及可回載的 checkpoint 格式。

## 範圍與本機資料

這是單人研究／教學工具，尚未提供多使用者權限、分散式訓練、任意神經網路層編輯、資料標註、跨鏡頭身份辨識或正式 benchmark 自動評測。應用工作檯提供同來源的 IoU 物件追蹤與模型／規則連線。不要直接把本機 API 暴露到 Internet。

除明確附帶來源與授權的 `examples/` 教學素材外，`runtime/`、模型權重、個人資料集與影片、訓練結果、環境及個人授權紀錄均不包含在本次發布中。要回報問題，請使用 [Issue 範本](.github/ISSUE_TEMPLATE/bug_report.md)，只附上已檢查過的環境資訊，不附 token、私有影像或完整個人路徑。

## 授權與來源

本專案原創程式採 **AGPL-3.0**，見 [LICENSE](LICENSE)。Ultralytics 的開源授權也是 [AGPL-3.0](https://www.ultralytics.com/license)。模型、權重、資料集與第三方套件仍遵循各自授權，不能由本專案的授權取代。

RF-DETR 2XL 的選配套件與權重遵循 [Platform Model License 1.0](https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE)。專案不附送其套件、權重或使用權利，也不預先替新使用者接受條款。

第三方程式與標籤來源見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
