# 工作流範例素材

在應用工作檯載入模板時，可一起選用與它對應的範例。每張模板卡片標示素材來源、可用狀態與預期能觀察的結果；載入只選取素材，不會自動開始推論或開攝影機。所有模板都可改用自己上傳的圖片／影片或即時攝影機。

後端以檔案大小與 SHA-256 比對已匯入素材，而不是只看檔名；重新命名仍可匹配，刪除或替換檔案後會顯示不可用。流程 JSON 不包含本機路徑，也不綁定其他電腦的 media ID。

## 原參考專案是否提供素材

核對日期：2026-11-01。

| 專案 | 可取得的原始輸入 | 對應工作流 |
|---|---|---|
| [ZoneGuard](https://github.com/K-saif/ZoneGuard/blob/0cb55cf27a54c89ae0148b43990b2373be459f22/data/0_te21_cropped.mp4) | `data/0_te21_cropped.mp4`，約 10.4 秒、8.13 MB，廠區走道的人員活動 | 區域闖入、區域順序、ROI／移動速度、雲端檢視 |
| [PYSKL](https://github.com/kennymckormick/pyskl/blob/f2bf3a6b08e2e8dec744692d64efdb187fd6719a/demo/ntu_sample.avi) | `demo/ntu_sample.avi`，72 幀、1.12 MB，NTU 雙人互動 | 骨架與時序動作 |
| [MMAction2](https://github.com/open-mmlab/mmaction2/blob/a5a167dff2399e2d182a60332325f9c0d4663517/demo/demo.mp4) | `demo/demo.mp4`，10 秒、636 KB，扳手腕示範 | RGB 影片動作；預設規則包含 `arm wrestling` |
| [AI Safety Monitoring](https://github.com/nusRying/ai-safety-monitoring#quick-start) | 原始樣本影片與輸出未提交；只有作品集展示圖片，並提供依其環境產生 demo 的程式 | PPE／跌倒模板需要自己的場景影片或攝影機，不把展示截圖當成原始測試影片 |

原專案影片只在使用者另外取得並匯入本機後匹配。ZoneGuard 沒有附影片授權，PYSKL 的 NTU 內容與程式授權不同，MMAction2 的程式 Apache-2.0 授權也不自動證明影片的再散布權利；因此這些原始影片不隨本工具 GitHub 或原始碼 ZIP 重新散布。AI Safety Monitoring 為專有授權，未複製其程式或展示圖片。

## 隨工具提供的替代範例

- **Freezy Traffic / LIFEOFVIDS / CC BY 3.0**：對應背景減除、光流、外觀比較。若尚未匯入 ZoneGuard 影片，區域相關模板可使用它，卡片會明示這是內附替代素材。
- **COCO 花椰菜圖**：對應裁切、前處理、HSV 綠色輪廓；jules:stonesoup / CC BY 2.0。
- **COCO 兩頭牛圖**：對應多物件關係；Melinda VanLone / CC BY-NC 2.0。

完整作者、來源與授權見 [內附素材署名](../examples/media/ATTRIBUTION.md)。這些圖片和影片沒有工作場所違規真值；場景不同時需修改 ROI、動作標籤、門檻與持續時間。沒有符合條件的事件可以是正確結果。

PYSKL 的範例包含雙人互動，而工作檯目前逐人提供骨架序列；這段可確認骨架與模型串接，不代表已支援完整雙人互動分類。一般走路、擁抱或街景影片也不能充當已標註的跌倒與 PPE 驗證集。

## 固定版本與內容驗證

原始素材的版本、位元組大小與 SHA-256 固定在 `backend/workflow_samples.py`。匯入時可以保留原檔名，也可以重新命名；程式只會將內容完全相同的檔案標示為該來源。此比對不會對外傳送素材，也不會在開啟模板清單時下載檔案。

| 原始素材 | SHA-256 |
|---|---|
| ZoneGuard `0_te21_cropped.mp4` | `0c64fdbeb2a49c252600aeac56290b6d5e3927021cba4abf789c8095e84c3e54` |
| PYSKL `ntu_sample.avi` | `5e6f0de7d4328aae50b3a9cdaf2a8353f1286721060cb00c49e4afafe4e896dd` |
| MMAction2 `demo.mp4` | `071e60535eaf0aed475ddac06269ee0cdfc4740158f22d9ccd2c3b93b42aa344` |
