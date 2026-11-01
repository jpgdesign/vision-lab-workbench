# OpenCV 功能與參考專案對照

核對日期：2026-11-01。以下按來源專案實際使用的 OpenCV 功能對照本工作檯的節點；工作檯程式為獨立實作，沒有移植這兩個專案的程式碼。

| 來源功能 | 工作檯節點／操作 | 說明 |
|---|---|---|
| 影片讀取、尺寸與 FPS、逐幀處理 | **影像／影片來源** | 圖片／影片共用來源節點；即時來源沿用瀏覽器攝影機及 VDO.Ninja。影片規則使用來源時間。 |
| 畫多邊形、區域內外判定 | **區域／停留規則** | 可編輯區域、選框中心或底部中心、串接進出／停留／人數事件。 |
| `fillPoly`、半透明 `addWeighted` | **OpenCV · 多邊形 ROI／疊色** | 可保留區域內、保留區域外或疊色；與只做事件判定的區域節點分開。 |
| 框、關節、文字、區域疊圖；寫出影片／快照 | **畫面與事件輸出** | 畫面預覽與下載；事件可匯出 JSON／CSV。原生 `imshow` 視窗與滑鼠回呼由網頁畫布及區域編輯器承接。 |
| RGB／HSV 轉換、直方圖正規化與相關性比較 | **OpenCV · 灰階／對比**、**OpenCV · HSV 顏色遮罩**、**OpenCV · 外觀直方圖比較** | 直方圖可比較整幅影像，或接追蹤後逐 ID 比較裁切區域；不會把色彩相似直接當作身分辨識。 |
| resize、GaussianBlur、旋轉／仿射旋轉 | **OpenCV · 縮放**、**OpenCV · 模糊／降噪**、**OpenCV · 旋轉** | 處理後影像可接模型；幾何節點保留回原圖的座標轉換。 |
| findContours、面積、外框、輪廓簡化 | **OpenCV · 輪廓／幾何物件** | 可調面積與簡化，輸出外框後接追蹤、區域或畫面輸出；不等於 SAM 分割模型。 |
| 追蹤位移與速度統計 | **OpenCV · 軌跡位移／速度** | 逐 ID 使用來源時間量測 px/s、累積像素路徑及畫面對角線比例。失聯重置；沒有相機校正時不輸出公尺或 km/h。 |

另提供二值化、Canny、形態學、MOG2／KNN 背景減除及 Farneback 光流，用於自行組合傳統視覺流程。這些是擴充功能，並非宣稱兩個參考專案皆有使用。

## 來源與範圍

- [ZoneGuard 主流程](https://github.com/K-saif/ZoneGuard/blob/main/main.py)、[多邊形處理](https://github.com/K-saif/ZoneGuard/blob/main/quadrilateral_tracker.py)、[影片工具](https://github.com/K-saif/ZoneGuard/blob/main/utils.py)、[區域與移動統計](https://github.com/K-saif/ZoneGuard/blob/main/zone_alert_manager.py)：對應影片 I/O、互動畫區域、遮罩疊色、區域判定、標註和追蹤統計。其原始以框高估距的方法未照搬；本站像素量測不宣稱物理距離。
- [AI Safety Monitoring 共用工具](https://github.com/nusRying/ai-safety-monitoring/blob/main/src/common.py)、[整合流程](https://github.com/nusRying/ai-safety-monitoring/blob/main/src/pipeline.py)、[區域](https://github.com/nusRying/ai-safety-monitoring/blob/main/src/stage5_zones.py)、[姿態旋轉](https://github.com/nusRying/ai-safety-monitoring/blob/main/src/stage6_fall.py)、[輪廓處理](https://github.com/nusRying/ai-safety-monitoring/blob/main/src/stage7_segment.py)、[示範影像處理](https://github.com/nusRying/ai-safety-monitoring/blob/main/src/make_demo_video.py)：核對 cv2 功能的用途；PPE、姿態模型、追蹤器、SAM、VLM 是其他模型或應用邏輯，不冒充 OpenCV 演算法。

ZoneGuard 核對時未提供授權檔；AI Safety Monitoring 採 [Proprietary / All Rights Reserved](https://github.com/nusRying/ai-safety-monitoring/blob/main/LICENSE)。因此本專案只參考公開功能描述並自行實作節點，不散布其來源、權重或示範素材。OpenCV 本身的授權與模型權重授權分別遵循各自上游。

## 可重現的串接方式

- 區域應用：來源 → 偵測／骨架 → 追蹤 → 軌跡位移／速度 → 區域／停留規則 → 輸出；來源另接 ROI 疊色 → 輸出畫面。
- 外觀比較：來源 → 外觀直方圖比較 → 條件規則（`appearance_change`）→ 輸出；需要逐物件比較時，再把追蹤結果接入直方圖節點。
- 傳統視覺：來源 → ROI → 灰階／模糊 → 二值化 → 輪廓 → 追蹤 → 區域 → 輸出。

像素位移、光流或直方圖變化只描述可觀察的影像變化。是否屬於跌倒、違規或特定動作，需接適合的動作模型與具體規則，並使用目標場景驗證。
