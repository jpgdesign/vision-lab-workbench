"""Match example graphs to verified local media, without downloading on GET.

Public-repository code licenses do not automatically license demonstration
footage. Original project clips are resolved only when already imported locally;
the public bundle keeps its individually attributed COCO and CC BY examples.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
from pathlib import Path
from urllib.parse import quote


SAMPLES = {
    "handwashing_user_clip": {
        "title": "指定洗手影片 · 原片 22–44 秒", "kind": "video", "bytes": 5216293,
        "sha256": "ec4fbd5de689af48d6e19f6f0dc176ee6323c178268248b228da0f6a3b729fce",
        "source_url": "https://www.youtube.com/watch?v=8wiBWkgZ3iI&t=22s",
        "license": "RTI／行政院防疫宣導影片；未確認再散布授權，僅解析使用者本機副本，不隨工具散布。",
        "origin": "user_provided", "notes": "22.022 秒手部近拍；搓手到擦乾，不能當作完整進入、沖洗、離開流程真值。"},
    "zoneguard": {
        "title": "ZoneGuard 原專案 · 廠區走道", "kind": "video", "bytes": 8129752,
        "sha256": "0c64fdbeb2a49c252600aeac56290b6d5e3927021cba4abf789c8095e84c3e54",
        "source_url": "https://github.com/K-saif/ZoneGuard/blob/0cb55cf27a54c89ae0148b43990b2373be459f22/data/0_te21_cropped.mp4",
        "license": "來源未附影片授權；僅解析使用者已匯入的本機副本，不隨本工具散布。", "origin": "original_project",
        "notes": "原專案的未標註輸入影片，約 10.4 秒；請依走道重畫區域。"},
    "ntu": {
        "title": "PYSKL 原專案 · NTU 雙人互動", "kind": "video", "bytes": 1119546,
        "sha256": "5e6f0de7d4328aae50b3a9cdaf2a8353f1286721060cb00c49e4afafe4e896dd",
        "source_url": "https://github.com/kennymckormick/pyskl/blob/f2bf3a6b08e2e8dec744692d64efdb187fd6719a/demo/ntu_sample.avi",
        "license": "NTU 資料條件獨立於 PYSKL 程式授權；僅解析本機副本，不隨本工具散布。", "origin": "original_project",
        "notes": "72 幀雙人互動，可觀察骨架與時序分類；目前模型逐人輸入，不能用這段證明完整雙人互動準確率。"},
    "mmaction": {
        "title": "MMAction2 原專案 · 扳手腕", "kind": "video", "bytes": 635539,
        "sha256": "071e60535eaf0aed475ddac06269ee0cdfc4740158f22d9ccd2c3b93b42aa344",
        "source_url": "https://github.com/open-mmlab/mmaction2/blob/a5a167dff2399e2d182a60332325f9c0d4663517/demo/demo.mp4",
        "license": "示範影片權利未單獨標明；僅解析本機副本，不隨本工具散布。", "origin": "original_project",
        "notes": "約 10 秒的原專案 RGB 動作示範；預設條件包含 arm wrestling，這是動作事件而非違規定義。"},
    "traffic": {
        "title": "內附範例 · Freezy Traffic", "kind": "video", "bytes": 13601697,
        "sha256": "b5b7da33afd548699a4f0bc8bd6d0d7f7307babb5c68b3f6ae5c1864b480d196",
        "source_url": "https://commons.wikimedia.org/wiki/File:Freezy_Traffic.webm",
        "license": "LIFEOFVIDS · CC BY 3.0", "origin": "bundled",
        "notes": "內附約 20 秒街頭交通影片；不是參考專案的原始素材，沒有違規或行為真值標籤。"},
    "cows": {
        "title": "內附 COCO · 兩頭牛", "kind": "image", "bytes": 197930,
        "sha256": "422552f71b460a7db60633b4037d9a3b97bf7a2c4eb4c91e4fb6b466b0619da0",
        "source_url": "https://www.flickr.com/photos/pixel97/7766356622/",
        "license": "Melinda VanLone · CC BY-NC 2.0", "origin": "bundled",
        "notes": "內附 COCO val2017 原圖；用於比較相鄰物件關係，不是 RelateAnything 原專案示範圖。"},
    "broccoli": {
        "title": "內附 COCO · 花椰菜", "kind": "image", "bytes": 247133,
        "sha256": "c3e455cd18621ea344f3f54f089ba29e7b31c8fdc16c768156eb17c23961f0e3",
        "source_url": "https://www.flickr.com/photos/stone-soup/7871176758/",
        "license": "jules:stonesoup · CC BY 2.0", "origin": "bundled",
        "notes": "內附 COCO val2017 原圖；可先觀察裁切與綠色區域，再換成自己的圖片或鏡頭。"},
}


# Each recommendation states what can actually be inspected. No unrelated
# available image is silently substituted for a temporal or safety sample.
TEMPLATE_SAMPLES = {
    "live-office-crossing": ([], "使用你的辦公室攝影機；左到右出去、右到左進來。坐姿需看得到腿部，這個桌面視角可能遮擋；中央線可拖拉。"),
    "handwashing-hybrid": (["handwashing_user_clip"], "比較實際手部關節、SAM 遮罩、RGB washing hands 分數與人體骨架可見性；缺證據時保留未知，沒有完整流程真值。"),
    "handwashing-detection-mask": (["handwashing_user_clip"], "比較文字提示物件框與實際遮罩；手重疊可能只找到一個區域，單張影像不代表動作完成。"),
    "live-multi-model": (["ntu", "zoneguard"], "同一畫面疊上 YOLO26、YOLOv12 與人體關節；可切換 USB／虛擬攝影機或 VDO.Ninja，並替換個別偵測節點的微調權重。"),
    "zone-guard": (["zoneguard", "traffic"], "觀察人員追蹤 ID、進出與停留；先讓區域涵蓋人員會經過的位置。"),
    "skeleton-actions": (["ntu"], "觀察同人骨架、累積至少 32 幀後的 NTU120 動作分數；短片沒有設定類別時可以沒有事件。"),
    "posture-fall": ([], "使用可看清全身、包含站立到躺下轉換的影片，或接鏡頭安全地測試姿態；一般走路或擁抱影片不當作跌倒範例。"),
    "zone-sequence": (["zoneguard", "traffic"], "觀察同一追蹤 ID 的區域進入訊號，再按現場路徑調整 check → work → exit；不是已標註的作業違規資料。"),
    "rgb-action-rules": (["mmaction"], "觀察片段分類是否出現 arm wrestling；只有分數與持續時間符合條件才產生動作事件。"),
    "ppe-safety": ([], "使用清楚拍到頭部、軀幹與安全帽／背心的場景影片，或接即時鏡頭；缺少裝備且可見性足夠時才會發出候選事件。"),
    "object-relations": (["cows"], "先確認偵測到兩個物件，再查看 next to 等關係與方向；關係可能因偵測框不同而改變。"),
    "cloud-review": (["zoneguard", "traffic"], "觀察區域事件與 API 對影像的描述；需已設定服務，執行後取樣畫面會送到該服務。"),
    "opencv-preprocess-detect": (["broccoli"], "比較裁切、縮放與模糊後的模型輸入，以及映回原圖的偵測框。"),
    "opencv-color-contours": (["broccoli"], "觀察綠色遮罩和輪廓框；green_region 是幾何標籤，不是食品分類，單張圖片沒有移動時間。"),
    "opencv-background-motion": (["traffic"], "前 10 幀建立背景；車流移動時觀察前景遮罩、比例與 motion 事件。"),
    "opencv-optical-flow": (["traffic"], "從第二幀開始觀察光流方向、亮度與平均像素速度；鏡頭移動也會影響數值。"),
    "opencv-roi-monitor": (["zoneguard", "traffic"], "觀察多邊形遮罩、人物追蹤、像素速度與停留；ROI 與區域需對應走道。"),
    "opencv-appearance-change": (["traffic"], "觀察相鄰影格的 HSV 相似度；色彩分布變化未達門檻時保持 appearance_stable，不保證每段影片都有事件。"),
}


@lru_cache(maxsize=256)
def _digest(path, size, mtime_ns):
    del size, mtime_ns  # Part of the cache key, so changed files are rechecked.
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _available_samples(entries):
    found = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("path"), str):
            continue
        try:
            path = Path(entry["path"])
            if not path.is_file():
                continue
            stat = path.stat()
            candidates = [(key, spec) for key, spec in SAMPLES.items()
                          if key not in found and spec["kind"] == entry.get("kind") and stat.st_size == spec["bytes"]]
            if not candidates:
                continue
            digest = _digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            for key, spec in candidates:
                if digest == spec["sha256"]:
                    found[key] = entry
        except OSError:
            continue
    return found


def with_samples(document, entries):
    """Attach portable, public metadata; local filesystem paths never leave API."""
    result = copy.deepcopy(document)
    available = _available_samples(entries)
    for template in result["templates"]:
        choices, expected = TEMPLATE_SAMPLES.get(template["id"], ([], "請選擇符合流程的素材或即時來源。"))
        chosen = next((key for key in choices if key in available), choices[0] if choices else None)
        spec = SAMPLES.get(chosen, {})
        entry = available.get(chosen)
        sample = {key: spec.get(key, "") for key in ("title", "kind", "source_url", "license", "notes", "origin")}
        sample.update(available=entry is not None, status="available" if entry else "missing", media_id=entry["id"] if entry else None,
                      name=entry.get("name") if entry else None,
                      url="/api/media/" + quote(entry["id"], safe="") + "/content" if entry else None, expected=expected)
        if not choices:
            sample.update(title="自備場景影片／即時鏡頭", kind="video", origin="user_required",
                          source_url="https://github.com/nusRying/ai-safety-monitoring#quick-start",
                          notes="參考專案未提交原始 PPE／跌倒影片；不會用作品集截圖或無關素材冒充輸入範例。")
            if template["id"] == "live-office-crossing":
                sample.update(source_url="", notes="使用自己的辦公室鏡頭；左右方向以送入模型的畫面為準。桌面遮擋腿部時，坐姿可能無法判定。")
        elif entry is None:
            sample["notes"] += " 此素材尚未匯入本機；可從來源頁取得後上傳，或選用自己的相符素材。"
        template["sample"] = sample
        template["recommended_options"] = {"frame_stride": 1, "max_frames": 300, "max_side": 960 if template["id"] == "skeleton-actions" else 640}
        if template["id"].startswith("handwashing-"):
            template["recommended_options"] = {"frame_stride": 3, "max_frames": 240, "max_side": 1280}
    return result
