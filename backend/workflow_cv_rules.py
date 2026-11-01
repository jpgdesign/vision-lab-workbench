"""Independent OpenCV building blocks for region and appearance workflows.

No code from the reference applications is reused. Metrics use source time and
image coordinates; pixel displacement is never presented as physical distance.
"""
from __future__ import annotations

from collections import deque
import copy
import math


CV_RULE_KINDS = {"cv_roi", "cv_histogram", "cv_track_motion"}
MAX_TRACKS = 512
MAX_HISTORY = 128


def cv_rules_catalog(node, param, port):
    return [
        node("cv_roi", "OpenCV · 多邊形 ROI／疊色", "OpenCV", "保留或遮掉多邊形區域，也可疊加半透明顏色；區域外遮黑不改變座標。", [port("frame", "frame")], [port("frame", "frame")], [
            param("points", "多邊形相對座標", [[.15, .15], [.85, .15], [.85, .85], [.15, .85]]),
            param("mode", "區域處理", "mask_inside", "select", options=["mask_inside", "mask_outside", "tint"]),
            param("color", "疊色 RGB", [100, 180, 150]), param("alpha", "疊色透明度", .3, min=0, max=1, step=.05)]),
        node("cv_histogram", "OpenCV · 外觀直方圖比較", "OpenCV", "比較 HSV 色彩分布；接入追蹤時逐物件比較，否則比較整個畫面。不能代替身分或行為辨識。", [port("frame", "frame"), port("tracks", "tracks", required=False)], [port("frame", "frame"), port("actions", "actions")], [
            param("reference", "比較基準", "previous", "select", options=["previous", "first"]),
            param("similarity_threshold", "外觀改變相似度門檻", .7, min=-1, max=1, step=.05),
            param("max_gap_seconds", "觀察最大間隔（秒）", 1., min=.1, max=30, step=.1),
            param("hue_bins", "色相分箱數", 24, min=8, max=64, step=1),
            param("saturation_bins", "飽和度分箱數", 16, min=8, max=64, step=1)]),
        node("cv_track_motion", "OpenCV · 軌跡位移／速度", "OpenCV", "依追蹤錨點與來源時間量測像素速度、路徑長與畫面對角線比例。失聯中斷計算；未校正相機時不輸出公尺或 km/h。", [port("tracks", "tracks")], [port("tracks", "tracks"), port("actions", "actions")], [
            param("anchor", "量測錨點", "foot", "select", options=["foot", "center"]),
            param("moving_threshold", "移動門檻（px/s）", 20., min=0, max=10000, step=1),
            param("window_seconds", "速度平滑時間窗（秒）", 1., min=.1, max=30, step=.1),
            param("max_gap_seconds", "觀察最大間隔（秒）", 1., min=.1, max=30, step=.1)]),
    ]


def _number(value, name, low, high, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f"{name} 必須介於 {low} 與 {high}" + (" 的整數。" if integer else "。"))
    return int(value) if integer else float(value)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on(a, b, p):
    return abs(_cross(a, b, p)) < 1e-10 and min(a[0], b[0]) - 1e-10 <= p[0] <= max(a[0], b[0]) + 1e-10 and min(a[1], b[1]) - 1e-10 <= p[1] <= max(a[1], b[1]) + 1e-10


def _intersect(a, b, c, d):
    return (_cross(a, b, c) * _cross(a, b, d) < 0 and _cross(c, d, a) * _cross(c, d, b) < 0) or any((_on(a, b, c), _on(a, b, d), _on(c, d, a), _on(c, d, b)))


def validate_cv_rule_params(kind, params):
    if kind not in CV_RULE_KINDS or not isinstance(params, dict):
        raise ValueError("未知 OpenCV 規則或無效參數。")
    p = copy.deepcopy(params)
    if kind == "cv_roi":
        points = p.get("points", [[.15, .15], [.85, .15], [.85, .85], [.15, .85]])
        if not isinstance(points, list) or not 3 <= len(points) <= 64:
            raise ValueError("ROI 必須有 3–64 個相對座標點。")
        clean = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("ROI 每個點需為 [x, y]。")
            clean.append([_number(v, "ROI 座標", 0, 1) for v in point])
        if len({tuple(point) for point in clean}) != len(clean):
            raise ValueError("ROI 不可包含重複座標點。")
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(clean, clean[1:] + clean[:1]))) / 2
        if area < 1e-8:
            raise ValueError("ROI 面積不可為零。")
        count = len(clean)
        for i in range(count):
            for j in range(i + 1, count):
                if j == i + 1 or (i == 0 and j == count - 1):
                    continue
                if _intersect(clean[i], clean[(i + 1) % count], clean[j], clean[(j + 1) % count]):
                    raise ValueError("ROI 邊線不可自相交。")
        mode = p.get("mode", "mask_inside")
        if mode not in {"mask_inside", "mask_outside", "tint"}:
            raise ValueError("無效的 ROI 處理模式。")
        color = p.get("color", [100, 180, 150])
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError("RGB 疊色需有三個整數。")
        p.update(points=clean, mode=mode, color=[_number(v, "RGB", 0, 255, integer=True) for v in color], alpha=_number(p.get("alpha", .3), "alpha", 0, 1))
    else:
        p["max_gap_seconds"] = _number(p.get("max_gap_seconds", 1), "max_gap_seconds", .1, 30)
        if kind == "cv_histogram":
            if p.get("reference", "previous") not in {"previous", "first"}:
                raise ValueError("無效的直方圖比較基準。")
            p["reference"] = p.get("reference", "previous")
            p["similarity_threshold"] = _number(p.get("similarity_threshold", .7), "similarity_threshold", -1, 1)
            p["hue_bins"] = _number(p.get("hue_bins", 24), "hue_bins", 8, 64, integer=True)
            p["saturation_bins"] = _number(p.get("saturation_bins", 16), "saturation_bins", 8, 64, integer=True)
        else:
            if p.get("anchor", "foot") not in {"foot", "center"}:
                raise ValueError("量測錨點需為 foot 或 center。")
            p["anchor"] = p.get("anchor", "foot")
            p["moving_threshold"] = _number(p.get("moving_threshold", 20), "moving_threshold", 0, 10000)
            p["window_seconds"] = _number(p.get("window_seconds", 1), "window_seconds", .1, 30)
    return p


def _packet(items, frame, **extras):
    return {"items": items, "frame": frame, "timestamp": frame["timestamp"], "fresh": True, **extras}


def _tracked(item):
    if not isinstance(item, dict) or item.get("observed", True) is False:
        return None
    identity = item.get("track_key")
    if identity is None:
        identity = item.get("track_id")
    box = item.get("box")
    if identity is None or not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return str(identity), list(map(float, box))


def _actor(item):
    return {"scope": "person", **{key: item[key] for key in ("track_key", "track_id") if key in item}}


class CVRuleNode:
    actual_device = "cpu"

    def __init__(self, kind, params, node_id=""):
        self.kind, self.node_id = kind, str(node_id)
        self.params = validate_cv_rule_params(kind, params)
        self.reset()

    def reset(self):
        self._states = {}
        self._context = self._last_time = None

    def close(self):
        self.reset()

    def _prepare(self, frame):
        stamp = frame["timestamp"]
        if type(stamp) not in (float, int) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("OpenCV 來源時間戳無效。")
        context = (frame.get("source_id"), frame.get("session_id"), frame["width"], frame["height"], repr(frame.get("to_source")))
        if context != self._context or (self._last_time is not None and (stamp <= self._last_time or stamp - self._last_time > self.params.get("max_gap_seconds", 1))):
            self._states.clear()
        self._context, self._last_time = context, stamp

    def process(self, inputs, frame):
        if self.kind != "cv_roi" and inputs.get("tracks", {}).get("fresh", True) is False:
            # Skipped inference is not new evidence. Preserve the timestamp of
            # the last real observation so the next update can detect a gap.
            result = {"actions": _packet([], frame, fresh=False)}
            result["frame" if self.kind == "cv_histogram" else "tracks"] = _packet([], frame, fresh=self.kind == "cv_histogram")
            return result
        self._prepare(frame)
        if self.kind == "cv_roi":
            return self._roi(frame)
        if self.kind == "cv_histogram":
            return self._histogram(inputs, frame)
        return self._motion(inputs, frame)

    def _roi(self, frame):
        import cv2
        import numpy as np
        from PIL import Image
        image = np.asarray(frame["image"].convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygon = np.rint(np.asarray(self.params["points"]) * [max(0, image.shape[1] - 1), max(0, image.shape[0] - 1)]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        mode = self.params["mode"]
        if mode == "tint":
            overlay = image.copy()
            overlay[mask > 0] = self.params["color"]
            result = cv2.addWeighted(image, 1 - self.params["alpha"], overlay, self.params["alpha"], 0)
        else:
            result = cv2.bitwise_and(image, image, mask=mask if mode == "mask_inside" else cv2.bitwise_not(mask))
        changed = {**frame, "image": Image.fromarray(result)}
        return {"frame": _packet([], changed)}

    def _histogram(self, inputs, frame):
        import cv2
        import numpy as np
        image = np.asarray(frame["image"].convert("RGB"))
        tracks = inputs.get("tracks")
        if tracks is not None and tracks.get("fresh", True) is False:
            return {"frame": _packet([], frame), "actions": _packet([], frame, fresh=False)}
        candidates = []
        if tracks is None:
            candidates.append(("scene", {"scope": "scene"}, image))
        else:
            for item in tracks.get("items", [])[:MAX_TRACKS]:
                tracked = _tracked(item)
                if tracked is None:
                    continue
                identity, box = tracked
                x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
                x2, y2 = min(image.shape[1], math.ceil(box[2])), min(image.shape[0], math.ceil(box[3]))
                if x2 > x1 and y2 > y1:
                    candidates.append((identity, _actor(item), image[y1:y2, x1:x2]))
        actions, states = [], {}
        for identity, actor, crop in candidates:
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [self.params["hue_bins"], self.params["saturation_bins"]], [0, 180, 0, 256])
            hist = cv2.normalize(hist, None, alpha=1, norm_type=cv2.NORM_L1)
            previous = self._states.get(identity)
            action = {**actor, "timestamp": frame["timestamp"], "label": "unknown", "score": 0., "reason": "需要下一次連續觀察"}
            if previous is not None:
                correlation = float(cv2.compareHist(previous, hist, cv2.HISTCMP_CORREL))
                if math.isfinite(correlation):
                    correlation = max(-1., min(1., correlation))
                    action.update(label="appearance_change" if correlation < self.params["similarity_threshold"] else "appearance_stable", score=1., similarity=correlation, reason="HSV 色彩分布比較；非身分或動作判定")
            actions.append(action)
            states[identity] = previous if previous is not None and self.params["reference"] == "first" else hist
        # A missing or invalid crop cannot preserve a reference across occlusion.
        self._states = states
        return {"frame": _packet([], frame), "actions": _packet(actions, frame)}

    def _motion(self, inputs, frame):
        incoming = inputs.get("tracks", {})
        if incoming.get("fresh", True) is False:
            return {"tracks": _packet([], frame, fresh=False), "actions": _packet([], frame, fresh=False)}
        stamp, diagonal = frame["timestamp"], math.hypot(frame["width"], frame["height"])
        states, tracks, actions = {}, [], []
        for item in incoming.get("items", [])[:MAX_TRACKS]:
            tracked = _tracked(item)
            if tracked is None:
                continue
            identity, box = tracked
            box = [max(0., min(frame["width"], box[0])), max(0., min(frame["height"], box[1])), max(0., min(frame["width"], box[2])), max(0., min(frame["height"], box[3]))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            point = ((box[0] + box[2]) / 2, box[3] if self.params["anchor"] == "foot" else (box[1] + box[3]) / 2)
            old = self._states.get(identity)
            state = old or {"history": deque(maxlen=MAX_HISTORY), "distance": 0.}
            action = {**_actor(item), "timestamp": stamp, "label": "unknown", "score": 0., "reason": "需要下一次連續觀察", "distance_px": state["distance"], "speed_px_s": None, "normalized_speed": None}
            history = state["history"]
            if history:
                state["distance"] += math.hypot(point[0] - history[-1][1][0], point[1] - history[-1][1][1])
            history.append((stamp, point, state["distance"]))
            # Retain one preceding sample to avoid a zero-width time window.
            while len(history) > 2 and stamp - history[1][0] >= self.params["window_seconds"]:
                history.popleft()
            span = stamp - history[0][0]
            if span > 0:
                speed = (state["distance"] - history[0][2]) / span
                action.update(label="moving" if speed > self.params["moving_threshold"] else "still", score=1., reason="像素位移規則；非真實世界速度", speed_px_s=speed, normalized_speed=speed / diagonal if diagonal else 0., window_seconds=span, distance_px=state["distance"])
            copied = copy.deepcopy(item)
            copied["motion"] = {k: action[k] for k in ("speed_px_s", "normalized_speed", "distance_px")}
            tracks.append(copied)
            actions.append(action)
            states[identity] = state
        self._states = states
        return {"tracks": _packet(tracks, frame, overlays=copy.deepcopy(incoming.get("overlays", []))), "actions": _packet(actions, frame)}
