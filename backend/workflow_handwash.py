"""Evidence-constrained, single-sink handwashing observation (not a hygiene score)."""
from __future__ import annotations

import math
import uuid


def handwash_catalog(node, param, port):
    return [
        node("hand_landmarks", "手部偵測／21 點骨架 · MediaPipe", "模型",
             "實際手掌偵測與手部關節；適合近拍。與 ST-GCN++ 的人體 17 點格式不同，不能直接互換。",
             [port("frame", "frame")], [{**port("poses", "poses"), "label": "手部骨架（21 點）"}], [
                 param("confidence", "偵測與手部存在門檻", .4, min=.1, max=.95, step=.05),
                 param("max_hands", "最多手數", 2, min=1, max=4, step=1)]),
        node("prompt_mask", "提示框 → 真實遮罩 · SAM 2.1 Tiny", "模型",
             "將手部或其他物件的框送入 SAM，產生實例遮罩。不是用矩形／骨架外包輪廓冒充遮罩；每次對新影格重新分割。",
             [port("frame", "frame"), port("objects", "observations", accepts=["poses", "detections", "tracks"])],
             [port("detections", "detections")], [
                 param("every_seconds", "分割取樣間隔（秒）", .5, min=0, max=10, step=.1),
                 param("max_objects", "最多提示物件", 4, min=1, max=10, step=1),
                 param("min_quality", "遮罩品質下限（非類別機率）", .5, min=0, max=1, step=.05)]),
        node("handwash", "洗手位置＋動作證據融合", "規則",
             "單一洗手台：雙手位置＋真實遮罩＋RGB 洗手分數。位置本身不等於洗手；搓手幾何候選、影片分數與人體骨架分開顯示。無沖洗證據時不宣稱流程完成。",
             [{**port("hands", "poses"), "label": "手部骨架"}, {**port("masks", "detections", required=False), "label": "手部遮罩"},
              port("actions", "actions", required=False, multiple=True)],
             [port("actions", "actions"), port("events", "events")], [
                 param("zones", "洗手 ROI（只使用第一個區域）", [{"name": "洗手區", "points": [[.28,.27],[.78,.27],[.78,.88],[.28,.88]]}]),
                 param("min_mask_overlap", "手遮罩在 ROI 內的最小比例", .5, min=.05, max=1, step=.05),
                 param("require_masks", "融合候選必須有 SAM 遮罩", True),
                 param("min_video_score", "RGB washing hands 分數下限", .25, min=.01, max=.99, step=.01),
                 param("hold_seconds", "持續證據後記錄事件（秒）", 1., min=.1, max=30, step=.1),
                 param("max_gap_seconds", "連續手部觀察最大間隔（秒）", 1., min=.1, max=5, step=.1),
                 param("evidence_max_age", "動作／遮罩證據有效期（秒）", 1.5, min=.1, max=5, step=.1),
                 param("motion_threshold", "雙手移動下限（畫面對角線／秒）", .005, min=.001, max=.5, step=.001)]),
    ]


def validate_handwash_params(params):
    zones = params.get("zones", [])
    if not isinstance(zones, list) or len(zones) != 1 or not isinstance(zones[0], dict):
        raise ValueError("洗手節點需要恰好一個 ROI。")
    points = zones[0].get("points")
    if not isinstance(points, list) or not 3 <= len(points) <= 32:
        raise ValueError("洗手 ROI 需要 3–32 個座標。")
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError("洗手 ROI 座標必須在 0–1。")
    area = abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points, points[1:]+points[:1])))/2
    if area < .0001:
        raise ValueError("洗手 ROI 面積太小或為退化多邊形。")
    return params


class HandwashNode:
    def __init__(self, params, node_id="handwash"):
        self.params = validate_handwash_params(params)
        self.node_id = node_id
        self.reset()

    def reset(self):
        self.context = self.last_time = self.last_centers = self.last_inside = None
        self.since = self.state = self.emitted_state = None
        self.mask_evidence = None
        self.video_evidence = {}
        self.event_count = 0

    @staticmethod
    def _current(packet, t):
        return (packet.get("fresh", True) and abs(float(packet.get("timestamp", t))-t) < 1e-6)

    def process(self, inputs, frame):
        import cv2
        import numpy as np
        from .segmentation import decode_mask
        t, w, h = float(frame["timestamp"]), frame["width"], frame["height"]
        context = (frame.get("source_id"), w, h)
        gap = self.last_time is not None and (t-self.last_time > self.params.get("max_gap_seconds", 1.) or t < self.last_time)
        if context != self.context or gap:
            self.reset()
        self.context = context
        overlay = {"kind": "polygon", "label": "洗手 ROI（位置證據）", "points": self.params["zones"][0]["points"]}
        empty = {"items": [], "timestamp": t, "frame": frame, "overlays": [overlay]}
        if self.last_time is not None and t == self.last_time:
            return {"actions": {**empty, "fresh": False}, "events": {**empty, "fresh": False}}
        dt = t-self.last_time if self.last_time is not None else None
        self.last_time = t
        polygon = np.asarray(overlay["points"], np.float32)*[w, h]
        polygon = polygon.astype(np.float32)
        roi = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi, [np.rint(polygon).astype(np.int32)], 1)
        source = inputs.get("hands", {})
        hands = source.get("items", []) if self._current(source, t) else []
        hands = [v for v in hands if v.get("observed", True) and len(v.get("keypoints", [])) == 21]
        centers = []
        for hand in hands:
            palm = np.asarray(hand["keypoints"], float)[[0, 5, 9, 13, 17], :]
            if not np.isfinite(palm).all() or (palm[:, 2] <= 0).any():
                continue
            centers.append(palm[:, :2].mean(0))
        inside = ([cv2.pointPolygonTest(polygon, tuple(map(float, center)), False) >= 0 for center in centers])
        two = len(centers) == 2
        both_inside = two and all(inside)
        motion = None
        if two and self.last_centers is not None and dt is not None and dt > 0:
            # Handedness may flip under occlusion. Match two observed palms by
            # minimum displacement, without claiming persistent person IDs.
            current, previous = np.asarray(centers), np.asarray(self.last_centers)
            motion = min(np.linalg.norm(current-previous, axis=1).mean(),
                         np.linalg.norm(current-previous[::-1], axis=1).mean())/math.hypot(w,h)/dt
        self.last_centers = centers if two else None
        masks = inputs.get("masks", {})
        if self._current(masks, t):
            overlaps = []
            for item in masks.get("items", []):
                if item.get("label") != "hand" or "mask" not in item:
                    continue
                pixels = decode_mask(item["mask"], (h, w))
                overlaps.append(float((pixels & (roi > 0)).sum()/max(1, pixels.sum())))
            self.mask_evidence = {"timestamp": t, "overlaps": overlaps}
        mask = self.mask_evidence
        age = self.params.get("evidence_max_age", 1.5)
        mask_valid = bool(mask and 0 <= t-mask["timestamp"] <= age and len(mask["overlaps"]) == 2
                          and min(mask["overlaps"]) >= self.params.get("min_mask_overlap", .5))
        packets = inputs.get("actions", [])
        if isinstance(packets, dict): packets = [packets]
        skeleton = []
        for index, packet in enumerate(packets):
            if self._current(packet, t):
                for row in packet.get("items", []):
                    candidates = row.get("candidates", [])
                    if row.get("scope") == "scene":
                        self.video_evidence[index] = {"timestamp": t, "label": row.get("label"), "candidates": candidates,
                                                      "reason": row.get("reason"), "model_id": packet.get("diagnostics", {}).get("model_id")}
            for row in packet.get("display_items", packet.get("items", [])):
                if row.get("scope") != "scene":
                    # Body tracks have no verified association to these hands.
                    # Keep their evidence separate, never silently fuse people.
                    skeleton.append({k: row.get(k) for k in ("track_id", "label", "score", "reason", "candidates")})
        video = [v for v in self.video_evidence.values() if 0 <= t-v["timestamp"] <= age]
        wash_score = None
        for evidence in video:
            candidates = evidence["candidates"]
            if candidates and str(candidates[0].get("label", "")).lower().strip() == "washing hands":
                wash_score = max(wash_score or 0, float(candidates[0]["score"]))
        rgb_valid = wash_score is not None and wash_score >= self.params.get("min_video_score", .25)
        spatial_valid = both_inside and (mask_valid or not self.params.get("require_masks", True))
        if not two:
            state, label = "unknown", "未知：雙手不足／多人手部混入"
        elif not both_inside:
            state, label = "outside", "雙手尚未同時進入洗手 ROI"
        elif spatial_valid and rgb_valid:
            state, label = "washing_candidate", "洗手候選（雙手＋遮罩＋RGB）" if self.params.get("require_masks", True) else "洗手候選（雙手＋RGB）"
        elif both_inside and motion is not None and motion >= self.params.get("motion_threshold", .005):
            state, label = "hand_motion_candidate", "雙手動作候選（未確認洗手）"
        else:
            state, label = "hands_in_roi", "雙手在洗手區（僅位置證據）"
        events = []
        def event(kind, name):
            self.event_count += 1
            events.append({"event_id": f"{self.node_id}-{uuid.uuid4().hex}", "kind": kind, "label": name,
                           "scope": "scene", "timestamp": t, "evidence_type": "rule", "node_id": self.node_id})
        if two:
            if self.last_inside is False and both_inside:
                event("hands_enter_roi", "雙手進入洗手 ROI")
            elif self.last_inside is True and not both_inside:
                event("hands_leave_roi", "手部離開 ROI（非完成判定）")
            self.last_inside = both_inside
        else:
            # Disappearance/occlusion is not proof of leaving the sink.
            self.last_inside = None
        if state != self.state or state == "unknown":
            self.state, self.since, self.emitted_state = state, t, None
        if state == "washing_candidate" and t-self.since >= self.params.get("hold_seconds", 1.) and self.emitted_state != state:
            event("washing_candidate", "持續洗手候選（未判定沖洗／流程完成）")
            self.emitted_state = state
        evidence = {"hand_count": len(centers), "both_hands_in_roi": both_inside, "mask_evidence": mask,
                    "mask_valid": mask_valid, "hand_motion_diagonals_s": motion, "rgb": video,
                    "rgb_washing_score": wash_score, "body_skeleton": skeleton,
                    "body_skeleton_status": "available_separate_identity" if any(row.get("label") not in (None, "unknown") for row in skeleton) else "insufficient_body_keypoints",
                    "rinsing": "unknown", "hygiene_compliance": "not_evaluated"}
        action = {"kind": "rule", "scope": "scene", "label": label, "state": state, "timestamp": t,
                  "score": wash_score if state == "washing_candidate" else 0., "score_kind": "rgb_score_not_fused_probability",
                  "evidence": evidence, "duration": max(0., t-self.since)}
        return {"actions": {**empty, "items": [action]}, "events": {**empty, "items": events}}
