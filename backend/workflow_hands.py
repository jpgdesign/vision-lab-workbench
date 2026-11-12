"""Real hand landmarks and box-prompted SAM masks, loaded once per graph job."""
from __future__ import annotations

import math
from pathlib import Path

HAND_MODEL_KINDS = {"hand_landmarks", "prompt_mask"}


class HandModelNode:
    def __init__(self, kind, params, resolved, device="cpu"):
        if kind not in HAND_MODEL_KINDS:
            raise ValueError("Unknown hand model node")
        self.kind, self.params, self.resolved = kind, params, resolved
        self.actual_device = "cpu" if kind == "hand_landmarks" else str(device)
        self.model = None
        self.reset()

    def reset(self):
        self.last_time = self.context = None

    def close(self):
        if self.kind == "hand_landmarks" and self.model is not None:
            self.model.close()
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        path = Path(self.resolved.get("weight_path", ""))
        if not path.is_file():
            raise ValueError("模型檔案不存在；請先下載工作流資產。")
        if self.kind == "hand_landmarks":
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            self.model = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=int(self.params.get("max_hands", 2)),
                min_hand_detection_confidence=self.params.get("confidence", .4),
                min_hand_presence_confidence=self.params.get("confidence", .4)))
        else:
            from ultralytics import SAM
            self.model = SAM(str(path))

    def process(self, inputs, frame):
        t = float(frame["timestamp"])
        if not math.isfinite(t) or t < 0:
            raise ValueError("Invalid frame timestamp")
        context = (frame.get("source_id"), frame["width"], frame["height"])
        if self.context != context or (self.last_time is not None and t < self.last_time):
            self.reset()
        self.context = context
        port = "poses" if self.kind == "hand_landmarks" else "detections"
        packet = {"items": [], "frame": frame, "timestamp": t, "fresh": True,
                  "diagnostics": {"model_id": self.resolved.get("model_id"), "actual_device": self.actual_device}}
        interval = float(self.params.get("every_seconds", 0))
        if self.last_time is not None and t - self.last_time < max(interval, 1e-9):
            # Never draw an old mask on a new video frame.
            packet["fresh"] = False
            return {port: packet}
        self.last_time = t
        if self.kind == "hand_landmarks":
            self._load()
            packet["items"] = self._hands(frame)
        else:
            observations = inputs.get("objects", {})
            if observations.get("fresh", True) and abs(float(observations.get("timestamp", t))-t) < 1e-6:
                objects = [v for v in observations.get("items", []) if v.get("observed", True)
                           and isinstance(v.get("box"), (list, tuple)) and len(v["box"]) == 4]
                objects = objects[:int(self.params.get("max_objects", 4))]
                if objects:
                    self._load()
                    packet["items"] = self._masks(frame, objects)
        return {port: packet}

    def _hands(self, frame):
        import numpy as np
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame["image"].convert("RGB")))
        result = self.model.detect(image)
        w, h = frame["width"], frame["height"]
        items = []
        for index, landmarks in enumerate(result.hand_landmarks):
            if len(landmarks) != 21:
                continue
            points = [[float(p.x*w), float(p.y*h), float(0 <= p.x <= 1 and 0 <= p.y <= 1)] for p in landmarks]
            if not np.isfinite(points).all():
                continue
            xy = np.asarray(points)[:, :2]
            extent = xy.max(0)-xy.min(0)
            low = np.maximum(0, xy.min(0)-extent*.12)
            high = np.minimum([w, h], xy.max(0)+extent*.12)
            if np.any(high-low < 2):
                continue
            handedness = result.handedness[index][0] if result.handedness[index] else None
            items.append({"label": "hand", "score": 1., "score_kind": "presence_gate_not_probability",
                          "box": [*low.tolist(), *high.tolist()], "keypoints": points,
                          "keypoint_format": "mediapipe21_xy_visibility_pixels", "observed": True,
                          "handedness": handedness.category_name if handedness else "unknown",
                          "handedness_score": float(handedness.score) if handedness else None,
                          "note": "關節第三欄為畫面內可見旗標；MediaPipe 不提供逐點信心。"})
        return items

    def _masks(self, frame, objects):
        import numpy as np
        from .segmentation import attach_mask
        result = self.model.predict(frame["image"], bboxes=[v["box"] for v in objects],
                                    device=self.actual_device, verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.detach().cpu().numpy()
        if len(masks) != len(objects):
            raise ValueError("SAM 遮罩數量與提示框不一致。")
        scores = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else [None]*len(objects)
        items = []
        for obj, mask, quality in zip(objects, masks, scores):
            if not np.isfinite(mask).all() or (quality is not None and (not math.isfinite(quality) or quality < self.params.get("min_quality", .5))):
                continue
            item = attach_mask({**obj, "mask_model": "sam2.1_tiny", "mask_quality": quality,
                                "mask_evidence_timestamp": frame["timestamp"]}, mask, frame["image"].size)
            if item["mask_area"]:
                items.append(item)
        return items
