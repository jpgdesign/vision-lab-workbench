"""OpenCV graph adapters. Images follow frame ports; geometry stays reversible."""
from __future__ import annotations

from copy import deepcopy
import math

from .workflow_opencv_catalog import OPENCV_KINDS, OPENCV_PARAMS


def validate_opencv_params(kind, params):
    if kind not in OPENCV_KINDS: raise ValueError(f"未知 OpenCV 節點：{kind}")
    if not isinstance(params, dict): raise ValueError("OpenCV 參數必須為物件。")
    definitions = {p["key"]: p for p in OPENCV_PARAMS[kind]}
    if set(params) - set(definitions): raise ValueError("包含未支援的 OpenCV 參數。")
    values = {key: deepcopy(params.get(key, spec["default"])) for key, spec in definitions.items()}
    for key, value in values.items():
        spec = definitions[key]
        if spec["type"] == "number":
            if type(value) not in (int, float) or not math.isfinite(value) or not spec.get("min", -math.inf) <= value <= spec.get("max", math.inf):
                raise ValueError(f"{key} 超出有效數值範圍。")
            if spec.get("step") == 1 and int(value) != value: raise ValueError(f"{key} 必須為整數。")
        elif spec["type"] == "boolean" and type(value) is not bool:
            raise ValueError(f"{key} 必須為勾選值。")
        elif spec["type"] in {"string", "select"}:
            if not isinstance(value, str) or not value.strip() or len(value) > 80: raise ValueError(f"{key} 必須為 1–80 字元文字。")
            if spec.get("options") and value not in spec["options"]: raise ValueError(f"{key} 選項無效。")
    for key in ("kernel_size", "block_size", "window_size"):
        if key in values and values[key] % 2 != 1: raise ValueError(f"{key} 必須為奇數。")
    if kind == "cv_edges":
        if values["low"] > values["high"]: raise ValueError("Canny 低閾值不可大於高閾值。")
        if values["aperture"] not in (3, 5, 7): raise ValueError("Canny aperture 必須為 3、5 或 7。")
    if kind == "cv_crop":
        roi = values["roi"]
        if not isinstance(roi, list) or len(roi) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in roi):
            raise ValueError("ROI 必須為四個 0–1 的數值 [左, 上, 右, 下]。")
        if roi[2] <= roi[0] or roi[3] <= roi[1]: raise ValueError("ROI 右／下界必須大於左／上界。")
    if kind == "cv_hsv":
        for key in ("lower", "upper"):
            value = values[key]
            if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or int(v) != v or not 0 <= v <= maximum for v, maximum in zip(value, (179, 255, 255))):
                raise ValueError("HSV 必須為 [H, S, V] 整數；H 0–179，S/V 0–255。")
        if any(values["lower"][i] > values["upper"][i] for i in (1, 2)):
            raise ValueError("HSV 的 S/V 下限不可大於上限；只有 H 支援跨色環。")
    if kind == "cv_background" and -1 < values["learning_rate"] < 0:
        raise ValueError("背景更新率請使用 -1（自動）或 0–1。")
    return values


def _packet(frame, items=None, **extra):
    return {"items": [] if items is None else items, "timestamp": frame["timestamp"], "frame": frame, "fresh": True, **extra}


def _frame(frame, pixels, local_to_parent=None):
    import numpy as np
    from PIL import Image
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
    result = {**frame, "image": image, "width": image.width, "height": image.height}
    if local_to_parent is not None:
        result["to_source"] = (np.asarray(frame.get("to_source", np.eye(3)), dtype=float) @ local_to_parent).tolist()
    return result


def _gray(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


class OpenCVNode:
    """One adapter per graph node. Temporal state is bounded to one image/model."""
    actual_device = "cpu"

    def __init__(self, kind, params, node_id="opencv"):
        self.kind, self.params, self.node_id = kind, validate_opencv_params(kind, params), str(node_id)
        self.reset()

    def reset(self):
        self.last_context = self.last_time = self.previous = self.background = None
        self.warmup_count = 0

    def close(self):
        self.reset()

    def _temporal_context(self, frame):
        matrix = tuple(tuple(row) for row in frame.get("to_source", ((1, 0, 0), (0, 1, 0), (0, 0, 1))))
        context = (frame.get("source_id"), frame.get("session_id"), frame["width"], frame["height"], matrix)
        stamp = float(frame["timestamp"])
        if not math.isfinite(stamp) or stamp < 0: raise ValueError("OpenCV 來源時間戳無效。")
        previous_time = self.last_time
        if self.last_context != context or previous_time is None or stamp <= previous_time or stamp - previous_time > self.params["max_gap_seconds"]:
            self.reset()
            previous_time = None
        self.last_context, self.last_time = context, stamp
        return None if previous_time is None else stamp - previous_time

    def _action(self, frame, label, **details):
        return {"label": label, "score": 0. if label == "unknown" else 1., "scope": "scene",
                "kind": "opencv_motion", "timestamp": frame["timestamp"], "evidence_timestamp": frame["timestamp"],
                "method": self.kind, "score_type": "geometric_rule", **details}

    def process(self, inputs, frame):
        import cv2
        import numpy as np
        from PIL import Image
        if not isinstance(frame.get("image"), Image.Image): raise ValueError("OpenCV 節點需要 PIL 影像。")
        if frame["image"].width < 1 or frame["image"].height < 1 or frame["image"].width * frame["image"].height > 8_388_608:
            raise ValueError("OpenCV 輸入尺寸無效或超過 8 MP。")
        rgb = np.asarray(frame["image"].convert("RGB"))
        h, w = rgb.shape[:2]
        if frame.get("width") != w or frame.get("height") != h: raise ValueError("OpenCV 影像尺寸與 frame 不符。")
        p, kind = self.params, self.kind
        transformed, pixels, extra = None, rgb, {}
        if kind == "cv_crop":
            left, top, right, bottom = p["roi"]
            x1, y1 = min(w - 1, int(math.floor(left * w))), min(h - 1, int(math.floor(top * h)))
            x2, y2 = max(x1 + 1, min(w, int(math.ceil(right * w)))), max(y1 + 1, min(h, int(math.ceil(bottom * h))))
            pixels = rgb[y1:y2, x1:x2]
            transformed = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]], dtype=float)
        elif kind == "cv_resize":
            width, height = int(p["width"]), int(p["height"])
            if p["mode"] == "fit":
                ratio = min(width / w, height / h)
                width, height = max(1, round(w * ratio)), max(1, round(h * ratio))
            pixels = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA if width < w and height < h else cv2.INTER_LINEAR)
            transformed = np.diag([w / width, h / height, 1.])
        elif kind == "cv_rotate":
            matrix = cv2.getRotationMatrix2D((w / 2., h / 2.), p["angle"], 1.)
            width, height = w, h
            if p["expand"]:
                cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
                width = max(1, math.ceil(w * cosine + h * sine - 1e-9))
                height = max(1, math.ceil(h * cosine + w * sine - 1e-9))
                if width * height > 8_388_608: raise ValueError("展開後的旋轉影像超過 8 MP；請先縮放。")
                matrix[0, 2] += (width - w) / 2.
                matrix[1, 2] += (height - h) / 2.
            # Geometry uses image-edge coordinates (boxes span 0..width),
            # whereas warpAffine samples pixel centers at integer indices.
            # Convert the translation between these conventions; otherwise a
            # 90-degree rotation loses a border column and inserts a black row.
            sampling_matrix = matrix.copy()
            sampling_matrix[:, 2] += matrix[:, :2] @ np.array([.5, .5]) - .5
            pixels = cv2.warpAffine(rgb, sampling_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            transformed = np.linalg.inv(np.vstack([matrix, [0., 0., 1.]]))
        elif kind == "cv_color":
            pixels = _gray(rgb)
            if p["mode"] == "equalize": pixels = cv2.equalizeHist(pixels)
            elif p["mode"] == "clahe": pixels = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(pixels)
        elif kind == "cv_blur":
            kernel = int(p["kernel_size"])
            if p["method"] == "gaussian": pixels = cv2.GaussianBlur(rgb, (kernel, kernel), p["sigma"])
            elif p["method"] == "median": pixels = cv2.medianBlur(rgb, kernel)
            else: pixels = cv2.bilateralFilter(rgb, kernel, p["sigma"], p["sigma"])
        elif kind == "cv_threshold":
            mode = cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY
            gray = _gray(rgb)
            if p["method"] == "adaptive": pixels = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, int(p["block_size"]), p["constant"])
            else: _, pixels = cv2.threshold(gray, p["threshold"], 255, mode | (cv2.THRESH_OTSU if p["method"] == "otsu" else 0))
        elif kind == "cv_edges":
            pixels = cv2.Canny(_gray(rgb), p["low"], p["high"], apertureSize=int(p["aperture"]), L2gradient=p["l2_gradient"])
        elif kind == "cv_morphology":
            operation = {"opening": cv2.MORPH_OPEN, "closing": cv2.MORPH_CLOSE, "erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "gradient": cv2.MORPH_GRADIENT}[p["operation"]]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(p["kernel_size"]),) * 2)
            pixels = cv2.morphologyEx(rgb, operation, kernel, iterations=int(p["iterations"]))
        elif kind == "cv_hsv":
            hsv, lower, upper = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), np.asarray(p["lower"], dtype=np.uint8), np.asarray(p["upper"], dtype=np.uint8)
            if lower[0] <= upper[0]: pixels = cv2.inRange(hsv, lower, upper)
            else:
                pixels = cv2.bitwise_or(cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]], dtype=np.uint8)), cv2.inRange(hsv, np.array([0, lower[1], lower[2]], dtype=np.uint8), upper))
            if p["invert"]: pixels = cv2.bitwise_not(pixels)
        elif kind == "cv_contours":
            _, mask = cv2.threshold(_gray(rgb), p["threshold"], 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            found = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if p["min_area"] <= area <= w * h * p["max_area_ratio"] and area > 0: found.append((area, contour))
            items = []
            for area, contour in sorted(found, key=lambda item: item[0], reverse=True)[:int(p["max_objects"])]:
                x, y, width, height = cv2.boundingRect(contour)
                polygon = cv2.approxPolyDP(contour, p["approx_epsilon"] * cv2.arcLength(contour, True), True).reshape(-1, 2)
                if len(polygon) > 128: polygon = polygon[np.linspace(0, len(polygon) - 1, 128, dtype=int)]
                items.append({"label": p["label"].strip(), "score": 1., "score_type": "geometric_rule", "box": [x, y, x + width, y + height],
                              "polygon": polygon.tolist(), "area": area, "area_ratio": area / (w * h), "timestamp": frame["timestamp"]})
            extra["detections"] = _packet(frame, items, diagnostics={"method": "external_contours", "semantic_classifier": False})
        elif kind == "cv_background":
            self._temporal_context(frame)
            if self.background is None:
                if p["method"] == "MOG2": self.background = cv2.createBackgroundSubtractorMOG2(history=int(p["history"]), varThreshold=p["var_threshold"], detectShadows=p["detect_shadows"])
                else: self.background = cv2.createBackgroundSubtractorKNN(history=int(p["history"]), dist2Threshold=p["var_threshold"], detectShadows=p["detect_shadows"])
            # Bound background-model memory regardless of output image dimensions.
            ratio = min(1., 640 / max(w, h))
            reduced = cv2.resize(rgb, (max(1, round(w * ratio)), max(1, round(h * ratio)))) if ratio < 1 else rgb
            mask = self.background.apply(reduced, learningRate=p["learning_rate"])
            foreground = (mask == 255).astype(np.uint8) * 255
            motion_ratio = float(np.count_nonzero(foreground) / foreground.size)
            self.warmup_count += 1
            label = "unknown" if self.warmup_count <= p["warmup_frames"] else "motion" if motion_ratio >= p["min_motion_ratio"] else "still"
            details = {"motion_ratio": motion_ratio, "warmup_frames_seen": self.warmup_count, "warmup_frames": p["warmup_frames"]}
            if label == "unknown": details["reason"] = "背景模型暖機中"
            extra["actions"] = _packet(frame, [self._action(frame, label, **details)])
            pixels = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_NEAREST) if foreground.shape != (h, w) else foreground
        elif kind == "cv_optical_flow":
            dt = self._temporal_context(frame)
            ratio = min(1., p["max_side"] / max(w, h))
            size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
            gray = cv2.resize(_gray(rgb), size, interpolation=cv2.INTER_AREA) if ratio < 1 else _gray(rgb)
            pixels = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            if self.previous is None or dt is None:
                action = self._action(frame, "unknown", reason="需要兩個連續來源影格才能計算光流")
            else:
                flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None, .5, int(p["levels"]), int(p["window_size"]), int(p["iterations"]), 5, 1.2, 0)
                flow *= np.array([w / gray.shape[1], h / gray.shape[0]], dtype=np.float32)
                linear = np.asarray(frame.get("to_source", np.eye(3)), dtype=float)[:2, :2]
                source_flow = flow @ linear.T
                speed = np.linalg.norm(source_flow, axis=2) / dt
                mean_speed = float(np.mean(speed))
                if not math.isfinite(mean_speed): raise ValueError("光流計算出現無效速度。")
                _, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(pixels)
                hsv[..., 0] = np.mod(angle * 90 / math.pi, 180).astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = np.minimum(255, speed * (127.5 / p["min_speed_px_s"])).astype(np.uint8)
                pixels = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                action = self._action(frame, "motion" if mean_speed >= p["min_speed_px_s"] else "still", mean_speed_px_s=mean_speed, dt_seconds=dt)
            self.previous = gray.copy()
            extra["actions"] = _packet(frame, [action])
            if pixels.shape[:2] != (h, w): pixels = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_LINEAR)
        output = _frame(frame, pixels, transformed)
        return {"frame": _packet(output), **extra}
