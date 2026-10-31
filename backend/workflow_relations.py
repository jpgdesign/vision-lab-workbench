"""Local learned relation inference from an image and existing object regions.

Based on RelateAnything by Maëlic Neau: https://github.com/Maelic/RelateAnything
The ONNX graph consumes pixels and boxes, not object class labels. Labels and
track keys are carried through only to associate outputs with their sources.
"""
import json
import math
from pathlib import Path
import threading
from .vendor.relateanything.windows_runtime import prepare_windows_runtime


MODEL_ID = "relateanything_convnext_large"
ATTRIBUTION = "Based on RelateAnything by Maëlic Neau"
ATTRIBUTION_URL = "https://github.com/Maelic/RelateAnything"
REQUIRED_FILES = ("relateanything.onnx", "relateanything.json", "predicate_bank.npz", "thresholds.json")


def _number(value, minimum, maximum, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} 必須介於 {minimum} 和 {maximum}。")
    return float(value)


def _track_key(detection):
    for key in ("track_key", "track_id", "key"):
        value = detection.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
            return value
    return None


class RelationModel:
    """One CPU ONNX session and the checkpoint's calibrated predicate bank.

    More than 32 regions are explicitly ranked by detector confidence. Original
    detection indices are retained; ``last_diagnostics`` reports the selection.
    An explicit threshold uses the model's calibrated relation-score units;
    no bank threshold silently overrides that user control.
    """
    model_id = MODEL_ID
    actual_device = "cpu"

    def __init__(self, model_dir, device="cpu"):
        prepare_windows_runtime()
        import numpy as np
        from .vendor.relateanything.runtime import OnnxRelationHead, load_predicate_bank

        if str(device) not in {"cpu", "auto"}:
            raise ValueError("目前安裝的是 RelateAnything CPU ONNX 接頭；請選擇 cpu 或 auto。")
        directory = Path(model_dir).expanduser().resolve()
        missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
        if missing:
            raise ValueError("RelateAnything 模型檔不完整：" + ", ".join(missing))
        metadata = json.loads((directory / "relateanything.json").read_text(encoding="utf-8"))
        if (metadata.get("img_size") != 448 or metadata.get("max_boxes") != 32
                or metadata.get("vocab_mode") != "input" or metadata.get("text_dim") != 512
                or metadata.get("output_kind") != "logits"):
            raise ValueError("此接頭需要完整的 RelateAnything ConvNeXt-Large 448px / 32 boxes ONNX bundle。")
        calibration = metadata.get("calibration") or {}
        if (not metadata.get("calibrated") or not isinstance(calibration, dict)
                or not isinstance(calibration.get("a"), (int, float)) or not math.isfinite(calibration["a"])
                or calibration["a"] <= 0 or not isinstance(calibration.get("b"), (int, float))
                or not math.isfinite(calibration["b"])):
            raise ValueError("RelateAnything 缺少有效的模型專屬分數校準。")
        # Validate before allocating the ~1 GB ONNX graph. Legacy object-array
        # banks may unpickle only when their bytes match the official SHA256.
        with load_predicate_bank(directory / "predicate_bank.npz") as bank:
            names = [str(value) for value in bank["names"]]
            defaults = [str(value) for value in bank["default"]]
            if (not names or len(set(names)) != len(names) or any(not name for name in names)
                    or bank["W"].shape != (len(names), 512) or bank["alpha"].shape != (len(names),)
                    or not np.isfinite(bank["W"]).all() or not np.isfinite(bank["alpha"]).all()
                    or not defaults or not set(defaults).issubset(names)):
                raise ValueError("RelateAnything predicate bank 的名稱、文字維度或數值無效。")
        self.head = OnnxRelationHead(str(directory / "relateanything.onnx"),
                                    str(directory / "predicate_bank.npz"), threads=2,
                                    providers=["CPUExecutionProvider"])
        if not self.head.emits_logits:
            raise ValueError("此接頭需要輸出原始 logits 的 ONNX 模型。")
        self.directory = directory
        self.default_predicates = tuple(defaults)
        self.predicate_labels = tuple(names)
        self._lock = threading.Lock()
        self.last_diagnostics = {"actual_device": "cpu"}

    def available_predicates(self):
        return list(self.predicate_labels)

    def predict(self, frame_bgr, detections, predicates=None, threshold=.4, top_k=20):
        """Return relation-scoped records; never infer a person's identity."""
        import numpy as np
        from .vendor.relateanything.postprocess import ThresholdConfig, decode

        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 1):
            raise ValueError("RelateAnything 影像必須是非空 uint8 BGR [H,W,3]。")
        threshold = _number(threshold, 0, 1, "關係信心閾值")
        if type(top_k) is not int or not 1 <= top_k <= 128:
            raise ValueError("關係 top_k 必須是 1–128 的整數。")
        names = list(self.default_predicates) if predicates is None else predicates
        if (not isinstance(names, list) or not names or len(names) > len(self.predicate_labels)
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("關係 predicates 必須是非空、不重複的名稱清單。")
        absent = [name for name in names if name not in self.predicate_labels]
        if absent:
            raise ValueError("模型 predicate bank 不包含：" + ", ".join(absent) + "；不會自動替換成其他關係。")
        if not isinstance(detections, list):
            raise ValueError("關係輸入 detections 必須是物件清單。")
        height, width = frame_bgr.shape[:2]
        regions = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ValueError("每個關係輸入必須包含物件框。")
            try:
                box = np.asarray(detection["box"], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("物件框必須是像素 [x1,y1,x2,y2]。") from exc
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("物件框必須是四個有限像素座標。")
            box = box.copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("物件框不能為空或完全位於影像外。")
            score = _number(detection.get("score", 1.), 0, 1, "物件信心值")
            regions.append({"index": index, "box": box, "score": score,
                            "label": str(detection.get("label", "object")), "source": detection})
        # Stable ordering is important for ties and for source-index mapping.
        selected = sorted(regions, key=lambda row: (-row["score"], row["index"]))[:self.head.max_boxes]
        self.last_diagnostics = {"actual_device": "cpu", "input_count": len(regions),
                                 "selected_count": len(selected), "dropped_count": len(regions) - len(selected),
                                 "selected_indices": [row["index"] for row in selected]}
        if len(selected) < 2:
            return []
        boxes = np.array([row["box"] for row in selected], dtype=np.float32)
        scores = np.array([row["score"] for row in selected], dtype=np.float32)
        labels = [row["label"] for row in selected]
        with self._lock:
            self.head.set_predicates(names)
            pred, pair, subject, obj, valid = self.head(frame_bgr, boxes)
            pred, pair = np.asarray(pred), np.asarray(pair)
            subject, obj, valid = np.asarray(subject), np.asarray(obj), np.asarray(valid, dtype=bool)
            if (pred.ndim != 2 or pred.shape[1] != len(names)
                    or any(value.shape != (pred.shape[0],) for value in (pair, subject, obj, valid))
                    or not np.issubdtype(subject.dtype, np.integer) or not np.issubdtype(obj.dtype, np.integer)):
                raise RuntimeError("RelateAnything 輸出格式與模型契約不符。")
            valid &= (subject >= 0) & (obj >= 0) & (subject < len(selected)) & (obj < len(selected))
            if not np.isfinite(pred[valid]).all() or not np.isfinite(pair[valid]).all():
                raise RuntimeError("RelateAnything 輸出包含非有限分數。")
            config = ThresholdConfig(threshold=threshold, topk=top_k,
                                     calib_a=self.head.contract.calib_a, calib_b=self.head.contract.calib_b)
            triplets = decode(pred, pair, subject, obj, valid, names, config,
                              boxes_xyxy=boxes, box_scores=scores, box_labels=labels)
        result = []
        for triplet in triplets:
            sub, target = selected[triplet.subject_idx], selected[triplet.object_idx]
            row = {"scope": "relation", "subject_index": sub["index"], "object_index": target["index"],
                   "subject_box": sub["box"].tolist(), "object_box": target["box"].tolist(),
                   "subject_label": sub["label"], "object_label": target["label"],
                   "predicate": triplet.predicate, "label": triplet.predicate, "score": triplet.score}
            subject_key, object_key = _track_key(sub["source"]), _track_key(target["source"])
            if subject_key is not None:
                row.update(track_key=subject_key, subject_track_key=subject_key)
            if object_key is not None:
                row["object_track_key"] = object_key
            if subject_key is not None and object_key is not None:
                row["relation_key"] = json.dumps([subject_key, triplet.predicate, object_key], ensure_ascii=False, separators=(",", ":"))
            result.append(row)
        return result
