"""Persistent learned-model nodes with bounded, source-time temporal buffers.

Packets marked fresh=False never contain new evidence in ``items``. Cached
scores may appear only in ``display_items`` with their original evidence time.
The caller must never feed display_items into a rule/condition node.
"""
from collections import deque
from contextlib import nullcontext
import math


KINDS = {"detector", "pose", "skeleton_action", "video_action", "relation"}
MAX_SKELETON_TRACKS = 32
MAX_SKELETON_FRAMES = 512
MAX_VIDEO_FRAMES = 64
MAX_VIDEO_SIDE = 384


def _number(value, name, low=0, high=1e15):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def _items(inputs, name, t):
    packet = inputs.get(name, {})
    if not isinstance(packet, dict) or not isinstance(packet.get("items", []), list):
        raise ValueError(f"{name} must be a packet with an items list")
    if packet.get("fresh", True) is False or packet.get("timestamp", t) != t:
        return []
    return [item for item in packet.get("items", []) if isinstance(item, dict) and item.get("observed", True) is not False]


def _track_key(item):
    value = item.get("track_key")
    if isinstance(value, str) and value:
        return value
    # A workflow tracker normally supplies a fully scoped key. A raw integer
    # ID is still local to this ModelNode/source/session, never merged globally.
    value = item.get("track_id")
    return f"track:{value}" if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "" else None


def _actor(track=None):
    if track is None:
        return {"scope": "scene"}
    result = {"scope": "person"}
    for key in ("track_key", "track_id", "box"):
        if key in track:
            result[key] = track[key]
    return result


def _unknown(reason, t, actor, **details):
    return {**actor, "label": "unknown", "score": 0., "kind": "unknown", "reason": reason,
            "timestamp": t, "evidence_timestamp": t, **details}


class ModelNode:
    """One model instance per graph node; no automatic weights/downloads."""

    def __init__(self, kind, params, resolved, device="cpu"):
        if kind not in KINDS:
            raise ValueError(f"Unsupported model node: {kind}; VLM is provided by the workflow engine")
        if not isinstance(params, dict) or not isinstance(resolved, dict):
            raise ValueError("ModelNode params/resolved must be objects")
        self.kind, self.params, self.resolved = kind, dict(params), dict(resolved)
        self.device = str(device)
        self.actual_device = "cpu" if kind == "relation" else self.device
        self.model = self.metadata = None
        self.every_seconds = _number(params.get("every_seconds", 0), "every_seconds", 0, 120)
        self.window = _number(params.get("window_seconds", 4 if kind == "skeleton_action" else 2), "window_seconds", .1, 60)
        self.min_frames = int(_number(params.get("min_frames", 32), "min_frames", 30, 100))
        if params.get("min_frames", 32) != self.min_frames:
            raise ValueError("min_frames must be an integer")
        self.min_score = _number(params.get("min_score", .35), "min_score", 0, 1)
        self.scope = params.get("scope", "scene")
        if kind == "video_action" and self.scope not in {"scene", "person"}:
            raise ValueError("video_action scope must be scene or person")
        self.max_people = int(_number(params.get("max_people", 3), "max_people", 1, 10))
        self.reset()

    def reset(self):
        """Reset temporal evidence, retaining an already loaded model."""
        self.buffers, self.display = {}, []
        self.last_run = self.last_time = self.source = None
        self.diagnostics = {}

    def close(self):
        self.reset()
        self.model = self.metadata = None

    def _load(self):
        if self.model is not None:
            return
        if self.kind == "relation":
            from .workflow_relations import RelationModel
            self.model = RelationModel(self.resolved["model_dir"], device="cpu")
            self.actual_device = self.model.actual_device
            return
        from .inference import resolve_device
        self.actual_device = resolve_device(self.device)
        if self.kind == "detector":
            from .inference import create_detector
            self.model = create_detector(self.resolved["model_id"], self.resolved["params"], self.actual_device,
                                         self.resolved.get("checkpoint_path"), self.resolved.get("model_weight_path"))
        elif self.kind == "pose":
            from pathlib import Path
            from ultralytics import YOLO
            path = Path(self.resolved["weight_path"])
            if not path.is_file():
                raise ValueError("YOLO pose weight file is missing; download the workflow assets first")
            model = YOLO(str(path))
            if model.task != "pose" or tuple(model.model.model[-1].kpt_shape) != (17, 3):
                raise ValueError("The pose node requires a COCO17 x/y/confidence pose checkpoint")
            self.model = model
        elif self.kind == "skeleton_action":
            from .workflow_skeleton import SkeletonActionModel
            self.model = SkeletonActionModel(self.resolved["weight_path"], self.actual_device)
        elif self.kind == "video_action":
            from .inference import create_action_model
            self.model, self.metadata = create_action_model(self.resolved["model_id"], self.actual_device,
                                                          self.resolved.get("checkpoint_path"), self.resolved.get("model_weight_path"))

    def _packet(self, items, frame, fresh=True):
        if fresh:
            self.display = [dict(item) for item in items]
        display = [dict(item, stale=not fresh) for item in self.display]
        port = {"detector": "detections", "pose": "poses", "relation": "relations"}.get(self.kind, "actions")
        return {port: {"items": items if fresh else [], "timestamp": frame["timestamp"], "frame": frame,
                       "fresh": fresh, "display_items": display,
                       "diagnostics": {"model_id": self.resolved.get("model_id"), "actual_device": self.actual_device,
                                       "loaded": self.model is not None, "buffer_count": len(self.buffers), **self.diagnostics}}}

    def _due(self, t):
        if self.last_run is not None and t - self.last_run + 1e-9 < self.every_seconds:
            return False
        self.last_run = t
        return True

    def process(self, inputs, frame):
        from PIL import Image
        if not isinstance(inputs, dict) or not isinstance(frame, dict):
            raise ValueError("inputs and frame must be objects")
        image = frame.get("image")
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("frame.image must be a PIL RGB image")
        if image.size != (frame.get("width"), frame.get("height")):
            raise ValueError("frame width/height must match the source image")
        t = _number(frame.get("timestamp"), "frame.timestamp")
        source = frame.get("source_id")
        if not isinstance(source, str) or not source:
            raise ValueError("frame.source_id must be a nonempty string")
        identity = (source, frame.get("session_id", ""), image.size)
        if self.source != identity or (self.last_time is not None and t < self.last_time):
            self.reset()
        elif self.last_time == t:
            return self._packet([], frame, fresh=False)
        self.source, self.last_time = identity, t
        self.diagnostics = {}
        self._history_reset = False
        if self.kind == "detector":
            self._load()
            return self._packet(self.model(image, self.resolved["params"]), frame)
        if self.kind == "pose":
            return self._packet(self._pose(image), frame)
        if self.kind == "skeleton_action":
            return self._skeleton(inputs, frame)
        if self.kind == "video_action":
            return self._video(inputs, frame)
        return self._relation(inputs, frame)

    def _pose(self, image):
        import numpy as np
        self._load()
        result = self.model.predict(image, device=self.actual_device, verbose=False,
                                    conf=self.params.get("conf", .35), imgsz=self.params.get("imgsz", 640),
                                    max_det=MAX_SKELETON_TRACKS)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.keypoints is None:
            raise RuntimeError("Pose checkpoint returned boxes without keypoints")
        points = result.keypoints.data.detach().cpu().numpy()
        if points.shape != (len(result.boxes), 17, 3) or not np.isfinite(points).all():
            raise RuntimeError("Pose checkpoint returned invalid COCO17 keypoints")
        return [{"label": "person", "score": float(score), "box": box.tolist(), "keypoints": joints.tolist(),
                 "keypoint_format": "coco17_xy_confidence_pixels", "observed": True}
                for box, score, joints in zip(result.boxes.xyxy.detach().cpu(), result.boxes.conf.detach().cpu(), points)]

    def _tracks(self, inputs, frame, limit):
        tracks = [item for item in _items(inputs, "tracks", frame["timestamp"])
                  if item.get("label") == "person" and _track_key(item) is not None]
        # Never concatenate duplicate IDs or arbitrarily combine two people.
        keys = [_track_key(item) for item in tracks]
        if len(set(keys)) != len(keys):
            raise ValueError("A temporal model received duplicate person track keys")
        tracks = sorted(tracks, key=lambda item: (-float(item.get("score", 0)), _track_key(item)))[:limit]
        active = {_track_key(item) for item in tracks}
        lost = set(self.buffers) - active
        self._lost_tracks = bool(lost or any(_track_key(item) not in active for item in self.display))
        for key in lost:
            self.buffers.pop(key)
        self.display = [item for item in self.display if _track_key(item) in active]
        return tracks

    def _append(self, key, t, value, cap, sampling_interval=0):
        history = self.buffers.setdefault(key, deque(maxlen=cap))
        # A long input gap is missing evidence, not continuous observed motion.
        gap_limit = min(1., self.window / 2)
        if history and t - history[-1][0] > gap_limit + 1e-9:
            history.clear()
            self._history_reset = True
        if not history or t - history[-1][0] + 1e-9 >= sampling_interval:
            history.append((t, value))
        # Keep one sample immediately preceding the window edge so uniform
        # source-time sampling can cover its entire requested duration.
        while len(history) > 1 and history[1][0] <= t - self.window:
            history.popleft()
        return history

    def _classified(self, candidates, t, actor, **details):
        if not candidates:
            return _unknown("empty_model_output", t, actor, candidates=[], **details)
        first = candidates[0]
        if first["score"] < self.min_score:
            return _unknown("below_confidence_threshold", t, actor, candidates=candidates, **details)
        return {**actor, **first, "kind": "learned_action", "timestamp": t, "evidence_timestamp": t,
                "candidates": candidates, **details}

    def _skeleton(self, inputs, frame):
        import numpy as np
        t = frame["timestamp"]
        tracks = self._tracks(inputs, frame, MAX_SKELETON_TRACKS)
        invalid = set()
        for track in tracks:
            key = _track_key(track)
            try:
                points = np.asarray(track.get("keypoints"), dtype=np.float32)
                valid = (points.shape == (17, 3) and np.isfinite(points).all()
                         and ((points[:, 2] >= 0) & (points[:, 2] <= 1)).all())
                visible = points[:, 2] > .1 if valid else np.zeros(17, bool)
                valid = valid and visible.sum() >= 6 and visible[[5, 6, 11, 12]].sum() >= 2
            except (ValueError, TypeError):
                valid = False
            if not valid:
                self.buffers.pop(key, None)
                invalid.add(key)
            else:
                self._append(key, t, points.copy(), MAX_SKELETON_FRAMES,
                             sampling_interval=self.window / (MAX_SKELETON_FRAMES - 2))
        if not self._due(t):
            if invalid or self._lost_tracks or self._history_reset:
                # Missing evidence is an observed change, not an unsampled
                # tick. A downstream condition must break its support timer.
                missing = [_unknown("missing_keypoints", t, _actor(track)) for track in tracks if _track_key(track) in invalid]
                if self._history_reset:
                    missing.extend(_unknown("warming_after_gap", t, _actor(track)) for track in tracks if _track_key(track) not in invalid)
                return self._packet(missing, frame, fresh=True)
            return self._packet([], frame, fresh=False)
        result = []
        for track in tracks:
            key, actor = _track_key(track), _actor(track)
            history = self.buffers.get(key, [])
            if key in invalid:
                result.append(_unknown("missing_keypoints", t, actor))
            elif len(history) < self.min_frames:
                result.append(_unknown("warming", t, actor, unique_frames=len(history), required_frames=self.min_frames))
            else:
                self._load()
                candidates = self.model.predict(np.stack([row[1] for row in history]), (frame["width"], frame["height"]))
                result.append(self._classified(candidates, t, actor, unique_frames=len(history),
                                               window_start=history[0][0], window_end=history[-1][0]))
        return self._packet(result, frame)

    @staticmethod
    def _small_image(image):
        from PIL import Image
        result = image.copy()
        result.thumbnail((MAX_VIDEO_SIDE, MAX_VIDEO_SIDE), Image.Resampling.BILINEAR)
        return result

    @staticmethod
    def _crop(image, track):
        box = track.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            values = [_number(value, "person crop coordinate", -1e9, 1e9) for value in box]
        except ValueError:
            return None
        x1, y1, x2, y2 = values
        w, h = image.size
        extent = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2)))
        return image.crop(extent) if extent[2] > extent[0] and extent[3] > extent[1] else None

    def _video_candidates(self, history, t):
        import numpy as np
        import torch
        from PIL import Image
        from .inference import classification_probabilities, prepare_action_tensor
        count = int(self.metadata["num_frames"])
        targets = np.linspace(t - self.window, t, count)
        selected = [min(history, key=lambda row: abs(row[0] - target)) for target in targets]
        # Genuine person crops may change dimensions. Match the newest crop's
        # dimensions before the existing model-specific spatial preparation.
        size = selected[-1][1].size
        pixels = np.stack([np.asarray(row[1].resize(size, Image.Resampling.BILINEAR)) for row in selected])
        tensor = prepare_action_tensor(pixels, self.metadata, self.actual_device)
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if self.actual_device.startswith("cuda") else nullcontext()
        with torch.inference_mode(), context:
            logits = self.model(pixel_values=tensor).logits if self.metadata["architecture"] == "vivit" else self.model(tensor)
            probabilities = classification_probabilities(logits, self.metadata["outputs_probabilities"])[0].cpu()
        scores, indices = probabilities.topk(min(5, probabilities.numel()))
        labels = self.metadata["labels"]
        return [{"class_id": int(index), "label": labels.get(int(index), str(int(index))), "score": float(score)}
                for score, index in zip(scores, indices)], len({row[0] for row in selected})

    def _video(self, inputs, frame):
        t, image = frame["timestamp"], frame["image"]
        if self.scope == "scene":
            actors = [("scene", None, image)]
            self._lost_tracks = False
        else:
            tracks = self._tracks(inputs, frame, self.max_people)
            actors = [(_track_key(track), track, self._crop(image, track)) for track in tracks]
        for key, track, crop in actors:
            if crop is None:
                self.buffers.pop(key, None)
                continue
            interval = self.window / (MAX_VIDEO_FRAMES - 2)
            history = self.buffers.get(key, [])
            if not history or t - history[-1][0] + 1e-9 >= interval:
                self._append(key, t, self._small_image(crop), MAX_VIDEO_FRAMES, sampling_interval=interval)
        self.diagnostics.update(max_buffer_frames=MAX_VIDEO_FRAMES, stored_max_side=MAX_VIDEO_SIDE)
        if not self._due(t):
            if self._lost_tracks or self._history_reset or any(crop is None for _, _, crop in actors):
                missing = [_unknown("invalid_person_crop", t, _actor(track)) for _, track, crop in actors if crop is None]
                if self._history_reset:
                    missing.extend(_unknown("warming_after_gap", t, _actor(track)) for _, track, crop in actors if crop is not None)
                return self._packet(missing, frame, fresh=True)
            return self._packet([], frame, fresh=False)
        result = []
        for key, track, crop in actors:
            actor, history = _actor(track), self.buffers.get(key, [])
            if crop is None:
                result.append(_unknown("invalid_person_crop", t, actor))
                continue
            if len(history) < 8 or history[0][0] > t - self.window + 1e-9:
                result.append(_unknown("warming", t, actor, unique_frames=len(history), window_seconds=self.window))
                continue
            self._load()
            candidates, unique = self._video_candidates(history, t)
            result.append(self._classified(candidates, t, actor, unique_frames=unique,
                                           sampled_frames=int(self.metadata["num_frames"]),
                                           window_start=t-self.window, window_end=history[-1][0]))
        return self._packet(result, frame)

    def _relation(self, inputs, frame):
        objects = _items(inputs, "objects", frame["timestamp"])
        if len(objects) < 2:
            return self._packet([], frame)
        from .workflow_relations import _track_key as relation_track_key
        active = {relation_track_key(item) for item in objects}
        lost = any((row.get("subject_track_key") is not None and row["subject_track_key"] not in active)
                   or (row.get("object_track_key") is not None and row["object_track_key"] not in active)
                   for row in self.display)
        if not self._due(frame["timestamp"]):
            # Explicitly invalidate vanished relation endpoints immediately.
            return self._packet([], frame, fresh=lost)
        import numpy as np
        self._load()
        bgr = np.ascontiguousarray(np.asarray(frame["image"])[:, :, ::-1])
        result = self.model.predict(bgr, objects, predicates=self.params.get("predicates"),
                                    threshold=self.min_score, top_k=int(self.params.get("top_k", 10)))
        self.diagnostics.update(self.model.last_diagnostics)
        for item in result:
            item.update(timestamp=frame["timestamp"], evidence_timestamp=frame["timestamp"])
        return self._packet(result, frame)
