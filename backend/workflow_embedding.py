"""Persistent DINOv3 CLS features and optional trained image classification."""
from __future__ import annotations

import math


class EmbeddingNode:
    def __init__(self, params, resolved, device="cpu"):
        self.params, self.resolved, self.actual_device = dict(params), dict(resolved), str(device)
        if self.resolved.get("model_id", self.params.get("model_id", "dinov3_7b")) != "dinov3_7b":
            raise ValueError("目前的視覺表徵接頭只支援 DINOv3。")
        self.every = self._number(params.get("every_seconds", 1.), "every_seconds", .1, 120)
        self.max_gap = self._number(params.get("max_gap_seconds", 5.), "max_gap_seconds", .1, 120)
        self.minimum = self._number(params.get("min_score", .25), "min_score", 0, 1)
        self.top_k = int(self._number(params.get("top_k", 5), "top_k", 1, 20))
        if self.top_k != params.get("top_k", 5):
            raise ValueError("top_k 必須為整數。")
        if type(params.get("normalize", True)) is not bool:
            raise ValueError("normalize 必須為勾選值。")
        self.normalize = params.get("normalize", True)
        self.model = self.processor = self.metadata = None
        self.reset()

    @staticmethod
    def _number(value, name, low, high):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} 超出有效範圍。")
        return float(value)

    def reset(self):
        self.context = self.last_time = self.last_run = None
        self.display_features, self.display_actions = [], []

    def close(self):
        self.reset()
        self.model = self.processor = self.metadata = None

    def _load(self):
        if self.model is None:
            from .inference import create_dinov3_model
            self.model, self.processor, self.metadata = create_dinov3_model(
                self.actual_device, self.resolved.get("checkpoint_path"), self.resolved.get("model_weight_path"))

    def _packets(self, frame, features=None, actions=None, fresh=True):
        if fresh:
            self.display_features = [{key: value for key, value in item.items() if key != "vector"} for item in features or []]
            self.display_actions = list(actions or [])
        def packet(items, display):
            return {"items": items or [], "timestamp": frame["timestamp"], "frame": frame, "fresh": fresh,
                    "display_items": [{**item, "stale": not fresh} for item in display],
                    "diagnostics": {"model_id": "dinov3_7b", "actual_device": self.actual_device, "loaded": self.model is not None}}
        return {"embeddings": packet(features, self.display_features), "actions": packet(actions, self.display_actions)}

    def process(self, inputs, frame):
        from PIL import Image
        image = frame.get("image")
        if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (frame.get("width"), frame.get("height")):
            raise ValueError("視覺表徵需要尺寸一致的 RGB 影格。")
        source = frame.get("source_id")
        if not isinstance(source, str) or not source:
            raise ValueError("視覺表徵需要有效來源 ID。")
        stamp = self._number(frame.get("timestamp"), "timestamp", 0, 1e15)
        context = (source, frame.get("session_id"), image.size, repr(frame.get("to_source")))
        if context != self.context or (self.last_time is not None and (stamp < self.last_time or stamp - self.last_time > self.max_gap)):
            self.reset()
        duplicate = self.last_time == stamp
        self.context, self.last_time = context, stamp
        if duplicate or self.last_run is not None and stamp - self.last_run + 1e-9 < self.every:
            return self._packets(frame, fresh=False)
        self._load()
        import torch
        from .training_dinov3 import prepare_image_pixels
        pixels = prepare_image_pixels(image, self.processor, self.actual_device)
        classified = self.metadata.get("mode") == "classification"
        with torch.inference_mode():
            backbone = self.model.backbone if classified else self.model
            features = backbone(pixel_values=pixels).pooler_output
            if features is None or features.ndim != 2 or features.shape[0] != 1 or not 1 <= features.shape[1] <= 32768 or not torch.isfinite(features).all():
                raise ValueError("DINOv3 未回傳有效且有限的 CLS 向量。")
            logits = self.model.classifier(features) if classified else None
        vector = features[0].detach().float().cpu()
        if self.normalize:
            vector = vector / vector.norm().clamp(min=1e-12)
        feature = {"model_id": "dinov3_7b", "pooling": "cls", "dimensions": int(vector.numel()),
                   "normalized": self.normalize, "vector": vector.tolist(), "timestamp": stamp, "evidence_timestamp": stamp}
        actions = [{"scope": "scene", "label": "unknown", "score": 0., "reason": "embedding_only",
                    "timestamp": stamp, "evidence_timestamp": stamp}]
        if classified:
            labels = self.metadata.get("labels")
            if logits is None or logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all() or not isinstance(labels, (list, dict)) or len(labels) != logits.shape[1]:
                raise ValueError("DINOv3 分類輸出與 checkpoint 標籤不一致。")
            scores, indices = torch.softmax(logits.float(), dim=-1)[0].cpu().topk(min(self.top_k, len(labels)))
            candidates = []
            for score, index in zip(scores, indices):
                index = int(index)
                label = labels[index] if isinstance(labels, list) else labels.get(index, labels.get(str(index)))
                if not isinstance(label, str) or not label:
                    raise ValueError("DINOv3 分類標籤無效。")
                candidates.append({"label": label, "score": float(score), "class_id": index})
            top = candidates[0]
            actions = [{"scope": "scene", "label": top["label"] if top["score"] >= self.minimum else "unknown",
                        "score": top["score"] if top["score"] >= self.minimum else 0., "candidates": candidates,
                        "reason": "image_classifier" if top["score"] >= self.minimum else "below_confidence_threshold",
                        "timestamp": stamp, "evidence_timestamp": stamp}]
        self.last_run = stamp
        return self._packets(frame, [feature], actions)
