"""Detection class expansion and explicit, held-out old/new class evaluation.

The native trainers still own their losses and optimization. This module copies
semantic output rows before optimization, and uses the same COCO metric and
inference settings for the source and the selected best checkpoint.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path


def expansion_spec(request):
    if request.get("training_mode", "standard") not in {"expand_classes", "expand_frozen"}:
        return None
    spec = request.get("class_expansion")
    if not isinstance(spec, dict):
        raise ValueError("類別擴充缺少已驗證的 class_expansion。")
    base, classes = spec.get("base_classes"), spec.get("classes")
    if not isinstance(base, list) or not base or not isinstance(classes, list) or classes[:len(base)] != base:
        raise ValueError("擴充類別必須保留原類別的完整順序。")
    ids = spec.get("base_class_ids")
    if not isinstance(ids, list) or len(ids) != len(base) or len(set(ids)) != len(ids):
        raise ValueError("類別擴充缺少來源分類頭的明確 base_class_ids。")
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("來源類別 ID 必須是非負整數。")
    if request.get("dataset", {}).get("classes") != classes:
        raise ValueError("擴充資料集與分類頭的類別順序不同。")
    if len(classes) <= len(base):
        raise ValueError("擴充模式至少需要一個新類別。")
    if spec.get("new_class_ids") != list(range(len(base), len(classes))):
        raise ValueError("新類別 ID 必須接續原類別，不能重排或省略。")
    return spec


def transplant_yolo_rows(source, target, spec):
    """Run after official shape transfer, before the optimizer is constructed."""
    import torch
    old_head, new_head = source.model[-1], target.model[-1]
    ids, count = spec["base_class_ids"], len(spec["base_classes"])
    if new_head.nc != len(spec["classes"]) or max(ids) >= old_head.nc:
        raise ValueError("YOLO 擴充類別與來源／目標分類頭大小不一致。")
    names = source.names
    if [names[i] for i in ids] != spec["base_classes"]:
        raise ValueError("YOLO 來源權重的類別名稱與擴充映射不同。")
    copied = []
    with torch.no_grad():
        for branch in ("cv3", "one2one_cv3"):
            previous, expanded = getattr(old_head, branch, None), getattr(new_head, branch, None)
            if previous is None and expanded is None:
                continue
            if previous is None or expanded is None or len(previous) != len(expanded):
                raise ValueError("YOLO 分類分支架構不一致，不能保證原類別保留。")
            for scale, (old, new) in enumerate(zip(previous, expanded)):
                old, new = old[-1], new[-1]
                if old.weight.shape[1:] != new.weight.shape[1:] or old.bias is None or new.bias is None:
                    raise ValueError("YOLO 分類特徵維度變更，無法逐行保留來源輸出。")
                # The freshly built target retains its native initialization for
                # every new class; only explicitly mapped source rows are copied.
                for destination, origin in enumerate(ids):
                    new.weight[destination].copy_(old.weight[origin])
                    new.bias[destination].copy_(old.bias[origin])
                    if not torch.equal(new.weight[destination], old.weight[origin]) or not torch.equal(new.bias[destination], old.bias[origin]):
                        raise RuntimeError("YOLO 原類別分類行移植驗證失敗。")
                copied.append({"branch": branch, "scale": scale, "verified_rows": count})
    target.names = dict(enumerate(spec["classes"]))
    return {"model": "yolo", "old_rows_verified": count, "new_rows": len(spec["classes"]) - count,
            "heads": copied, "verification": "torch.equal for every copied weight and bias row"}


def expand_rfdetr_rows(network, spec):
    """RF-DETR uses sigmoid logits, sparse category IDs, and no softmax EOS row.

Category 0 is unused in this adapter's one-based COCO data. Preserve that row
separately; move semantic COCO rows (including row 90) by their explicit IDs.
Unmatched queries are trained against all-zero sigmoid targets in native IA-BCE.
"""
    import torch
    from torch import nn
    ids, classes = spec["base_class_ids"], spec["classes"]
    if not ids or 0 in ids:
        raise ValueError("RF-DETR 來源必須明確提供非零 category ID；row 0 不能當物件類別。")
    copied = []

    def expand(old, name):
        if not isinstance(old, nn.Linear) or old.bias is None or max(ids) >= old.weight.shape[0]:
            raise ValueError(f"RF-DETR {name} 不符合已驗證的來源分類頭映射。")
        result = nn.Linear(old.in_features, len(classes) + 1, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            # Native RF-DETR initializes class biases with a 1% sigmoid prior.
            result.bias.fill_(-math.log((1 - 0.01) / 0.01))
            pairs = [(0, 0), *[(index + 1, origin) for index, origin in enumerate(ids)]]
            for destination, origin in pairs:
                result.weight[destination].copy_(old.weight[origin])
                result.bias[destination].copy_(old.bias[origin])
                if not torch.equal(result.weight[destination], old.weight[origin]) or not torch.equal(result.bias[destination], old.bias[origin]):
                    raise RuntimeError("RF-DETR 原分類行移植驗證失敗。")
        result.requires_grad_(old.weight.requires_grad)
        copied.append({"head": name, "verified_rows": len(ids), "unused_row_0_verified": True})
        return result

    network.class_embed = expand(network.class_embed, "class_embed")
    if getattr(network, "two_stage", False):
        heads = network.transformer.enc_out_class_embed
        for index in range(len(heads)):
            heads[index] = expand(heads[index], f"transformer.enc_out_class_embed.{index}")
    return {"model": "rfdetr", "old_rows_verified": len(ids), "new_rows": len(classes) - len(ids),
            "class_id_offset": 1, "heads": copied,
            "no_object": "all-zero sigmoid target; no explicit softmax background row",
            "verification": "torch.equal for every copied weight and bias row"}


def yolo_predictor(model, settings, device, source_ids=None):
    mapping = None if source_ids is None else {origin: index for index, origin in enumerate(source_ids)}
    def predict(image):
        result = model.predict(image, device=str(device), verbose=False, imgsz=settings["imgsz"],
                               conf=settings["confidence"], iou=settings["nms_iou"], max_det=settings["max_detections"])[0]
        return [{"class_id": int(cls) if mapping is None else mapping[int(cls)], "score": float(score), "box": box.tolist()}
                for box, score, cls in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu())
                if mapping is None or int(cls) in mapping]
    return predict


def rfdetr_predictor(model, settings, source_ids=None):
    mapping = None if source_ids is None else {origin: index for index, origin in enumerate(source_ids)}
    def predict(image):
        model.model.model.eval()
        result = model.predict(image, threshold=settings["confidence"])
        return [{"class_id": int(cls) - 1 if mapping is None else mapping[int(cls)], "score": float(score), "box": box.tolist()}
                for box, score, cls in zip(result.xyxy, result.confidence, result.class_id)
                if (int(cls) > 0 if mapping is None else int(cls) in mapping)]
    return predict


def grounding_predictor(model, processor, caption, settings, device, *, caption_chunks=None):
    import torch
    from PIL import Image
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    chunks = caption_chunks if caption_chunks is not None else [{"caption": caption, "global_class_ids": None}]
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Grounding DINO 缺少完整文字提示。")
    def predict(image):
        model.eval()
        # Match the training square resize, using the same caption groups before
        # and after. Numeric phrase maps avoid ambiguous decoded text labels.
        resized = image.resize((settings["imgsz"], settings["imgsz"]), Image.Resampling.BILINEAR)
        predictions = []
        for chunk in chunks:
            inputs = processor(images=resized, text=chunk["caption"], do_resize=False, return_tensors="pt").to(device)
            if caption_chunks is not None:
                maximum = model.config.max_text_len
                if inputs["input_ids"].shape[1] > maximum:
                    raise ValueError("分組文字提示超過已載入 Grounding DINO 的上限，不能截斷類別。")
            with torch.inference_mode():
                output = model(**inputs)
            phrase_map = build_label_maps(output.logits, inputs["input_ids"])[0]
            ids = chunk["global_class_ids"]
            if ids is not None and phrase_map.shape[0] != len(ids):
                raise ValueError("Grounding DINO 分組詞組與全域類別對應不一致。")
            phrase_map = phrase_map.to(output.logits.dtype)
            phrase_map = phrase_map / phrase_map.sum(-1, keepdim=True).clamp(min=1)
            scores = output.logits[0].sigmoid() @ phrase_map.T
            scores, labels = scores.max(-1)
            boxes = output.pred_boxes[0]
            boxes = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)
            boxes *= boxes.new_tensor([image.width, image.height, image.width, image.height])
            predictions.extend({"class_id": int(label) if ids is None else ids[int(label)],
                                "score": float(score), "box": box.cpu().tolist()}
                               for box, score, label in zip(boxes, scores, labels) if float(score) >= settings["confidence"])
        return predictions
    return predict


class DetectionRetention:
    def __init__(self, request, output, params):
        self.spec = expansion_spec(request)
        self.model_id = request.get("model_id")
        self.mode = request.get("training_mode", "standard")
        self.output = Path(output)
        self.datasets = request.get("evaluation_datasets", {})
        self.has_old = bool(self.datasets.get("old"))
        if self.spec and (not self.datasets.get("new") or (not self.has_old and self.mode != "expand_frozen")):
            raise ValueError("類別擴充必須包含獨立 old/new 驗證資料，不能以訓練資料代替。")
        self.settings = {"imgsz": params["imgsz"], "confidence": 0.001, "nms_iou": 0.7,
                         "max_detections": 100, "iou_thresholds": [round(0.5 + i * 0.05, 2) for i in range(10)]}
        self.before = None

    def evaluate(self, split, predict):
        if split == "old" and not self.has_old:
            return None
        import numpy as np
        import torch
        from PIL import Image
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from torchvision.ops import batched_nms
        from .training_grounding import _detection_samples
        classes = self.spec["classes"]
        dataset = self.datasets[split]
        if dataset["classes"] != classes:
            raise ValueError("保留能力評估資料的 union 類別順序不同。")
        _, samples = _detection_samples(dataset)
        selected = list(range(len(self.spec["base_classes"]))) if split == "old" else list(self.spec["new_class_ids"])
        support = {i: 0 for i in selected}
        document = {"info": {}, "images": [], "annotations": [],
                    "categories": [{"id": i, "name": classes[i]} for i in selected]}
        detections = []
        for image_id, sample in enumerate(samples, 1):
            with Image.open(sample["path"]) as original:
                image = original.convert("RGB")
            width, height = image.size
            document["images"].append({"id": image_id, "width": width, "height": height})
            for label, (cx, cy, bw, bh) in zip(sample["labels"], sample["boxes"]):
                if label not in support:
                    continue
                support[label] += 1
                document["annotations"].append({"id": len(document["annotations"]) + 1, "image_id": image_id,
                    "category_id": label, "bbox": [(cx-bw/2)*width, (cy-bh/2)*height, bw*width, bh*height],
                    "area": bw*bh*width*height, "iscrowd": 0})
            rows = predict(image)
            valid = []
            for row in rows:
                box = row["box"]
                if (int(row["class_id"]) not in support or len(box) != 4 or
                        not all(math.isfinite(float(v)) for v in [*box, row["score"]]) or
                        float(row["score"]) < self.settings["confidence"]):
                    continue
                x1, y1, x2, y2 = box
                clipped = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
                           max(0.0, min(width, x2)), max(0.0, min(height, y2))]
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    valid.append({**row, "box": clipped})
            if valid:
                boxes = torch.tensor([r["box"] for r in valid], dtype=torch.float32)
                scores = torch.tensor([r["score"] for r in valid], dtype=torch.float32)
                labels = torch.tensor([r["class_id"] for r in valid], dtype=torch.int64)
                keep = batched_nms(boxes, scores, labels, self.settings["nms_iou"])[:self.settings["max_detections"]]
                for index in keep.tolist():
                    x1, y1, x2, y2 = valid[index]["box"]
                    detections.append({"image_id": image_id, "category_id": valid[index]["class_id"],
                                       "score": valid[index]["score"], "bbox": [x1, y1, x2-x1, y2-y1]})
        missing = [classes[i] for i, count in support.items() if not count]
        summary = {"sample_count": len(samples), "annotation_count": sum(support.values()),
                   "class_count": sum(bool(n) for n in support.values()), "requested_class_count": len(selected),
                   "missing_classes": missing, "class_support": {classes[i]: n for i, n in support.items()}}
        if not sum(support.values()):
            return {**summary, "map50": None, "map50_95": None, "status": "no_ground_truth_for_requested_classes"}
        with contextlib.redirect_stdout(io.StringIO()):
            ground_truth = COCO()
            ground_truth.dataset = document
            ground_truth.createIndex()
            if detections:
                predictions = ground_truth.loadRes(detections)
            else:
                predictions = COCO()
                predictions.dataset = {**document, "annotations": []}
                predictions.createIndex()
            evaluator = COCOeval(ground_truth, predictions, "bbox")
            evaluator.params.catIds = selected
            evaluator.params.iouThrs = np.array(self.settings["iou_thresholds"])
            evaluator.params.maxDets = [1, 10, self.settings["max_detections"]]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        return {**summary, "map50": float(evaluator.stats[1]), "map50_95": float(evaluator.stats[0]), "status": "evaluated"}

    def baseline(self, predict):
        from .training import _write_json
        self.before = {"old": self.evaluate("old", predict)}
        _write_json(self.output / "retention_before.json", {"before": self.before, "settings": self.settings})

    def finish(self, predict):
        from .training import _write_json
        if self.before is None:
            raise RuntimeError("類別擴充缺少訓練前保留能力基準。")
        after = {split: self.evaluate(split, predict) for split in ("old", "new")}
        before_score = self.before["old"]["map50_95"] if self.before["old"] is not None else None
        after_score = after["old"]["map50_95"] if after["old"] is not None else None
        warnings = [f"{split} 驗證缺少 {len(metrics['missing_classes'])} 類；指標只包含有標註的類別。"
                    for split, metrics in after.items() if metrics is not None and metrics["missing_classes"]]
        if not self.has_old:
            warnings.append("未提供舊類別驗證集：舊能力與變化尚未量測，不能宣稱已保留。")
        if self.mode == "expand_frozen":
            warnings.append("凍結擴充固定骨幹；分類／定位頭仍會更新，不保證舊任務準確率不下降。")
        result = {"metric": "map50_95", "training_mode": self.mode, "before": self.before, "after": after,
                  "delta": {"old": None if before_score is None or after_score is None else after_score-before_score},
                  "scope": ("指定 old/new 驗證集" if self.has_old else "僅新驗證集；舊能力未量測") + ("；僅代表保存的 union caption，非全部開放詞彙。" if self.model_id == "grounding_dino_base" else "。"),
                  "warnings": warnings, "settings": self.settings,
                  "coverage": {split: None if metrics is None else {key: metrics[key] for key in ("sample_count", "annotation_count", "class_count", "requested_class_count", "missing_classes")}
                               for split, metrics in after.items()}}
        _write_json(self.output / "retention.json", result)
        return result
