"""Shared data audits and COCO evaluation for independent Bento research experiments.

This is a separate coarse-label task, not a reproduction of the unpublished
SlideShare training recipe or an improvement score for the original 91 classes.
No training is started on import. Test data are evaluated only after best-weight
selection on validation mAP50-95. Source annotations remain untouched.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "outputs/BentoCal-training/coarse-experiment-20261104"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolved_files(value, root):
    result = []
    for entry in value if isinstance(value, list) else [value]:
        p = Path(entry)
        p = p if p.is_absolute() else root / p
        p = p.resolve()
        if p.is_dir():
            result.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))
        elif p.suffix.lower() == ".txt" and p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Path(line.strip())
                    result.append(item.resolve() if item.is_absolute() else (p.parent / item).resolve())
        elif p.is_file():
            result.append(p)
        else:
            raise ValueError(f"Missing split input: {p}")
    if not result or any(not p.is_file() for p in result):
        raise ValueError("Every train/val/test split must contain existing image files")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate image paths inside split")
    return result


def audit_dataset(yaml_path):
    import yaml
    from PIL import Image, ImageOps
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "download" in config:
        raise ValueError("Dataset must be a local YAML without a download directive")
    names = config["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("This experiment requires exactly six distinct coarse classes")
    base = Path(config.get("path") or yaml_path.parent)
    base = base.resolve() if base.is_absolute() else (yaml_path.parent / base).resolve()
    splits = {s: resolved_files(config[s], base) for s in ("train", "val", "test")}
    manifest_path = yaml_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("The curated dataset manifest.json is required for group-split verification")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = {s: set() for s in splits}
    manifest_paths = {s: set() for s in splits}
    holdout_groups = set(manifest.get("grouping", {}).get("legacy_holdout_group_ids", []))
    for row in manifest["images"]:
        split = row["split"]
        if split not in splits:
            continue
        group = str(row["group_id"])
        if group in holdout_groups:
            raise ValueError("Legacy Bento validation group leaked into the new experiment")
        groups[split].add(group)
        entry = Path(row["image"])
        manifest_paths[split].add(entry.resolve() if entry.is_absolute() else (yaml_path.parent / entry).resolve())
    for split in splits:
        if set(splits[split]) != manifest_paths[split]:
            raise ValueError(f"YAML and manifest disagree for {split}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Group leakage: {left}/{right}")
    hashes = {}
    for split, images in splits.items():
        for path in images:
            with Image.open(path) as original:
                im = ImageOps.exif_transpose(original).convert("RGB")
                value = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            if value in hashes:
                raise ValueError(f"Exact RGB duplicate after curation: {hashes[value]} / {split}:{path}")
            hashes[value] = f"{split}:{path}"
    return names, splits, {"yaml": str(yaml_path), "yaml_sha256": digest(yaml_path),
                          "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
                          "split_counts": {s: len(v) for s, v in splits.items()},
                          "group_counts": {s: len(v) for s, v in groups.items()},
                          "group_disjoint": True, "exact_rgb_unique": len(hashes),
                          "excluded_legacy_group_count": len(holdout_groups)}


def ground_truth(images, names):
    from PIL import Image
    from ultralytics.data.utils import img2label_paths
    doc = {"info": {}, "images": [], "annotations": [],
           "categories": [{"id": i, "name": name} for i, name in enumerate(names)]}
    for image_id, path in enumerate(images, 1):
        with Image.open(path) as im:
            width, height = im.size
        doc["images"].append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        label = Path(img2label_paths([str(path)])[0])
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(map(math.isfinite, values)):
                raise ValueError(f"Detection labels must contain five finite numbers: {label}")
            cls, cx, cy, bw, bh = values
            if int(cls) != cls or not 0 <= cls < len(names) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid class or box in {label}")
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            doc["annotations"].append({"id": len(doc["annotations"]) + 1, "image_id": image_id,
                                       "category_id": int(cls), "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
    if not doc["annotations"]:
        raise ValueError("Evaluation split contains no ground truth")
    return doc


def coco_score(doc, detections):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = doc; gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO(); dt.dataset = {**doc, "annotations": []}; dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.catIds = [c["id"] for c in doc["categories"]]
        ev.params.iouThrs = np.array([round(.5 + .05 * i, 2) for i in range(10)])
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in ev.params.catIds}
    for row in ev.evalImgs:
        if row is None or list(row["aRng"]) != list(ev.params.areaRng[0]):
            continue
        support = int((~row["gtIgnore"].astype(bool)).sum())
        keep = (np.asarray(row["dtScores"]) >= .25) & ~row["dtIgnore"][0].astype(bool)
        tp = int(((row["dtMatches"][0] > 0) & keep).sum())
        fp = int(((row["dtMatches"][0] == 0) & keep).sum())
        c = counts[row["category_id"]]
        c["tp"] += tp; c["fp"] += fp; c["fn"] += support - tp; c["support"] += support
    def prf(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return {**c, "precision": tp / (tp + fp) if tp + fp else 0.,
                "recall": tp / (tp + fn) if tp + fn else 0.,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
    per_class = {}
    for i, category in enumerate(doc["categories"]):
        precision = ev.eval["precision"][:, :, i, 0, 2]
        p50 = precision[0]
        per_class[category["name"]] = {**prf(counts[category["id"]]),
            "map50_95": float(precision[precision > -1].mean()) if (precision > -1).any() else None,
            "map50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None}
    return {"map50": float(ev.stats[1]), "map50_95": float(ev.stats[0]), "coco_ar100": float(ev.stats[8]),
            "fixed_025_iou_050": {"micro": prf({k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn", "support")}),
                                 "per_class": per_class},
            "ap_confidence_floor": .001, "nms_iou": .7, "max_detections_per_image": 100,
            "threshold_selection": "Fixed at 0.25 before evaluation; no test-set threshold tuning"}
