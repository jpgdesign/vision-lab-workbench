"""Audits for an explicit, arbitrary-size food taxonomy; no training on import.

This module deliberately leaves the earlier six-class experiment unchanged.
Source-label consistency is not proof of complete or correct human annotation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .common import ROOT, coco_score as _coco_score, digest, now, resolved_files, save


def validate_names(config):
    """Require contiguous IDs and preserve the supplied names exactly."""
    supplied = config.get("names")
    if isinstance(supplied, dict):
        if any(str(key) != str(int(key)) for key in supplied):
            raise ValueError("Class-name dictionary keys must be integer IDs")
        indexed = {int(key): value for key, value in supplied.items()}
        if len(indexed) != len(supplied) or sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Class IDs must be contiguous from zero")
        names = [indexed[index] for index in range(len(indexed))]
    elif isinstance(supplied, list):
        names = list(supplied)
    else:
        raise ValueError("Dataset names must be a list or integer-keyed dictionary")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("At least one nonempty class name is required")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique, without silently renaming labels")
    if "nc" in config and (type(config["nc"]) is not int or config["nc"] != len(names)):
        raise ValueError("Dataset nc and explicit class names disagree")
    return names


def label_path(image):
    """Match native YOLO's last /images/ -> /labels/ convention."""
    parts = list(Path(image).parts)
    positions = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if not positions:
        raise ValueError(f"YOLO images must reside beneath an images directory: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(path, class_count):
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file is not an assumed negative: {path}")
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = tuple(float(value) for value in line.split())
        except ValueError as exc:
            raise ValueError(f"Non-numeric label at {path}:{number}") from exc
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Expected one finite YOLO bbox row at {path}:{number}")
        cls, x, y, width, height = values
        if cls != int(cls) or not 0 <= cls < class_count or width <= 0 or height <= 0:
            raise ValueError(f"Invalid class ID or box dimensions at {path}:{number}")
        if any(value < -1e-5 or value > 1 + 1e-5
               for value in (x - width / 2, y - height / 2, x + width / 2, y + height / 2)):
            raise ValueError(f"Box extends beyond image bounds at {path}:{number}")
        boxes.append((int(cls), x, y, width, height))
    if len(set(boxes)) != len(boxes):
        raise ValueError(f"Repeated identical label rows must be resolved before training: {path}")
    return boxes


def audit_dataset(yaml_path, allowed_untrained_classes=()):
    import yaml
    from PIL import Image, ImageOps

    yaml_path = Path(yaml_path).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or "download" in config:
        raise ValueError("A local dataset YAML without download directives is required")
    names = validate_names(config)
    base = Path(config.get("path") or yaml_path.parent)
    base = base.resolve() if base.is_absolute() else (yaml_path.parent / base).resolve()
    splits = {split: resolved_files(config[split], base) for split in ("train", "val", "test")}
    manifest_path = yaml_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("manifest.json beside the dataset YAML is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if "taxonomy" in manifest:
        if validate_names({"names": manifest["taxonomy"]}) != names:
            raise ValueError("Manifest taxonomy and dataset class-name ordering disagree")
    groups = {split: set() for split in splits}
    manifest_paths = {split: set() for split in splits}
    by_path = {}
    holdout_groups = set(manifest.get("grouping", {}).get("legacy_holdout_group_ids", []))
    for row in manifest["images"]:
        split = row["split"]
        if split not in splits:
            raise ValueError(f"Unexpected manifest split: {split}")
        if row.get("group_id") is None or not str(row["group_id"]).strip():
            raise ValueError("Every image requires an explicit nonempty group_id")
        group = str(row["group_id"])
        if group in holdout_groups:
            raise ValueError("An explicitly excluded legacy holdout group entered the dataset")
        path = Path(row["image"])
        path = path.resolve() if path.is_absolute() else (yaml_path.parent / path).resolve()
        if path in by_path:
            raise ValueError(f"Repeated image in manifest: {path}")
        by_path[path] = row
        groups[split].add(group)
        manifest_paths[split].add(path)
    for split, paths in splits.items():
        if set(paths) != manifest_paths[split]:
            raise ValueError(f"Dataset YAML and manifest images disagree for {split}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Related image group leaked across {left}/{right}")
    hashes, counts, label_hashes = {}, {}, {}
    for split, paths in splits.items():
        image_counts, box_counts = Counter(), Counter()
        empty = 0
        fingerprint = hashlib.sha256()
        for path in paths:
            with Image.open(path) as original:
                im = ImageOps.exif_transpose(original).convert("RGB")
                value = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            if value in hashes:
                raise ValueError(f"Exact RGB duplicates must be resolved first: {hashes[value]} / {split}:{path}")
            hashes[value] = f"{split}:{path}"
            label = label_path(path)
            boxes = read_boxes(label, len(names))
            present = {box[0] for box in boxes}
            if "class_ids" in by_path[path] and present != set(by_path[path]["class_ids"]):
                raise ValueError(f"Manifest class IDs and actual labels disagree: {path}")
            image_counts.update(present)
            box_counts.update(box[0] for box in boxes)
            empty += not boxes
            fingerprint.update(str(label).encode())
            fingerprint.update(b"\0")
            fingerprint.update(label.read_bytes())
            fingerprint.update(b"\0")
        counts[split] = {"images": len(paths), "objects": sum(box_counts.values()),
                         "empty_label_images": empty,
                         "per_class": {name: {"images": image_counts[i], "objects": box_counts[i]}
                                       for i, name in enumerate(names)},
                         "absent_classes": [name for i, name in enumerate(names) if not box_counts[i]]}
        label_hashes[split] = fingerprint.hexdigest()
        if not counts[split]["objects"]:
            raise ValueError(f"The {split} split has no annotated objects")
    allowed_untrained_classes = list(allowed_untrained_classes)
    if len(set(allowed_untrained_classes)) != len(allowed_untrained_classes):
        raise ValueError("The explicit untrained-class allowance contains duplicate names")
    missing_train = counts["train"]["absent_classes"]
    if set(missing_train) != set(allowed_untrained_classes):
        raise ValueError("Training-absent classes must exactly equal the explicit allowance. "
                         f"Actual missing={missing_train}; allowed={allowed_untrained_classes}. "
                         "Do not move held-out images into training to satisfy this check.")
    audit = {"yaml": str(yaml_path), "yaml_sha256": digest(yaml_path),
             "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
             "class_count": len(names), "classes": names,
             "split_counts": {split: len(paths) for split, paths in splits.items()},
             "group_counts": {split: len(value) for split, value in groups.items()},
             "label_set_sha256": label_hashes, "class_distribution": counts,
             "group_disjoint": True, "exact_rgb_unique": len(hashes),
             "excluded_legacy_group_count": len(holdout_groups),
             "allowed_untrained_classes": allowed_untrained_classes,
             "actual_train_absent_classes": missing_train,
             "train_absence_exactly_matches_allowance": True,
             "trained_classes": [name for name in names if name not in missing_train],
             "limits": ["Counts and consistency do not establish complete or semantically correct labels",
                        "Coarse and fine labels can overlap; this is a source-label baseline, not automatic ontology reconciliation",
                        "Empty labels are not confirmed negative scenes",
                        "Explicitly allowed train-absent classes have no positive training examples; their test metrics must not be advertised as learned capability",
                        "Classes absent from val/test have no AP/recall evidence on that split"]}
    return names, splits, audit


def bind_native_dataset(output, names, splits, audit):
    """Bind native training to exactly the files that passed the manifest audit."""
    import yaml
    output = Path(output).resolve()
    native = {"path": str(output), "nc": len(names), "names": dict(enumerate(names))}
    lists = {}
    for split, paths in splits.items():
        listing = output / f"audited_{split}_images.txt"
        listing.write_text("\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8")
        native[split] = str(listing)
        lists[split] = {"path": str(listing), "sha256": digest(listing)}
    yaml_path = output / "audited_dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(native, sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit.update(native_dataset_yaml=str(yaml_path), native_dataset_yaml_sha256=digest(yaml_path),
                 native_image_lists=lists)
    return yaml_path


def ground_truth(images, names):
    from PIL import Image, ImageOps
    doc = {"info": {}, "images": [], "annotations": [],
           "categories": [{"id": index, "name": name} for index, name in enumerate(names)]}
    for image_id, path in enumerate(images, 1):
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        doc["images"].append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        for cls, x, y, bw, bh in read_boxes(label_path(path), len(names)):
            box = [(x - bw / 2) * width, (y - bh / 2) * height, bw * width, bh * height]
            doc["annotations"].append({"id": len(doc["annotations"]) + 1, "image_id": image_id,
                                       "category_id": cls, "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
    return doc


def coco_score(doc, detections):
    result = _coco_score(doc, detections)
    per_class = result["fixed_025_iou_050"]["per_class"]
    for value in per_class.values():
        value["evaluation_supported"] = value["support"] > 0
        if not value["support"]:
            value["recall"] = None
            value["f1"] = None
    result.update(nms_iou=None,
                  postprocessing="Native YOLO26 end-to-end predictions; no additional NMS",
                  classes_without_ground_truth=[name for name, value in per_class.items() if not value["support"]],
                  macro_ap_scope="COCO AP averages only categories with ground truth in this split; absent categories are not validated")
    return result


def trained_classes_only_score(doc, detections, trained_classes):
    """An additional declared subset view; never replaces all-taxonomy scores."""
    kept_names = set(trained_classes)
    known_names = {category["name"] for category in doc["categories"]}
    if not kept_names or not kept_names <= known_names:
        raise ValueError("Trained-only metric categories must be a nonempty subset of the original taxonomy")
    categories = [category for category in doc["categories"] if category["name"] in kept_names]
    kept_ids = {category["id"] for category in categories}
    annotations = [row for row in doc["annotations"] if row["category_id"] in kept_ids]
    predictions = [row for row in detections if row["category_id"] in kept_ids]
    subset = {**doc, "categories": categories, "annotations": annotations}
    result = coco_score(subset, predictions)
    result["subset_protocol"] = {
        "scope": "Additional trained-classes-only diagnostic; all-taxonomy score is retained separately",
        "images_retained": len(doc["images"]), "kept_classes": [category["name"] for category in categories],
        "excluded_classes": [category["name"] for category in doc["categories"] if category["id"] not in kept_ids],
        "original_gt_objects": len(doc["annotations"]), "kept_gt_objects": len(annotations),
        "excluded_gt_objects": len(doc["annotations"]) - len(annotations),
        "original_predictions": len(detections), "kept_predictions": len(predictions),
        "excluded_predictions": len(detections) - len(predictions),
        "filter": "Only train-observed class IDs kept in both GT and predictions; every original image retained"}
    return result
