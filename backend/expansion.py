"""Preflight and immutable dataset preparation for supervised class expansion.

No ML tensors are loaded into the API process. YOLO label inspection runs in a
short-lived CPU subprocess. Every job recomputes its plan from validated inputs.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

COCO_NAMES = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(",")
COCO_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]


def _names(value):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError("類別必須是非空名稱清單。")
    if len({x.strip().casefold() for x in value}) != len(value):
        raise ValueError("類別名稱重複（忽略大小寫及前後空白）。")
    return list(value)


@functools.lru_cache(maxsize=16)
def _yolo_names(path, size, modified):
    result = subprocess.run([sys.executable, "-m", "backend.expansion", "--inspect-yolo", path],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8",
        timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "YOLO_OFFLINE": "true", "PYTHONUTF8": "1"})
    if result.returncode:
        raise ValueError("無法讀取 YOLO 來源類別：" + result.stderr[-1500:])
    line = next((x for x in reversed(result.stdout.splitlines()) if x.startswith("CLASS_NAMES=")), None)
    if line is None:
        raise ValueError("YOLO 來源沒有可用的類別對照表。")
    return _names(json.loads(line.split("=", 1)[1]))


def source_classes(request):
    from .model_registry import checkpoint_metadata, validate_checkpoint, validate_weight
    model_id = request["model_id"]
    if request.get("checkpoint_path"):
        path = validate_checkpoint(model_id, request["checkpoint_path"])
        classes = _names(checkpoint_metadata(path)["classes"])
        return classes, list(range(1, len(classes) + 1)) if model_id == "rf_detr_2xl" else list(range(len(classes)))
    if model_id == "grounding_dino_base":
        classes = _names(request.get("replay_dataset", {}).get("classes"))
        return classes, list(range(len(classes)))
    if model_id == "dinov3_7b":
        raise ValueError("原始 DINOv3 是表徵模型，沒有舊分類頭；請先用一般微調建立分類器，再選擇該 checkpoint 擴充。")
    if model_id in {"yolov12x", "yolo26x"}:
        path = validate_weight(model_id, request["model_weight_path"])
        stat = path.stat()
        classes = _yolo_names(str(path), stat.st_size, stat.st_mtime_ns)
    elif model_id == "rf_detr_2xl":
        return list(COCO_NAMES), list(COCO_IDS)
    elif model_id in {"x3d_l", "slowfast_r101_8x8", "vivit_base_k400"}:
        from .inference import kinetics400_labels
        labels = kinetics400_labels()
        if model_id == "vivit_base_k400":
            path = validate_weight(model_id, request["model_weight_path"])
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            values = config.get("id2label", {})
            if values and not all(str(x).startswith("LABEL_") for x in values.values()):
                labels = {int(k): v for k, v in values.items()}
        classes = [labels[i] for i in range(len(labels))]
    elif model_id == "videomaev2_giant_k710":
        classes = (Path(__file__).parent / "assets/kinetics710.txt").read_text(encoding="utf-8").splitlines()
    else:
        raise ValueError("此模型尚無類別擴充接頭。")
    return _names(classes), list(range(len(classes)))


def dataset_samples(dataset):
    if dataset["format"] in {"yolo_detection", "coco_detection"}:
        from .training_grounding import _detection_samples
        return dict(zip(("train", "val"), _detection_samples(dataset)))
    return {split: copy.deepcopy(dataset["details"][f"{split}_samples"]) for split in ("train", "val")}


def _coverage(samples, classes, mapping, expected):
    counts = {name: 0 for name in expected}
    for sample in samples:
        for label in sample.get("labels", [sample.get("label")]):
            name = classes[mapping[int(label)]]
            counts[name] = counts.get(name, 0) + 1
    return {"counts": counts, "missing": [name for name in expected if not counts[name]],
            "sample_count": len(samples), "represented_classes": sum(v > 0 for v in counts.values())}


def preview_expansion(request):
    errors, warnings = [], []
    plan = {}
    try:
        if request.get("training_mode", "standard") == "standard":
            if request.get("replay_dataset") or request.get("class_aliases"):
                raise ValueError("舊資料重播與類別別名僅可用於類別擴充模式。")
            return {"valid": True, "errors": [], "warnings": ["一般微調使用新資料集的類別輸出，不會自動保留原類別。"], "plan": {}}
        replay, new = request.get("replay_dataset"), request["dataset"]
        if not replay:
            raise ValueError("類別擴充需要已驗證的舊任務資料集（含獨立 train / val）。")
        detection = new["format"] in {"yolo_detection", "coco_detection"}
        if (detection and replay["format"] not in {"yolo_detection", "coco_detection"}) or (not detection and replay["format"] != new["format"]):
            raise ValueError("新舊資料集必須是相同任務；偵測可混用 YOLO 與 COCO。")
        base, ids = source_classes(request)
        base = _names(base)
        old_names, new_names = _names(replay["classes"]), _names(new["classes"])
        lookup = {name.strip().casefold(): i for i, name in enumerate(base)}
        old_map = []
        for name in old_names:
            if name.strip().casefold() not in lookup:
                raise ValueError(f"舊資料集類別 {name!r} 不在來源模型類別中；請修正標註名稱。")
            old_map.append(lookup[name.strip().casefold()])
        aliases = request.get("class_aliases", {})
        if not isinstance(aliases, dict) or any(key not in new_names or value not in base for key, value in aliases.items()):
            raise ValueError("類別別名只能把新資料集的現有名稱對應至來源模型的現有名稱。")
        classes, new_map = list(base), []
        for name in new_names:
            target = aliases.get(name, name)
            key = target.strip().casefold()
            if key not in lookup:
                lookup[key] = len(classes)
                classes.append(target)
            new_map.append(lookup[key])
        old_samples, new_samples = dataset_samples(replay), dataset_samples(new)
        coverage = {}
        for label, samples, mapping, expected in (("old", old_samples, old_map, base), ("new", new_samples, new_map, [classes[i] for i in new_map])):
            for split in ("train", "val"):
                coverage[f"{label}_{split}"] = _coverage(samples[split], classes, mapping, expected)
        plan = {"base_classes": base, "base_class_ids": ids, "classes": classes,
                "old_class_ids": list(range(len(base))), "new_class_ids": list(range(len(base), len(classes))),
                "old_label_map": old_map, "new_label_map": new_map, "coverage": coverage,
                "class_aliases": aliases, "source_dataset_ids": {"old": replay.get("id"), "new": new.get("id")}}
        if len(classes) == len(base):
            errors.append("沒有新增類別；若只訓練原有類別，請使用一般微調。")
        for part in ("old_train", "old_val"):
            missing = coverage[part]["missing"]
            if missing:
                errors.append(f"{part} 缺少 {len(missing)} 個舊類別樣本：" + ", ".join(missing[:12]))
        if coverage["new_train"]["missing"]:
            errors.append("新訓練資料有類別沒有任何標註：" + ", ".join(coverage["new_train"]["missing"][:12]))
        if coverage["new_val"]["missing"]:
            warnings.append("新驗證資料未涵蓋所有類別，指標無法代表缺少的類別：" + ", ".join(coverage["new_val"]["missing"][:12]))
        # Detect cross-dataset leakage before creating any files. Content hashes
        # also catch the same image copied into different source folders.
        split_keys = {}
        for split in ("train", "val"):
            keys = set()
            for sample in old_samples[split] + new_samples[split]:
                path = Path(sample["path"]).resolve()
                keys.add(("path", os.path.normcase(str(path))))
                if detection or new["format"] == "imagefolder":
                    keys.add(("sha256", _digest(str(path), path.stat().st_size, path.stat().st_mtime_ns)))
            split_keys[split] = keys
        if split_keys["train"] & split_keys["val"]:
            errors.append("新舊資料混合後 train / val 有重複檔案或相同影像內容，會造成驗證洩漏；請重新切分。")
        warnings.append("保留分類權重與舊資料混合可降低遺忘，但不保證舊任務準確率不下降；請比較訓練前後指標。")
        if detection:
            warnings.append("混合訓練影像中出現的舊類別與新類別都應完整標註；漏標物件會被當作背景。")
        if request["model_id"] == "grounding_dino_base":
            warnings.append("Grounding DINO 的舊能力評估僅涵蓋舊資料集提供的文字類別，不能代表所有開放詞彙。")
        plan["warnings"] = warnings
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "warnings": warnings, "plan": plan}


@functools.lru_cache(maxsize=16384)
def _digest(path, size, modified):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remap(samples, mapping):
    result = copy.deepcopy(samples)
    for sample in result:
        if "labels" in sample:
            sample["labels"] = [mapping[i] for i in sample["labels"]]
        else:
            sample["label"] = mapping[sample["label"]]
    return result


def _materialize(samples, classes, directory, format):
    """Copy independent data into the job: training never edits source labels."""
    import yaml
    directory.mkdir(parents=True, exist_ok=True)
    details = {"root": str(directory)}
    for split, values in samples.items():
        images, mapped = [], []
        for index, sample in enumerate(values):
            source = Path(sample["path"])
            target = directory / split / "images" / f"{index:07d}{source.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Native detection loaders can repair JPEGs on disk. Never link
            # images to source data; only read-only video decoding uses links.
            if format == "video_csv":
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
            if format == "yolo_detection":
                label = directory / split / "labels" / f"{index:07d}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("\n".join(f"{i} " + " ".join(str(v) for v in box) for i, box in zip(sample["labels"], sample["boxes"])), encoding="utf-8")
                images.append(str(target))
            else:
                mapped.append({"path": str(target), "label": sample["label"]})
        details[f"{split}_images" if format == "yolo_detection" else f"{split}_samples"] = images if format == "yolo_detection" else mapped
    path = directory
    if format == "yolo_detection":
        path = directory / "dataset.yaml"
        path.write_text(yaml.safe_dump({"path": str(directory), "train": "train/images", "val": "val/images", "names": classes}, allow_unicode=True), encoding="utf-8")
        details["yaml_path"] = str(path)
    result = {"valid": True, "id": directory.name, "name": directory.name, "path": str(path), "format": format,
              "classes": classes, "train_count": len(samples["train"]), "val_count": len(samples["val"]), "details": details}
    (directory / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def prepare_expansion(request, output, emit):
    checked = preview_expansion(request)
    if not checked["valid"]:
        raise ValueError("類別擴充檢查失敗：" + "；".join(checked["errors"]))
    if request.get("training_mode", "standard") == "standard":
        return request
    plan = checked["plan"]
    emit({"event": "log", "message": f"類別擴充：保留 {len(plan['base_classes'])} 類，新增 {len(plan['new_class_ids'])} 類；準備獨立舊／新驗證集。"})
    mapped = {key: {split: _remap(samples, plan[f"{key}_label_map"]) for split, samples in dataset_samples(request["replay_dataset"] if key == "old" else request["dataset"]).items()} for key in ("old", "new")}
    format = "yolo_detection" if request["dataset"]["format"] in {"yolo_detection", "coco_detection"} else request["dataset"]["format"]
    root = Path(output) / "expansion_data"
    merged = {split: mapped["old"][split] + mapped["new"][split] for split in ("train", "val")}
    result = {**request, "class_expansion": plan, "dataset": _materialize(merged, plan["classes"], root / "mixed", format),
              "evaluation_datasets": {key: _materialize(samples, plan["classes"], root / key, format) for key, samples in mapped.items()}}
    (Path(output) / "class_expansion.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--inspect-yolo":
        from ultralytics import YOLO
        names = YOLO(sys.argv[2]).names
        print("CLASS_NAMES=" + json.dumps([names[i] for i in range(len(names))], ensure_ascii=False))
