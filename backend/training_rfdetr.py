"""Offline, supervised RF-DETR 2XL fine-tuning using its native trainer.

Inputs are registered and validated datasets. Exported checkpoints carry a
one-based category map, matching the native Roboflow COCO trainer and predictor.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from .model_licenses import is_model_license_accepted
from .system_info import available_device_options


MODEL_ID = "rf_detr_2xl"


def training_capabilities() -> dict:
    from .training import _parameter
    p = _parameter
    params = [
        p("epochs", "訓練輪數", 10, min=1, max=10000, step=1),
        p("batch_size", "每批樣本數", 1, min=1, max=256, step=1),
        p("imgsz", "輸入圖片尺寸（40 倍數）", 320, min=160, max=1280, step=40),
        p("lr", "初始學習率", 0.0001, min=1e-9, max=1, step=0.00001),
        p("lr_encoder", "主幹學習率（完整微調時）", 0.00005, min=1e-9, max=1, step=0.00001),
        p("optimizer", "梯度最佳化器", "AdamW", "select", options=["AdamW"]),
        p("loss", "損失函數", "rf_detr_composite", "select", options=["rf_detr_composite"]),
        p("scheduler", "學習率排程", "cosine", "select", options=["cosine", "step"]),
        p("strategy", "微調策略", "freeze_backbone", "select", options=["freeze_backbone", "full"]),
        p("weight_decay", "權重衰減", 0.0001, min=0, max=1, step=0.0001),
        p("patience", "提前停止耐心輪數（0 停用）", 5, min=0, max=10000, step=1),
        p("seed", "隨機種子", 42, min=0, max=2147483647, step=1),
        p("device", "運算裝置", "auto", "select", options=available_device_options()),
        p("amp", "CUDA 混合精度（BF16）", True, "boolean"),
        p("workers", "資料載入工作程序", 0, min=0, max=16, step=1),
        p("accumulate", "梯度累積批數", 1, min=1, max=256, step=1),
        p("augmentation", "訓練資料增強（翻轉與隨機裁切）", True, "boolean"),
        p("warmup_epochs", "暖身輪數", 0.0, min=0, max=1000, step=0.5),
        p("lrf", "Cosine 最後學習率比例", 0.01, min=0, max=1, step=0.01),
        p("step_size", "Step 排程衰減輪數", 5, min=1, max=10000, step=1),
        p("gradient_clip", "梯度範數上限（0 停用）", 0.1, min=0, max=1000, step=0.1),
        p("box", "L1 定位損失權重", 5.0, min=0, max=1000, step=0.1),
        p("giou", "GIoU 損失權重", 2.0, min=0, max=1000, step=0.1),
        p("cls", "分類損失權重", 1.0, min=0, max=1000, step=0.1),
        p("group_detr", "訓練查詢群組數（較大需更多記憶體）", 1, min=1, max=13, step=1),
        p("gradient_checkpointing", "主幹梯度檢查點（節省顯示記憶體）", False, "boolean"),
        p("accept_platform_license", "我已閱讀並接受 RF-DETR Platform Model License", is_model_license_accepted(MODEL_ID), "boolean"),
    ]
    return {"dataset_formats": ["yolo_detection", "coco_detection"], "optimizers": ["AdamW"],
            "losses": ["rf_detr_composite"], "schedulers": ["cosine", "step"],
            "strategies": ["freeze_backbone", "full"],
            "defaults": {x["key"]: x["default"] for x in params}, "parameters": params,
            "supported_params": [x["key"] for x in params],
            "early_stopping_monitor": {"metric": "val mAP50–95", "mode": "max", "label": "驗證 mAP50–95（越高越好）"}}


def normalize_training_params(params: dict | None) -> dict:
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params 必須是 JSON 物件。")
    caps = training_capabilities()
    specs = {p["key"]: p for p in caps["parameters"]}
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"RF-DETR 不支援參數：{', '.join(sorted(unknown))}")
    result = {**caps["defaults"], **params}
    integers = {"epochs", "batch_size", "imgsz", "patience", "seed", "workers", "accumulate", "step_size", "group_detr"}
    for key, value in result.items():
        spec = specs[key]
        if spec["type"] == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必須是 true / false。")
        elif spec["type"] == "select":
            if value not in spec["options"]:
                raise ValueError(f"{key} 不支援 {value!r}。")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} 必須是有限數字。")
            if not spec["min"] <= value <= spec["max"] or (key in integers and int(value) != value):
                raise ValueError(f"{key} 超出允許範圍或不是整數。")
            result[key] = int(value) if key in integers else float(value)
    if result["imgsz"] % 40:
        raise ValueError("RF-DETR 2XL 圖片尺寸必須是 40 的倍數。")
    if caps["defaults"]["accept_platform_license"]:
        result["accept_platform_license"] = True
    if not result["accept_platform_license"]:
        raise ValueError("請先閱讀並接受 RF-DETR Platform Model License。")
    if result["box"] + result["giou"] + result["cls"] == 0:
        raise ValueError("至少一個損失權重必須大於零。")
    return result


def _prepare_coco(dataset: dict, output: Path, emit) -> Path:
    """Copy validated images and convert both formats to contiguous 1-based COCO."""
    from PIL import Image
    from .datasets import _inside, validate_dataset
    from .training import _write_json
    check = validate_dataset(dataset["path"], dataset["format"])
    if not check["valid"]:
        raise ValueError("資料集已變更或不再有效：" + "; ".join(check["errors"]))
    fresh = check["dataset"]
    if fresh["classes"] != dataset["classes"]:
        raise ValueError("資料集類別在驗證後已變更，請重新驗證。")
    details, classes = fresh["details"], fresh["classes"]
    root = Path(details["root"]).resolve()
    target = output / "rf_dataset"
    docs = {}
    total_bytes = 0
    for split, native_split in (("train", "train"), ("val", "valid")):
        doc = {"info": {"description": "Workbench supervised training data"}, "licenses": [],
               "images": [], "annotations": [],
               "categories": [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(classes)]}
        sources = []
        if fresh["format"] == "coco_detection":
            original = json.loads(_inside(Path(details[f"{split}_json"]), root).read_text(encoding="utf-8-sig"))
            mapping = {int(k): int(v) + 1 for k, v in details["category_id_to_index"].items()}
            image_map = {}
            for index, entry in enumerate(original["images"], 1):
                source = _inside(Path(details[f"{split}_images"][str(entry["id"])]), root)
                image_map[entry["id"]] = index
                name = f"{index:08d}{source.suffix.lower()}"
                sources.append((source, name))
                doc["images"].append({"id": index, "file_name": name, "width": entry["width"], "height": entry["height"]})
            for index, entry in enumerate(original["annotations"], 1):
                box = [float(v) for v in entry["bbox"]]
                doc["annotations"].append({"id": index, "image_id": image_map[entry["image_id"]],
                    "category_id": mapping[entry["category_id"]], "bbox": box, "area": box[2] * box[3],
                    "iscrowd": int(entry.get("iscrowd", 0))})
        else:
            for index, image_name in enumerate(details[f"{split}_images"], 1):
                source = _inside(Path(image_name), root)
                with Image.open(source) as image:
                    width, height = image.size
                name = f"{index:08d}{source.suffix.lower()}"
                sources.append((source, name))
                doc["images"].append({"id": index, "file_name": name, "width": width, "height": height})
                label_parts = list(source.relative_to(root).parts)
                label_parts[max(i for i, v in enumerate(label_parts) if v == "images")] = "labels"
                label = _inside(Path(*label_parts).with_suffix(".txt"), root)
                for line in label.read_text(encoding="utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    values = line.split()
                    cx, cy, bw, bh = map(float, values[1:])
                    # Clamp only tiny boundary tolerance already accepted by validation.
                    x1, y1 = max(0.0, (cx - bw / 2) * width), max(0.0, (cy - bh / 2) * height)
                    x2, y2 = min(float(width), (cx + bw / 2) * width), min(float(height), (cy + bh / 2) * height)
                    box = [x1, y1, x2 - x1, y2 - y1]
                    doc["annotations"].append({"id": len(doc["annotations"]) + 1, "image_id": index,
                        "category_id": int(values[0]) + 1, "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
        total_bytes += sum(source.stat().st_size for source, _ in sources)
        docs[native_split] = (doc, sources)
    if shutil.disk_usage(output).free < total_bytes + 2 * 1024**3:
        raise ValueError("磁碟空間不足以複製資料集並保存 RF-DETR 最佳與最後權重。")
    for split, (doc, sources) in docs.items():
        destination = target / split
        destination.mkdir(parents=True, exist_ok=True)
        for source, name in sources:
            shutil.copy2(source, destination / name)
        _write_json(destination / "_annotations.coco.json", doc)
    report = {"source_format": fresh["format"], "class_id_offset": 1, "classes": classes,
              "copied_image_bytes": total_bytes,
              "splits": {split: {"images": len(doc["images"]), "annotations": len(doc["annotations"]),
                                "crowd_annotations": sum(a["iscrowd"] != 0 for a in doc["annotations"])} for split, (doc, _) in docs.items()}}
    _write_json(output / "dataset_conversion.json", report)
    emit({"event": "log", "message": f"已將 {fresh['train_count']} 張訓練與 {fresh['val_count']} 張驗證影像複製到工作資料夾；RF-DETR 使用 1 起算的 {len(classes)} 個類別，COCO crowd 區域保留供評估。"})
    return target


def run_training(request: dict, output_dir: str | Path, emit) -> dict:
    from .training import EarlyStoppingTracker, _catalog_weight, _resolve_device, _validated_dataset, _write_json
    import torch
    from rfdetr_plus import RFDETR2XLarge
    from rfdetr.datasets import build_dataset as native_build_dataset
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64

    p = normalize_training_params(request.get("params"))
    dataset = _validated_dataset(request, ("yolo_detection", "coco_detection"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg")
    classes = dataset["classes"]
    device = _resolve_device(p["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(p["seed"])
    checkpoint = _catalog_weight(MODEL_ID, request.get("model_weight_path"))
    _write_json(output / "training_config.json", {"model_id": MODEL_ID, "params": p,
                "dataset_id": dataset.get("id"), "dataset_classes": classes, "checkpoint_source": "local_catalog", "schema_version": 1})
    converted = _prepare_coco(dataset, output, emit)
    emit({"event": "progress", "progress": 0.03, "message": "載入本機 RF-DETR 2XL 權重"})

    class LocalRFDETR2XLarge(RFDETR2XLarge):
        def maybe_download_pretrain_weights(self):
            if not Path(self.model_config.pretrain_weights).is_file():
                raise ValueError("RF-DETR 本機權重不存在，請先完成下載。")

    def no_download(*args, **kwargs):
        raise RuntimeError("工作台使用本機權重，禁止訓練期間自動下載模型。")

    with patch("rfdetr.main.download_pretrain_weights", side_effect=no_download):
        model = LocalRFDETR2XLarge(pretrain_weights=str(checkpoint), device=device.type,
                  accept_platform_model_license=p["accept_platform_license"], resolution=p["imgsz"],
                  group_detr=p["group_detr"], amp=bool(p["amp"] and device.type == "cuda"),
                  gradient_checkpointing=p["gradient_checkpointing"])
    frozen_count = 0
    if p["strategy"] == "freeze_backbone":
        for parameter in model.model.model.backbone.parameters():
            frozen_count += parameter.numel()
            parameter.requires_grad_(False)
        model.callbacks["on_train_batch_start"].append(lambda event: event["model"].backbone.eval())
    stopper = EarlyStoppingTracker(p["patience"], "max", "val mAP50–95", emit)
    metrics = []
    started = time.monotonic()
    trainable_count = sum(x.numel() for x in model.model.model.parameters() if x.requires_grad)
    # Bbox regression is stable across class-head replacement inside native train().
    update_name, update_parameter = next((name, value) for name, value in model.model.model.named_parameters()
                                         if value.requires_grad and "bbox_embed" in name and value.numel() > 1)
    update_before = update_parameter.detach().cpu().clone()
    emit({"event": "log", "message": f"RF-DETR：{device}，AdamW，輸入 {p['imgsz']}，可訓練 {trainable_count:,} / 凍結 {frozen_count:,} 個參數；使用分類 + L1 + GIoU 原生損失。"})

    def save_checkpoint(folder: Path, epoch: int, score: float):
        folder.mkdir(exist_ok=True)
        state = {name: tensor.detach().cpu() for name, tensor in model.model.model.state_dict().items()}
        args = argparse.Namespace(class_names=list(classes), num_classes=len(classes) + 1,
                                  resolution=p["imgsz"], group_detr=p["group_detr"])
        # Native format, produced by this trusted worker. No optimizer pickle or external file path is accepted.
        temp = folder / "model.pth.tmp"
        torch.save({"model": state, "args": args, "epoch": epoch - 1}, temp)
        temp.replace(folder / "model.pth")
        _write_json(folder / "workbench_training.json", {"schema_version": 1, "model_id": MODEL_ID,
                    "classes": classes, "class_id_offset": 1, "num_classes": len(classes) + 1,
                    "resolution": p["imgsz"], "group_detr": p["group_detr"], "positional_encoding_size": 44,
                    "epoch": epoch, "val_map50_95": score, "strategy": p["strategy"],
                    "early_stopping": stopper.summary(), "format": "rfdetr_pth"})

    def epoch_end(stats):
        epoch = int(stats["epoch"]) + 1
        map_stats = stats["test_coco_eval_bbox"]
        row = {"epoch": epoch, "train_loss": float(stats["train_loss"]), "val_loss": float(stats["test_loss"]),
               "map50_95": float(map_stats[0]), "map50": float(map_stats[1]),
               "lr": float(stats.get("train_lr", p["lr"])), "elapsed_seconds": round(time.monotonic() - started, 3)}
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError("RF-DETR 訓練產生非有限損失或指標，未保存為成功權重。")
        stop = stopper(epoch, row["map50_95"])
        save_checkpoint(output / "last", epoch, row["map50_95"])
        if stopper.improved:
            save_checkpoint(output / "best", epoch, row["map50_95"])
        metrics.append(row)
        _write_json(output / "metrics.json", metrics)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(metrics)
        emit({"event": "metric", **row})
        emit({"event": "progress", "progress": min(0.98, epoch / p["epochs"]),
              "message": f"RF-DETR epoch {epoch}/{p['epochs']}；val loss {row['val_loss']:.4f}，mAP50–95 {row['map50_95']:.4f}"})
        if stop:
            model.model.request_early_stop()

    model.callbacks["on_fit_epoch_end"].append(epoch_end)
    native_dir = output / "native_training"
    native_dir.mkdir(exist_ok=True)

    def build_dataset(image_set, args, resolution):
        built = native_build_dataset(image_set=image_set, args=args, resolution=resolution)
        if image_set == "train" and not p["augmentation"]:
            built._transforms = make_coco_transforms_square_div_64("val", resolution, patch_size=20, num_windows=2)
        return built

    kwargs = {"dataset_dir": str(converted), "dataset_file": "roboflow", "output_dir": str(native_dir),
        "epochs": p["epochs"], "batch_size": p["batch_size"], "grad_accum_steps": p["accumulate"],
        "resolution": p["imgsz"], "lr": p["lr"], "lr_encoder": p["lr_encoder"],
        "weight_decay": p["weight_decay"], "num_workers": p["workers"], "seed": p["seed"],
        "warmup_epochs": p["warmup_epochs"], "lr_scheduler": p["scheduler"], "lr_min_factor": p["lrf"],
        "lr_drop": p["step_size"], "clip_max_norm": p["gradient_clip"],
        "bbox_loss_coef": p["box"], "giou_loss_coef": p["giou"], "cls_loss_coef": p["cls"],
        "group_detr": p["group_detr"], "gradient_checkpointing": p["gradient_checkpointing"],
        "amp": bool(p["amp"] and device.type == "cuda"), "multi_scale": False, "expanded_scales": False,
        "use_ema": False, "run_test": False, "early_stopping": False, "tensorboard": False, "wandb": False,
        "do_benchmark": False, "dont_save_weights": True, "min_batches": 1, "fp16_eval": False,
        "class_names": classes, "square_resize_div_64": True}
    with patch("rfdetr.main.build_dataset", side_effect=build_dataset), patch("rfdetr.main.download_pretrain_weights", side_effect=no_download):
        model.train(**kwargs)
    best, last = output / "best" / "model.pth", output / "last" / "model.pth"
    if not metrics or not best.is_file() or not last.is_file():
        raise RuntimeError("RF-DETR 未產生完整驗證指標與 best / last 權重，不能標記成功。")
    update_evidence = {"parameter": update_name,
                       "max_abs_delta": float((update_parameter.detach().cpu() - update_before).abs().max().item())}
    # The native trainer resizes classification heads to the dataset class count.
    trainable_count = sum(x.numel() for x in model.model.model.parameters() if x.requires_grad)
    effective = {"device": str(device), "amp": kwargs["amp"], "epochs_completed": len(metrics),
                 "trainable_parameters": trainable_count, "frozen_parameters": frozen_count,
                 "native_training_arguments": kwargs, "early_stopping": stopper.summary(), "requested_params": p,
                 "weight_update": update_evidence}
    _write_json(output / "effective_training.json", effective)
    emit({"event": "progress", "progress": 1.0, "message": "RF-DETR 訓練與驗證完成，已保存最佳與最後權重"})
    return {"checkpoint_path": str(best),
            "artifacts": {"best": str(best), "last": str(last), "metrics": str(output / "metrics.csv"),
                          "metrics_json": str(output / "metrics.json"), "config": str(output / "training_config.json"),
                          "effective_config": str(output / "effective_training.json"),
                          "dataset_conversion": str(output / "dataset_conversion.json"),
                          "checkpoint_metadata": str(output / "best" / "workbench_training.json")},
            "summary": f"完成 {len(metrics)} 輪 RF-DETR 2XL 監督微調；最佳 epoch {stopper.best_epoch}。",
            "training": {"model_id": MODEL_ID, "checkpoint_format": "rfdetr_pth", "checkpoint_directory": str(best.parent),
                         "classes": classes, "class_id_offset": 1, "best_epoch": stopper.best_epoch,
                         "early_stopping": stopper.summary(), "weight_update": update_evidence}}
