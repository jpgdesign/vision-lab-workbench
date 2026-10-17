"""Offline Grounding DINO supervised box fine-tuning with native HF loss.

Class indices refer to period-delimited phrases in a shared, checked caption.
The dataset is read only; all checkpoints and metrics belong to the job.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import time
import zipfile
from pathlib import Path

from .system_info import available_device_options


def training_capabilities() -> dict:
    from .training import _parameter as param
    optimizers = ["SGD", "Adam", "AdamW", "Adagrad", "RMSprop"]
    schedulers = ["cosine", "step", "none"]
    strategies = ["freeze_backbone", "full"]
    losses = ["grounding_dino_composite"]
    parameters = [
        param("epochs", "訓練輪數", 10, min=1, max=10000, step=1),
        param("batch_size", "每批樣本數", 1, min=1, max=64, step=1),
        param("imgsz", "輸入圖片尺寸（正方形，32 倍數）", 256, min=224, max=1024, step=32),
        param("lr", "初始學習率", 0.00001, min=1e-9, max=0.1, step=0.000001),
        param("optimizer", "梯度最佳化器", "AdamW", "select", options=optimizers),
        param("loss", "損失函數", losses[0], "select", options=losses),
        param("scheduler", "學習率排程", "cosine", "select", options=schedulers),
        param("strategy", "微調策略", strategies[0], "select", options=strategies),
        param("weight_decay", "權重衰減", 0.0001, min=0, max=1, step=0.0001),
        param("momentum", "SGD / RMSprop 動量", 0.9, min=0, max=0.9999, step=0.01),
        param("patience", "提前停止耐心輪數（0 停用）", 5, min=0, max=10000, step=1),
        param("seed", "隨機種子", 42, min=0, max=2147483647, step=1),
        param("device", "運算裝置", "auto", "select", options=available_device_options()),
        param("amp", "CUDA 混合精度", True, "boolean"),
        param("workers", "資料載入工作程序", 0, min=0, max=16, step=1),
        param("accumulate", "梯度累積批數", 1, min=1, max=256, step=1),
        param("augmentation", "訓練資料水平翻轉", True, "boolean"),
        param("gradient_clip", "梯度範數上限（0 停用）", 0.1, min=0, max=1000, step=0.1),
        param("step_size", "Step 排程間隔輪數", 5, min=1, max=10000, step=1),
        param("step_gamma", "Step 排程衰減比例", 0.1, min=0.0001, max=1, step=0.01),
        param("min_lr", "Cosine 最低學習率", 0.000001, min=0, max=0.1, step=0.000001),
        param("bbox_loss_coefficient", "L1 定位損失權重", 5.0, min=0, max=100, step=0.1),
        param("giou_loss_coefficient", "GIoU 損失權重", 2.0, min=0, max=100, step=0.1),
        param("focal_alpha", "Focal alpha", 0.25, min=0, max=1, step=0.01),
    ]
    return {"dataset_formats": ["yolo_detection", "coco_detection"], "optimizers": optimizers,
            "losses": losses, "schedulers": schedulers, "strategies": strategies,
            "defaults": {p["key"]: p["default"] for p in parameters}, "parameters": parameters,
            "supported_params": [p["key"] for p in parameters],
            "early_stopping_monitor": {"metric": "val_loss", "mode": "min", "label": "驗證損失 val_loss（越低越好）"}}


def normalize_training_params(params: dict | None) -> dict:
    caps = training_capabilities()
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params 必須是 JSON 物件。")
    specs = {p["key"]: p for p in caps["parameters"]}
    if set(params) - set(specs):
        raise ValueError(f"此模型不支援參數：{', '.join(sorted(set(params) - set(specs)))}")
    result = {**caps["defaults"], **params}
    integers = {"epochs", "batch_size", "imgsz", "patience", "seed", "workers", "accumulate", "step_size"}
    for key, value in result.items():
        spec = specs[key]
        if spec["type"] == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必須是 true / false。")
        elif spec["type"] == "select":
            if key == "device" and isinstance(value, str) and re.fullmatch(r"(?:cuda:)?\d+", value):
                result[key] = "cuda:" + value.removeprefix("cuda:")
            elif value not in spec["options"]:
                raise ValueError(f"{key} 不支援 {value!r}。")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} 必須是有限數字。")
            if key in integers and int(value) != value:
                raise ValueError(f"{key} 必須是整數。")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{key} 必須介於 {spec['min']} 和 {spec['max']}。")
            result[key] = int(value) if key in integers else float(value)
    if result["imgsz"] % 32:
        raise ValueError("Grounding DINO 圖片尺寸必須是 32 的倍數。")
    if result["min_lr"] > result["lr"]:
        raise ValueError("min_lr 不能大於初始 lr。")
    return result


def _caption(classes: list[str], tokenizer, max_text_len: int) -> tuple[str, int]:
    """Require one unambiguous native loss label-map row per dataset class."""
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    if any(re.search(r"[.?!\[\]]", name) for name in classes):
        raise ValueError("Grounding DINO 類別名稱不可含句點、問號、驚嘆號或特殊 token 括號；請改用完整單一詞組。")
    normalized = [name.strip().lower() for name in classes]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Grounding DINO 類別名稱轉小寫後不可重複。")
    caption = " . ".join(normalized) + " ."
    encoded = tokenizer(caption, return_tensors="pt", truncation=False)
    ids = encoded["input_ids"]
    count = ids.shape[1]
    if count > max_text_len:
        raise ValueError(f"所有類別提示共 {count} tokens，超過本機 Grounding DINO 上限 {max_text_len}；請縮短類別名稱或拆分資料集。")
    maps = build_label_maps(torch.empty((1, 1, max_text_len)), ids)
    if maps[0].shape[0] != len(classes) or (maps[0].sum(-1) == 0).any():
        raise ValueError("類別名稱無法與 Grounding DINO 的文字 token 一一對應，請移除特殊標記與分隔符。")
    return caption, int(count)


def _detection_samples(dataset: dict) -> tuple[list[dict], list[dict]]:
    """Map both supported formats to 0-based classes and normalized cxcywh."""
    from PIL import Image
    from .datasets import _inside
    details, classes = dataset["details"], dataset["classes"]
    root = Path(details["root"]).resolve()
    splits = []
    for split in ("train", "val"):
        samples = []
        if dataset["format"] == "yolo_detection":
            for raw in details[f"{split}_images"]:
                path = _inside(Path(raw), root)
                parts = list(path.relative_to(root).parts)
                parts[max(i for i, part in enumerate(parts) if part == "images")] = "labels"
                label = _inside(Path(*parts).with_suffix(".txt"), root)
                boxes, labels = [], []
                for line in label.read_text(encoding="utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    entries = line.split()
                    if len(entries) != 5:
                        raise ValueError("YOLO 標註已變更，請重新驗證資料集。")
                    labels.append(int(entries[0]))
                    boxes.append([float(v) for v in entries[1:]])
                samples.append({"path": str(path), "labels": labels, "boxes": boxes})
        else:
            mapping = {int(key): int(value) for key, value in details["category_id_to_index"].items()}
            if sorted(mapping.values()) != list(range(len(classes))):
                raise ValueError("COCO 類別對應表無效，請重新驗證。")
            document = json.loads(_inside(Path(details[f"{split}_json"]), root).read_text(encoding="utf-8-sig"))
            by_id = {}
            for annotation in document["annotations"]:
                if annotation.get("iscrowd", 0):
                    continue
                by_id.setdefault(annotation["image_id"], []).append(annotation)
            for info in document["images"]:
                path = _inside(Path(details[f"{split}_images"][str(info["id"])]), root)
                with Image.open(path) as image:
                    width, height = image.size
                if width != info["width"] or height != info["height"]:
                    raise ValueError("COCO 影像尺寸已變更，請重新驗證資料集。")
                boxes, labels = [], []
                for annotation in by_id.get(info["id"], []):
                    x, y, w, h = annotation["bbox"]
                    boxes.append([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height])
                    labels.append(mapping[annotation["category_id"]])
                samples.append({"path": str(path), "labels": labels, "boxes": boxes})
        if not samples:
            raise ValueError(f"{split} 不可為空。")
        for sample in samples:
            if not Path(sample["path"]).is_file():
                raise ValueError("影像不存在，請重新驗證資料集。")
            for label, box in zip(sample["labels"], sample["boxes"]):
                x, y, w, h = box
                if not 0 <= label < len(classes) or not all(math.isfinite(v) for v in box):
                    raise ValueError("類別或座標無效，請重新驗證資料集。")
                if not (0 < w <= 1 and 0 < h <= 1 and x - w / 2 >= -1e-5 and y - h / 2 >= -1e-5
                        and x + w / 2 <= 1 + 1e-5 and y + h / 2 <= 1 + 1e-5):
                    raise ValueError("邊界框超出影像範圍，請重新驗證資料集。")
        splits.append(samples)
    return tuple(splits)


class _DetectionSamples:
    def __init__(self, samples, size, augment):
        self.samples, self.size, self.augment = samples, size, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        from PIL import Image, ImageOps
        sample = self.samples[index]
        with Image.open(sample["path"]) as source:
            image = source.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        boxes = torch.tensor(sample["boxes"], dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            boxes[:, 0] = 1 - boxes[:, 0]
        return image, {"class_labels": torch.tensor(sample["labels"], dtype=torch.long), "boxes": boxes}


class _Collator:
    def __init__(self, processor, caption):
        self.processor, self.caption = processor, caption

    def __call__(self, samples):
        images, labels = zip(*samples)
        inputs = self.processor(images=list(images), text=[self.caption] * len(images),
                                do_resize=False, padding=True, return_tensors="pt")
        inputs["labels"] = list(labels)
        return inputs


def _to_device(inputs, device):
    return {key: [{k: v.to(device) for k, v in label.items()} for label in value] if key == "labels"
            else value.to(device) for key, value in inputs.items()}


def _supervised_loss(model, inputs):
    """Use the native criterion with differentiable encoder proposal outputs.

    Transformers 4.57.6's convenience `labels=` forward supplies raw learned
    queries to encoder classification and detached boxes to encoder regression.
    Select the real top-k encoder proposals instead. This preserves the native
    matcher/loss and restores gradient flow into the encoder box head.
    """
    import torch
    from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps, build_text_mask
    outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
    logits = outputs.logits
    encoder_logits, encoder_boxes = None, None
    if model.config.two_stage:
        selected = outputs.enc_outputs_class.max(-1)[0].topk(logits.shape[1], dim=1).indices
        encoder_logits = torch.gather(outputs.enc_outputs_class, 1,
                                      selected.unsqueeze(-1).expand(-1, -1, outputs.enc_outputs_class.shape[-1]))
        encoder_boxes = torch.gather(outputs.enc_outputs_coord_logits, 1,
                                     selected.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()
    loss, components, _ = GroundingDinoForObjectDetectionLoss(
        logits, inputs["labels"], logits.device, outputs.pred_boxes, model.config,
        build_label_maps(logits, inputs["input_ids"]), build_text_mask(logits, inputs["attention_mask"]),
        encoder_logits=encoder_logits, encoder_pred_boxes=encoder_boxes)
    return loss, components


def _archive_checkpoint(directory: Path) -> Path:
    """Package already-dense safetensors without costly deflate compression."""
    archive = directory.with_suffix(".zip")
    temporary = archive.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Checkpoint 封裝不接受符號連結。")
            if path.is_file():
                bundle.write(path, arcname=path.relative_to(directory).as_posix())
    temporary.replace(archive)
    return archive


def run_training(request: dict, output_dir: str | Path, emit) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from .training import (EarlyStoppingTracker, _build_optimizer, _build_scheduler, _catalog_weight,
                           _resolve_device, _seed_worker, _validated_dataset, _write_json)

    p = normalize_training_params(request.get("params"))
    dataset = _validated_dataset(request, ("yolo_detection", "coco_detection"))
    train_samples, val_samples = _detection_samples(dataset)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    classes = dataset["classes"]
    weight = _catalog_weight(request["model_id"], request.get("model_weight_path"))
    source = weight if weight.is_dir() else weight.parent
    processor = AutoProcessor.from_pretrained(str(source), local_files_only=True)
    # Use the portable PyTorch attention operator; Windows does not need a CUDA compiler.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(source), local_files_only=True)
    config.disable_custom_kernels = True
    config.auxiliary_loss = False
    config.bbox_loss_coefficient = p["bbox_loss_coefficient"]
    config.giou_loss_coefficient = p["giou_loss_coefficient"]
    config.focal_alpha = p["focal_alpha"]
    caption, text_tokens = _caption(classes, processor.tokenizer, config.max_text_len)
    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(p["device"])
    emit({"event": "progress", "progress": 0.01, "message": "載入 Grounding DINO 本機權重與監督框"})
    model, loading = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(source), config=config, local_files_only=True, weights_only=True, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Grounding DINO 本機權重不完整：{loading}")
    if p["strategy"] == "freeze_backbone":
        for module in (model.model.backbone, model.model.text_backbone):
            module.requires_grad_(False)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    audit_name, audit_parameter = next((name, parameter) for name, parameter in model.named_parameters()
                                      if parameter.requires_grad and "bbox_embed" in name and name.endswith("weight"))
    audit_initial = audit_parameter.detach().float().cpu().clone()
    optimizer = _build_optimizer(trainable, p)
    scheduler = _build_scheduler(optimizer, p)
    amp = bool(p["amp"] and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    common = dict(batch_size=p["batch_size"], num_workers=p["workers"], worker_init_fn=_seed_worker,
                  collate_fn=_Collator(processor, caption), persistent_workers=p["workers"] > 0)
    train_loader = DataLoader(_DetectionSamples(train_samples, p["imgsz"], p["augmentation"]),
                              shuffle=True, generator=torch.Generator().manual_seed(p["seed"]), **common)
    val_loader = DataLoader(_DetectionSamples(val_samples, p["imgsz"], False), shuffle=False, **common)
    stopper = EarlyStoppingTracker(p["patience"], "min", "val_loss", emit)
    effective = {"device": str(device), "amp": amp, "amp_dtype": str(amp_dtype) if amp else "float32",
                 "classes": classes, "caption": caption, "text_tokens": text_tokens, "params": p,
                 "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                 "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                 "preprocessing": "square bilinear resize; normalized cxcywh; ImageNet normalization",
                 "augmentation": "random horizontal flip" if p["augmentation"] else "none",
                 "crowd_annotations": "excluded", "loss": "native HF Hungarian matcher + token focal + L1 + GIoU",
                 "encoder_supervision": "differentiable top-k encoder proposal logits and boxes",
                 "cardinality_error": "unweighted native diagnostic; not part of the optimization objective; not an open-vocabulary detection count metric",
                 "class_loss_coefficient": 2.0, "checkpoint_format": "huggingface_safetensors"}
    _write_json(output / "training_config.json", {"model_id": request["model_id"], "params": p,
                "dataset_id": dataset.get("id"), "dataset_classes": classes, "schema_version": 1})
    _write_json(output / "effective_training.json", effective)
    emit({"event": "log", "message": f"Grounding DINO：{len(classes)} 類、{text_tokens} tokens；"
          f"{effective['trainable_parameters']:,} 個可訓練參數，{device}，AMP={amp}。"})
    metrics = []
    csv_path = output / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "lr", "seconds"]
    csv_path.write_text(",".join(fields) + "\n", encoding="utf-8")

    def save_checkpoint(name, epoch):
        emit({"event": "progress", "progress": 0.01 + 0.89 * epoch / p["epochs"],
              "message": f"Epoch {epoch}/{p['epochs']} · 保存 {'最佳 best' if name == 'best' else '最後 last'} 權重與前處理設定"})
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(destination), safe_serialization=True)
        processor.save_pretrained(str(destination))
        _write_json(destination / "workbench_training.json", {"model_id": request["model_id"], "epoch": epoch,
                    "classes": classes, "caption": caption, "params": p, "dataset_id": dataset.get("id"),
                    "inference_input": "image_or_video", "label_format": "zero_based_caption_phrase_index"})

    for epoch in range(1, p["epochs"] + 1):
        started = time.monotonic()
        model.train()
        if p["strategy"] == "freeze_backbone":
            model.model.backbone.eval()
            model.model.text_backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        train_total, train_seen, train_components = 0.0, 0, {}
        batch_count = len(train_loader)
        for index, inputs in enumerate(train_loader):
            inputs = _to_device(inputs, device)
            count = len(inputs["labels"])
            group_start = (index // p["accumulate"]) * p["accumulate"]
            group_end = min(group_start + p["accumulate"], batch_count)
            group_samples = min(len(train_samples), group_end * p["batch_size"]) - group_start * p["batch_size"]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                loss, components = _supervised_loss(model, inputs)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Grounding DINO 訓練損失出現 NaN / Inf；請停用 AMP 或降低學習率。")
            scaler.scale(loss * (count / group_samples)).backward()
            if index + 1 == group_end:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, p["gradient_clip"] or float("inf"), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_total += loss.detach().item() * count
            train_seen += count
            for key, value in components.items():
                train_components[key] = train_components.get(key, 0.0) + value.detach().item() * count
            emit({"event": "progress", "progress": 0.01 + 0.89 * (epoch - 1 + 0.8 * (index + 1) / batch_count) / p["epochs"],
                  "message": f"Epoch {epoch}/{p['epochs']} · train batch {index + 1}/{batch_count}"})
        model.eval()
        emit({"event": "progress", "progress": 0.01 + 0.89 * (epoch - 0.2) / p["epochs"],
              "message": f"Epoch {epoch}/{p['epochs']} · 驗證資料集"})
        val_total, val_seen, val_components = 0.0, 0, {}
        with torch.no_grad():
            for inputs in val_loader:
                inputs = _to_device(inputs, device)
                count = len(inputs["labels"])
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    loss, components = _supervised_loss(model, inputs)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Grounding DINO 驗證損失出現 NaN / Inf。")
                val_total += loss.item() * count
                val_seen += count
                for key, value in components.items():
                    val_components[key] = val_components.get(key, 0.0) + value.item() * count
        row = {"epoch": epoch, "train_loss": train_total / train_seen, "val_loss": val_total / val_seen,
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.monotonic() - started, 3)}
        metrics.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        _write_json(output / "metrics.json", metrics)
        should_stop = stopper(epoch, row["val_loss"])
        save_checkpoint("last", epoch)
        if stopper.improved:
            save_checkpoint("best", epoch)
        emit({"event": "metric", **row})
        emit({"event": "log", "message": f"Epoch {epoch}：train loss={row['train_loss']:.5f}，val loss={row['val_loss']:.5f}"})
        component_summary = {"epoch": epoch, "train": {key: value / train_seen for key, value in train_components.items()},
                             "val": {key: value / val_seen for key, value in val_components.items()}}
        _write_json(output / "loss_components.json", component_summary)
        emit({"event": "log", "message": "Loss components: " + json.dumps(component_summary, ensure_ascii=False)})
        if scheduler is not None:
            scheduler.step()
        if should_stop:
            break
    effective.update(epochs_completed=len(metrics), early_stopping=stopper.summary())
    update_evidence = {"parameter": audit_name,
                       "max_abs_delta": float((audit_parameter.detach().float().cpu() - audit_initial).abs().max())}
    effective["parameter_update"] = update_evidence
    _write_json(output / "effective_training.json", effective)
    emit({"event": "progress", "progress": 0.95, "message": "微調完成，正在封裝最佳權重 best.zip"})
    best_archive = _archive_checkpoint(output / "best")
    emit({"event": "progress", "progress": 0.98, "message": "正在封裝最後權重 last.zip"})
    last_archive = _archive_checkpoint(output / "last")
    emit({"event": "progress", "progress": 1.0, "message": "Grounding DINO 微調完成，已保存最佳與最後權重"})
    return {"checkpoint_path": str(best_archive), "artifacts": {"best": str(best_archive), "last": str(last_archive),
            "metrics": str(csv_path), "metrics_json": str(output / "metrics.json"),
            "loss_components": str(output / "loss_components.json"),
            "config": str(output / "training_config.json"), "effective_config": str(output / "effective_training.json")},
            "summary": f"完成 {len(metrics)} 輪 Grounding DINO 監督微調；最佳 epoch {stopper.best_epoch}，val loss {stopper.best_value:.5f}。",
            "training": {"model_id": request["model_id"], "checkpoint_format": "huggingface_safetensors",
                         "checkpoint_directory": str(output / "best"), "classes": classes, "caption": caption,
                         "best_epoch": stopper.best_epoch, "early_stopping": stopper.summary(),
                         "parameter_update": update_evidence}}
