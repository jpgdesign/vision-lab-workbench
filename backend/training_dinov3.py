"""Offline DINOv3 image classification: a CLS embedding and a trainable Linear head.

The official AutoModel API exposes ``pooler_output`` (the normalized CLS token):
https://huggingface.co/docs/transformers/model_doc/dinov3
Checkpoints contain the complete backbone, processor and safetensors classifier;
no remote downloads or executable checkpoint formats are needed.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path


CHECKPOINT_FORMAT = "dinov3_classifier_safetensors"


def _classes(value):
    if (not isinstance(value, list) or not value or
            any(not isinstance(x, str) or not x.strip() for x in value) or len(set(value)) != len(value)):
        raise ValueError("DINOv3 分類 checkpoint 需要不重複的非空 classes。")
    return list(value)


def _make_classifier(backbone, count):
    import torch

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Linear(int(backbone.config.hidden_size), count)

        def forward(self, pixel_values):
            features = self.backbone(pixel_values=pixel_values).pooler_output
            if features is None or features.ndim != 2:
                raise RuntimeError("DINOv3 backbone 未回傳預期的 CLS pooler_output。")
            return self.classifier(features)

    return Classifier()


def _load_backbone(folder):
    from transformers import AutoModel
    backbone, info = AutoModel.from_pretrained(
        str(folder), local_files_only=True, weights_only=True, output_loading_info=True)
    if getattr(backbone.config, "model_type", None) != "dinov3_vit":
        raise ValueError("此接頭需要 DINOv3 ViT backbone。")
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("DINOv3 backbone 權重不完整或不相容，不能以隨機權重替代。")
    return backbone.float()


def checkpoint_metadata(path):
    """Read/validate the portable classifier layout without loading ML weights."""
    folder = Path(path)
    folder = folder if folder.is_dir() else folder.parent
    config = json.loads((folder / "model_config.json").read_text(encoding="utf-8"))
    training = json.loads((folder / "workbench_training.json").read_text(encoding="utf-8"))
    classes = _classes(config.get("classes"))
    if (config.get("checkpoint_format") != CHECKPOINT_FORMAT or config.get("model_id") != "dinov3_7b"
            or training.get("model_id") != "dinov3_7b" or training.get("classes") != classes
            or config.get("pooling") != "cls" or config.get("num_classes") != len(classes)):
        raise ValueError("DINOv3 分類 checkpoint 中繼資料不一致。")
    from .model_registry import _hf_weight
    _hf_weight(folder / "backbone")
    for file in (folder / "backbone" / "config.json", folder / "classifier.safetensors",
                 folder / "preprocessor_config.json"):
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f"DINOv3 分類 checkpoint 缺少 {file.name}。")
    return folder, config


def load_classifier(checkpoint_path, device="cpu"):
    """Return (logits model, image processor, metadata) for image/frame inference."""
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor
    folder, config = checkpoint_metadata(checkpoint_path)
    model = _make_classifier(_load_backbone(folder / "backbone"), len(config["classes"]))
    if model.classifier.in_features != config.get("hidden_size"):
        raise ValueError("DINOv3 分類頭維度與 backbone 不符。")
    model.classifier.load_state_dict(load_file(str(folder / "classifier.safetensors"), device="cpu"), strict=True)
    processor = AutoImageProcessor.from_pretrained(str(folder), local_files_only=True)
    model.to(device).eval()
    return model, processor, {**config, "labels": dict(enumerate(config["classes"])),
                              "outputs_probabilities": False, "inference_input": "image"}


def prepare_image_pixels(images, processor, device="cpu"):
    """Apply the same saved processor in training, retention and frame inference.

PIL images and numpy arrays must be RGB; OpenCV BGR frames must be converted by
the caller. Video inference is independent frame classification, not temporal
action recognition.
"""
    from PIL import Image, ImageOps
    batch = images if isinstance(images, (list, tuple)) else [images]
    batch = [ImageOps.exif_transpose(x).convert("RGB") if isinstance(x, Image.Image) else x for x in batch]
    return processor(images=batch, return_tensors="pt")["pixel_values"].to(device)


class _ImageSamples:
    def __init__(self, samples, processor, augment=False):
        self.samples, self.processor, self.augment = samples, processor, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image, ImageOps
        sample = self.samples[index]
        with Image.open(sample["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
        return prepare_image_pixels(image, self.processor)[0], sample["label"]


def _samples(dataset, split):
    if dataset.get("format") != "imagefolder":
        raise ValueError("DINOv3 分類微調需要 ImageFolder 資料集。")
    root = Path(dataset["details"]["root"]).resolve()
    samples = dataset["details"].get(f"{split}_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"DINOv3 {split} split 沒有已驗證影像。")
    checked = []
    for sample in samples:
        path, label = Path(sample["path"]).resolve(), sample.get("label")
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("影像不存在或超出資料集範圍，請重新驗證。")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(dataset["classes"]):
            raise ValueError("影像類別索引超出資料集範圍。")
        checked.append({"path": str(path), "label": label})
    return checked


def _evaluate(model, loader, classes, params, device, class_ids=None):
    import torch
    from .training import _classification_loss
    model.eval()
    total = correct = 0
    loss_total = 0.0
    counts, hits = {}, {}
    if class_ids is not None:
        lookup = {value: index for index, value in enumerate(class_ids)}
    with torch.inference_mode():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            # Retention and validation deliberately use FP32, for identical
            # preprocessing/numerical precision before and after fine-tuning.
            logits = model(pixels)
            if not torch.isfinite(logits).all():
                raise RuntimeError("DINOv3 驗證 logits 出現 NaN / Inf。")
            if class_ids is not None:
                if any(int(x) not in lookup for x in labels):
                    raise ValueError("舊類驗證集包含非舊類標籤。")
                targets = torch.tensor([lookup[int(x)] for x in labels], device=device)
                loss = _classification_loss(logits[:, class_ids], targets, params)
                predicted = torch.tensor(class_ids, device=device)[logits[:, class_ids].argmax(-1)]
            else:
                loss = _classification_loss(logits, labels, params)
                predicted = logits.argmax(-1)
            if not torch.isfinite(loss):
                raise RuntimeError("DINOv3 驗證 loss 出現 NaN / Inf。")
            total += labels.numel()
            correct += int((predicted == labels).sum())
            loss_total += loss.item() * labels.numel()
            for truth, guess in zip(labels.tolist(), predicted.tolist()):
                counts[truth] = counts.get(truth, 0) + 1
                hits[truth] = hits.get(truth, 0) + int(truth == guess)
    if not total:
        raise ValueError("DINOv3 驗證集不能為空。")
    return {"accuracy": correct / total, "loss": loss_total / total, "sample_count": total,
            "class_count": len(counts), "class_accuracy": {classes[k]: hits[k] / v for k, v in counts.items()},
            "class_sample_counts": {classes[k]: v for k, v in counts.items()}}


def run_training(request, output, emit):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor
    from .training import (_validated_dataset, _catalog_weight, _resolve_device, _seed_worker,
                           _write_json, _build_optimizer, _build_scheduler, _classification_loss,
                           _expanded_linear_classifier, EarlyStoppingTracker)

    output, p = Path(output), request["params"]
    output.mkdir(parents=True, exist_ok=True)
    dataset = _validated_dataset(request, "imagefolder")
    classes = _classes(dataset["classes"])
    train_samples, val_samples = _samples(dataset, "train"), _samples(dataset, "val")
    mode = request.get("training_mode", "standard")
    if mode not in {"standard", "expand_classes"}:
        raise ValueError("未知的 training_mode。")
    device = _resolve_device(p["device"])
    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    checkpoint = request.get("checkpoint_path")
    # Explicit checkpoint selection always takes precedence over catalog weights.
    if checkpoint:
        model, processor, source = load_classifier(checkpoint)
        base_classes = source["classes"]
    else:
        if mode == "expand_classes":
            raise ValueError("原始 DINOv3 沒有分類類別；擴充類別必須選擇已訓練的 DINOv3 分類 checkpoint。")
        weight = _catalog_weight(request["model_id"], request.get("model_weight_path"))
        folder = weight if weight.is_dir() else weight.parent
        model = _make_classifier(_load_backbone(folder), len(classes))
        processor = AutoImageProcessor.from_pretrained(str(folder), local_files_only=True)
        base_classes = []
    expansion = request.get("class_expansion", {})
    if mode == "expand_classes":
        if expansion.get("base_classes") != base_classes or expansion.get("classes") != classes:
            raise ValueError("DINOv3 類別擴充對應與來源 checkpoint 不一致。")
        old_ids = [classes.index(name) for name in base_classes if name in classes]
        new_ids = [index for index, name in enumerate(classes) if name not in base_classes]
        if (expansion.get("old_class_ids") != old_ids or expansion.get("new_class_ids") != new_ids
                or len(old_ids) != len(base_classes) or not new_ids):
            raise ValueError("DINOv3 舊／新增類別索引與名稱不一致，且必須至少新增一類。")
        model.classifier = _expanded_linear_classifier(model.classifier, base_classes, classes)
    elif checkpoint and base_classes != classes:
        # Preserve the learned head when continuing the exact same task; equal
        # counts alone do not imply matching label names or row order.
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(classes))
    if p["strategy"] == "head_only":
        model.backbone.requires_grad_(False)
    elif p["strategy"] != "full":
        raise ValueError("DINOv3 strategy 必須是 head_only 或 full。")
    model.to(device)
    trainable = [x for x in model.parameters() if x.requires_grad]
    optimizer = _build_optimizer(trainable, p)
    scheduler = _build_scheduler(optimizer, p)
    amp = bool(p["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    common = {"batch_size": p["batch_size"], "num_workers": p["workers"],
              "worker_init_fn": _seed_worker, "persistent_workers": p["workers"] > 0,
              "pin_memory": device.type == "cuda"}
    def loader(samples, training=False):
        return DataLoader(_ImageSamples(samples, processor, training and p["augmentation"]),
                          shuffle=training, generator=torch.Generator().manual_seed(p["seed"]), **common)
    train_loader, val_loader = loader(train_samples, True), loader(val_samples)
    retention, evaluation_loaders = None, {}
    if mode == "expand_classes":
        for name in ("old", "new"):
            evaluation = request.get("evaluation_datasets", {}).get(name)
            if not isinstance(evaluation, dict) or evaluation.get("classes") != classes:
                raise ValueError("DINOv3 類別擴充需要以聯集標籤對應的 old/new 驗證集。")
            evaluation_loaders[name] = loader(_samples(evaluation, "val"))
        old_ids = [classes.index(name) for name in base_classes]
        before = _evaluate(model, evaluation_loaders["old"], classes, p, device, old_ids)
        retention = {"metric": "accuracy", "before": {"old": before}, "after": {}, "delta": {},
                     "classes": classes, "base_classes": base_classes, "training_mode": mode,
                     "scope": "fixed old/new validation; source old head versus best union head; identical processor and FP32 evaluation",
                     "warnings": [], "coverage": {}}
        for name, evaluation_loader in evaluation_loaders.items():
            observed = {sample["label"] for sample in evaluation_loader.dataset.samples}
            expected = old_ids if name == "old" else new_ids
            missing = [classes[index] for index in expected if index not in observed]
            retention["coverage"][name] = {"observed_class_ids": sorted(observed), "missing_classes": missing,
                "sample_count": len(evaluation_loader.dataset), "expected_class_count": len(expected)}
            if missing:
                retention["warnings"].append(f"{name} 驗證集缺少 {len(missing)} 類，評估不代表未出現的類別。")
            if name == "new" and observed.intersection(old_ids):
                retention["warnings"].append("new accuracy 代表新資料集（含原有重疊類別），不是僅新增類別的 accuracy。")
        _write_json(output / "retention.json", retention)
    effective = {"device": str(device), "amp": amp, "classes": classes, "params": p,
                 "training_mode": mode, "base_classes": base_classes, "pooling": "cls",
                 "head_reset": mode == "standard" and (not checkpoint or base_classes != classes),
                 "validation": "saved processor; RGB; no augmentation; FP32",
                 "augmentation": "horizontal flip" if p["augmentation"] else "none",
                 "checkpoint_format": CHECKPOINT_FORMAT, "trainable_parameters": sum(x.numel() for x in trainable)}
    _write_json(output / "effective_training.json", effective)
    emit({"event": "log", "message": f"DINOv3 CLS 分類：{len(classes)} 類，{p['strategy']}，"
          f"訓練 {effective['trainable_parameters']:,} 參數，裝置 {device}，AMP={amp}。"})
    saved_backbones = set()
    def save_checkpoint(name, epoch):
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        # A frozen 7B backbone is identical across epochs. Each standalone
        # best/last directory gets one complete copy, not another 27GB write
        # every time its small Linear head is updated.
        if p["strategy"] != "head_only" or name not in saved_backbones:
            model.backbone.save_pretrained(str(destination / "backbone"), safe_serialization=True)
            saved_backbones.add(name)
        processor.save_pretrained(str(destination))
        save_file({key: value.detach().cpu().contiguous() for key, value in model.classifier.state_dict().items()},
                  str(destination / "classifier.safetensors"))
        _write_json(destination / "model_config.json", {"model_id": request["model_id"], "classes": classes,
                    "num_classes": len(classes), "hidden_size": model.classifier.in_features,
                    "pooling": "cls", "checkpoint_format": CHECKPOINT_FORMAT})
        _write_json(destination / "workbench_training.json", {"model_id": request["model_id"], "epoch": epoch,
                    "classes": classes, "base_classes": base_classes, "training_mode": mode,
                    "class_expansion": expansion if mode == "expand_classes" else None,
                    "checkpoint_format": CHECKPOINT_FORMAT, "params": p, "dataset_id": dataset.get("id"),
                    "inference_input": "image", "video_mode": "independent_frames"})

    metrics = []
    fields = ["epoch", "train_loss", "val_loss", "accuracy", "train_accuracy", "lr", "seconds"]
    csv_path = output / "metrics.csv"
    csv_path.write_text(",".join(fields) + "\n", encoding="utf-8")
    stopper = EarlyStoppingTracker(p["patience"], "min", "val_loss", emit)
    for epoch in range(1, p["epochs"] + 1):
        started = time.monotonic()
        model.train()
        if p["strategy"] == "head_only":
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        train_loss = train_correct = train_seen = 0
        for index, (pixels, labels) in enumerate(train_loader):
            pixels, labels = pixels.to(device), labels.to(device)
            group_start = index // p["accumulate"] * p["accumulate"]
            group_end = min(group_start + p["accumulate"], len(train_loader))
            group_samples = min(len(train_samples), group_end * p["batch_size"]) - group_start * p["batch_size"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(pixels)
                loss = _classification_loss(logits, labels, p)
            if not torch.isfinite(loss):
                raise RuntimeError("DINOv3 訓練 loss 出現 NaN / Inf，請降低學習率或停用 AMP。")
            scaler.scale(loss * labels.numel() / group_samples).backward()
            if index + 1 == group_end:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, p["gradient_clip"] or float("inf"), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_seen += labels.numel()
            train_loss += loss.item() * labels.numel()
            train_correct += int((logits.detach().argmax(-1) == labels).sum())
            emit({"event": "progress", "progress": min(.98, (epoch - 1 + .8 * (index + 1) / len(train_loader)) / p["epochs"]),
                  "message": f"Epoch {epoch}/{p['epochs']} · train batch {index + 1}/{len(train_loader)}"})
        validation = _evaluate(model, val_loader, classes, p, device)
        row = {"epoch": epoch, "train_loss": train_loss / train_seen, "val_loss": validation["loss"],
               "accuracy": validation["accuracy"], "train_accuracy": train_correct / train_seen,
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.monotonic() - started, 3)}
        metrics.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        _write_json(output / "metrics.json", metrics)
        save_checkpoint("last", epoch)
        stop = stopper(epoch, row["val_loss"])
        if stopper.improved:
            save_checkpoint("best", epoch)
        emit({"event": "metric", **row})
        if scheduler is not None:
            scheduler.step()
        if stop:
            break
    if retention is not None:
        # Release optimizer/last model before loading the selected best full model.
        del optimizer, scheduler, trainable, model, logits, loss
        model, _, _ = load_classifier(output / "best", device)
        for name, evaluation_loader in evaluation_loaders.items():
            retention["after"][name] = _evaluate(model, evaluation_loader, classes, p, device)
        retention["delta"]["old"] = retention["after"]["old"]["accuracy"] - retention["before"]["old"]["accuracy"]
        retention["best_epoch"] = stopper.best_epoch
        _write_json(output / "retention.json", retention)
        _write_json(output / "best" / "retention.json", retention)
        metadata_path = output / "best" / "workbench_training.json"
        best_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _write_json(metadata_path, {**best_metadata, "retention": retention})
    effective["early_stopping"] = stopper.summary()
    _write_json(output / "effective_training.json", effective)
    training = {"model_id": request["model_id"], "checkpoint_format": CHECKPOINT_FORMAT,
                "checkpoint_directory": str(output / "best"), "classes": classes,
                "best_epoch": stopper.best_epoch, "early_stopping": stopper.summary()}
    artifacts = {"best": str(output / "best"), "last": str(output / "last"), "metrics": str(csv_path),
                 "metrics_json": str(output / "metrics.json"), "effective_config": str(output / "effective_training.json")}
    if retention is not None:
        training["retention"] = retention
        training["class_expansion"] = expansion
        artifacts["retention"] = str(output / "retention.json")
    emit({"event": "progress", "progress": 1., "message": "DINOv3 分類微調完成，已保存完整 backbone、分類頭與前處理設定"})
    return {"checkpoint_path": str(output / "best"), "artifacts": artifacts, "training": training,
            "summary": f"完成 {len(metrics)} 輪 DINOv3 分類微調；最佳 epoch {stopper.best_epoch}。影片以逐影格分類使用此模型。"}
