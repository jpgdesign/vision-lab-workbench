"""Real, offline supervised training adapters for the visual workbench.

No ML package is imported until a job runs. Inputs are trusted model catalog
entries and validated dataset registry records, never user-supplied pickle files.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Callable

from .system_info import available_device_options

YOLO_IDS = {"yolov12x", "yolo12x", "yolo26x"}
VIVIT_IDS = {"vivit_base_k400", "vivit_base"}
ACTION_IDS = {"x3d_l", "slowfast_r101_8x8", "videomaev2_giant_k710"}
Emit = Callable[[dict], None]


def _detector_training_adapter(model_id: str):
    if model_id == "rf_detr_2xl":
        from . import training_rfdetr
        return training_rfdetr
    if model_id == "grounding_dino_base":
        from . import training_grounding
        return training_grounding
    return None


class EarlyStoppingTracker:
    """Patience over validated epochs, including an initial metric of zero.

    ``possible_stop`` and the callable signature match the Ultralytics stopper
    interface. Checkpoint selection stays with each model's existing trainer.
    """
    def __init__(self, patience: int, mode: str, metric_name: str, emit: Emit | None = None):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("patience 必須是非負整數。")
        if mode not in {"min", "max"}:
            raise ValueError("early stopping mode 必須是 min 或 max。")
        self.patience = patience
        self.mode = mode
        self.metric_name = metric_name
        self.emit = emit
        self.best_value = None
        self.best_epoch = 0
        self.bad_epochs = 0
        self.improved = False
        self.possible_stop = False
        self.stop_epoch = None
        self.reason = None

    def __call__(self, epoch: int, value: float | None) -> bool:
        if value is None:
            self.improved = False
            return False
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.metric_name} 不是有限數值，無法判定提前停止。")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("epoch 必須是從 1 開始的整數。")
        self.improved = self.best_value is None or (
            value < self.best_value if self.mode == "min" else value > self.best_value)
        if self.improved:
            self.best_value, self.best_epoch = value, epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs = epoch - self.best_epoch
        self.possible_stop = bool(self.patience and self.bad_epochs >= self.patience - 1)
        should_stop = bool(self.patience and self.bad_epochs >= self.patience)
        if should_stop and self.stop_epoch is None:
            self.stop_epoch = epoch
            self.reason = (f"提前停止：{self.metric_name} 連續 {self.bad_epochs} 輪未改善；"
                           f"最佳 epoch {self.best_epoch}（{self.best_value:.6g}），於 epoch {epoch} 停止。")
            if self.emit is not None:
                self.emit({"event": "log", "message": self.reason})
        return should_stop

    def summary(self) -> dict:
        return {"monitor": self.metric_name, "mode": self.mode, "patience": self.patience,
                "best_value": self.best_value, "best_epoch": self.best_epoch,
                "stop_epoch": self.stop_epoch, "reason": self.reason}


def _install_yolo_stopper(trainer, patience: int, emit: Emit):
    # Setup creates the native stopper before on_pretrain_routine_end. Replace
    # only that object: trainer.best_fitness and best/last saving are untouched.
    trainer.stopper = EarlyStoppingTracker(patience, "max", "val mAP50–95", emit)


def _parameter(key, label, default, kind="number", **kwargs):
    return {"key": key, "label": label, "default": default, "type": kind, **kwargs}


def training_capabilities(model_id: str) -> dict:
    """Single source of truth for supported UI controls and worker validation."""
    adapter = _detector_training_adapter(model_id)
    if adapter is not None:
        return adapter.training_capabilities()
    yolo = model_id in YOLO_IDS
    if not yolo and model_id not in VIVIT_IDS | ACTION_IDS:
        return {"dataset_formats": [], "optimizers": [], "losses": [], "schedulers": [],
                "strategies": [], "defaults": {}, "parameters": []}
    optimizers = (["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "MuSGD"] if yolo
                  else ["SGD", "Adam", "AdamW", "Adagrad", "RMSprop"])
    losses = ["yolo_composite"] if yolo else ["cross_entropy", "focal"]
    schedulers = ["cosine", "linear"] if yolo else ["cosine", "step", "none"]
    strategies = ["full", "freeze_backbone"] if yolo else ["head_only", "full"]
    params = [
        _parameter("epochs", "訓練輪數", 10, min=1, max=10000, step=1),
        _parameter("batch_size", "每批樣本數", 2 if yolo else 1, min=1, max=256, step=1),
        _parameter("lr", "初始學習率", 0.001 if yolo else 0.0001, min=0.000000001, max=1, step=0.00001),
        _parameter("optimizer", "梯度最佳化器", "AdamW", "select", options=optimizers),
        _parameter("loss", "損失函數", losses[0], "select", options=losses),
        _parameter("scheduler", "學習率排程", "cosine", "select", options=schedulers),
        _parameter("strategy", "微調策略", strategies[0], "select", options=strategies),
        _parameter("weight_decay", "權重衰減", 0.0005 if yolo else 0.01, min=0, max=1, step=0.0001),
        _parameter("momentum", "SGD / RMSprop 動量；YOLO 的 Adam beta1", 0.9, min=0, max=0.9999, step=0.01),
        _parameter("patience", "提前停止耐心輪數（0 停用）", 5, min=0, max=10000, step=1),
        _parameter("seed", "隨機種子", 42, min=0, max=2147483647, step=1),
        _parameter("device", "運算裝置", "auto", "select", options=available_device_options()),
        _parameter("amp", "CUDA 混合精度", True, "boolean"),
        _parameter("workers", "資料載入工作程序", 0, min=0, max=16, step=1),
        _parameter("accumulate", "梯度累積批數（YOLO 為目標值，暖身時逐步增加）", 1, min=1, max=256, step=1),
        _parameter("augmentation", "訓練資料增強", True, "boolean"),
    ]
    if yolo:
        params += [
            _parameter("imgsz", "輸入圖片尺寸（32 倍數）", 640, min=64, max=1536, step=32),
            _parameter("freeze", "凍結前 N 層（策略為凍結主幹時；0 自動取主幹）", 0, min=0, max=1000, step=1),
            _parameter("warmup_epochs", "暖身輪數", 3.0, min=0, max=1000, step=0.5),
            _parameter("lrf", "最後學習率 / 初始學習率", 0.01, min=0, max=1, step=0.01),
            _parameter("box", "定位損失權重", 7.5, min=0, max=1000, step=0.1),
            _parameter("cls", "分類損失權重", 0.5, min=0, max=1000, step=0.1),
        ]
        if model_id != "yolo26x":
            params.append(_parameter("dfl", "DFL 損失權重", 1.5, min=0, max=1000, step=0.1))
    else:
        params += [
            _parameter("label_smoothing", "標籤平滑", 0.0, min=0, max=0.5, step=0.01),
            _parameter("gradient_clip", "梯度範數上限（0 停用）", 1.0, min=0, max=1000, step=0.1),
            _parameter("focal_gamma", "Focal gamma（僅 focal loss）", 2.0, min=0, max=10, step=0.1),
            _parameter("step_size", "Step 排程間隔輪數", 5, min=1, max=10000, step=1),
            _parameter("step_gamma", "Step 排程衰減比例", 0.1, min=0.0001, max=1, step=0.01),
            _parameter("min_lr", "Cosine 最低學習率", 0.000001, min=0, max=1, step=0.000001),
        ]
    return {"dataset_formats": ["yolo_detection", "coco_detection"] if yolo else ["video_csv"],
            "optimizers": optimizers, "losses": losses, "schedulers": schedulers,
            "strategies": strategies, "defaults": {p["key"]: p["default"] for p in params},
            "parameters": params,
            "early_stopping_monitor": {"metric": "val mAP50–95" if yolo else "val_loss",
                                       "mode": "max" if yolo else "min",
                                       "label": "驗證 mAP50–95（越高越好）" if yolo else "驗證損失 val_loss（越低越好）"}}


def normalize_training_params(model_id: str, params: dict | None) -> dict:
    adapter = _detector_training_adapter(model_id)
    if adapter is not None:
        return adapter.normalize_training_params(params)
    caps = training_capabilities(model_id)
    if not caps["dataset_formats"]:
        raise ValueError(f"模型 {model_id} 尚未實作微調，請選擇已接入訓練的模型。")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params 必須是 JSON 物件。")
    specs = {p["key"]: p for p in caps["parameters"]}
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"此模型不支援參數：{', '.join(sorted(unknown))}")
    result = dict(caps["defaults"])
    result.update(params)
    integers = {"epochs", "batch_size", "patience", "seed", "workers", "accumulate", "imgsz", "freeze", "step_size"}
    for key, value in result.items():
        spec = specs[key]
        if spec["type"] == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必須是 true / false。")
        elif spec["type"] == "select":
            if key == "device" and isinstance(value, str) and re.fullmatch(r"(?:cuda:)?\d+", value):
                result[key] = "cuda:" + value.removeprefix("cuda:")
            elif value not in spec["options"]:
                raise ValueError(f"{key} 不支援 {value!r}；可用值：{', '.join(spec['options'])}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} 必須是有限數字。")
            if key in integers and int(value) != value:
                raise ValueError(f"{key} 必須是整數。")
            if value < spec["min"] or value > spec["max"]:
                raise ValueError(f"{key} 必須介於 {spec['min']} 和 {spec['max']}。")
            result[key] = int(value) if key in integers else float(value)
    if model_id in YOLO_IDS:
        if result["imgsz"] % 32:
            raise ValueError("YOLO 圖片尺寸必須是 32 的倍數。")
        if result["strategy"] == "full" and result["freeze"]:
            raise ValueError("freeze 必須搭配 freeze_backbone 策略；full 策略不能凍結層。")
        if sum(result[k] for k in ("box", "cls", "dfl") if k in result) == 0:
            raise ValueError("至少一個損失權重必須大於零。")
    elif result["min_lr"] > result["lr"]:
        raise ValueError("min_lr 不能大於初始 lr。")
    return result


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _catalog_weight(model_id: str, model_weight_path=None) -> Path:
    # Paths are snapshotted by the server from its explicit local registry.
    if model_weight_path:
        from .model_registry import validate_weight
        return validate_weight(model_id, model_weight_path)
    from .catalog import get_models
    model = next((m for m in get_models() if m["id"] == model_id), None)
    if model is None or not model.get("weight_path"):
        raise ValueError(f"找不到 {model_id} 的本機權重。")
    path = Path(model["weight_path"])
    if not path.exists():
        raise FileNotFoundError(f"本機模型權重不存在：{path.name}")
    return path


def _validated_dataset(request: dict, expected_format: str | tuple[str, ...]) -> dict:
    dataset = request.get("dataset")
    formats = (expected_format,) if isinstance(expected_format, str) else expected_format
    if not isinstance(dataset, dict) or dataset.get("format") not in formats:
        raise ValueError(f"此模型需要已驗證的 {expected_format} 資料集。")
    classes = dataset.get("classes")
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c.strip() for c in classes):
        raise ValueError("資料集需要非空的類別名稱清單。")
    if len(set(classes)) != len(classes):
        raise ValueError("資料集類別名稱重複。")
    if not dataset.get("details") or not dataset.get("train_count") or not dataset.get("val_count"):
        raise ValueError("資料集必須有非空的 train 與 val split。")
    return dataset


def _coco_to_yolo(dataset: dict, output: Path, emit: Emit) -> dict:
    """Convert validated COCO boxes into an isolated job-owned YOLO dataset.

    Copy images rather than hard-linking them: third-party loaders may repair a
    JPEG on disk, which must never modify the user's original dataset.
    """
    details = dataset["details"]
    root = Path(details["root"]).resolve()
    image_root = Path(details["image_root"]).resolve()
    mapping = {int(key): int(value) for key, value in details["category_id_to_index"].items()}
    if sorted(mapping.values()) != list(range(len(dataset["classes"]))):
        raise ValueError("COCO category 對應表無效，請重新驗證。")
    documents = {}
    source_paths = {}
    required_bytes = 0
    for split in ("train", "val"):
        annotations_path = Path(details[f"{split}_json"]).resolve()
        if not annotations_path.is_relative_to(root):
            raise ValueError("COCO 標註檔超出資料集範圍。")
        doc = json.loads(annotations_path.read_text(encoding="utf-8-sig"))
        documents[split] = doc
        resolved = {}
        for info in doc["images"]:
            source = (image_root / info["file_name"]).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError("COCO 影像不存在或超出資料集範圍，請重新驗證。")
            known = details.get(f"{split}_images", {}).get(str(info["id"]))
            if known is not None and Path(known).resolve() != source:
                raise ValueError("COCO 標註在驗證後已變更，請重新驗證。")
            resolved[info["id"]] = source
            required_bytes += source.stat().st_size
        if not resolved:
            raise ValueError(f"COCO {split} split 為空。")
        source_paths[split] = resolved
    if shutil.disk_usage(output).free < required_bytes + 512 * 1024 ** 2:
        raise ValueError(f"COCO 轉換需要複製約 {required_bytes / 1e9:.2f} GB 影像，目前磁碟空間不足。")
    target = output / "converted_dataset"
    converted = {**dataset, "format": "yolo_detection", "path": str(target),
                 "details": {"root": str(target)}}
    report = {"source_format": "coco_detection", "category_id_to_index": mapping,
              "copied_image_bytes": required_bytes, "ignored_crowd_annotations": 0,
              "annotation_counts": {}}
    for split, doc in documents.items():
        image_dir = target / "images" / split
        label_dir = target / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        dimensions = {info["id"]: (float(info["width"]), float(info["height"])) for info in doc["images"]}
        labels = {image_id: [] for image_id in dimensions}
        for annotation in doc["annotations"]:
            if annotation.get("iscrowd", 0):
                report["ignored_crowd_annotations"] += 1
                continue
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            if image_id not in dimensions or category_id not in mapping:
                raise ValueError("COCO 標註參照無效，請重新驗證。")
            width, height = dimensions[image_id]
            x, y, box_width, box_height = map(float, annotation["bbox"])
            if (width <= 0 or height <= 0 or not all(map(math.isfinite, (x, y, box_width, box_height)))
                    or x < 0 or y < 0 or box_width <= 0 or box_height <= 0
                    or x + box_width > width + 0.001 or y + box_height > height + 0.001):
                raise ValueError("COCO bbox 不合法，請重新驗證。")
            # Clamp only the validator's permitted floating-point edge tolerance.
            box_width = min(box_width, width - x)
            box_height = min(box_height, height - y)
            box = [(x + box_width / 2) / width, (y + box_height / 2) / height,
                   box_width / width, box_height / height]
            labels[image_id].append(str(mapping[category_id]) + " " + " ".join(f"{v:.10f}" for v in box))
        images = []
        for index, (image_id, source) in enumerate(source_paths[split].items()):
            name = f"{index:08d}{source.suffix.lower()}"
            destination = image_dir / name
            shutil.copy2(source, destination)
            (label_dir / f"{index:08d}.txt").write_text("\n".join(labels[image_id]) + ("\n" if labels[image_id] else ""), encoding="utf-8")
            images.append(str(destination))
        converted["details"][f"{split}_images"] = images
        report["annotation_counts"][split] = sum(map(len, labels.values()))
    _write_json(output / "coco_conversion.json", report)
    emit({"event": "log", "message": f"COCO 已轉換為 YOLO：影像複製到工作資料夾，原資料保留；"
          f"{len(mapping)} 類，忽略 {report['ignored_crowd_annotations']} 個 iscrowd 標註（YOLO 格式不支援 crowd 忽略區域）。"})
    return converted


def _resolve_device(value: str):
    import torch
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda"):
        index = int(value.split(":")[1]) if ":" in value else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"要求 {value}，但該 CUDA 裝置目前不可用。")
    return torch.device(value)


def run_training(request: dict, output_dir: str | Path, emit: Emit) -> dict:
    model_id = request["model_id"]
    params = normalize_training_params(model_id, request.get("params"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "true"
    _write_json(output / "training_config.json", {"model_id": model_id, "params": params,
                "dataset_id": request.get("dataset", {}).get("id"),
                "dataset_classes": request.get("dataset", {}).get("classes"),
                "checkpoint_source": "local_catalog", "schema_version": 1})
    emit({"event": "progress", "progress": 0.01, "message": "載入本機預訓練權重與監督資料集"})
    try:
        adapter = _detector_training_adapter(model_id)
        if adapter is not None:
            return adapter.run_training({**request, "params": params}, output, emit)
        if model_id in YOLO_IDS:
            return _train_yolo(request, params, output, emit)
        if model_id in VIVIT_IDS | ACTION_IDS:
            return _train_classifier(request, params, output, emit)
        raise ValueError(f"尚未實作模型 {model_id} 的訓練介面。")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("GPU / RAM 記憶體不足。請降低 batch_size 或偵測模型 imgsz，並選擇凍結主幹；行為模型可改用 head_only。已保存的 checkpoint 仍保留。") from exc
        raise


def _train_yolo(request: dict, p: dict, output: Path, emit: Emit) -> dict:
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    dataset = _validated_dataset(request, ("yolo_detection", "coco_detection"))
    if dataset["format"] == "coco_detection":
        dataset = _coco_to_yolo(dataset, output, emit)
    weight = _catalog_weight(request["model_id"], request.get("model_weight_path"))
    device = _resolve_device(p["device"])
    details = dataset["details"]
    # Rebuild only the validated fields. Ultralytics dataset YAML supports a
    # download directive that must never execute from a user-uploaded dataset.
    safe_data = {"path": str(Path(details["root"]).resolve()), "nc": len(dataset["classes"]),
                 "names": dict(enumerate(dataset["classes"])), "channels": 3}
    for split in ("train", "val"):
        images = details.get(f"{split}_images")
        if not images:
            raise ValueError(f"缺少已驗證的 {split}_images 清單。")
        resolved = [Path(s).resolve() for s in images]
        if any(not x.is_file() or not x.is_relative_to(Path(details["root"]).resolve()) for x in resolved):
            raise ValueError(f"{split} 影像已移除，請重新驗證資料集。")
        listing = output / f"{split}_images.txt"
        listing.write_text("\n".join(x.as_posix() for x in resolved) + "\n", encoding="utf-8")
        safe_data[split] = str(listing)
    safe_yaml = output / "dataset.yaml"
    safe_yaml.write_text(yaml.safe_dump(safe_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    class LocalDetectionTrainer(DetectionTrainer):
        def get_dataset(self):
            return dict(safe_data)

        def final_eval(self):
            # Ultralytics emits on_fit_epoch_end again for the final best-weight
            # evaluation. It is not an additional training epoch.
            self._workbench_final_eval = True
            try:
                return super().final_eval()
            finally:
                self._workbench_final_eval = False

    model = YOLO(str(weight), task="detect")
    from .inference import validate_yolo_architecture
    validate_yolo_architecture(model, request["model_id"])
    freeze = 0
    if p["strategy"] == "freeze_backbone":
        freeze = p["freeze"] or len(model.model.yaml.get("backbone", []))
        if freeze <= 0 or freeze >= len(model.model.model):
            raise ValueError("凍結層數無效，必須保留可訓練的偵測頭。")
    metrics = []
    csv_path = output / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "map50", "map50_95", "lr"]
    csv_path.write_text(",".join(fields) + "\n", encoding="utf-8")

    def activate_amp(trainer):
        # Ultralytics' AMP probe downloads a separate small model. Configure
        # AMP after setup instead; every actual loss is checked for finiteness.
        enabled = bool(p["amp"] and trainer.device.type == "cuda")
        trainer.args.amp = enabled
        trainer.amp = enabled
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        _install_yolo_stopper(trainer, p["patience"], emit)
        emit({"event": "log", "message": f"使用本機權重；AMP={enabled}，freeze={freeze}，目標梯度累積={p['accumulate']}。"})

    def batch_finished(trainer):
        loss = getattr(trainer, "loss", None)
        if loss is not None and not torch.isfinite(loss).all():
            raise RuntimeError("訓練損失出現 NaN / Inf，請降低學習率或停用 AMP。")

    def epoch_finished(trainer):
        if getattr(trainer, "_workbench_final_eval", False):
            return
        raw = {str(k): float(v) for k, v in (trainer.metrics or {}).items()}
        losses = trainer.tloss
        if isinstance(losses, dict):
            train_loss = sum(float(v) for v in losses.values())
        elif losses is not None:
            train_loss = float(losses.sum())
        else:
            return
        val_values = [v for k, v in raw.items() if k.startswith("val/") and "loss" in k]
        row = {"epoch": int(trainer.epoch) + 1, "train_loss": train_loss,
               "lr": float(trainer.optimizer.param_groups[0]["lr"])}
        if val_values:
            row["val_loss"] = sum(val_values)
        for key, value in raw.items():
            if "mAP50-95" in key:
                row["map50_95"] = value
            elif "mAP50" in key:
                row["map50"] = value
        if any(not math.isfinite(x) for x in row.values()):
            raise RuntimeError("訓練或驗證指標出現 NaN / Inf。")
        metrics.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        _write_json(output / "metrics.json", metrics)
        emit({"event": "metric", **row})
        emit({"event": "progress", "progress": min(0.98, row["epoch"] / p["epochs"]),
              "message": f"Epoch {row['epoch']} / {p['epochs']} 完成"})

    model.add_callback("on_pretrain_routine_end", activate_amp)
    model.add_callback("on_train_batch_end", batch_finished)
    model.add_callback("on_fit_epoch_end", epoch_finished)
    kwargs = dict(data=str(safe_yaml), epochs=p["epochs"], batch=p["batch_size"], imgsz=p["imgsz"],
                  lr0=p["lr"], lrf=p["lrf"], optimizer=p["optimizer"], weight_decay=p["weight_decay"],
                  momentum=p["momentum"], patience=p["patience"], seed=p["seed"], workers=p["workers"],
                  device=str(device.index or 0) if device.type == "cuda" else "cpu", amp=False,
                  freeze=freeze, cos_lr=p["scheduler"] == "cosine", warmup_epochs=p["warmup_epochs"],
                  nbs=p["batch_size"] * p["accumulate"], box=p["box"], cls=p["cls"],
                  project=str(output), name="train", exist_ok=True, save=True, val=True,
                  plots=False, pretrained=True, resume=False, cache=False, deterministic=True)
    if "dfl" in p:
        kwargs["dfl"] = p["dfl"]
    if not p["augmentation"]:
        kwargs.update(mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0, scale=0,
                      shear=0, perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0)
    model.train(trainer=LocalDetectionTrainer, **kwargs)
    trainer = model.trainer
    best = Path(trainer.best)
    last = Path(trainer.last)
    if not best.is_file() or not last.is_file() or not metrics:
        raise RuntimeError("訓練未產生完整的 best / last checkpoint 與指標，無法標記成功。")
    final_best = output / "best.pt"
    final_last = output / "last.pt"
    shutil.copy2(best, final_best)
    shutil.copy2(last, final_last)
    _write_json(output / "workbench_training.json", {"schema_version": 1, "model_id": request["model_id"],
                "classes": dataset["classes"], "checkpoint_format": "ultralytics"})
    _write_json(output / "effective_training.json", {"device": str(device), "amp": bool(p["amp"] and device.type == "cuda"),
                "freeze_layers": freeze, "actual_batch_size": int(trainer.batch_size), "epochs_completed": len(metrics),
                "early_stopping": trainer.stopper.summary(),
                "requested_params": p, "ultralytics_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(trainer.args).items()}})
    emit({"event": "progress", "progress": 1.0, "message": "訓練與驗證完成，已保存最佳與最後權重"})
    artifacts = {"best": str(final_best), "last": str(final_last), "checkpoint_metadata": str(output / "workbench_training.json"),
            "metrics": str(csv_path), "metrics_json": str(output / "metrics.json"),
            "config": str(output / "training_config.json"), "effective_config": str(output / "effective_training.json")}
    if (output / "coco_conversion.json").is_file():
        artifacts["coco_conversion"] = str(output / "coco_conversion.json")
    return {"checkpoint_path": str(final_best), "artifacts": artifacts,
            "summary": f"完成 {len(metrics)} 輪 YOLO 監督微調；最佳權重 best.pt，最後權重 last.pt。",
            "training": {"model_id": request["model_id"], "checkpoint_format": "ultralytics", "classes": dataset["classes"],
                         "early_stopping": trainer.stopper.summary()}}


def sample_frame_indices(total: int, count: int, training: bool = False, rng=None) -> list[int]:
    """Uniform temporal coverage; repeat real frames only when a clip is short."""
    if total < 1 or count < 1:
        raise ValueError("影片沒有可解碼影格，或影格數無效。")
    rng = rng or random
    if training and total >= count:
        return [rng.randrange(i * total // count, max(i * total // count + 1, (i + 1) * total // count))
                for i in range(count)]
    return [min(total - 1, int(i * (total - 1) / max(count - 1, 1))) for i in range(count)]


def _decode_video(path: str, count: int, augment: bool):
    import cv2
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"無法開啟影片：{Path(path).name}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total, count, augment)
        frames = []
        cached = {}
        for index in indices:
            if index not in cached:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"影片影格 {index} 無法解碼：{Path(path).name}")
                cached[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cached[index])
        # One shared spatial transform across the clip preserves temporal order.
        if augment and random.random() < 0.5:
            frames = [cv2.flip(frame, 1) for frame in frames]
        return frames
    finally:
        capture.release()


class _VideoSamples:
    """Picklable Windows DataLoader dataset without eager torch imports."""
    def __init__(self, samples, processor, frames, augment):
        self.samples = samples
        self.processor = processor
        self.frames = frames
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        video = _decode_video(sample["path"], self.frames, self.augment)
        pixels = self.processor(video, return_tensors="pt")["pixel_values"][0]
        return pixels, sample["label"]


class _ActionSamples:
    def __init__(self, samples, metadata, augment):
        self.samples = samples
        self.metadata = metadata
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from .inference import prepare_action_tensor
        sample = self.samples[index]
        frames = _decode_video(sample["path"], int(self.metadata["num_frames"]), self.augment)
        tensor = prepare_action_tensor(frames, self.metadata, "cpu")
        # Inference preprocessing includes a batch dimension; DataLoader adds it.
        tensor = [part.squeeze(0) for part in tensor] if isinstance(tensor, (list, tuple)) else tensor.squeeze(0)
        return tensor, sample["label"]


def _to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return [_to_device(part, device) for part in tensor]
    return tensor.to(device, non_blocking=True)


def _seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _classification_loss(logits, target, p):
    import torch
    import torch.nn.functional as functional
    loss = functional.cross_entropy(logits, target, reduction="none", label_smoothing=p["label_smoothing"])
    if p["loss"] == "focal":
        probability = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        loss = (1.0 - probability).pow(p["focal_gamma"]) * loss
    elif p["loss"] != "cross_entropy":
        raise ValueError(f"未知的分類損失：{p['loss']}")
    return loss.mean()


def _build_optimizer(parameters, p):
    import torch
    name = p["optimizer"]
    common = {"lr": p["lr"], "weight_decay": p["weight_decay"]}
    if name == "SGD":
        return torch.optim.SGD(parameters, momentum=p["momentum"], **common)
    if name == "Adam":
        return torch.optim.Adam(parameters, **common)
    if name == "AdamW":
        return torch.optim.AdamW(parameters, **common)
    if name == "Adagrad":
        return torch.optim.Adagrad(parameters, **common)
    if name == "RMSprop":
        return torch.optim.RMSprop(parameters, momentum=p["momentum"], **common)
    raise ValueError(f"不支援的 optimizer：{name}")


def _build_scheduler(optimizer, p):
    import torch
    if p["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"], eta_min=p["min_lr"])
    if p["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=p["step_size"], gamma=p["step_gamma"])
    if p["scheduler"] == "none":
        return None
    raise ValueError(f"不支援的 scheduler：{p['scheduler']}")


def _check_video_samples(dataset):
    root = Path(dataset["details"]["root"]).resolve()
    splits = []
    for split in ("train", "val"):
        samples = dataset["details"].get(f"{split}_samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{split} 沒有已驗證的影片樣本。")
        checked = []
        for sample in samples:
            path = Path(sample["path"]).resolve()
            label = sample.get("label")
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"{split} 影片不存在或超出資料集範圍，請重新驗證。")
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(dataset["classes"]):
                raise ValueError(f"{split} 的 label 超出類別範圍。")
            checked.append({"path": str(path), "label": label})
        splits.append(checked)
    return splits


def _load_vivit(weight, classes):
    import torch
    from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
    model_dir = weight if weight.is_dir() else weight.parent
    config = VivitConfig.from_pretrained(str(model_dir), local_files_only=True)
    num_frames = int(getattr(config, "num_frames", 32))
    processor = VivitImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    config.num_labels = len(classes)
    config.id2label = dict(enumerate(classes))
    config.label2id = {label: index for index, label in enumerate(classes)}
    model, loading = VivitForVideoClassification.from_pretrained(
        str(model_dir), config=config, local_files_only=True,
        ignore_mismatched_sizes=True, output_loading_info=True, weights_only=True)
    # Only the classification head may change shape. A partially missing trunk
    # must not quietly become a random-init training run.
    missing = list(loading.get("missing_keys", []))
    mismatched = loading.get("mismatched_keys", [])
    bad = [key for key in missing if not key.startswith("classifier.")]
    for item in mismatched:
        key = item[0] if isinstance(item, (tuple, list)) else item
        if not str(key).startswith("classifier."):
            bad.append(str(key))
    if bad or loading.get("error_msgs"):
        raise RuntimeError(f"預訓練主幹權重不相容：{bad or loading['error_msgs']}")
    # Even equal class counts do not imply identical label semantics.
    model.classifier.reset_parameters()
    return model, model.classifier, processor, {"num_frames": num_frames}


def _load_action_classifier(model_id, classes, model_weight_path=None):
    import torch
    from .inference import create_action_model
    model, metadata = create_action_model(model_id, device="cpu", model_weight_path=model_weight_path)
    model.float()
    if model_id in {"x3d_l", "slowfast_r101_8x8"}:
        head = model.blocks[-1]
        if not isinstance(head.proj, torch.nn.Linear):
            raise RuntimeError("模型分類頭不是預期的 Linear 結構。")
        head.proj = torch.nn.Linear(head.proj.in_features, len(classes))
        head.activation = torch.nn.Identity()
        classifier = head.proj
    elif model_id == "videomaev2_giant_k710":
        if not isinstance(model.head, torch.nn.Linear):
            raise RuntimeError("VideoMAE 分類頭不是預期的 Linear 結構。")
        model.head = torch.nn.Linear(model.head.in_features, len(classes))
        classifier = model.head
    else:
        raise ValueError(f"不支援的行為分類器：{model_id}")
    metadata = dict(metadata)
    metadata["labels"] = list(classes)
    metadata["outputs_probabilities"] = False
    return model, classifier, metadata


def _train_classifier(request: dict, p: dict, output: Path, emit: Emit) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    dataset = _validated_dataset(request, "video_csv")
    train_samples, val_samples = _check_video_samples(dataset)
    weight = _catalog_weight(request["model_id"], request.get("model_weight_path"))
    device = _resolve_device(p["device"])
    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(p["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classes = dataset["classes"]
    is_vivit = request["model_id"] in VIVIT_IDS
    if is_vivit:
        model, classifier, processor, metadata = _load_vivit(weight, classes)
        checkpoint_format = "huggingface_safetensors"
    else:
        model, classifier, metadata = _load_action_classifier(request["model_id"], classes, weight)
        processor = None
        checkpoint_format = "action_state_dict_safetensors"
    num_frames = int(metadata["num_frames"])
    if p["strategy"] == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)
    model.to(device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = _build_optimizer(trainable, p)
    scheduler = _build_scheduler(optimizer, p)
    amp = bool(p["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(p["seed"])
    common = dict(batch_size=p["batch_size"], num_workers=p["workers"], pin_memory=device.type == "cuda",
                  worker_init_fn=_seed_worker, persistent_workers=p["workers"] > 0)
    if is_vivit:
        train_data = _VideoSamples(train_samples, processor, num_frames, p["augmentation"])
        val_data = _VideoSamples(val_samples, processor, num_frames, False)
    else:
        train_data = _ActionSamples(train_samples, metadata, p["augmentation"])
        val_data = _ActionSamples(val_samples, metadata, False)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    metrics = []
    csv_path = output / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "accuracy", "train_accuracy", "lr", "seconds"]
    csv_path.write_text(",".join(fields) + "\n", encoding="utf-8")
    stopper = EarlyStoppingTracker(p["patience"], "min", "val_loss", emit)
    emit({"event": "log", "message": f"{request['model_id']}：{num_frames} 個真實影格 / 影片，{len(classes)} 類，"
          f"訓練 {sum(x.numel() for x in trainable):,} 參數，裝置 {device}，AMP={amp}。"})
    _write_json(output / "effective_training.json", {"device": str(device), "amp": amp,
                "num_frames": num_frames, "classes": classes, "params": p,
                "augmentation": "temporal stratified sampling + clip-consistent horizontal flip" if p["augmentation"] else "none",
                "validation": "uniform full-clip sampling, no augmentation",
                "head_reset": True, "checkpoint_format": checkpoint_format})

    def save_checkpoint(name, epoch):
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        if is_vivit:
            model.save_pretrained(str(destination), safe_serialization=True)
            processor.save_pretrained(str(destination))
        else:
            from safetensors.torch import save_file
            state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
            save_file(state, str(destination / "model.safetensors"))
            _write_json(destination / "model_config.json", {"model_id": request["model_id"],
                        "num_classes": len(classes), "classes": classes, "preprocessing": metadata,
                        "checkpoint_format": checkpoint_format, "head_activation": "identity"})
        _write_json(destination / "workbench_training.json", {"model_id": request["model_id"], "epoch": epoch,
                    "classes": classes, "params": p, "dataset_id": dataset.get("id"), "inference_input": "video"})

    for epoch in range(1, p["epochs"] + 1):
        started = time.monotonic()
        model.train()
        if p["strategy"] == "head_only":
            model.eval()
            classifier.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        train_correct = 0
        train_seen = 0
        batch_count = len(train_loader)
        for index, (pixels, labels) in enumerate(train_loader):
            pixels, labels = _to_device(pixels, device), labels.to(device, non_blocking=True)
            group_start = (index // p["accumulate"]) * p["accumulate"]
            group_end = min(group_start + p["accumulate"], batch_count)
            group_sample_count = min(len(train_samples), group_end * p["batch_size"]) - group_start * p["batch_size"]
            with torch.autocast(device_type=device.type, dtype=torch.float16 if amp else torch.bfloat16, enabled=amp):
                logits = model(pixel_values=pixels).logits if is_vivit else model(pixels)
                loss = _classification_loss(logits, labels, p)
            if not torch.isfinite(loss):
                raise RuntimeError("訓練損失出現 NaN / Inf，請降低學習率或停用 AMP。")
            scaler.scale(loss * (labels.shape[0] / group_sample_count)).backward()
            if index + 1 == group_end:
                if p["gradient_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, p["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss_total += loss.detach().item() * labels.shape[0]
            train_correct += (logits.detach().argmax(-1) == labels).sum().item()
            train_seen += labels.shape[0]
            emit({"event": "progress", "progress": min(0.98, (epoch - 1 + 0.8 * (index + 1) / batch_count) / p["epochs"]),
                  "message": f"Epoch {epoch}/{p['epochs']} · train batch {index + 1}/{batch_count}"})
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_seen = 0
        with torch.inference_mode():
            for pixels, labels in val_loader:
                pixels, labels = _to_device(pixels, device), labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16 if amp else torch.bfloat16, enabled=amp):
                    logits = model(pixel_values=pixels).logits if is_vivit else model(pixels)
                    loss = _classification_loss(logits, labels, p)
                if not torch.isfinite(loss):
                    raise RuntimeError("驗證損失出現 NaN / Inf。")
                val_loss_total += loss.item() * labels.shape[0]
                val_correct += (logits.argmax(-1) == labels).sum().item()
                val_seen += labels.shape[0]
        row = {"epoch": epoch, "train_loss": train_loss_total / train_seen,
               "val_loss": val_loss_total / val_seen, "accuracy": val_correct / val_seen,
               "train_accuracy": train_correct / train_seen, "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.monotonic() - started, 3)}
        metrics.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        _write_json(output / "metrics.json", metrics)
        save_checkpoint("last", epoch)
        should_stop = stopper(epoch, row["val_loss"])
        if stopper.improved:
            save_checkpoint("best", epoch)
        emit({"event": "metric", **row})
        emit({"event": "log", "message": f"Epoch {epoch}：train loss={row['train_loss']:.5f}，"
              f"val loss={row['val_loss']:.5f}，val accuracy={row['accuracy']:.1%}"})
        if scheduler is not None:
            scheduler.step()
        if should_stop:
            break
    best_archive = Path(shutil.make_archive(str(output / "best"), "zip", root_dir=output / "best"))
    last_archive = Path(shutil.make_archive(str(output / "last"), "zip", root_dir=output / "last"))
    emit({"event": "progress", "progress": 1.0, "message": "微調完成，已保存 safetensors、前處理設定與類別名稱"})
    return {"checkpoint_path": str(best_archive), "artifacts": {"best": str(best_archive), "last": str(last_archive),
            "metrics": str(csv_path), "metrics_json": str(output / "metrics.json"),
            "config": str(output / "training_config.json"), "effective_config": str(output / "effective_training.json")},
            "summary": f"完成 {len(metrics)} 輪行為分類微調；最佳 epoch {stopper.best_epoch}，val loss {stopper.best_value:.5f}。"
                       "ZIP 含 safetensors、模型與前處理設定。",
            "training": {"model_id": request["model_id"], "checkpoint_format": checkpoint_format,
                         "checkpoint_directory": str(output / "best"), "classes": classes, "best_epoch": stopper.best_epoch,
                         "early_stopping": stopper.summary()}}
