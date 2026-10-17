"""Strict dataset validation and bounded ZIP extraction.

Validation never creates labels or silently splits a user's data. Paths in a
dataset must stay inside its declared root; ZIP uploads reject links and escapes.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FORMATS = {"yolo_detection", "coco_detection", "imagefolder", "video_csv"}
MAX_DATASET_FILES = 100_000
MAX_METADATA_BYTES = 100 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED = 8 * 1024**3
MAX_ZIP_ENTRY = 2 * 1024**3


class DatasetError(ValueError):
    pass


def _inside(path: Path, root: Path) -> Path:
    """Containment must hold before and after resolving Windows/Unix symlinks."""
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DatasetError(f"資料路徑不可超出資料集根目錄：{path}")
    current = candidate
    while current != current.parent and current != root:
        if current.is_symlink():
            raise DatasetError(f"資料集內不接受符號連結：{path}")
        current = current.parent
    return resolved


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise DatasetError(f"找不到必要檔案：{path.name}")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise DatasetError(f"標註檔過大：{path.name}")
    return path.read_text(encoding="utf-8-sig")


def _classes(values: Any) -> list[str]:
    if isinstance(values, dict):
        try:
            indexed = {int(k): v for k, v in values.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetError("類別索引必須是從 0 起的連續整數。") from exc
        if set(indexed) != set(range(len(indexed))):
            raise DatasetError("類別索引必須是從 0 起的連續整數。")
        values = [indexed[i] for i in range(len(indexed))]
    if not isinstance(values, list) or not values:
        raise DatasetError("必須提供非空的類別名稱清單。")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise DatasetError("類別名稱必須是非空字串。")
    values = [v.strip() for v in values]
    if len(set(values)) != len(values):
        raise DatasetError("類別名稱不可重複。")
    return values


def _image_size(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise DatasetError(f"找不到支援的影像：{path.name}")
    try:
        with Image.open(path) as image:
            size = image.size
            expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".bmp": "BMP"}
            if image.format != expected[path.suffix.lower()]:
                raise DatasetError(f"影像副檔名與內容格式不符：{path.name}")
            image.verify()
        if not all(0 < v <= 32768 for v in size) or size[0] * size[1] > 100_000_000:
            raise DatasetError(f"影像尺寸超出限制：{path.name}")
        return size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise DatasetError(f"影像無法讀取：{path.name}") from exc


def media_kind(path: Path, *, verify_image: bool = True) -> str:
    """Recognize media by both a known extension and actual container signature."""
    if not path.is_file() or path.stat().st_size == 0:
        raise DatasetError("媒體檔案不存在或為空。")
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        if verify_image:
            _image_size(path)
        return "image"
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise DatasetError("僅接受 JPG、PNG、WEBP、BMP 或 MP4、MOV、AVI、MKV、WEBM。")
    with path.open("rb") as stream:
        head = stream.read(64)
    ext = path.suffix.lower()
    valid = (
        (ext in {".mp4", ".mov"} and len(head) >= 12 and head[4:8] == b"ftyp")
        or (ext == ".avi" and head[:4] == b"RIFF" and head[8:12] == b"AVI ")
        or (ext in {".mkv", ".webm"} and head[:4] == b"\x1aE\xdf\xa3")
    )
    if not valid:
        raise DatasetError(f"影片副檔名與內容格式不符：{path.name}")
    return "video"


def _ensure_separate(train: list[str], val: list[str]) -> None:
    if not train or not val:
        raise DatasetError("train 與 val 都必須包含實際樣本，不能為空。")
    if len(train) + len(val) > MAX_DATASET_FILES:
        raise DatasetError(f"資料集最多接受 {MAX_DATASET_FILES:,} 個樣本。")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise DatasetError("同一 split 不可包含重複樣本。")
    if set(train) & set(val):
        raise DatasetError("train 與 val 不可使用相同檔案，請先正確拆分資料。")


def _scan_images(folder: Path, root: Path) -> list[Path]:
    if not folder.is_dir():
        raise DatasetError(f"找不到影像資料夾：{folder}")
    result: list[Path] = []
    for current, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs:
            _inside(Path(current) / name, root)
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                result.append(_inside(Path(current) / name, root))
                if len(result) > MAX_DATASET_FILES:
                    raise DatasetError("資料集檔案數超過限制。")
    return sorted(result)


def _yolo_split(value: Any, root: Path) -> list[Path]:
    entries = value if isinstance(value, list) else [value]
    images: list[Path] = []
    if not entries or any(not isinstance(v, str) or not v for v in entries):
        raise DatasetError("YAML 的 train / val 必須是資料夾、圖片清單 .txt 或其清單。")
    for entry in entries:
        item = _inside(Path(entry), root)
        if item.is_dir():
            images.extend(_scan_images(item, root))
        elif item.suffix.lower() == ".txt":
            for raw in _read_text(item).splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                image = _inside(Path(raw), root)
                if not image.is_file() and not Path(raw).is_absolute():
                    image = _inside(item.parent / raw, root)
                images.append(image)
        else:
            raise DatasetError(f"不支援的 YOLO split 路徑：{entry}")
        if len(images) > MAX_DATASET_FILES:
            raise DatasetError("資料集檔案數超過限制。")
    return images


def _validate_yolo(path: Path) -> dict[str, Any]:
    if path.is_dir():
        candidates = [p for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml") if (p := path / name).is_file()]
        if len(candidates) != 1:
            raise DatasetError("YOLO 請指定資料集 YAML；ZIP 根目錄須有唯一 data.yaml 或 dataset.yaml。")
        path = candidates[0]
    config = yaml.safe_load(_read_text(path))
    if not isinstance(config, dict):
        raise DatasetError("YOLO YAML 必須是設定物件。")
    if "download" in config:
        raise DatasetError("資料集 YAML 不接受 download 指令，請先備妥本機資料。")
    root_value = config.get("path", ".")
    if not isinstance(root_value, str):
        raise DatasetError("YOLO path 必須是本機資料夾字串。")
    root = (path.parent / root_value).resolve()
    if not root.is_dir():
        raise DatasetError("YOLO path 指向不存在的資料夾。")
    classes = _classes(config.get("names"))
    if "nc" in config and config["nc"] != len(classes):
        raise DatasetError("YOLO nc 與 names 數量不一致。")
    train, val = _yolo_split(config.get("train"), root), _yolo_split(config.get("val"), root)
    _ensure_separate([str(p) for p in train], [str(p) for p in val])
    box_count = 0
    for image in train + val:
        _image_size(image)
        parts = list(image.relative_to(root).parts)
        if "images" not in parts:
            raise DatasetError("YOLO 影像必須放在 images/ 路徑下，以對應 labels/ 標註。")
        parts[max(i for i, part in enumerate(parts) if part == "images")] = "labels"
        label = _inside(Path(*parts).with_suffix(".txt"), root)
        if not label.is_file():
            raise DatasetError(f"缺少標註 {label.relative_to(root)}；背景圖片也需有空白 .txt。")
        for line_number, line in enumerate(_read_text(label).splitlines(), 1):
            if not line.strip():
                continue
            parts_ = line.split()
            if len(parts_) != 5:
                raise DatasetError(f"{label.name}:{line_number} 應為 class x_center y_center width height 五欄。")
            try:
                cls = int(parts_[0])
                x, y, w, h = [float(v) for v in parts_[1:]]
            except ValueError as exc:
                raise DatasetError(f"{label.name}:{line_number} 標註含無效數字。") from exc
            if not 0 <= cls < len(classes):
                raise DatasetError(f"{label.name}:{line_number} 類別索引超出範圍。")
            if not all(math.isfinite(v) for v in (x, y, w, h)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise DatasetError(f"{label.name}:{line_number} 邊界框需為 0–1 正規化座標且寬高大於 0。")
            if x - w / 2 < -1e-5 or y - h / 2 < -1e-5 or x + w / 2 > 1 + 1e-5 or y + h / 2 > 1 + 1e-5:
                raise DatasetError(f"{label.name}:{line_number} 邊界框超出影像範圍。")
            box_count += 1
    return dict(path=str(path.resolve()), classes=classes, train_count=len(train), val_count=len(val), details={"root": str(root), "yaml_path": str(path.resolve()), "train_images": [str(p) for p in train], "val_images": [str(p) for p in val], "box_count": box_count})


def _validate_imagefolder(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise DatasetError("ImageFolder 路徑必須是資料夾。")
    train_dir, val_dir = _inside(root / "train", root), _inside(root / "val", root)
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise DatasetError("ImageFolder 必須有 train/類別/圖片 與 val/類別/圖片。")
    classes = _classes(sorted(p.name for p in train_dir.iterdir() if p.is_dir()))
    if set(classes) != {p.name for p in val_dir.iterdir() if p.is_dir()}:
        raise DatasetError("train 與 val 的類別資料夾必須完全一致。")
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split in splits:
        for label, name in enumerate(classes):
            images = _scan_images(_inside(root / split / name, root), root)
            if not images:
                raise DatasetError(f"{split}/{name} 沒有影像。")
            for image in images:
                _image_size(image)
                splits[split].append({"path": str(image), "label": label})
    _ensure_separate([x["path"] for x in splits["train"]], [x["path"] for x in splits["val"]])
    return dict(path=str(root), classes=classes, train_count=len(splits["train"]), val_count=len(splits["val"]), details={"root": str(root), "train_dir": str(train_dir), "val_dir": str(val_dir), "train_samples": splits["train"], "val_samples": splits["val"]})


def _validate_video_csv(root: Path) -> dict[str, Any]:
    classes = _classes([x.strip() for x in _read_text(_inside(root / "classes.txt", root)).splitlines() if x.strip()])
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split in splits:
        csv_path = _inside(root / f"{split}.csv", root)
        reader = csv.DictReader(io.StringIO(_read_text(csv_path)))
        if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise DatasetError(f"{split}.csv 必須包含唯一的 path,label 欄位。")
        for row_number, row in enumerate(reader, 2):
            raw, label = row.get("path", ""), row.get("label", "")
            if not isinstance(raw, str) or not raw.strip() or not isinstance(label, str) or not label.strip():
                raise DatasetError(f"{split}.csv:{row_number} path 與 label 不可空白。")
            video = _inside(Path(raw.strip()), root)
            if media_kind(video) != "video":
                raise DatasetError(f"{split}.csv:{row_number} 必須指向影片。")
            label = label.strip()
            if label in classes:
                label_id = classes.index(label)
            else:
                try:
                    label_id = int(label)
                except ValueError as exc:
                    raise DatasetError(f"{split}.csv:{row_number} 未知類別 {label}。") from exc
            if not 0 <= label_id < len(classes):
                raise DatasetError(f"{split}.csv:{row_number} 類別索引超出範圍。")
            splits[split].append({"path": str(video), "label": label_id})
            if len(splits[split]) > MAX_DATASET_FILES:
                raise DatasetError("資料集檔案數超過限制。")
    _ensure_separate([x["path"] for x in splits["train"]], [x["path"] for x in splits["val"]])
    return dict(path=str(root), classes=classes, train_count=len(splits["train"]), val_count=len(splits["val"]), details={"root": str(root), "train_csv": str(root / "train.csv"), "val_csv": str(root / "val.csv"), "train_samples": splits["train"], "val_samples": splits["val"]})


def _validate_coco(root: Path) -> dict[str, Any]:
    documents = {split: json.loads(_read_text(_inside(root / f"{split}.json", root))) for split in ("train", "val")}
    categories = documents["train"].get("categories", [])
    if not isinstance(categories, list) or not categories:
        raise DatasetError("COCO categories 不可為空。")
    if any(not isinstance(x, dict) or not isinstance(x.get("id"), int) for x in categories):
        raise DatasetError("COCO 類別需要整數 id。")
    categories = sorted(categories, key=lambda x: x["id"])
    classes = _classes([x.get("name") for x in categories])
    ids = [x["id"] for x in categories]
    if len(set(ids)) != len(ids):
        raise DatasetError("COCO category id 不可重複。")
    if {x.get("id"): x.get("name") for x in documents["val"].get("categories", [])} != {x["id"]: x["name"] for x in categories}:
        raise DatasetError("COCO train 與 val 的 categories 必須一致。")
    image_root = _inside(root / "images", root) if (root / "images").is_dir() else root
    paths: dict[str, list[str]] = {"train": [], "val": []}
    resolved_images: dict[str, dict[str, str]] = {"train": {}, "val": {}}
    annotation_counts: dict[str, int] = {}
    crowd_counts: dict[str, int] = {}
    for split, doc in documents.items():
        image_ids: dict[int, tuple[int, int]] = {}
        if not isinstance(doc.get("images"), list) or not isinstance(doc.get("annotations"), list):
            raise DatasetError(f"{split}.json 必須有 images 與 annotations 清單。")
        for info in doc["images"]:
            if not isinstance(info, dict) or not isinstance(info.get("id"), int) or not isinstance(info.get("file_name"), str):
                raise DatasetError(f"{split}.json 的 image 需要整數 id 與 file_name。")
            if info["id"] in image_ids:
                raise DatasetError(f"{split}.json 的 image id 重複。")
            image = _inside(image_root / info["file_name"], root)
            size = _image_size(image)
            if tuple(info.get(key) for key in ("width", "height")) != size:
                raise DatasetError(f"{info['file_name']} 的標註尺寸與實際影像不同。")
            image_ids[info["id"]] = size
            paths[split].append(str(image))
            resolved_images[split][str(info["id"])] = str(image)
        seen_annotations = set()
        crowd_count = 0
        for annotation in doc["annotations"]:
            if not isinstance(annotation, dict) or not isinstance(annotation.get("id"), int) or annotation["id"] in seen_annotations:
                raise DatasetError(f"{split}.json 的 annotation id 缺少或重複。")
            seen_annotations.add(annotation["id"])
            if annotation.get("iscrowd", 0) not in (0, 1):
                raise DatasetError("COCO iscrowd 必須是 0 或 1。")
            crowd_count += int(annotation.get("iscrowd", 0))
            if annotation.get("image_id") not in image_ids or annotation.get("category_id") not in ids:
                raise DatasetError(f"{split}.json 的 annotation 參照不存在的影像或類別。")
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox):
                raise DatasetError("COCO bbox 必須是 [x,y,width,height] 四個有限數字。")
            x, y, w, h = bbox
            width, height = image_ids[annotation["image_id"]]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
                raise DatasetError("COCO bbox 超出實際影像範圍。")
        annotation_counts[split] = len(doc["annotations"])
        crowd_counts[split] = crowd_count
    _ensure_separate(paths["train"], paths["val"])
    return dict(path=str(root), classes=classes, train_count=len(paths["train"]), val_count=len(paths["val"]), details={"root": str(root), "train_json": str(root / "train.json"), "val_json": str(root / "val.json"), "image_root": str(image_root), "category_id_to_index": {str(k): i for i, k in enumerate(ids)}, "annotation_counts": annotation_counts, "crowd_counts": crowd_counts, "train_images": resolved_images["train"], "val_images": resolved_images["val"]})


def validate_dataset(path: str | Path, format: str, name: str | None = None, allowed_root: Path | None = None) -> dict[str, Any]:
    try:
        if format not in FORMATS:
            raise DatasetError("不支援的資料集格式。")
        if not str(path).strip():
            raise DatasetError("請輸入資料集路徑。")
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise DatasetError("資料集路徑不存在。")
        if allowed_root is not None:
            boundary = allowed_root.resolve()
            _inside(resolved, boundary)
            if format == "yolo_detection":
                config_path = resolved
                if resolved.is_dir():
                    candidates = [resolved / name for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml") if (resolved / name).is_file()]
                    if len(candidates) != 1:
                        raise DatasetError("ZIP 根目錄須有唯一 data.yaml 或 dataset.yaml。")
                    config_path = candidates[0]
                config = yaml.safe_load(_read_text(config_path))
                if not isinstance(config, dict) or not isinstance(config.get("path", "."), str):
                    raise DatasetError("YOLO YAML 格式不正確。")
                _inside(config_path.parent / config.get("path", "."), boundary)
        validators = {"yolo_detection": _validate_yolo, "imagefolder": _validate_imagefolder, "video_csv": _validate_video_csv, "coco_detection": _validate_coco}
        dataset = validators[format](resolved)
        dataset.update(format=format, name=(name or resolved.stem).strip()[:160], num_classes=len(dataset["classes"]))
        warnings = []
        if dataset["train_count"] < 20:
            warnings.append("訓練樣本少於 20，適合流程測試，無法代表泛化表現。")
        if any(dataset["details"].get("crowd_counts", {}).values()):
            warnings.append("YOLO 轉換會忽略 iscrowd 標註；YOLO 格式不支援 crowd 忽略區域。")
        return {"valid": True, "dataset": dataset, "errors": [], "warnings": warnings}
    except (DatasetError, OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, yaml.YAMLError) as exc:
        return {"valid": False, "errors": [str(exc) or "資料集格式不正確。"], "warnings": []}


def extract_zip_safely(archive: Path, destination: Path) -> Path:
    """Preflight all entries before extraction, then independently bound reads."""
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if not entries or len(entries) > MAX_DATASET_FILES:
            raise DatasetError("ZIP 為空或檔案數超過 100,000。")
        total = 0
        seen: set[str] = set()
        for entry in entries:
            raw = entry.filename.replace("\\", "/")
            name = PurePosixPath(raw)
            if not raw or raw.startswith("/") or any(p in {"..", ""} for p in name.parts) or any(":" in p for p in name.parts):
                raise DatasetError("ZIP 包含不安全路徑。")
            if any(p.rstrip(" .").split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]} or p != p.rstrip(" .") for p in name.parts):
                raise DatasetError("ZIP 包含不支援的 Windows 檔名。")
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise DatasetError("ZIP 不接受符號連結或特殊裝置。")
            key = str(name).casefold().rstrip("/")
            if key in seen:
                raise DatasetError("ZIP 包含重複或大小寫衝突的路徑。")
            seen.add(key)
            if entry.flag_bits & 1:
                raise DatasetError("ZIP 不接受加密檔案。")
            total += entry.file_size
            if entry.file_size > MAX_ZIP_ENTRY or total > MAX_ZIP_UNCOMPRESSED:
                raise DatasetError("ZIP 解壓縮後超過限制（單檔 2 GB、合計 8 GB）。")
            if entry.file_size > 10 * 1024**2 and entry.file_size / max(1, entry.compress_size) > 200:
                raise DatasetError("ZIP 壓縮比異常，拒絕解壓縮。")
            _inside(root / Path(*name.parts), root)
        written = 0
        for entry in entries:
            target = _inside(root / Path(*PurePosixPath(entry.filename.replace("\\", "/")).parts), root)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("xb") as output:
                entry_written = 0
                while chunk := source.read(1024 * 1024):
                    entry_written += len(chunk)
                    written += len(chunk)
                    if entry_written > MAX_ZIP_ENTRY or written > MAX_ZIP_UNCOMPRESSED:
                        raise DatasetError("ZIP 解壓縮內容超過容量限制。")
                    output.write(chunk)
    # One outer folder is a common archive layout. Do not guess recursively.
    visible = [p for p in root.iterdir() if p.name != "__MACOSX"]
    return visible[0] if len(visible) == 1 and visible[0].is_dir() else root
