import csv
import json
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from backend.datasets import DatasetError, extract_zip_safely, media_kind, validate_dataset


def image(path: Path, size=(20, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def yolo(root: Path):
    for split in ("train", "val"):
        image(root / "images" / split / f"{split}.png")
        label = root / "labels" / split / f"{split}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    config = root / "data.yaml"
    config.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [cat]\n", encoding="utf-8")
    return config


def test_yolo_valid_counts_and_no_silent_labels(tmp_path):
    config = yolo(tmp_path)
    result = validate_dataset(config, "yolo_detection")
    assert result["valid"]
    assert result["dataset"]["train_count"] == 1
    assert result["dataset"]["details"]["box_count"] == 2
    label = tmp_path / "labels/val/val.txt"
    label.unlink()
    result = validate_dataset(config, "yolo_detection")
    assert not result["valid"] and "缺少標註" in result["errors"][0]
    assert not label.exists()


@pytest.mark.parametrize("annotation", ["4 0.5 0.5 0.2 0.2", "0 nan 0.2 0.2 0.2", "0 0.05 0.05 0.5 0.5", "0 0.5 0.5 0 0.1", "0 0.5 0.5 0.2 0.2 0.4"])
def test_yolo_rejects_invalid_annotations(tmp_path, annotation):
    config = yolo(tmp_path)
    (tmp_path / "labels/train/train.txt").write_text(annotation, encoding="utf-8")
    assert not validate_dataset(config, "yolo_detection")["valid"]


def test_yolo_accepts_explicit_background_labels(tmp_path):
    config = yolo(tmp_path)
    (tmp_path / "labels/train/train.txt").write_text("", encoding="utf-8")
    assert validate_dataset(config, "yolo_detection")["valid"]


def test_yolo_rejects_train_val_leak_and_external_path(tmp_path):
    config = yolo(tmp_path)
    config.write_text("train: images/train\nval: images/train\nnames: [cat]\n")
    assert "相同檔案" in validate_dataset(config, "yolo_detection")["errors"][0]
    config.write_text("train: ../outside\nval: images/val\nnames: [cat]\n")
    assert "根目錄" in validate_dataset(config, "yolo_detection")["errors"][0]


def test_yolo_yaml_download_never_runs(tmp_path):
    config = yolo(tmp_path)
    with config.open("a") as stream:
        stream.write("download: echo unsafe\n")
    assert "download" in validate_dataset(config, "yolo_detection")["errors"][0]


def test_imagefolder_requires_matching_nonempty_classes(tmp_path):
    image(tmp_path / "train/cat/one.png")
    image(tmp_path / "val/cat/two.png")
    assert validate_dataset(tmp_path, "imagefolder")["valid"]
    (tmp_path / "val/dog").mkdir()
    assert not validate_dataset(tmp_path, "imagefolder")["valid"]
    (tmp_path / "train/dog").mkdir()
    assert "沒有影像" in validate_dataset(tmp_path, "imagefolder")["errors"][0]


def test_invalid_image_signature_is_rejected(tmp_path):
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not an image")
    with pytest.raises(DatasetError):
        media_kind(fake)
    Image.new("RGB", (10, 10)).save(fake, format="JPEG")
    with pytest.raises(DatasetError):
        media_kind(fake)


def video_csv(root: Path):
    (root / "classes.txt").write_text("walking\nrunning\n", encoding="utf-8")
    for split, label in (("train", "walking"), ("val", "1")):
        (root / f"{split}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 24)
        (root / f"{split}.csv").write_text(f"path,label\n{split}.mp4,{label}\n", encoding="utf-8")


def test_video_csv_normalizes_classes_and_checks_paths(tmp_path):
    video_csv(tmp_path)
    result = validate_dataset(tmp_path, "video_csv")
    assert result["valid"]
    assert result["dataset"]["details"]["train_samples"][0]["label"] == 0
    assert result["dataset"]["details"]["val_samples"][0]["label"] == 1
    (tmp_path / "val.csv").write_text("path,label\nval.mp4,missing\n")
    assert "未知類別" in validate_dataset(tmp_path, "video_csv")["errors"][0]
    (tmp_path / "val.csv").write_text("path,label\n../outside.mp4,0\n")
    assert "根目錄" in validate_dataset(tmp_path, "video_csv")["errors"][0]


def test_video_csv_prevents_empty_or_duplicate_samples(tmp_path):
    video_csv(tmp_path)
    (tmp_path / "val.csv").write_text("path,label\n")
    assert not validate_dataset(tmp_path, "video_csv")["valid"]
    (tmp_path / "val.csv").write_text("path,label\ntrain.mp4,0\n")
    assert "相同檔案" in validate_dataset(tmp_path, "video_csv")["errors"][0]


def coco(root: Path):
    for split in ("train", "val"):
        image(root / "images" / f"{split}.png")
        doc = {"categories": [{"id": 5, "name": "cat"}], "images": [{"id": 1, "file_name": f"{split}.png", "width": 20, "height": 10}], "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 5, 5]}]}
        (root / f"{split}.json").write_text(json.dumps(doc))


def test_coco_validates_reference_and_actual_dimensions(tmp_path):
    coco(tmp_path)
    result = validate_dataset(tmp_path, "coco_detection")
    assert result["valid"]
    assert result["dataset"]["details"]["category_id_to_index"] == {"5": 0}
    path = tmp_path / "train.json"
    doc = json.loads(path.read_text())
    doc["annotations"][0]["image_id"] = 7
    path.write_text(json.dumps(doc))
    assert "參照不存在" in validate_dataset(tmp_path, "coco_detection")["errors"][0]
    doc["annotations"][0]["image_id"] = 1
    doc["images"][0]["width"] = 21
    path.write_text(json.dumps(doc))
    assert "實際影像" in validate_dataset(tmp_path, "coco_detection")["errors"][0]


def test_coco_rejects_bbox_outside_image(tmp_path):
    coco(tmp_path)
    path = tmp_path / "val.json"
    doc = json.loads(path.read_text())
    doc["annotations"][0]["bbox"] = [19, 0, 5, 5]
    path.write_text(json.dumps(doc))
    assert not validate_dataset(tmp_path, "coco_detection")["valid"]


@pytest.mark.parametrize("name", ["../escaped.txt", "/absolute.txt", "C:/escape.txt", "folder/../../escape.txt", "folder\\..\\..\\escape.txt", "file.txt:stream", "NUL.txt", "folder. /file.txt"])
def test_zip_rejects_unsafe_paths_before_writing(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("safe.txt", "safe")
        bundle.writestr(name, "unsafe")
    destination = tmp_path / "unpacked"
    with pytest.raises(DatasetError):
        extract_zip_safely(archive, destination)
    assert not (destination / "safe.txt").exists()


def test_zip_rejects_symlink_and_case_alias(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "../../outside")
    with pytest.raises(DatasetError):
        extract_zip_safely(archive, tmp_path / "symlink")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("file.txt", "1")
        bundle.writestr("FILE.TXT", "2")
    with pytest.raises(DatasetError):
        extract_zip_safely(archive, tmp_path / "alias")


def test_zip_limits_and_wrapper_folder(tmp_path, monkeypatch):
    import backend.datasets as module
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dataset/classes.txt", "walking\n")
    extracted = extract_zip_safely(archive, tmp_path / "good")
    assert extracted.name == "dataset"
    monkeypatch.setattr(module, "MAX_ZIP_UNCOMPRESSED", 2)
    with pytest.raises(DatasetError):
        extract_zip_safely(archive, tmp_path / "too-big")


def test_uploaded_yaml_cannot_reference_other_local_dataset(tmp_path):
    outside = tmp_path / "outside"
    yolo(outside)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "data.yaml").write_text(f"path: {outside.as_posix()}\ntrain: images/train\nval: images/val\nnames: [cat]\n")
    # Explicit local YAML may choose a separate local root, but an uploaded ZIP
    # is confined to its extraction directory before reading any outside media.
    assert validate_dataset(extracted, "yolo_detection")["valid"]
    result = validate_dataset(extracted, "yolo_detection", allowed_root=extracted)
    assert not result["valid"] and "根目錄" in result["errors"][0]
