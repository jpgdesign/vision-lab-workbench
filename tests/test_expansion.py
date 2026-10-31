import json
from pathlib import Path

import pytest
from PIL import Image

from backend import expansion
from backend.datasets import validate_dataset


def detection(root, names, offset=0):
    import yaml
    for split, seed in (("train", 10), ("val", 90)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for index, name in enumerate(names):
            Image.new("RGB", (32, 32), (seed + offset + index, index, offset)).save(root / split / "images" / f"{index}.png")
            (root / split / "labels" / f"{index}.txt").write_text(f"{index} 0.5 0.5 0.5 0.5\n")
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "val/images", "names": names}))
    result = validate_dataset(str(config), "yolo_detection")
    assert result["valid"], result
    return result["dataset"]


def request(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion, "source_classes", lambda _: (["person", "dog"], [0, 1]))
    return {"kind": "training", "model_id": "yolo26x", "training_mode": "expand_classes",
            "replay_dataset": detection(tmp_path / "old", ["dog", "person"]),
            "dataset": detection(tmp_path / "new", ["people", "bento"], 20),
            "class_aliases": {"people": "person"}}


def test_union_preserves_source_order_and_explicit_alias(tmp_path, monkeypatch):
    result = expansion.preview_expansion(request(tmp_path, monkeypatch))
    assert result["valid"], result
    plan = result["plan"]
    assert plan["classes"] == ["person", "dog", "bento"]
    assert plan["old_label_map"] == [1, 0]
    assert plan["new_label_map"] == [0, 2]
    assert plan["coverage"]["old_train"]["counts"] == {"person": 1, "dog": 1}


def test_materialized_labels_match_heads_and_sources_never_mutate(tmp_path, monkeypatch):
    raw = request(tmp_path, monkeypatch)
    output = tmp_path / "job"
    output.mkdir()
    result = expansion.prepare_expansion(raw, output, lambda _: None)
    checked = validate_dataset(result["dataset"]["path"], "yolo_detection")
    assert checked["valid"], checked
    assert checked["dataset"]["classes"] == ["person", "dog", "bento"]
    train = expansion.dataset_samples(result["dataset"])["train"]
    assert [x["labels"] for x in train] == [[1], [0], [0], [2]]
    assert result["evaluation_datasets"]["old"]["val_count"] == 2
    source = Path(raw["replay_dataset"]["details"]["train_images"][0])
    original = source.read_bytes()
    Path(train[0]["path"]).write_bytes(b"native loader repair")
    assert source.read_bytes() == original
    assert json.loads((output / "class_expansion.json").read_text())["base_class_ids"] == [0, 1]


def test_old_coverage_and_missing_replay_block_training(tmp_path, monkeypatch):
    raw = request(tmp_path, monkeypatch)
    raw["replay_dataset"]["details"]["val_images"] = raw["replay_dataset"]["details"]["val_images"][:1]
    result = expansion.preview_expansion(raw)
    assert not result["valid"] and any("old_val" in x for x in result["errors"])
    raw.pop("replay_dataset")
    assert not expansion.preview_expansion(raw)["valid"]


def test_different_paths_identical_train_val_content_rejected(tmp_path, monkeypatch):
    raw = request(tmp_path, monkeypatch)
    old_train = Path(raw["replay_dataset"]["details"]["train_images"][0])
    new_val = Path(raw["dataset"]["details"]["val_images"][0])
    new_val.write_bytes(old_train.read_bytes())
    result = expansion.preview_expansion(raw)
    assert not result["valid"] and any("洩漏" in x for x in result["errors"])


def test_unknown_alias_and_no_added_class_are_errors(tmp_path, monkeypatch):
    raw = request(tmp_path, monkeypatch)
    raw["class_aliases"] = {"people": "invented"}
    assert not expansion.preview_expansion(raw)["valid"]
    raw["class_aliases"] = {"people": "person", "bento": "dog"}
    assert any("沒有新增" in x for x in expansion.preview_expansion(raw)["errors"])


def test_rfdetr_sparse_coco_ids_and_grounding_scope():
    classes, ids = expansion.source_classes({"model_id": "rf_detr_2xl"})
    assert len(classes) == len(ids) == 80
    assert ids[classes.index("toothbrush")] == 90
    assert 12 not in ids and 0 not in ids
    assert expansion.source_classes({"model_id": "grounding_dino_base", "replay_dataset": {"classes": ["food"]}}) == (["food"], [0])
    with pytest.raises(ValueError, match="分類頭"):
        expansion.source_classes({"model_id": "dinov3_7b"})


def test_standard_never_silently_uses_replay_or_alias():
    assert expansion.preview_expansion({"training_mode": "standard"})["valid"]
    assert not expansion.preview_expansion({"training_mode": "standard", "replay_dataset": {"id": "a"}})["valid"]
