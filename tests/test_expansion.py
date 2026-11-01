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
            (root / split / "labels" / f"{index}.txt").write_text(f"{index} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "val/images", "names": names}), encoding="utf-8")
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
    assert json.loads((output / "class_expansion.json").read_text(encoding="utf-8"))["base_class_ids"] == [0, 1]


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


@pytest.mark.parametrize("mode", ["expand_frozen", "dual_branch"])
def test_no_replay_modes_plan_and_materialize_only_new_training(tmp_path, monkeypatch, mode):
    raw = request(tmp_path, monkeypatch)
    raw["training_mode"] = mode
    raw.pop("replay_dataset")
    checked = expansion.preview_expansion(raw)
    assert checked["valid"], checked
    assert checked["plan"]["classes"] == ["person", "dog", "bento"]
    assert checked["plan"]["coverage"]["old_val"]["provided"] is False
    assert checked["plan"]["old_evaluation_available"] is False
    assert any("未測量" in item for item in checked["warnings"])
    output = tmp_path / "job"
    output.mkdir()
    prepared = expansion.prepare_expansion(raw, output, lambda _: None)
    assert set(prepared["evaluation_datasets"]) == {"new"}
    assert prepared["dataset"]["train_count"] == 2
    labels = [item["labels"] for item in expansion.dataset_samples(prepared["dataset"])["train"]]
    assert labels == ([[0], [2]] if mode == "expand_frozen" else [[0], [1]])
    assert prepared["dataset"]["classes"] == (["person", "dog", "bento"] if mode == "expand_frozen" else ["people", "bento"])


@pytest.mark.parametrize("mode", ["expand_frozen", "dual_branch"])
def test_optional_partial_old_validation_warns_without_replay_training(tmp_path, monkeypatch, mode):
    raw = request(tmp_path, monkeypatch)
    raw["training_mode"] = mode
    raw["replay_dataset"]["details"]["val_images"] = raw["replay_dataset"]["details"]["val_images"][:1]
    checked = expansion.preview_expansion(raw)
    assert checked["valid"], checked
    assert any("old_val" in warning for warning in checked["warnings"])
    assert not checked["plan"]["uses_old_training_data"]
    output = tmp_path / "job"
    output.mkdir()
    prepared = expansion.prepare_expansion(raw, output, lambda _: None)
    assert prepared["dataset"]["train_count"] == raw["dataset"]["train_count"]
    assert prepared["evaluation_datasets"]["old"]["val_count"] == 1


def test_freeze_mode_cannot_keep_partial_or_full_training_params():
    assert expansion.training_mode_params("yolo26x", "expand_frozen", {"strategy": "full", "freeze": 3}) == {"strategy": "freeze_backbone", "freeze": 0}
    assert expansion.training_mode_params("x3d_l", "expand_frozen", {"strategy": "full"}) == {"strategy": "head_only"}
    assert expansion.training_mode_params("x3d_l", "dual_branch", {"strategy": "full"}) == {"strategy": "full"}


def test_grounding_prompt_scope_can_be_declared_without_old_data():
    assert expansion.source_classes({"model_id": "grounding_dino_base", "base_class_names": ["person", "car"]}) == (["person", "car"], [0, 1])
    assert expansion.source_classes({"model_id": "grounding_dino_base"})[0] == expansion.COCO_NAMES
    for mode in ("expand_frozen", "dual_branch"):
        assert expansion.source_classes({"model_id": "grounding_dino_base", "training_mode": mode,
                                         "replay_dataset": {"classes": ["person"]}})[0] == expansion.COCO_NAMES


@pytest.mark.parametrize("mode", expansion.TRAINING_MODES)
def test_bundle_cannot_be_recursively_used_as_training_source(tmp_path, mode):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "branch_bundle.json").write_text("{}", encoding="utf-8")
    checked = expansion.preview_expansion({"model_id": "yolo26x", "training_mode": mode, "checkpoint_path": str(source)})
    assert not checked["valid"] and "遞迴" in checked["errors"][0]
