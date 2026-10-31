"""Portable bundle validation without importing or deserializing ML weights."""
import copy
import json
import os
from pathlib import Path
import shutil

import pytest

from backend import dual_branch


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(root, classes, content):
    root.mkdir(parents=True)
    weight = root / "best.pt"
    weight.write_bytes(content)
    _write_json(root / "workbench_training.json", {"model_id": "yolo26x", "classes": classes})
    return weight


@pytest.fixture
def bundle(tmp_path):
    original = _checkpoint(tmp_path / "source-original", ["person", "dog"], b"original-weight-bytes")
    new = _checkpoint(tmp_path / "source-new", ["person_alias", "bento"], b"new-weight-bytes")
    plan = {"base_classes": ["person", "dog"], "classes": ["person", "dog", "bento"],
            "new_label_map": [0, 2], "class_aliases": {"person_alias": "person"},
            "training_mode": "dual_branch"}
    root = dual_branch.create_bundle("yolo26x", original, new, plan, tmp_path / "bundle",
                                     original_is_checkpoint=True)
    return root, original, new, plan


def test_bundle_copies_are_portable_and_never_alias_source_weights(bundle, tmp_path):
    root, original, new, _ = bundle
    original_bytes, new_bytes = original.read_bytes(), new.read_bytes()
    manifest = _read_json(root / dual_branch.MANIFEST)
    copied_original = root / manifest["branches"]["original"]["path"]
    copied_new = root / manifest["branches"]["new"]["path"]
    assert copied_original.read_bytes() == original_bytes
    assert copied_new.read_bytes() == new_bytes
    assert not os.path.samefile(copied_original, original)
    assert not os.path.samefile(copied_new, new)
    assert all(not Path(branch["path"]).is_absolute() for branch in manifest["branches"].values())
    portable = tmp_path / "another-location" / "portable"
    shutil.copytree(root, portable)
    assert dual_branch.validate_bundle(portable, "yolo26x") == portable.resolve()
    # Changing source originals after export cannot mutate the bundle snapshot.
    original.write_bytes(b"source changed after export")
    new.write_bytes(b"source changed after export")
    assert copied_original.read_bytes() == original_bytes
    assert copied_new.read_bytes() == new_bytes
    assert dual_branch.validate_bundle(root, "yolo26x") == root


def test_same_size_original_mutation_invalidates_cached_sha(bundle):
    root, original, _, _ = bundle
    manifest = _read_json(root / dual_branch.MANIFEST)
    snapshot = root / manifest["branches"]["original"]["path"]
    original_bytes = original.read_bytes()
    dual_branch.validate_bundle(root, "yolo26x")  # Populate the stat-aware cache.
    snapshot.write_bytes(b"X" + original_bytes[1:])
    stat = snapshot.stat()
    os.utime(snapshot, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert snapshot.stat().st_size == len(original_bytes)
    with pytest.raises(ValueError, match="SHA256"):
        dual_branch.validate_bundle(root, "yolo26x")
    assert original.read_bytes() == original_bytes


def test_explicit_aliases_preserve_overlapping_predictions_and_independent_scores(bundle):
    root, _, _, _ = bundle
    manifest = _read_json(root / dual_branch.MANIFEST)
    box = [1, 2, 3, 4]
    old = dual_branch.mapped_predictions([{"label": "person", "score": .8, "box": box}], manifest, "original")
    new = dual_branch.mapped_predictions([{"label": "person_alias", "score": .9, "box": box}], manifest, "new")
    assert old[0]["union_class_id"] == new[0]["union_class_id"] == 0
    assert old[0]["label"] == new[0]["label"] == "person"
    assert new[0]["source_label"] == "person_alias"
    assert [item["score"] for item in old + new] == [.8, .9]
    assert [item["branch"] for item in old + new] == ["original", "new"]
    assert old[0]["box"] == new[0]["box"] == box
    assert manifest["cross_branch_suppression"] is False


def test_exact_source_label_wins_over_underscore_space_normalization():
    manifest = {"classes": ["red_bean", "red bean"], "branches": {"new": {
        "classes": ["red_bean", "red bean"], "label_map": [0, 1]}}}
    result = dual_branch.mapped_predictions([
        {"label": "red_bean", "score": .3}, {"label": "red bean", "score": .6},
        {"label": "RED BEAN", "score": .7}], manifest, "new")
    assert [value["union_class_id"] for value in result] == [0, 1, None]
    assert [value["label"] for value in result] == ["red_bean", "red bean", "RED BEAN"]
    assert result[2]["mapping_status"] == "ambiguous_label"


def test_unique_normalized_fallback_maps_only_unambiguous_source_labels():
    manifest = {"classes": ["bean_curd"], "branches": {"new": {
        "classes": ["bean_curd"], "label_map": [0]}}}
    result = dual_branch.mapped_predictions([
        {"label": "Bean Curd", "score": .3}, {"label": "unknown", "score": .6}], manifest, "new")
    assert result[0]["union_class_id"] == 0
    assert result[1]["union_class_id"] is None
    assert result[1]["label"] == "unknown"
    assert result[1]["mapping_status"] == "outside_scope"


@pytest.mark.parametrize("mutation", ["wrong_new_map", "wrong_plan_map", "wrong_alias", "wrong_base_order"])
def test_manifest_semantic_tampering_is_rejected(bundle, mutation):
    root, _, _, _ = bundle
    path = root / dual_branch.MANIFEST
    manifest = _read_json(path)
    if mutation == "wrong_new_map":
        manifest["branches"]["new"]["label_map"] = [1, 2]
    elif mutation == "wrong_plan_map":
        manifest["class_expansion"]["new_label_map"] = [1, 2]
    elif mutation == "wrong_alias":
        manifest["class_expansion"]["class_aliases"]["person_alias"] = "dog"
    else:
        manifest["base_classes"] = ["dog", "person"]
    _write_json(path, manifest)
    with pytest.raises(ValueError):
        dual_branch.validate_bundle(root, "yolo26x")


def test_alias_mapping_mismatch_fails_before_copying_sources(bundle, tmp_path):
    _, original, new, plan = bundle
    plan = copy.deepcopy(plan)
    plan["new_label_map"] = [1, 2]
    destination = tmp_path / "invalid-mapping"
    with pytest.raises(ValueError, match="名稱／別名"):
        dual_branch.create_bundle("yolo26x", original, new, plan, destination, original_is_checkpoint=True)
    assert original.is_file() and new.is_file()
    assert not (destination / "original").exists()


@pytest.mark.parametrize("relative", ["../outside.pt", "original/../../outside.pt", "C:/outside.pt", "/outside.pt", "new\\best.pt"])
def test_manifest_rejects_absolute_traversal_and_windows_paths(bundle, relative):
    root, _, _, _ = bundle
    path = root / dual_branch.MANIFEST
    manifest = _read_json(path)
    manifest["branches"]["new"]["path"] = relative
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="路徑|範圍"):
        dual_branch.validate_bundle(root, "yolo26x")


def test_nested_bundle_manifest_is_rejected(bundle):
    root, _, _, _ = bundle
    _write_json(root / "new" / dual_branch.MANIFEST, {})
    with pytest.raises(ValueError, match="巢狀"):
        dual_branch.validate_bundle(root, "yolo26x")


def test_symlink_inside_branch_is_rejected(bundle):
    root, _, _, _ = bundle
    link = root / "new" / "linked.pt"
    try:
        link.symlink_to(root / "new" / "best.pt")
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Host cannot create symlinks: {error}")
    with pytest.raises(ValueError, match="連結"):
        dual_branch.validate_bundle(root, "yolo26x")


@pytest.mark.parametrize("field", ["branch", "inventory", "plan"])
def test_malformed_nested_metadata_returns_validation_error_not_attribute_error(bundle, field):
    root, _, _, _ = bundle
    path = root / dual_branch.MANIFEST
    manifest = _read_json(path)
    if field == "branch":
        manifest["branches"]["new"] = "invalid"
    elif field == "inventory":
        manifest["original_inventory"]["best.pt"] = "invalid"
    else:
        manifest["class_expansion"] = "invalid"
    _write_json(path, manifest)
    with pytest.raises(ValueError):
        dual_branch.validate_bundle(root, "yolo26x")
