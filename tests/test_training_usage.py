"""Provenance follows actual train bytes, not names or current dataset paths."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.training_usage import MANIFEST, TrainingUsageIndex, record_training_inputs


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def dataset(train, val=(), *, samples=False):
    if samples:
        details = {f"{split}_samples": [{"path": str(path), "label": 0} for path in paths]
                   for split, paths in (("train", train), ("val", val))}
    else:
        details = {"train_images": [str(path) for path in train], "val_images": [str(path) for path in val]}
    return {"format": "imagefolder" if samples else "yolo_detection", "details": details}


def job(name="trained", status="completed"):
    return {"id": name, "kind": "training", "status": status,
            "started_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()}


def entry(path):
    return {"id": path.name, "path": str(path), "name": path.name}


def test_content_matches_renamed_copy_but_not_val_or_name(tmp_path):
    train = write(tmp_path / "train/a.jpg", b"actual-train-content")
    val = write(tmp_path / "val/a.jpg", b"actual-val-content")
    renamed = write(tmp_path / "media/renamed.jpg", train.read_bytes())
    fake = write(tmp_path / "media/便當訓練集_fake.jpg", b"not-trained")
    root = tmp_path / "jobs"
    record_training_inputs({"dataset": dataset([train], [val])}, root / "trained")
    entries = [entry(path) for path in (renamed, val, fake)]
    result = TrainingUsageIndex(root).annotate(entries, [job()])
    assert [row["training_usage"] for row in result] == [
        {"used": True, "job_ids": ["trained"]}, {"used": False, "job_ids": []}, {"used": False, "job_ids": []}]
    assert all("training_usage" not in row for row in entries)
    assert set(result[0]["training_usage"]) == {"used", "job_ids"}


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled"])
def test_unfinished_job_never_marks_all_planned_inputs_as_used(tmp_path, status):
    train = write(tmp_path / "train.jpg", b"train")
    root = tmp_path / "jobs"
    record_training_inputs({"dataset": dataset([train])}, root / "trained")
    index = TrainingUsageIndex(root)
    assert index.annotate([entry(train)], [job(status=status)])[0]["training_usage"]["used"] is False
    assert index.annotate([entry(train)], [job()])[0]["training_usage"]["used"] is True


@pytest.mark.parametrize("mode,expected", [("standard", False), ("expand_frozen", False),
                                           ("dual_branch", False), ("expand_classes", True)])
def test_replay_train_used_only_by_replay_strategy(tmp_path, mode, expected):
    new = write(tmp_path / "new.mp4", b"new")
    replay = write(tmp_path / "replay.mp4", b"replay")
    old_val = write(tmp_path / "old-val.mp4", b"old-val")
    root = tmp_path / "jobs"
    record_training_inputs({"training_mode": mode, "dataset": dataset([new], samples=True),
                            "replay_dataset": dataset([replay], [old_val], samples=True)}, root / "trained")
    result = TrainingUsageIndex(root).annotate([entry(p) for p in (new, replay, old_val)], [job()])
    assert [row["training_usage"]["used"] for row in result] == [True, expected, False]


def test_saved_manifest_immutable_and_media_hash_cache_invalidates(tmp_path):
    source = write(tmp_path / "source.jpg", b"aaaa")
    renamed = write(tmp_path / "renamed.jpg", b"aaaa")
    root = tmp_path / "jobs"
    request = {"dataset": dataset([source])}
    manifest = record_training_inputs(request, root / "trained")
    original = manifest.read_bytes()
    index = TrainingUsageIndex(root)
    assert index.annotate([entry(source)], [job()])[0]["training_usage"]["used"]
    # Same size must not reuse the old digest after a content edit.
    source.write_bytes(b"bbbb")
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1_000_000))
    record_training_inputs(request, root / "trained")
    assert manifest.read_bytes() == original
    results = index.annotate([entry(source), entry(renamed)], [job()])
    assert [row["training_usage"]["used"] for row in results] == [False, True]
    source.unlink()
    assert index.annotate([entry(renamed)], [job()])[0]["training_usage"]["used"]


def test_repeated_listing_and_restarted_index_do_not_rehash_training_dataset(tmp_path):
    source = write(tmp_path / "source.jpg", b"source")
    copy = write(tmp_path / "copy.jpg", source.read_bytes())
    root = tmp_path / "jobs"
    record_training_inputs({"dataset": dataset([source])}, root / "trained")
    index = TrainingUsageIndex(root)
    index.annotate([entry(copy)], [job()])
    with patch("backend.training_usage._sha256", side_effect=AssertionError("should be cached")):
        assert index.annotate([entry(copy)], [job()])[0]["training_usage"]["used"]
    source.unlink()
    assert TrainingUsageIndex(root).annotate([entry(copy)], [job()])[0]["training_usage"]["used"]


def test_legacy_backfill_only_exact_unchanged_train_paths(tmp_path):
    train = write(tmp_path / "train.jpg", b"legacy")
    changed = write(tmp_path / "changed.jpg", b"newer")
    val = write(tmp_path / "val.jpg", b"val")
    missing = tmp_path / "missing.jpg"
    future = datetime.now(timezone.utc).timestamp() + 3600
    os.utime(changed, (future, future))
    root = tmp_path / "jobs"
    request = {"dataset": dataset([train, changed, missing], [val])}
    write(root / "trained/request.json", json.dumps(request).encode())
    index = TrainingUsageIndex(root)
    result = index.annotate([entry(p) for p in (train, changed, val)], [job()])
    assert [row["training_usage"]["used"] for row in result] == [True, False, False]
    saved = json.loads((root / "trained" / MANIFEST).read_text())
    assert saved["basis"] == "legacy_request_unchanged_files"
    assert saved["skipped_unverifiable_files"] == 2
    assert len(saved["files"]) == 1
    # Backfill is persisted, so later source edits cannot change past usage.
    copy = write(tmp_path / "copy.jpg", train.read_bytes())
    train.write_bytes(b"edited")
    assert TrainingUsageIndex(root).annotate([entry(copy)], [job()])[0]["training_usage"]["used"]


def test_run_training_snapshots_before_materialization_and_internal_child_skips(tmp_path):
    from backend import training
    source = write(tmp_path / "original.jpg", b"original")
    request = {"model_id": "yolo26x", "dataset": dataset([source])}
    output = tmp_path / "job"
    observed = []
    def prepare(current, directory, emit):
        snapshot = json.loads((directory / MANIFEST).read_text())
        observed.append(snapshot["files"][0]["source_path"])
        assert current["_training_inputs_recorded"] is True
        return current
    with patch.object(training, "normalize_training_params", return_value={}), \
         patch("backend.expansion.prepare_expansion", side_effect=prepare), \
         patch.object(training, "_detector_training_adapter", return_value=None), \
         patch.object(training, "_train_yolo", return_value={"ok": True}):
        assert training.run_training(request, output, lambda event: None) == {"ok": True}
    assert observed == [str(source)]
    with patch.object(training, "normalize_training_params", return_value={}), \
         patch("backend.expansion.prepare_expansion", side_effect=lambda current, *args: current), \
         patch.object(training, "_detector_training_adapter", return_value=None), \
         patch.object(training, "_train_yolo", return_value={}), \
         patch("backend.training_usage.record_training_inputs", side_effect=AssertionError("duplicate snapshot")):
        training.run_training({**request, "_training_inputs_recorded": True}, tmp_path / "child", lambda event: None)
    assert not (tmp_path / "child" / MANIFEST).exists()


@pytest.mark.parametrize("document", [[], None, {"files": None}, {"files": "invalid"}])
def test_malformed_manifest_never_breaks_media_listing(tmp_path, document):
    source = write(tmp_path / "image.jpg", b"image")
    root = tmp_path / "jobs"
    write(root / "trained" / MANIFEST, json.dumps(document).encode())
    result = TrainingUsageIndex(root).annotate([entry(source)], [job()])
    assert result[0]["training_usage"] == {"used": False, "job_ids": []}


@pytest.mark.parametrize("document", [[], None, {"dataset": []}, {"dataset": {"details": []}}])
def test_malformed_legacy_request_never_breaks_media_listing(tmp_path, document):
    source = write(tmp_path / "image.jpg", b"image")
    root = tmp_path / "jobs"
    write(root / "trained/request.json", json.dumps(document).encode())
    result = TrainingUsageIndex(root).annotate([entry(source)], [job()])
    assert result[0]["training_usage"] == {"used": False, "job_ids": []}
