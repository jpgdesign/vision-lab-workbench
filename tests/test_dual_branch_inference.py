"""CPU native model checks for independent bundle outputs and frozen training."""
import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from backend import dual_branch, inference
from backend.inference_cache import InferenceModelCache


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def document(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def tiny_yolo(root, classes, model=None):
    from ultralytics.nn.tasks import DetectionModel
    root.mkdir()
    if model is None:
        cfg = {"nc": 2, "end2end": True, "reg_max": 1, "scale": "x",
            "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 1, "C2PSA", [128]]],
            "head": [[[0, 1, 2], 1, "Detect", ["nc"]]]}
        model = DetectionModel(cfg, nc=2, verbose=False)
        with torch.no_grad():
            for head in model.model[-1].one2one_cv3:
                head[-1].bias.fill_(3.)
    model.names = dict(enumerate(classes))
    path = root / "best.pt"
    torch.save({"model": model, "epoch": -1}, path)
    document(root / "workbench_training.json", {"model_id": "yolo26x", "classes": classes})
    return path, model


def test_real_tiny_yolo_bundle_preserves_original_outputs_and_cache(tmp_path):
    original, model = tiny_yolo(tmp_path / "original", ["cat", "dog"])
    new, _ = tiny_yolo(tmp_path / "new", ["cat", "rice"], copy.deepcopy(model))
    plan = {"classes": ["cat", "dog", "rice"], "base_classes": ["cat", "dog"], "new_label_map": [0, 2]}
    bundle = dual_branch.create_bundle("yolo26x", original, new, plan, tmp_path / "bundle", original_is_checkpoint=True)
    image = tmp_path / "input.png"
    Image.new("RGB", (64, 64), (20, 140, 180)).save(image)
    params = {"device": "cpu", "conf": .01, "imgsz": 64, "max_det": 5, "frame_stride": 1, "max_frames": 3}
    request = {"model_id": "yolo26x", "checkpoint_path": str(original), "media_path": str(image), "params": params}
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    direct = inference.run_inference(request, direct_dir, lambda event: None)
    assert direct["predictions"], "test must compare actual detections, not two empty outputs"
    cache = InferenceModelCache(max_entries=3)
    try:
        request["checkpoint_path"] = str(bundle)
        result = inference.run_inference(request, tmp_path / "combined", lambda event: None, model_cache=cache)
        raw = json.loads(Path(result["artifacts"]["original_predictions"]).read_text(encoding="utf-8"))
        assert raw == direct["predictions"]
        old = [row for row in result["predictions"] if row["branch"] == "original"]
        assert [{key: row[key] for key in ("label", "score", "box")} for row in old] == direct["predictions"]
        assert len(result["predictions"]) == sum(len(group["predictions"]) for group in result["prediction_groups"])
        assert {row["branch"] for row in result["predictions"]} == {"original", "new"}
        Image.new("RGB", (64, 64), (170, 60, 20)).save(image)
        second = inference.run_inference(request, tmp_path / "combined2", lambda event: None, model_cache=cache)
        assert second["runtime"]["model_cache_hit"] is True
        import cv2
        import numpy as np
        video = tmp_path / "input.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5., (64, 64))
        assert writer.isOpened()
        for value in (50, 100, 150, 200):
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
        writer.release()
        video_request = {**request, "media_path": str(video), "params": {**params, "max_frames": 2}}
        compared_video = inference.run_inference(video_request, tmp_path / "video", lambda event: None, model_cache=cache)
        assert compared_video["media_kind"] == "video" and compared_video["frames"] == 2
        assert Path(compared_video["media_path"]).is_file()
        frames = json.loads(Path(compared_video["artifacts"]["frames"]).read_text())["frames"]
        assert [frame["frame"] for frame in frames] == [0, 1]
        assert all({group["branch"] for group in frame["prediction_groups"]} == {"original", "new"} for frame in frames)
        assert compared_video["runtime"]["model_cache_hit"] is True
        key = inference.inference_model_key(request, "cpu")
        manifest = json.loads((bundle / dual_branch.MANIFEST).read_text())
        (bundle / manifest["branches"]["new"]["path"]).touch()
        assert key != inference.inference_model_key(request, "cpu")
    finally:
        cache.close()
    from backend.live_inference import _load_runner
    with pytest.raises(ValueError, match="即時攝影機尚未接入雙分支"):
        _load_runner(request, "cpu")


def test_live_api_rejects_bundle_before_enqueuing(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.jobs import JobManager
    from backend.live import register_live_routes
    folder = tmp_path / "bundle"
    folder.mkdir()
    document(folder / dual_branch.MANIFEST, {})
    manager = JobManager(tmp_path / "runtime", tmp_path)
    app = FastAPI()
    catalog = [{"id": "yolo26x", "available": True, "inference_supported": True,
                "capabilities": {"input_types": ["image", "video"]}}]
    register_live_routes(app, manager, lambda: catalog, lambda *args, **kwargs: {},
                         request_weights=lambda *args: {"checkpoint_path": str(folder)})
    with TestClient(app) as client:
        response = client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x", "checkpoint_id": "bundle"}]})
    assert response.status_code == 422
    assert "即時攝影機尚未接入雙分支" in response.json()["detail"]
    assert not manager.jobs


def test_grounding_decoded_punctuation_maps_without_losing_original_label():
    manifest = {"classes": ["kai-lan", "rice"], "branches": {"original": {
        "classes": ["kai-lan"], "label_map": [0]}}}
    predictions = dual_branch.mapped_predictions([{"label": "kai - lan", "score": .9}], manifest, "original")
    assert predictions[0]["union_class_id"] == 0
    assert predictions[0]["source_label"] == "kai - lan"
    assert predictions[0]["label"] == "kai-lan"


def test_grounding_old_scope_overflow_fails_before_model_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import transformers
    folder = tmp_path / "original"
    folder.mkdir()
    classes = ["kai-lan", "tofu"]
    branch = {"kind": "pretrained", "path": "original", "classes": ["kai-lan"], "label_map": [0]}
    manifest = {"classes": classes, "branches": {"original": branch}}
    document(tmp_path / dual_branch.MANIFEST, manifest)
    tokenizer = Mock(return_value={"input_ids": torch.ones((1, 10), dtype=torch.long)})
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(tokenizer=tokenizer))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(max_text_len=5))
    load = Mock(side_effect=AssertionError("Model allocation must not precede prompt validation"))
    monkeypatch.setattr(transformers.AutoModelForZeroShotObjectDetection, "from_pretrained", load)
    request = {"model_id": "grounding_dino_base", "params": {"device": "cpu", "imgsz": 64},
        "dataset": {"classes": ["tofu"]}, "evaluation_datasets": {"old": {"classes": classes}, "new": {"classes": classes}},
        "class_expansion": {"classes": classes, "base_classes": ["kai-lan"], "base_class_ids": [0], "new_class_ids": [1]}}
    with pytest.raises(ValueError, match="超過本機 Grounding DINO 上限"):
        dual_branch._evaluate_branch(request, tmp_path, manifest, "original", "old")
    load.assert_not_called()


def image_data(root, classes, labels):
    root.mkdir()
    records = {}
    for split in ("train", "val"):
        records[split] = []
        for index, label in enumerate(labels):
            path = root / f"{split}_{index}.png"
            Image.new("RGB", (16, 16), (50 + index * 80, 30, 110)).save(path)
            records[split].append({"path": str(path), "label": label})
    return {"id": root.name, "format": "imagefolder", "classes": classes, "train_count": len(labels), "val_count": len(labels),
            "details": {"root": str(root), "train_samples": records["train"], "val_samples": records["val"]}}


def test_dual_training_real_dino_keeps_original_and_independent_probability_groups(tmp_path):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast
    from backend import training_dinov3
    from backend.training import normalize_training_params
    from backend.jobs import JobManager
    base = tmp_path / "backbone"
    DINOv3ViTModel(DINOv3ViTConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, image_size=16, patch_size=4)).save_pretrained(base)
    DINOv3ViTImageProcessorFast(size={"height": 16, "width": 16}).save_pretrained(base)
    params = normalize_training_params("dinov3_7b", {"device": "cpu", "amp": False, "epochs": 1,
        "workers": 0, "batch_size": 1, "augmentation": False, "scheduler": "none"})
    source = training_dinov3.run_training({"model_id": "dinov3_7b", "model_weight_path": str(base), "params": params,
        "dataset": image_data(tmp_path / "base_data", ["red", "green"], [0, 1])}, tmp_path / "source", lambda event: None)
    new_data = image_data(tmp_path / "new_data", ["red", "blue"], [0, 1])
    plan = {"classes": ["red", "green", "blue"], "base_classes": ["red", "green"], "base_class_ids": [0, 1],
            "old_class_ids": [0, 1], "new_class_ids": [2], "new_label_map": [0, 2], "class_aliases": {}}
    request = {"model_id": "dinov3_7b", "training_mode": "dual_branch", "params": params,
        "checkpoint_path": source["checkpoint_path"], "dataset": new_data, "class_expansion": plan,
        "evaluation_datasets": {"new": image_data(tmp_path / "new_eval", plan["classes"], [0, 2])}}
    manager = JobManager(tmp_path / "runtime", tmp_path)
    output = manager.job_root / "dual"
    result = dual_branch.run_training(request, output, lambda event: None)
    audit = json.loads(Path(result["artifacts"]["integrity_audit"]).read_text())
    assert audit["old_branch_unchanged"] is True
    assert audit["old_accuracy_status"] == "not_measured"
    assert result["training"]["retention"]["before"]["old"] is None
    assert result["training"]["retention"]["delta"]["old"] is None
    assert result["training"]["retention"]["after"]["new"]["sample_count"] == 2
    assert Path(result["training"]["new_branch"]["checkpoint_directory"]).is_dir()
    public, _ = manager._public_result("dual", result)
    assert public["checkpoint_is_directory"] is True and "checkpoint_url" not in public
    manager.jobs["dual"] = {"kind": "training", "model_id": "dinov3_7b", "status": "completed"}
    assert manager.checkpoint_for("dual", "dinov3_7b") == Path(result["checkpoint_path"])
    inference_request = {"model_id": "dinov3_7b", "checkpoint_path": result["checkpoint_path"],
        "media_path": new_data["details"]["val_samples"][0]["path"], "params": {"device": "cpu", "top_k": 2}}
    combined = inference.run_inference(inference_request, tmp_path / "inference", lambda event: None)
    assert combined["runtime"]["memory_policy"] == "sequential_eviction"
    assert len(combined["prediction_groups"]) == 2
    assert sum(row["score"] for row in combined["predictions"]) == pytest.approx(2.)
    for group in combined["prediction_groups"]:
        assert sum(row["score"] for row in group["predictions"]) == pytest.approx(1.)
