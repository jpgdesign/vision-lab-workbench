"""CPU integration with a real tiny DINOv3 architecture, without downloads."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")
from PIL import Image

from backend import training_dinov3 as adapter
from backend.training import normalize_training_params


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def tiny_backbone(tmp_path):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast
    folder = tmp_path / "pretrained"
    config = DINOv3ViTConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=4, patch_size=4, image_size=16, num_register_tokens=1)
    torch.manual_seed(7)
    model = DINOv3ViTModel(config)
    model.save_pretrained(folder, safe_serialization=True)
    DINOv3ViTImageProcessorFast(size={"height": 16, "width": 16}).save_pretrained(folder)
    return folder


def dataset(root, classes, labels=None):
    root.mkdir(parents=True)
    samples = {}
    labels = list(range(len(classes))) if labels is None else labels
    for split in ("train", "val"):
        samples[split] = []
        for index, label in enumerate(labels):
            path = root / f"{split}_{index}.png"
            Image.new("RGB", (19, 17), [(230, 20, 10), (20, 230, 10), (10, 20, 230)][label]).save(path)
            samples[split].append({"path": str(path), "label": label})
    return {"id": root.name, "format": "imagefolder", "classes": classes, "train_count": len(labels),
            "val_count": len(labels), "details": {"root": str(root), "train_samples": samples["train"],
                                                    "val_samples": samples["val"]}}


def params(**overrides):
    return normalize_training_params("vivit_base_k400", {"device": "cpu", "amp": False, "epochs": 2,
        "batch_size": 1, "accumulate": 2, "workers": 0, "augmentation": False, "lr": .01,
        "patience": 0, "scheduler": "none", **overrides})


@pytest.mark.parametrize("strategy,loss", [("head_only", "cross_entropy"), ("full", "focal")])
def test_real_training_checkpoint_reload_and_freezing(tmp_path, tiny_backbone, strategy, loss):
    from transformers import AutoModel
    before = AutoModel.from_pretrained(tiny_backbone, local_files_only=True).state_dict()
    data = dataset(tmp_path / "data", ["red", "green"])
    events = []
    request = {"model_id": "dinov3_7b", "params": params(strategy=strategy, loss=loss), "dataset": data,
               "model_weight_path": str(tiny_backbone)}
    result = adapter.run_training(request, tmp_path / "job", events.append)
    model, processor, metadata = adapter.load_classifier(result["checkpoint_path"], "cpu")
    image = Image.open(data["details"]["val_samples"][0]["path"])
    with torch.inference_mode():
        prediction = model(adapter.prepare_image_pixels(image, processor))
    assert prediction.shape == (1, 2) and torch.isfinite(prediction).all()
    assert metadata["classes"] == ["red", "green"]
    assert result["training"]["checkpoint_format"] == adapter.CHECKPOINT_FORMAT
    assert len([x for x in events if x["event"] == "metric"]) == 2
    assert "retention" not in result["training"]
    same = all(torch.equal(before[key], value) for key, value in model.backbone.state_dict().items())
    assert same == (strategy == "head_only")
    model2, processor2, _ = adapter.load_classifier(tmp_path / "job" / "best")
    with torch.inference_mode():
        reloaded = model2(adapter.prepare_image_pixels(image, processor2))
    torch.testing.assert_close(prediction, reloaded, rtol=0, atol=0)


def test_expansion_retention_replays_original_head_and_best(tmp_path, tiny_backbone):
    old_classes, union = ["red", "green"], ["red", "green", "blue"]
    source = adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1),
        "dataset": dataset(tmp_path / "base_data", old_classes), "model_weight_path": str(tiny_backbone)},
        tmp_path / "base_job", lambda event: None)
    old_eval = dataset(tmp_path / "old_eval", union, [0, 1])
    # New data may include known classes as well as genuinely new classes.
    new_eval = dataset(tmp_path / "new_eval", union, [1, 2])
    request = {"model_id": "dinov3_7b", "params": params(epochs=1),
        "checkpoint_path": source["checkpoint_path"], "model_weight_path": str(tmp_path / "missing"),
        "dataset": dataset(tmp_path / "union_data", union), "training_mode": "expand_classes",
        "class_expansion": {"base_classes": old_classes, "classes": union, "old_class_ids": [0, 1], "new_class_ids": [2]},
        "evaluation_datasets": {"old": old_eval, "new": new_eval}}
    result = adapter.run_training(request, tmp_path / "expanded", lambda event: None)
    report = result["training"]["retention"]
    assert report == json.loads((tmp_path / "expanded" / "retention.json").read_text(encoding="utf-8"))
    assert report["metric"] == "accuracy"
    assert report["before"]["old"]["sample_count"] == 2
    assert report["after"]["new"]["class_count"] == 2
    assert report["delta"]["old"] == pytest.approx(report["after"]["old"]["accuracy"] - report["before"]["old"]["accuracy"])
    model, processor, _ = adapter.load_classifier(source["checkpoint_path"])
    pixels = adapter.prepare_image_pixels([Image.open(x["path"]) for x in old_eval["details"]["val_samples"]], processor)
    with torch.inference_mode():
        expected = float((model(pixels).argmax(-1) == torch.tensor([0, 1])).float().mean())
    assert report["before"]["old"]["accuracy"] == expected
    best, best_processor, _ = adapter.load_classifier(result["checkpoint_path"])
    loader = torch.utils.data.DataLoader(adapter._ImageSamples(new_eval["details"]["val_samples"], best_processor), batch_size=1)
    after = adapter._evaluate(best, loader, union, request["params"], torch.device("cpu"))
    assert report["after"]["new"] == after


def test_unlabelled_backbone_cannot_expand(tmp_path, tiny_backbone):
    with pytest.raises(ValueError, match="原始 DINOv3 沒有分類類別"):
        adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1),
            "training_mode": "expand_classes", "model_weight_path": str(tiny_backbone),
            "dataset": dataset(tmp_path / "data", ["red", "green"])}, tmp_path / "job", lambda event: None)


@pytest.mark.parametrize("target_classes,preserved", [(["red", "green"], True), (["green", "red"], False), (["cat", "dog"], False)])
def test_standard_dinov3_continuation_preserves_only_identical_semantics(tmp_path, tiny_backbone, monkeypatch, target_classes, preserved):
    from backend import training
    source = adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1),
        "model_weight_path": str(tiny_backbone), "dataset": dataset(tmp_path / "old_data", ["red", "green"])},
        tmp_path / "source", lambda event: None)
    original, _, _ = adapter.load_classifier(source["checkpoint_path"])
    before = [parameter.detach().clone() for parameter in original.classifier.parameters()]
    real_optimizer = training._build_optimizer
    def checked_optimizer(parameters, config):
        assert len(parameters) == len(before)  # head_only: classifier weight and bias
        assert all(torch.equal(old, current) for old, current in zip(before, parameters)) == preserved
        return real_optimizer(parameters, config)
    monkeypatch.setattr(training, "_build_optimizer", checked_optimizer)
    adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1), "checkpoint_path": source["checkpoint_path"],
        "dataset": dataset(tmp_path / "new_data", target_classes)}, tmp_path / "continued", lambda event: None)
    effective = json.loads((tmp_path / "continued" / "effective_training.json").read_text(encoding="utf-8"))
    assert effective["head_reset"] == (not preserved)


def test_missing_shard_rejected_without_download(tmp_path, tiny_backbone):
    (tiny_backbone / "model.safetensors").unlink()
    (tiny_backbone / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="缺少完整檔案"):
        adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1),
            "model_weight_path": str(tiny_backbone), "dataset": dataset(tmp_path / "data", ["red", "green"])},
            tmp_path / "job", lambda event: None)


def test_nonfinite_training_is_rejected(tmp_path, tiny_backbone, monkeypatch):
    from backend import training
    monkeypatch.setattr(training, "_classification_loss", lambda logits, target, p: logits.sum() * float("nan"))
    with pytest.raises(RuntimeError, match="NaN / Inf"):
        adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1),
            "model_weight_path": str(tiny_backbone), "dataset": dataset(tmp_path / "data", ["red", "green"])},
            tmp_path / "job", lambda event: None)


def test_original_embedding_inference_reuses_cache_and_processes_video(tmp_path, tiny_backbone):
    import cv2
    import numpy as np
    from backend.inference import run_inference
    from backend.inference_cache import InferenceModelCache
    data = dataset(tmp_path / "data", ["red", "green"])
    cache = InferenceModelCache(max_entries=2)
    request = {"model_id": "dinov3_7b", "model_weight_path": str(tiny_backbone),
               "media_path": data["details"]["val_samples"][0]["path"],
               "params": {"device": "cpu", "top_k": 2, "frame_stride": 1, "max_frames": 3}}
    try:
        first = run_inference(request, tmp_path / "first", lambda event: None, model_cache=cache)
        assert first["output_mode"] == "embedding" and first["predictions"] == []
        values = np.load(first["artifacts"]["embeddings"], allow_pickle=False)
        assert values.shape == (1, 16) and np.isfinite(values).all()
        assert first["runtime"]["model_cache_hit"] is False
        request["media_path"] = data["details"]["val_samples"][1]["path"]
        second = run_inference(request, tmp_path / "second", lambda event: None, model_cache=cache)
        assert second["runtime"]["model_cache_hit"] is True
        assert not np.allclose(values, np.load(second["artifacts"]["embeddings"]))
        video = tmp_path / "input.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5., (32, 32))
        assert writer.isOpened()
        for channel in (0, 1, 2, 0):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:, :, channel] = 200
            writer.write(frame)
        writer.release()
        request["media_path"] = str(video)
        result = run_inference(request, tmp_path / "video", lambda event: None, model_cache=cache)
        assert result["frames"] == 3 and result["media_kind"] == "video"
        assert Path(result["media_path"]).is_file()
        assert np.load(result["artifacts"]["embeddings"]).shape == (3, 16)
        assert result["runtime"]["model_cache_hit"] is True
    finally:
        cache.close()


def test_classifier_registry_jobs_inference_and_nested_cache(tmp_path, tiny_backbone):
    import cv2
    import numpy as np
    from backend.inference import run_inference, inference_model_key
    from backend.jobs import JobManager
    from backend.model_registry import validate_checkpoint, checkpoint_metadata
    data = dataset(tmp_path / "data", ["red", "green"])
    manager = JobManager(tmp_path / "runtime", tmp_path)
    output = manager.job_root / "dino_test"
    result = adapter.run_training({"model_id": "dinov3_7b", "params": params(epochs=1),
        "model_weight_path": str(tiny_backbone), "dataset": data}, output, lambda event: None)
    best = validate_checkpoint("dinov3_7b", result["checkpoint_path"])
    assert checkpoint_metadata(best)["video_mode"] == "independent_frames"
    public, allowed = manager._public_result("dino_test", result)
    assert public["checkpoint_is_directory"] is True
    assert public["checkpoint_directory_name"] == "best"
    assert "checkpoint_url" not in public and "best" not in public["artifact_urls"]
    assert "best" not in allowed
    manager.jobs["dino_test"] = {"id": "dino_test", "kind": "training", "model_id": "dinov3_7b", "status": "completed"}
    assert manager.checkpoint_for("dino_test", "dinov3_7b") == best
    request = {"model_id": "dinov3_7b", "checkpoint_path": str(best), "model_weight_path": str(tmp_path / "missing"),
               "media_path": data["details"]["val_samples"][0]["path"], "params": {"device": "cpu", "top_k": 2}}
    inference = run_inference(request, tmp_path / "inference", lambda event: None)
    assert inference["output_mode"] == "classification"
    assert {row["label"] for row in inference["predictions"]} == {"red", "green"}
    assert "embeddings" not in inference["artifacts"]
    video = tmp_path / "classifier_input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5., (32, 32))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((32, 32, 3), 100, dtype=np.uint8))
    writer.release()
    video_request = {**request, "media_path": str(video), "params": {**request["params"], "frame_stride": 1, "max_frames": 2}}
    video_result = run_inference(video_request, tmp_path / "classifier_video", lambda event: None)
    video_records = json.loads(Path(video_result["artifacts"]["predictions"]).read_text(encoding="utf-8"))
    assert video_result["media_kind"] == "video" and video_result["frames"] == 2
    assert video_records["video_mode"] == "independent_frames"
    assert all(len(frame["predictions"]) == 2 for frame in video_records["frames"])
    before = inference_model_key(request, "cpu")
    path = best / "backbone" / "config.json"
    config = json.loads(path.read_text())
    path.write_text(json.dumps({**config, "cache_test": 1}))
    after = inference_model_key(request, "cpu")
    assert before.identity == after.identity and before.fingerprint != after.fingerprint
    with pytest.raises(ValueError, match="工作範圍"):
        manager._public_result("other_job", result)
    (best / "backbone" / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="完整權重"):
        validate_checkpoint("dinov3_7b", str(best))


def test_dinov3_catalog_exposes_real_capabilities_and_checkpoint_only_training(monkeypatch):
    from backend import catalog
    model = next(x for x in catalog.get_models() if x["id"] == "dinov3_7b")
    assert model["inference_supported"] is True
    assert model["training_supported"] is True
    assert model["capabilities"]["input_types"] == ["image", "video"]
    model.update(available=False, environment_available=True, reason="missing base")
    monkeypatch.setattr(catalog, "get_models", lambda: [model])
    clean = catalog.validate_job_request("dinov3_7b", "training", {"device": "cpu"}, allow_missing_weights=True)
    assert clean["strategy"] == "head_only"
    with pytest.raises(ValueError, match="missing base"):
        catalog.validate_job_request("dinov3_7b", "training", {"device": "cpu"})
