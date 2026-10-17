"""Model ownership, invalidation, parameter isolation and bounded OOM recovery."""
import gc
import json
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from backend import inference
from backend.inference_cache import InferenceModelCache, ModelCacheKey, model_cache_key
from backend.model_registry import FILES, HF_TYPES


MODELS = ("yolov12x", "yolo26x", "grounding_dino_base", "rf_detr_2xl",
          "vivit_base_k400", "videomaev2_giant_k710", "slowfast_r101_8x8", "x3d_l")


@pytest.fixture(autouse=True)
def no_gpu_allocator(monkeypatch):
    monkeypatch.setattr("backend.inference_cache.release_unused_memory", gc.collect)
    monkeypatch.setattr(inference, "resolve_device", lambda value: "cpu")


def weight(tmp_path, model_id):
    folder = tmp_path / model_id
    folder.mkdir(exist_ok=True)
    path = folder / FILES[model_id][1]
    path.write_bytes(b"test-weights")
    if model_id in HF_TYPES:
        (folder / "config.json").write_text(json.dumps({"model_type": HF_TYPES[model_id]}))
        (folder / "preprocessor_config.json").write_text("{}")
        if model_id == "grounding_dino_base":
            (folder / "tokenizer_config.json").write_text("{}")
            (folder / "vocab.txt").write_text("object")
    return path


def request_for(tmp_path, model_id="yolov12x"):
    return {"model_id": model_id, "model_weight_path": str(weight(tmp_path, model_id)),
            "media_path": "first.jpg", "params": {"device": "cpu", "conf": .2,
            "prompt": "person", "top_k": 1, "accept_platform_license": True}}


def key(name, version=1):
    return ModelCacheKey((name,), (version,))


def test_lru_is_bounded_and_evicts_before_loading():
    cache = InferenceModelCache(max_entries=2)
    first, _ = cache.acquire(key("a"), object)
    cache.acquire(key("b"), object)
    assert cache.acquire(key("a"), object) == (first, True)
    def load_third():
        assert len(cache) == 1
        return object()
    cache.acquire(key("c"), load_third)
    assert cache.acquire(key("a"), object)[1] is True
    assert cache.acquire(key("b"), object)[1] is False
    assert len(cache) == 2


def test_replaced_weights_evict_old_version_before_load():
    cache = InferenceModelCache()
    class Model:
        pass
    loaded, _ = cache.acquire(key("model", 1), Model)
    old = weakref.ref(loaded)
    del loaded
    def load_new():
        assert old() is None
        return Model()
    assert cache.acquire(key("model", 2), load_new)[1] is False
    assert len(cache) == 1


def test_close_and_clear_are_idempotent():
    cache = InferenceModelCache()
    cache.acquire(key("a"), object)
    cache.clear()
    cache.close()
    cache.close()
    assert len(cache) == 0


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_cache_limits_fail(limit):
    with pytest.raises(ValueError):
        InferenceModelCache(limit)


def test_weight_stat_and_metadata_contents_invalidate(tmp_path):
    path = weight(tmp_path, "yolov12x")
    old = model_cache_key("yolov12x", "cpu", path)
    path.write_bytes(b"updated weights")
    changed = model_cache_key("yolov12x", "cpu", path)
    assert old.identity == changed.identity
    assert old.fingerprint != changed.fingerprint
    metadata = path.parent / "workbench_training.json"
    metadata.write_text('{"classes":["cat"]}')
    first = model_cache_key("yolov12x", "cpu", path)
    metadata.write_text('{"classes":["dog"]}')
    assert first != model_cache_key("yolov12x", "cpu", path)


def test_hf_shards_tokenizer_and_config_are_fingerprinted(tmp_path):
    path = weight(tmp_path, "grounding_dino_base")
    shard = path.parent / "model-00002-of-00002.safetensors"
    shard.write_bytes(b"second")
    def current():
        return model_cache_key("grounding_dino_base", "cpu", path, folder_model=True)
    first = current()
    shard.write_bytes(b"second changed")
    assert first != current()
    first = current()
    (path.parent / "vocab.txt").write_text("vehicle")
    assert first != current()
    first = current()
    (path.parent / "config.json").write_text('{"model_type":"grounding-dino","revision":2}')
    assert first != current()
    first = current()
    (path.parent / "inference-results.json").write_text('{"run":123}')
    assert first == current()


def test_keys_separate_model_device_and_checkpoint(tmp_path):
    path = weight(tmp_path, "yolov12x")
    keys = {model_cache_key("yolov12x", "cpu", path),
            model_cache_key("yolo26x", "cpu", path),
            model_cache_key("yolov12x", "cuda:0", path),
            model_cache_key("yolov12x", "cuda:1", path),
            model_cache_key("yolov12x", "cpu", path, checkpoint=True)}
    assert len(keys) == 5


def test_rf_checkpoint_architecture_and_labels_change_key(tmp_path):
    folder = tmp_path / "finetuned"
    folder.mkdir()
    (folder / "model.pth").write_bytes(b"model")
    metadata = {"model_id": "rf_detr_2xl", "class_id_offset": 1, "classes": ["cat"],
                "resolution": 560, "group_detr": 13, "positional_encoding_size": 37}
    metadata_path = folder / "workbench_training.json"
    metadata_path.write_text(json.dumps(metadata))
    request = {"model_id": "rf_detr_2xl", "checkpoint_path": str(folder),
               "params": {"accept_platform_license": True}}
    first = inference.inference_model_key(request, "cpu")
    metadata["resolution"] = 672
    metadata_path.write_text(json.dumps(metadata))
    assert first != inference.inference_model_key(request, "cpu")


@pytest.mark.parametrize("model_id", MODELS)
def test_all_models_reuse_across_inputs_with_current_parameters(tmp_path, monkeypatch, model_id):
    request = request_for(tmp_path, model_id)
    loads, runs = [], []
    resource = object()
    def load(*args):
        loads.append(args)
        return resource
    def run(current, output, emit, device, **kwargs):
        assert next(iter(kwargs.values())) is resource
        runs.append((current["media_path"], dict(current["params"])))
        return {"inference_ms": 12.5}
    monkeypatch.setattr(inference, "create_detector", load)
    monkeypatch.setattr(inference, "create_action_model", load)
    monkeypatch.setattr(inference, "run_detection", run)
    monkeypatch.setattr(inference, "run_action", run)
    cache = InferenceModelCache()
    first = inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    request = {**request, "media_path": "second.mp4", "params": {**request["params"],
               "conf": .8, "iou": .4, "prompt": "vehicle", "top_k": 5}}
    second = inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    assert len(loads) == 1
    assert first["runtime"]["model_cache_hit"] is False
    assert second["runtime"]["model_cache_hit"] is True
    assert second["runtime"]["model_load_seconds"] == 0
    assert second["runtime"]["total_seconds"] >= 0
    assert second["runtime"]["worker_pid"] > 0
    assert second["inference_ms"] == 12.5
    assert runs[1][0] == "second.mp4"
    assert runs[1][1]["prompt"] == "vehicle"
    assert runs[1][1]["top_k"] == 5
    assert runs[1][1]["conf"] == .8


def test_live_yolo_callable_keeps_original_api_but_new_job_overrides_parameters(tmp_path, monkeypatch):
    calls = []
    class YOLO:
        def __init__(self, path):
            pass
        def predict(self, image, **kwargs):
            calls.append(kwargs)
            boxes = SimpleNamespace(xyxy=torch.empty((0, 4)), conf=torch.empty(0), cls=torch.empty(0))
            return [SimpleNamespace(boxes=boxes, names={})]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=YOLO))
    monkeypatch.setattr(inference, "validate_yolo_architecture", lambda *_: None)
    initial = {"conf": .2, "imgsz": 640, "max_det": 100, "iou": .7}
    predict = inference.create_detector("yolov12x", initial, "cpu", model_weight_path=weight(tmp_path, "yolov12x"))
    predict(object())
    current = {"conf": .8, "imgsz": 320, "max_det": 10, "iou": .4}
    predict(object(), current)
    assert all(calls[0][name] == value for name, value in initial.items())
    assert all(calls[1][name] == value for name, value in current.items())


def test_grounding_prompt_and_thresholds_change_without_reload(tmp_path, monkeypatch):
    prompts, thresholds = [], []
    class Inputs(dict):
        input_ids = None
        def to(self, device):
            return self
    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def __call__(self, **kwargs):
            prompts.append(kwargs["text"])
            return Inputs()
        def post_process_grounded_object_detection(self, *args, **kwargs):
            thresholds.append((kwargs["threshold"], kwargs["text_threshold"]))
            return [{"text_labels": [], "scores": [], "boxes": []}]
    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, **kwargs):
            return None
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=Processor, AutoModelForZeroShotObjectDetection=Model))
    predict = inference.create_detector("grounding_dino_base", {"prompt": "Person", "conf": .2, "text_threshold": .3},
        "cpu", model_weight_path=weight(tmp_path, "grounding_dino_base"))
    image = SimpleNamespace(height=100, width=200)
    predict(image)
    predict(image, {"prompt": "CAR.", "conf": .8, "text_threshold": .7})
    assert prompts == ["person .", "car."]
    assert thresholds == [(.2, .3), (.8, .7)]


def test_cache_does_not_bypass_rf_license(tmp_path):
    request = request_for(tmp_path, "rf_detr_2xl")
    request["params"]["accept_platform_license"] = False
    with pytest.raises(ValueError, match="PML"):
        inference.inference_model_key(request, "cpu")


def test_weight_update_reloads_same_model(tmp_path, monkeypatch):
    request = request_for(tmp_path)
    loads = []
    monkeypatch.setattr(inference, "create_detector", lambda *args: loads.append(args) or object())
    monkeypatch.setattr(inference, "run_detection", lambda *args, **kwargs: {})
    cache = InferenceModelCache()
    inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    from pathlib import Path
    Path(request["model_weight_path"]).write_bytes(b"new weights, different size")
    result = inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    assert len(loads) == 2
    assert len(cache) == 1
    assert result["runtime"]["model_cache_hit"] is False


@pytest.mark.parametrize("already_cached", [False, True])
def test_oom_releases_failed_model_and_other_models_before_single_retry(tmp_path, monkeypatch, already_cached):
    request = request_for(tmp_path)
    cache = InferenceModelCache()
    refs, loads, runs = [], [], []
    class Model:
        pass
    other, _ = cache.acquire(key("other"), Model)
    other_ref = weakref.ref(other)
    del other
    def load(*args):
        if loads:
            assert refs[0]() is None  # Includes completed traceback frames.
            assert other_ref() is None
            assert len(cache) == 0
        model = Model()
        refs.append(weakref.ref(model))
        loads.append(1)
        return model
    def run(*args, predict, **kwargs):
        runs.append(1)
        if len(runs) == 1:
            tensor_owner = predict
            assert tensor_owner is not None
            raise torch.OutOfMemoryError("CUDA out of memory. test")
        return {}
    monkeypatch.setattr(inference, "create_detector", load)
    monkeypatch.setattr(inference, "run_detection", run)
    if already_cached:
        cache.acquire(inference.inference_model_key(request, "cpu"), load)
    result = inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    assert len(loads) == len(runs) == 2
    assert len(cache) == 1
    assert result["runtime"]["oom_retry"] is True
    assert result["runtime"]["model_cache_hit"] is False


def test_typed_torch_oom_is_recognized_without_relying_on_message():
    assert inference._is_out_of_memory(torch.OutOfMemoryError("allocation failed"))
    assert not inference._is_out_of_memory(ValueError("out of memory in user text"))


def test_loading_oom_retries_once_and_preserves_failure_cause(tmp_path, monkeypatch):
    request = request_for(tmp_path)
    calls = []
    def load(*args):
        calls.append(1)
        raise RuntimeError(f"CUDA out of memory: attempt {len(calls)}")
    monkeypatch.setattr(inference, "create_detector", load)
    cache = InferenceModelCache()
    with pytest.raises(RuntimeError, match="attempt 2") as failed:
        inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    assert len(calls) == 2
    assert "attempt 1" in str(failed.value.__cause__)
    assert failed.value.__cause__.__traceback__ is None
    assert len(cache) == 0


def test_ordinary_failure_discards_only_suspect_model_and_does_not_retry(tmp_path, monkeypatch):
    request = request_for(tmp_path)
    cache = InferenceModelCache()
    other, _ = cache.acquire(key("other"), object)
    calls = []
    monkeypatch.setattr(inference, "create_detector", lambda *args: calls.append(1) or object())
    def broken(*args, **kwargs):
        raise ValueError("invalid input")
    monkeypatch.setattr(inference, "run_detection", broken)
    with pytest.raises(ValueError, match="invalid input"):
        inference.run_inference(request, tmp_path, lambda _: None, model_cache=cache)
    assert calls == [1]
    assert len(cache) == 1
    assert cache.acquire(key("other"), object) == (other, True)


def test_calls_without_worker_cache_do_not_retain_models(tmp_path, monkeypatch):
    request = request_for(tmp_path)
    calls = []
    monkeypatch.setattr(inference, "create_detector", lambda *args: calls.append(1) or object())
    monkeypatch.setattr(inference, "run_detection", lambda *args, **kwargs: {})
    for _ in range(2):
        result = inference.run_inference(request, tmp_path, lambda _: None)
        assert result["runtime"]["model_cache_hit"] is False
    assert len(calls) == 2


def test_real_image_then_video_path_reuses_detector_and_applies_new_params(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from PIL import Image
    request = request_for(tmp_path)
    still = tmp_path / "input.png"
    Image.new("RGB", (32, 32)).save(still)
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    loads = []
    def load(*args):
        loads.append(1)
        return lambda image, params: [{"label": params["prompt"], "score": params["conf"], "box": [1, 1, 12, 12]}]
    monkeypatch.setattr(inference, "create_detector", load)
    monkeypatch.setattr(inference, "h264_video", lambda raw, final: raw.replace(final))
    cache = InferenceModelCache()
    image_output, video_output = tmp_path / "still", tmp_path / "video"
    image_output.mkdir()
    video_output.mkdir()
    request["media_path"] = str(still)
    first = inference.run_inference(request, image_output, lambda _: None, model_cache=cache)
    request["media_path"] = str(video)
    request["params"].update(prompt="vehicle", conf=.85, frame_stride=1, max_frames=2)
    second = inference.run_inference(request, video_output, lambda _: None, model_cache=cache)
    assert loads == [1]
    assert first["media_kind"] == "image"
    assert first["predictions"][0]["label"] == "person"
    assert second["media_kind"] == "video"
    assert second["frames"] == 2
    assert second["predictions"][0]["label"] == "vehicle"
    assert second["predictions"][0]["score"] == .85
    assert second["runtime"]["model_cache_hit"] is True
    lines = (video_output / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_real_action_path_reuses_weights_but_updates_topk_and_clips(tmp_path, monkeypatch):
    import numpy as np
    request = request_for(tmp_path, "x3d_l")
    request["media_path"] = "first.mp4"
    request["params"].update(start_seconds=0, clip_seconds=1, num_clips=1)
    loads, reads = [], []
    class Model:
        def __call__(self, tensor):
            return torch.tensor([[3., 2., 1.]])
    metadata = {"num_frames": 2, "side": 4, "mean": [0, 0, 0], "std": [1, 1, 1],
                "architecture": "pytorchvideo", "outputs_probabilities": False,
                "labels": {0: "cat", 1: "dog", 2: "horse"}}
    def load(*args):
        loads.append(1)
        return Model(), metadata
    def read(path, num_frames, start, duration):
        reads.append((str(path), num_frames, start, duration))
        return np.zeros((2, 16, 16, 3), dtype=np.uint8), 3.0
    monkeypatch.setattr(inference, "create_action_model", load)
    monkeypatch.setattr(inference, "read_clip", read)
    monkeypatch.setattr(inference.subprocess, "run", lambda *args, **kwargs: None)
    cache = InferenceModelCache()
    first_output, second_output = tmp_path / "first", tmp_path / "second"
    first_output.mkdir()
    second_output.mkdir()
    first = inference.run_inference(request, first_output, lambda _: None, model_cache=cache)
    request["media_path"] = "second.mp4"
    request["params"].update(start_seconds=1, clip_seconds=2, num_clips=2, top_k=2)
    second = inference.run_inference(request, second_output, lambda _: None, model_cache=cache)
    assert loads == [1]
    assert len(first["predictions"]) == len(first["clips"]) == 1
    assert len(second["predictions"]) == len(second["clips"]) == 2
    assert second["frames"] == 4
    assert reads[1] == ("second.mp4", 2, 1, 2)
    assert second["runtime"]["model_cache_hit"] is True
