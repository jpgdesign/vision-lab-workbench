import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchEncoding

from backend.grounding_inference import create_grounding_predictor
from backend.grounding_prompts import build_caption_chunks
from backend.model_registry import checkpoint_metadata


class Tokenizer:
    def __call__(self, caption, **kwargs):
        return {"input_ids": torch.tensor([[101] + [1012 if word == "." else 200 for word in caption.split()] + [102]])}


class Processor:
    tokenizer = Tokenizer()
    def __call__(self, images, text, **kwargs):
        return BatchEncoding(self.tokenizer(text))
    def post_process_grounded_object_detection(self, *args, **kwargs):
        return [{"text_labels": ["custom"], "scores": [.8], "boxes": [torch.tensor([1, 2, 3, 4])]}]


class Model:
    config = SimpleNamespace(max_text_len=8)
    def __call__(self, **kwargs):
        return None


def test_saved_complete_classes_keep_global_names_and_live_parameter_changes(tmp_path, monkeypatch):
    classes = ["one", "two", "three", "four", "five"]
    chunks = build_caption_chunks(classes, Tokenizer(), 8)
    metadata = {"model_id": "grounding_dino_base", "classes": classes, "caption": "",
                "caption_mode": "chunked", "caption_chunks": chunks, "params": {"imgsz": 384}}
    (tmp_path / "workbench_training.json").write_text(json.dumps(metadata))
    calls = []
    def native(model, processor, caption, settings, device, *, caption_chunks):
        calls.append((settings, caption_chunks))
        return lambda image: [{"class_id": 4, "score": .9, "box": [1, 2, 3, 4]}]
    monkeypatch.setattr("backend.detection_expansion.grounding_predictor", native)
    params = {"use_training_classes": True, "prompt": "dog", "conf": .3, "text_threshold": .25}
    predict = create_grounding_predictor(Model(), Processor(), tmp_path, params, "cpu", is_checkpoint=True)
    image = Image.new("RGB", (40, 30))
    assert predict(image)[0]["label"] == "five"
    assert predict(image, {**params, "conf": .7})[0]["class_id"] == 4
    assert [call[0]["confidence"] for call in calls] == [.3, .7]
    assert all(call[0]["imgsz"] == 384 and call[1] == chunks for call in calls)
    assert predict(image, {**params, "use_training_classes": False})[0]["label"] == "custom"
    public = checkpoint_metadata(tmp_path)
    assert public["caption_chunks"] == chunks and public["caption_mode"] == "chunked"


def test_original_weights_cannot_pretend_to_have_training_classes_and_text_is_never_truncated(tmp_path):
    params = {"use_training_classes": True, "prompt": "food " * 12, "conf": .3, "text_threshold": .25}
    predict = create_grounding_predictor(Model(), Processor(), tmp_path, params, "cpu", is_checkpoint=False)
    image = Image.new("RGB", (40, 30))
    with pytest.raises(ValueError, match="checkpoint"):
        predict(image)
    with pytest.raises(ValueError, match="tokens"):
        predict(image, {**params, "use_training_classes": False})


def test_partial_saved_chunks_are_rejected_before_predicting(tmp_path):
    classes = ["one", "two", "three", "four", "five"]
    chunks = build_caption_chunks(classes, Tokenizer(), 8)
    (tmp_path / "workbench_training.json").write_text(json.dumps({"classes": classes, "caption_chunks": chunks[:1]}))
    predict = create_grounding_predictor(Model(), Processor(), tmp_path,
        {"use_training_classes": True, "conf": .3}, "cpu", is_checkpoint=True)
    with pytest.raises(ValueError, match="完整"):
        predict(Image.new("RGB", (40, 30)))
