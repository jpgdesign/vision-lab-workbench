"""Chunked branch evaluation retains source capacity and semantic IDs on CPU."""
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchEncoding

from backend import dual_branch
from backend.grounding_prompts import build_caption_chunks


class Tokenizer:
    def __call__(self, text, **kwargs):
        ids = [101] + [1012 if word == "." else 200 for word in text.split()] + [102]
        return {"input_ids": torch.tensor([ids])}


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("branch_name", ["original", "new"])
def test_long_branch_evaluation_uses_all_groups_without_expanding_source_config(tmp_path, monkeypatch, saved, branch_name):
    import transformers
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

    classes = [f"food item {i}" for i in range(170)]
    label_map = list(range(170)) if branch_name == "original" else list(range(2, 172))
    union = classes + ["extra"] if branch_name == "original" else ["old one", "old two"] + classes
    source = tmp_path / branch_name
    source.mkdir()
    expected_chunks = build_caption_chunks(classes, Tokenizer(), 128 if saved else 256)
    metadata = {"model_id": "grounding_dino_base", "classes": classes, "caption": ""}
    if saved:
        metadata.update(caption_mode="chunked", caption_chunks=expected_chunks)
    if branch_name == "new" or saved:
        (source / "workbench_training.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {"classes": union, "branches": {branch_name: {
        "kind": "checkpoint" if branch_name == "new" or saved else "pretrained",
        "path": branch_name, "classes": classes, "label_map": label_map}}}
    (tmp_path / dual_branch.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    config = SimpleNamespace(max_text_len=256)
    observed = []

    class Processor:
        tokenizer = Tokenizer()
        def __call__(self, images, text, **kwargs):
            observed.append(text)
            return BatchEncoding(self.tokenizer(text))

    class Model:
        def __init__(self): self.config = config
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self): return self
        def __call__(self, **inputs):
            assert inputs["input_ids"].shape[1] <= 256
            logits = torch.full((1, 1, 256), -10.)
            label_map = build_label_maps(logits, inputs["input_ids"])[0]
            logits[0, 0, label_map[-1].bool()] = 2.
            return SimpleNamespace(logits=logits, pred_boxes=torch.tensor([[[.5, .5, .4, .4]]]))

    class Retention:
        def __init__(self, *args):
            self.settings = {"imgsz": 64, "confidence": .1}
            self.spec = {}
        def evaluate(self, split, predict):
            predictions = predict(Image.new("RGB", (100, 50)))
            assert [row["class_id"] for row in predictions] == [label_map[c["global_class_ids"][-1]] for c in expected_chunks]
            return {"map": .5}

    def reject_single_caption(*args, **kwargs):
        raise AssertionError("Long branch evaluation must not request a single caption")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(transformers.AutoModelForZeroShotObjectDetection, "from_pretrained", lambda *a, **k: Model())
    monkeypatch.setattr("backend.detection_expansion.DetectionRetention", Retention)
    monkeypatch.setattr("backend.training_grounding._caption", reject_single_caption)
    split = "old" if branch_name == "original" else "new"
    request = {"model_id": "grounding_dino_base", "params": {"device": "cpu", "imgsz": 64},
               "dataset": {"classes": classes}, "evaluation_datasets": {split: {"classes": union}},
               "class_expansion": {}}
    result = dual_branch._evaluate_branch(request, tmp_path, manifest, branch_name, split)
    assert config.max_text_len == result["source_max_text_len"] == 256
    assert result["caption"] == "" and result["caption_chunks"] == expected_chunks
    assert observed == [chunk["caption"] for chunk in expected_chunks]
    assert [i for chunk in result["caption_chunks"] for i in chunk["global_class_ids"]] == list(range(170))


@pytest.mark.parametrize("original_kind", ["pretrained", "checkpoint"])
@pytest.mark.parametrize("outer_flag", [False, True])
def test_dual_branch_preserves_explicit_original_checkpoint_mode_and_routes_grouped_new(tmp_path, original_kind, outer_flag):
    new = tmp_path / "new"
    new.mkdir()
    (new / "workbench_training.json").write_text(json.dumps({
        "model_id": "grounding_dino_base", "classes": ["food"],
        "caption_mode": "chunked", "caption_chunks": [{"caption": "food .", "global_class_ids": [0], "text_tokens": 4}]}))
    manifest = {"branches": {
        "original": {"kind": original_kind, "path": "original/model.safetensors"},
        "new": {"kind": "checkpoint", "path": "new"}}}
    (tmp_path / dual_branch.MANIFEST).write_text(json.dumps(manifest))
    request = {"model_id": "grounding_dino_base", "params": {
        "prompt": "custom dog .", "conf": .2, "use_training_classes": outer_flag}}
    original = dual_branch.branch_request(request, tmp_path, "original")
    trained = dual_branch.branch_request(request, tmp_path, "new")
    assert original["params"]["use_training_classes"] is (outer_flag if original_kind == "checkpoint" else False)
    assert trained["params"]["use_training_classes"] is True
    assert original["params"]["prompt"] == "custom dog ."
    assert request["params"]["use_training_classes"] is outer_flag
