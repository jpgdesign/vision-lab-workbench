"""CPU tests for semantic head transfer and held-out retention evaluation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from backend.training import (
    _action_replacement_head,
    _evaluate_action_classifier,
    _expanded_linear_classifier,
    _load_action_classifier,
    _load_vivit,
    _train_classifier,
    normalize_training_params,
)


class TinyAction(torch.nn.Module):
    def __init__(self, model_id="x3d_l", count=2):
        super().__init__()
        self.trunk = torch.nn.Linear(3, 3, bias=False)
        if model_id == "videomaev2_giant_k710":
            self.head = torch.nn.Linear(3, count)
        else:
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[-1].proj = torch.nn.Linear(3, count)
            self.blocks[-1].activation = torch.nn.Softmax(-1)
        with torch.no_grad():
            self.trunk.weight.copy_(torch.eye(3))
            self.classifier().weight.copy_(torch.eye(count, 3) * 4)
            self.classifier().bias.zero_()

    def classifier(self):
        return self.head if hasattr(self, "head") else self.blocks[-1].proj

    def forward(self, tensor):
        logits = self.classifier()(self.trunk(tensor))
        return logits if hasattr(self, "head") else self.blocks[-1].activation(logits)


class TinySamples:
    def __init__(self, samples, metadata, augment):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label = self.samples[index]["label"]
        return torch.eye(3)[label], label


@pytest.mark.parametrize("bias", [True, False])
def test_expanded_head_copies_by_name_with_reordered_union(bias):
    original = torch.nn.Linear(3, 2, bias=bias, dtype=torch.float64)
    before = copy.deepcopy(original.state_dict())
    expanded = _expanded_linear_classifier(original, ["walk", "sit"], ["new", "sit", "walk"])
    assert expanded.out_features == 3
    assert expanded.weight.dtype == torch.float64
    assert torch.equal(expanded.weight[2], before["weight"][0])
    assert torch.equal(expanded.weight[1], before["weight"][1])
    assert torch.isfinite(expanded.weight[0]).all()
    if bias:
        assert torch.equal(expanded.bias[[2, 1]], before["bias"])
    else:
        assert expanded.bias is None
    for name, value in original.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("base,classes", [(["a"], ["a", "b"]), (["a", "a"], ["a", "b"]),
                                          (["a", "b"], ["a", "c"]), (["a", "b"], ["a", "a", "b"])])
def test_expanded_head_rejects_ambiguous_or_lost_classes(base, classes):
    with pytest.raises(ValueError):
        _expanded_linear_classifier(torch.nn.Linear(3, 2), base, classes)


def test_source_semantics_must_match_even_if_class_count_matches():
    with pytest.raises(ValueError, match="base_classes"):
        _action_replacement_head(torch.nn.Linear(3, 2), ["a", "b", "c"], ["b", "a"], ["a", "b"])


@pytest.mark.parametrize("classes,preserved", [(["a", "b"], True), (["b", "a"], False), (["x", "y"], False)])
@pytest.mark.parametrize("model_id", ["x3d_l", "slowfast_r101_8x8", "videomaev2_giant_k710"])
def test_standard_action_same_task_retains_head_other_semantics_reset(model_id, classes, preserved):
    model = TinyAction(model_id)
    original = model.classifier()
    before = copy.deepcopy(original.state_dict())
    with patch("backend.inference.create_action_model", return_value=(model, {"num_frames": 2, "labels": {0: "a", 1: "b"}})):
        _, head, _ = _load_action_classifier(model_id, classes, checkpoint_path="source")
    assert (head is original) == preserved
    assert torch.equal(head.weight, before["weight"]) == preserved
    if preserved:
        assert torch.equal(head.bias, before["bias"])


@pytest.mark.parametrize("classes,preserved", [(["a", "b"], True), (["b", "a"], False), (["x", "y"], False)])
def test_standard_vivit_same_task_retains_head_other_semantics_reset(tmp_path, classes, preserved):
    config = SimpleNamespace(num_frames=2, num_labels=2, id2label={0: "a", 1: "b"})
    model = torch.nn.Module()
    model.config = config
    model.classifier = torch.nn.Linear(3, 2)
    original = model.classifier
    before = copy.deepcopy(original.state_dict())
    fake = SimpleNamespace(VivitConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
        VivitImageProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: "processor"),
        VivitForVideoClassification=SimpleNamespace(from_pretrained=lambda *a, **k: (model, {})))
    with patch.dict("sys.modules", {"transformers": fake}):
        _, head, _, _ = _load_vivit(tmp_path, classes)
    assert (head is original) == preserved
    assert torch.equal(head.weight, before["weight"]) == preserved
    if preserved:
        assert torch.equal(head.bias, before["bias"])


@pytest.mark.parametrize("model_id", ["x3d_l", "slowfast_r101_8x8", "videomaev2_giant_k710"])
def test_native_action_loaders_preserve_original_rows(model_id):
    model = TinyAction(model_id)
    old = copy.deepcopy(model.classifier().state_dict())
    with patch("backend.inference.create_action_model", return_value=(model, {"num_frames": 2, "labels": {0: "a", 1: "b"}})) as load:
        loaded, head, metadata = _load_action_classifier(model_id, ["a", "b", "c"], "unused.pt",
                                                        checkpoint_path="source", base_classes=["a", "b"])
    assert load.call_args.kwargs["checkpoint_path"] == "source"
    assert load.call_args.kwargs["model_weight_path"] is None
    assert loaded is model
    assert torch.equal(head.weight[:2], old["weight"])
    assert torch.equal(head.bias[:2], old["bias"])
    assert metadata["labels"] == ["a", "b", "c"]
    if hasattr(model, "blocks"):
        assert isinstance(model.blocks[-1].activation, torch.nn.Identity)


def test_vivit_loads_complete_source_before_expanding(tmp_path):
    config = SimpleNamespace(num_frames=2, num_labels=2, id2label={0: "a", 1: "b"})
    model = torch.nn.Module()
    model.config = config
    model.classifier = torch.nn.Linear(3, 2)
    old = copy.deepcopy(model.classifier.state_dict())

    def load_model(*args, **kwargs):
        assert kwargs["config"].num_labels == 2
        assert "ignore_mismatched_sizes" not in kwargs
        return model, {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

    fake = SimpleNamespace(VivitConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
                           VivitImageProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: "processor"),
                           VivitForVideoClassification=SimpleNamespace(from_pretrained=load_model))
    with patch.dict("sys.modules", {"transformers": fake}):
        loaded, head, _, metadata = _load_vivit(tmp_path, ["a", "b", "c"], ["a", "b"])
    assert loaded.config.num_labels == 3
    assert loaded.config.id2label == {0: "a", 1: "b", 2: "c"}
    assert torch.equal(head.weight[:2], old["weight"])
    assert torch.equal(head.bias[:2], old["bias"])
    assert metadata["source_classes"] == ["a", "b"]


def test_cpu_training_retention_and_saved_checkpoint(tmp_path):
    torch.set_num_threads(2)
    source = tmp_path / "source"
    source.mkdir()
    model = TinyAction()
    original_trunk = model.trunk.weight.detach().clone()
    original_head = model.classifier().weight.detach().clone()
    save_file(model.state_dict(), str(source / "model.safetensors"))
    metadata = {"model_id": "x3d_l", "classes": ["a", "b"]}
    for filename in ("model_config.json", "workbench_training.json"):
        (source / filename).write_text(json.dumps(metadata), encoding="utf-8")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"decoder mocked; real training tensors below")
    classes = ["a", "b", "new"]

    def dataset(labels):
        samples = [{"path": str(clip), "label": label} for label in labels]
        return {"format": "video_csv", "classes": classes, "train_count": len(samples), "val_count": len(samples),
                "details": {"root": str(tmp_path), "train_samples": samples, "val_samples": samples}}

    request = {"model_id": "x3d_l", "training_mode": "expand_classes", "checkpoint_path": str(source),
               "model_weight_path": "must-not-be-used.pt", "dataset": dataset([0, 1, 2, 0, 1, 2]),
               "class_expansion": {"base_classes": ["a", "b"], "classes": classes,
                                   "old_class_ids": [0, 1], "new_class_ids": [2]},
               "evaluation_datasets": {"old": dataset([0, 1]), "new": dataset([2, 0])}}
    p = normalize_training_params("x3d_l", {"epochs": 2, "device": "cpu", "amp": False,
                                           "batch_size": 2, "workers": 0, "lr": .05, "scheduler": "none",
                                           "strategy": "head_only", "augmentation": False, "loss": "focal"})
    output = tmp_path / "job"
    output.mkdir()
    events = []
    with patch("backend.training._catalog_weight", side_effect=AssertionError("checkpoint must take priority")), \
         patch("backend.inference.create_action_model", return_value=(model, {"num_frames": 2, "labels": {0: "a", 1: "b"}})), \
         patch("backend.training._ActionSamples", TinySamples):
        result = _train_classifier(request, p, output, events.append)
    retention = json.loads((output / "retention.json").read_text(encoding="utf-8"))
    assert retention == result["training"]["retention"]
    assert retention["metric"] == "accuracy"
    assert retention["before"]["old"]["accuracy"] == 1.0
    assert retention["before"]["old"]["sample_count"] == 2
    assert retention["after"]["new"]["sample_count"] == 2
    assert retention["coverage"]["new"]["observed_class_ids"] == [0, 2]
    assert retention["delta"]["old"] == retention["after"]["old"]["accuracy"] - 1.0
    assert torch.equal(model.trunk.weight, original_trunk)
    assert not torch.equal(model.classifier().weight[:2], original_head)
    assert Path(result["checkpoint_path"]).is_file()
    saved = load_file(str(output / "best" / "model.safetensors"))
    restored = TinyAction(count=3)
    restored.blocks[-1].activation = torch.nn.Identity()
    restored.load_state_dict(saved, strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, saved[name])  # after eval used best, not last
    assert torch.isfinite(restored(torch.eye(3))).all()
    assert (output / "best" / "retention.json").is_file()


def test_retention_rejects_new_labels_in_original_validation():
    model = TinyAction(count=3)
    model.blocks[-1].activation = torch.nn.Identity()
    loader = [(torch.eye(3)[[2]], torch.tensor([2]))]
    p = {"amp": False, "loss": "cross_entropy", "label_smoothing": 0.0}
    with pytest.raises(ValueError, match="原類別驗證集"):
        _evaluate_action_classifier(model, loader, p, torch.device("cpu"), False, ["a", "b", "new"], [0, 1])
