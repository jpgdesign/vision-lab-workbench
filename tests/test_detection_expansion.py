"""CPU native-head/loss checks; no pretrained downloads or GPU allocation."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image

from backend.detection_expansion import (DetectionRetention, expand_rfdetr_rows, expansion_spec,
                                         grounding_predictor, transplant_yolo_rows)


def spec(base, classes, ids):
    return {"base_classes": base, "classes": classes, "base_class_ids": ids,
            "old_class_ids": list(range(len(base))), "new_class_ids": list(range(len(base), len(classes)))}


def test_standard_mode_does_not_require_expansion_metadata():
    assert expansion_spec({}) is None
    with pytest.raises(ValueError, match="class_expansion"):
        expansion_spec({"training_mode": "expand_classes"})


@pytest.mark.parametrize("end2end", [False, True])
def test_native_yolo_both_branches_keep_old_rows_and_new_rows_train_and_reload(tmp_path, end2end):
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = {"nc": 2, "end2end": end2end, "reg_max": 1,
               "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]],
                            [-1, 1, "Conv", [64, 3, 2]]],
               "head": [[[0, 1, 2], 1, "Detect", ["nc"]]]}
        source = DetectionModel(copy.deepcopy(cfg), nc=2, verbose=False)
        source.names = {0: "cat", 1: "dog"}
        target = DetectionModel(copy.deepcopy(cfg), nc=4, verbose=False)
        target.load(source, verbose=False)
        expansion = spec(["cat", "dog"], ["cat", "dog", "rice", "tofu"], [0, 1])
        branches = ("cv3", "one2one_cv3") if end2end else ("cv3",)
        initial_new = {branch: [(block[-1].weight[2:].clone(), block[-1].bias[2:].clone())
                               for block in getattr(target.model[-1], branch)] for branch in branches}
        audit = transplant_yolo_rows(source, target, expansion)
        assert len(audit["heads"]) == 3 * len(branches) and audit["old_rows_verified"] == 2
        for branch in branches:
            for index, (old, new) in enumerate(zip(getattr(source.model[-1], branch), getattr(target.model[-1], branch))):
                assert torch.equal(old[-1].weight, new[-1].weight[:2])
                assert torch.equal(old[-1].bias, new[-1].bias[:2])
                assert torch.equal(initial_new[branch][index][0], new[-1].weight[2:])
                assert torch.equal(initial_new[branch][index][1], new[-1].bias[2:])
        target.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        target.train()
        batch = {"img": torch.rand(2, 3, 32, 32), "batch_idx": torch.tensor([0, 1]),
                 "cls": torch.tensor([[0.], [3.]]), "bboxes": torch.tensor([[.5, .5, .4, .4], [.5, .5, .4, .4]])}
        optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4)
        loss, _ = target.loss(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all()
        assert any(block[-1].weight.grad[3].abs().sum() > 0 for block in getattr(target.model[-1], branches[-1]))
        optimizer.step()
        checkpoint = tmp_path / "expanded.pt"
        torch.save({"model": target, "names": target.names}, checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        assert restored.model[-1].nc == 4
        assert restored.names == dict(enumerate(expansion["classes"]))
        assert torch.equal(restored.model[-1].cv3[0][-1].weight, target.model[-1].cv3[0][-1].weight)
        bad = {**expansion, "base_classes": ["dog", "cat"]}
        with pytest.raises(ValueError, match="類別名稱"):
            transplant_yolo_rows(source, target, bad)
    finally:
        torch.set_num_threads(previous_threads)


class TinyRF(nn.Module):
    def __init__(self, rows=91):
        super().__init__()
        self.class_embed = nn.Linear(8, rows)
        self.two_stage = True
        self.transformer = nn.Module()
        self.transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(8, rows) for _ in range(2)])


def test_rfdetr_sparse_category_rows_and_unused_zero_preserved_in_every_head(tmp_path):
    from rfdetr.models.lwdetr import SetCriterion
    model = TinyRF()
    before = copy.deepcopy(model)
    expansion = spec(["person", "car", "toothbrush"], ["person", "car", "toothbrush", "rice", "tofu"], [1, 3, 90])
    audit = expand_rfdetr_rows(model, expansion)
    assert len(audit["heads"]) == 3
    heads = [model.class_embed, *model.transformer.enc_out_class_embed]
    old_heads = [before.class_embed, *before.transformer.enc_out_class_embed]
    for head, original in zip(heads, old_heads):
        assert head.out_features == 6 and head.weight.shape == (6, 8)
        for destination, origin in [(0, 0), (1, 1), (2, 3), (3, 90)]:
            assert torch.equal(head.weight[destination], original.weight[origin])
            assert torch.equal(head.bias[destination], original.bias[origin])
        assert torch.allclose(head.bias[4:].sigmoid(), torch.full((2,), .01))
    # Exercise the installed native sigmoid IA-BCE path with the appended
    # class, and an unmatched query. No output row is removed as "background".
    features = torch.rand(1, 3, 8)
    criterion = SetCriterion(num_classes=6, matcher=None, weight_dict={}, focal_alpha=.25,
                             losses=["labels"], group_detr=1, ia_bce_loss=True)
    logits = model.class_embed(features)
    boxes = torch.tensor([[[.5, .5, .3, .3]] * 3])
    targets = [{"labels": torch.tensor([5]), "boxes": torch.tensor([[.5, .5, .3, .3]])}]
    output = {"pred_logits": logits, "pred_boxes": boxes}
    loss = criterion.loss_labels(output, targets, [(torch.tensor([0]), torch.tensor([0]))], 1)["loss_ce"]
    loss.backward()
    assert torch.isfinite(loss) and model.class_embed.weight.grad[5].abs().sum() > 0
    torch.save(model.state_dict(), tmp_path / "rf.pth")
    restored = TinyRF(rows=6)
    restored.load_state_dict(torch.load(tmp_path / "rf.pth", weights_only=True))
    assert torch.equal(restored.class_embed.weight, model.class_embed.weight)
    with pytest.raises(ValueError, match="row 0"):
        expand_rfdetr_rows(TinyRF(), {**expansion, "base_class_ids": [0, 3, 90]})


def test_rfdetr_saved_source_head_and_position_configuration_are_not_guessed():
    from backend.training_rfdetr import _checkpoint_model_options
    metadata = {"classes": [f"class_{i}" for i in range(91)], "positional_encoding_size": 44, "group_detr": 2}
    assert _checkpoint_model_options(metadata, {"group_detr": 1}) == {"num_classes": 91, "positional_encoding_size": 44}
    with pytest.raises(ValueError, match="群組"):
        _checkpoint_model_options(metadata, {"group_detr": 3})


def test_native_rfdetr_train_entry_does_not_repeat_expanded_semantic_rows(tmp_path, monkeypatch):
    from collections import defaultdict
    from rfdetr.detr import RFDETR
    from rfdetr.config import TrainConfig
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig
    expansion = spec(["person", "car", "toothbrush"], ["person", "car", "toothbrush", "rice", "tofu"], [1, 3, 90])
    network = TinyRF()
    expand_rfdetr_rows(network, expansion)
    expected = network.class_embed.weight.detach().clone()
    captured = {}
    def forbidden_resize(*args):
        raise AssertionError("Native entry must not repeat or crop preserved rows")
    wrapper = object.__new__(RFDETR)
    wrapper.model = SimpleNamespace(model=network, class_names=None, reinitialize_detection_head=forbidden_resize,
                                    train=lambda **kwargs: captured.update(kwargs))
    wrapper.model_config = RFDETR2XLargeConfig(num_classes=len(expansion["classes"]) + 1)
    wrapper.callbacks = defaultdict(list)
    wrapper._load_classes = lambda _: expansion["classes"]
    monkeypatch.setattr("rfdetr.detr.MetricsPlotSink", lambda **kwargs: SimpleNamespace(update=lambda *_: None, save=lambda *_: None))
    config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), dataset_file="roboflow",
                         epochs=1, tensorboard=False, wandb=False, early_stopping=False)
    wrapper.train_from_config(config)
    assert captured["num_classes"] == 6
    assert wrapper.model.class_names == expansion["classes"]
    assert torch.equal(network.class_embed.weight, expected)


def dataset(tmp_path, name, classes, labels):
    root = tmp_path / name
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.new("RGB", (100, 50)).save(root / split / "images" / "image.png")
        (root / split / "labels" / "image.txt").write_text("\n".join(f"{label} .5 .5 .4 .4" for label in labels))
    return {"format": "yolo_detection", "classes": classes, "details": {"root": str(root),
            **{f"{split}_images": [str(root / split / "images" / "image.png")] for split in ("train", "val")}}}


def test_retention_measures_selected_classes_and_reports_missing_coverage(tmp_path):
    expansion = spec(["cat", "dog"], ["cat", "dog", "rice"], [0, 1])
    classes = expansion["classes"]
    old, new = dataset(tmp_path, "old", classes, [0]), dataset(tmp_path, "new", classes, [2])
    request = {"training_mode": "expand_classes", "class_expansion": expansion, "dataset": {"classes": classes},
               "evaluation_datasets": {"old": old, "new": new}}
    evaluation = DetectionRetention(request, tmp_path, {"imgsz": 64})
    evaluation.baseline(lambda _: [{"class_id": 0, "score": .9, "box": [30, 15, 70, 35]}])
    result = evaluation.finish(lambda _: [{"class_id": 2, "score": .9, "box": [30, 15, 70, 35]}])
    assert result["before"]["old"]["map50_95"] == pytest.approx(1)
    assert result["after"]["old"]["map50_95"] == pytest.approx(0)
    assert result["after"]["new"]["map50_95"] == pytest.approx(1)
    assert result["delta"]["old"] == pytest.approx(-1)
    assert result["coverage"]["old"]["missing_classes"] == ["dog"]
    assert result["coverage"]["old"]["class_count"] == 1
    assert result["warnings"]
    assert json.loads((tmp_path / "retention.json").read_text(encoding="utf-8"))["settings"] == evaluation.settings
    assert evaluation.evaluate("new", lambda _: [])["map50"] == 0
    empty = dataset(tmp_path, "empty", classes, [])
    evaluation.datasets["new"] = empty
    assert evaluation.evaluate("new", lambda _: [])["map50_95"] is None


@pytest.mark.parametrize("with_old", [False, True])
def test_frozen_detection_retention_without_replay_or_with_partial_old_validation(tmp_path, with_old):
    expansion = spec(["cat", "dog"], ["cat", "dog", "rice"], [0, 1])
    classes = expansion["classes"]
    evaluations = {"new": dataset(tmp_path, "new", classes, [2])}
    if with_old:
        evaluations["old"] = dataset(tmp_path, "old", classes, [0])
    request = {"training_mode": "expand_frozen", "class_expansion": expansion,
               "dataset": {"classes": classes}, "evaluation_datasets": evaluations}
    evaluation = DetectionRetention(request, tmp_path, {"imgsz": 64})
    calls = []
    def before(image):
        calls.append(image)
        return [{"class_id": 0, "score": .9, "box": [30, 15, 70, 35]}]
    evaluation.baseline(before)
    assert len(calls) == int(with_old)
    result = evaluation.finish(lambda _: [{"class_id": 2, "score": .9, "box": [30, 15, 70, 35]}])
    assert result["after"]["new"]["map50_95"] == pytest.approx(1)
    if with_old:
        assert result["coverage"]["old"]["missing_classes"] == ["dog"]
        assert result["delta"]["old"] == pytest.approx(-1)
    else:
        assert result["before"]["old"] is result["after"]["old"] is result["delta"]["old"] is None
        assert result["coverage"]["old"] is None
        assert any("未量測" in warning for warning in result["warnings"])
    assert result["training_mode"] == "expand_frozen"
    if not with_old:
        with pytest.raises(ValueError, match="old/new"):
            DetectionRetention({**request, "training_mode": "expand_classes"}, tmp_path, {"imgsz": 64})


def test_grounding_retention_uses_normalized_phrase_scores_and_original_image_boxes():
    from transformers import BatchEncoding
    class Model:
        def eval(self):
            return self
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[-10., 1., 1., -10., -5., -10., -10.]]]),
                                   pred_boxes=torch.tensor([[[.5, .5, .4, .4]]]))
    class Processor:
        def __call__(self, **kwargs):
            assert kwargs["images"].size == (64, 64)
            return BatchEncoding({"input_ids": torch.tensor([[101, 1000, 1001, 1012, 1002, 1012, 102]])})
    predict = grounding_predictor(Model(), Processor(), "two words . rice .", {"imgsz": 64, "confidence": .001}, "cpu")
    rows = predict(Image.new("RGB", (100, 50)))
    assert len(rows) == 1 and rows[0]["class_id"] == 0
    assert rows[0]["score"] == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())
    assert rows[0]["box"] == pytest.approx([30, 15, 70, 35])


@pytest.mark.parametrize("mode", ["expand_classes", "expand_frozen"])
def test_native_yolo_full_expansion_adapter_one_cpu_epoch(tmp_path, monkeypatch, mode):
    """Actual native trainer + root preparation; all files stay in pytest tmp."""
    import yaml
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    from backend.datasets import validate_dataset
    from backend.training import run_training
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "settings"))
    # This is deliberately a tiny architecture, not a disguised production X.
    # Only the size/family catalog guard is replaced; the trainer, source load,
    # expansion planner, head transfer, losses, optimizer, checkpoint, and AP
    # evaluator are all the actual implementation.
    monkeypatch.setattr("backend.inference.validate_yolo_architecture", lambda *_: None)
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: False)
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *args, **kwargs: None)
    try:
        cfg = {"nc": 2, "end2end": True, "reg_max": 1,
               "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]],
                            [-1, 1, "Conv", [64, 3, 2]]],
               "head": [[[0, 1, 2], 1, "Detect", ["nc"]]]}
        model = DetectionModel(cfg, nc=2, verbose=False)
        model.names = {0: "cat", 1: "dog"}
        model.args = dict(DEFAULT_CFG_DICT)
        # Production YOLO checkpoints are serialized in FP16. Compare frozen
        # tensors in that same representation, including all BN buffers.
        model.half()
        frozen_before = {name: value.clone() for name, value in model.state_dict().items()
                         if any(name.startswith(f"model.{i}.") for i in range(3))}
        source = tmp_path / "source"
        source.mkdir()
        torch.save({"model": model, "train_args": dict(DEFAULT_CFG_DICT)}, source / "best.pt")
        (source / "workbench_training.json").write_text(json.dumps({"model_id": "yolo26x", "classes": ["cat", "dog"]}))

        def make_dataset(name, names, color_offset):
            root = tmp_path / name
            for split, split_color in (("train", 20), ("val", 120)):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir()
                for i in range(max(2, len(names))):
                    Image.new("RGB", (64, 64), (split_color+color_offset+i, 10+i, color_offset)).save(root / split / "images" / f"{i}.png")
                    (root / split / "labels" / f"{i}.txt").write_text(f"{i % len(names)} .5 .5 .4 .4\n")
            path = root / "dataset.yaml"
            path.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "val/images", "names": names}))
            checked = validate_dataset(str(path), "yolo_detection")
            assert checked["valid"], checked
            return checked["dataset"]

        request = {"kind": "training", "model_id": "yolo26x", "checkpoint_path": str(source / "best.pt"),
                   "training_mode": mode,
                   "dataset": make_dataset("new", ["rice"], 30),
                   "params": {"epochs": 1, "batch_size": 2, "imgsz": 64, "device": "cpu", "amp": False,
                              "workers": 0, "augmentation": False, "optimizer": "AdamW", "lr": 1e-4,
                              "strategy": "full", "accumulate": 1, "warmup_epochs": 0}}
        if mode == "expand_classes":
            request["replay_dataset"] = make_dataset("old", ["dog", "cat"], 0)
        output = tmp_path / "job"
        result = run_training(request, output, lambda _: None)
        assert result["training"]["classes"] == ["cat", "dog", "rice"]
        audit = json.loads((output / "expansion_transfer.json").read_text())
        assert audit["old_rows_verified"] == 2 and len(audit["heads"]) == 6
        retention = result["training"]["retention"]
        if mode == "expand_classes":
            assert retention["before"]["old"]["class_count"] == 2
            assert retention["after"]["old"]["class_count"] == 2
            assert retention["delta"]["old"] is not None
        else:
            assert retention["before"]["old"] is retention["after"]["old"] is retention["delta"]["old"] is None
            assert len((output / "train_images.txt").read_text().splitlines()) == 2
        assert retention["after"]["new"]["class_count"] == 1
        restored = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)["model"]
        assert restored.names == {0: "cat", 1: "dog", 2: "rice"}
        assert restored.model[-1].nc == 3
        if mode == "expand_frozen":
            for name, value in frozen_before.items():
                assert torch.equal(restored.state_dict()[name], value), name
            effective = json.loads((output / "effective_training.json").read_text(encoding="utf-8"))
            assert effective["requested_params"]["strategy"] == "freeze_backbone"
            assert effective["freeze_layers"] == 3
        metadata = json.loads((output / "workbench_training.json").read_text(encoding="utf-8"))
        assert metadata["class_expansion"]["base_class_ids"] == [0, 1]
        assert metadata["training_mode"] == mode
    finally:
        torch.set_num_threads(previous_threads)
