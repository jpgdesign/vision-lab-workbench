"""No-replay detection training on tiny CPU models and native loss/loops."""
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_native_rf_engine_freezes_backbone_parameters_and_batchnorm_buffers(tmp_path):
    from rfdetr.engine import train_one_epoch
    from rfdetr.models.lwdetr import SetCriterion
    from rfdetr.models.matcher import HungarianMatcher
    from rfdetr.util.misc import NestedTensor
    from backend.training_rfdetr import _freeze_backbone
    from backend.detection_expansion import expand_rfdetr_rows

    class TinyRF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1), torch.nn.BatchNorm2d(8))
            self.class_embed = torch.nn.Linear(8, 3)
            self.bbox_embed = torch.nn.Linear(8, 4)
            self.two_stage = False
        def forward(self, pixels, targets=None):
            features = self.backbone(pixels.tensors).mean((-2, -1))[:, None, :].expand(-1, 3, -1)
            return {"pred_logits": self.class_embed(features), "pred_boxes": self.bbox_embed(features).sigmoid()}

    model = TinyRF()
    frozen = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    expansion = {"base_classes": ["cat", "dog"], "classes": ["cat", "dog", "rice"], "base_class_ids": [1, 2]}
    expand_rfdetr_rows(model, expansion)
    head_before = model.class_embed.weight.detach().clone()
    callbacks = defaultdict(list)
    wrapper = SimpleNamespace(model=SimpleNamespace(model=model), callbacks=callbacks)
    assert _freeze_backbone(wrapper) > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    criterion = SetCriterion(num_classes=4, matcher=HungarianMatcher(),
        weight_dict={"loss_ce": 1., "loss_bbox": 5., "loss_giou": 2.},
        focal_alpha=.25, losses=["labels", "boxes", "cardinality"], group_detr=1, ia_bce_loss=True)
    pixels = NestedTensor(torch.rand(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.bool))
    targets = [{"labels": torch.tensor([3]), "boxes": torch.tensor([[.5, .5, .3, .3]])} for _ in range(2)]
    args = SimpleNamespace(print_freq=100, amp=False, grad_accum_steps=1, multi_scale=False)
    for epoch in range(2):
        model.train()  # Reproduce the native outer trainer's epoch transition.
        metrics = train_one_epoch(model, criterion, scheduler, [(pixels, targets)], optimizer,
            torch.device("cpu"), epoch, batch_size=2, args=args, callbacks=callbacks, num_training_steps_per_epoch=1)
        assert metrics["loss"] > 0
    for name, value in frozen.items():
        assert torch.equal(value, model.backbone.state_dict()[name]), name
    assert not torch.equal(head_before[3], model.class_embed.weight[3])
    torch.save(model.state_dict(), tmp_path / "new_classes.pth")
    saved = torch.load(tmp_path / "new_classes.pth", weights_only=True)
    assert torch.equal(saved["backbone.1.running_mean"], frozen["1.running_mean"])
    assert torch.equal(saved["backbone.1.num_batches_tracked"], frozen["1.num_batches_tracked"])


def test_real_tiny_grounding_frozen_expansion_no_replay_backbone_and_text_unchanged(tmp_path):
    import yaml
    from transformers import (BertConfig, BertTokenizerFast, GroundingDinoConfig, GroundingDinoForObjectDetection,
                              GroundingDinoImageProcessor, GroundingDinoProcessor, SwinConfig)
    from backend.datasets import validate_dataset
    from backend.training import run_training

    source = tmp_path / "source"
    source.mkdir()
    vocab = [f"[unused{i}]" for i in range(1100)]
    for index, token in {0: "[PAD]", 100: "[UNK]", 101: "[CLS]", 102: "[SEP]", 103: "[MASK]",
                         200: "cat", 201: "dog", 202: "rice", 1012: ".", 1029: "?"}.items():
        vocab[index] = token
    (source / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    processor = GroundingDinoProcessor(GroundingDinoImageProcessor(),
                                       BertTokenizerFast(vocab_file=str(source / "vocab.txt")))
    processor.save_pretrained(source)
    config = GroundingDinoConfig(
        backbone_config=SwinConfig(image_size=64, embed_dim=8, depths=[1, 1, 1, 1],
            num_heads=[1, 1, 2, 4], window_size=2, out_features=["stage2", "stage3", "stage4"]),
        text_config=BertConfig(vocab_size=1100, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, max_position_embeddings=512),
        d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
        decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
        encoder_n_points=2, decoder_n_points=2, num_feature_levels=3, num_queries=3,
        two_stage=True, auxiliary_loss=False, disable_custom_kernels=True,
        dropout=0., attention_dropout=0., max_text_len=256)
    model = GroundingDinoForObjectDetection(config)
    model.save_pretrained(source)
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if name.startswith(("model.backbone.", "model.text_backbone."))}
    (source / "workbench_training.json").write_text(json.dumps({"model_id": "grounding_dino_base", "classes": ["cat", "dog"]}))
    root = tmp_path / "data"
    for split, color in (("train", "red"), ("val", "blue")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.new("RGB", (64, 64), color).save(root / split / "images" / "image.png")
        (root / split / "labels" / "image.txt").write_text("0 .5 .5 .4 .4\n")
    path = root / "dataset.yaml"
    path.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "val/images", "names": ["rice"]}))
    checked = validate_dataset(str(path), "yolo_detection")
    assert checked["valid"], checked
    request = {"model_id": "grounding_dino_base", "training_mode": "expand_frozen", "checkpoint_path": str(source),
               "dataset": checked["dataset"], "_skip_checkpoint_archives": True,
               "params": {"epochs": 1, "batch_size": 1, "imgsz": 224, "device": "cpu", "amp": False,
                          "workers": 0, "augmentation": False, "strategy": "full"}}
    output = tmp_path / "job"
    result = run_training(request, output, lambda _: None)
    assert Path(result["checkpoint_path"]).is_dir() and not (output / "best.zip").exists()
    restored = GroundingDinoForObjectDetection.from_pretrained(result["checkpoint_path"], local_files_only=True)
    for name, value in frozen.items():
        assert torch.equal(value, restored.state_dict()[name]), name
    report = result["training"]["retention"]
    assert report["before"]["old"] is report["after"]["old"] is report["delta"]["old"] is None
    assert report["after"]["new"]["sample_count"] == 1
    metadata = json.loads((output / "best" / "workbench_training.json").read_text(encoding="utf-8"))
    assert metadata["training_mode"] == "expand_frozen"
    assert metadata["classes"] == ["cat", "dog", "rice"]
    assert metadata["params"]["strategy"] == "freeze_backbone"
