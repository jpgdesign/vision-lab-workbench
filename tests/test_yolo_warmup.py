"""Fine-tuning must not inherit a 0.1 warmup LR from the native trainer."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend import training


@pytest.mark.parametrize("model_id", ["yolo26x", "yolov12x"])
@pytest.mark.parametrize("optimizer", ["AdamW", "SGD", "MuSGD"])
def test_native_warmup_starts_all_groups_at_zero_without_disabling_warmup(model_id, optimizer):
    params = training.normalize_training_params(model_id, {"optimizer": optimizer, "lr": .0003, "warmup_epochs": 3})
    options = training._yolo_warmup_options(params)
    assert options["warmup_epochs"] == 3
    # Native interpolation applies warmup_bias_lr to group zero and zero to
    # the other groups. Test the effective LR curve, not just dictionary keys.
    for group in range(3):
        start = options["warmup_bias_lr"] if group == 0 else 0.
        curve = np.interp([0, 28, 50, 100], [0, 100], [start, params["lr"]])
        assert curve[0] == 0
        assert curve[-1] == .0003
        assert np.all(curve <= .0003)
        assert curve[1] == pytest.approx(.000084)


def test_explicit_warmup_disabled_is_preserved():
    assert training._yolo_warmup_options({"warmup_epochs": 0}) == {"warmup_epochs": 0, "warmup_bias_lr": 0.0}


def fake_yolo_training(tmp_path, monkeypatch, *, failure=None, amp=True):
    # Exercise the real adapter and callbacks without importing Ultralytics,
    # allocating CUDA tensors, or loading/downloading actual model weights.
    images = []
    for split in ("train", "val"):
        path = tmp_path / f"{split}.jpg"
        path.write_bytes(b"dataset validation is supplied by this fixture")
        images.append(str(path))
    dataset = {"format": "yolo_detection", "classes": ["rice", "carrot"],
               "details": {"root": str(tmp_path), "train_images": images[:1], "val_images": images[1:]}}
    monkeypatch.setattr(training, "_validated_dataset", lambda *args: dataset)
    monkeypatch.setattr(training, "_catalog_weight", lambda *args: tmp_path / "pretrained.pt")
    monkeypatch.setattr(training, "_resolve_device", lambda value: torch.device("cuda:0"))
    monkeypatch.setattr("backend.inference.validate_yolo_architecture", lambda *args: None)
    scalers, captured = [], {}
    monkeypatch.setattr(torch.amp, "GradScaler", lambda device, *, enabled: scalers.append((device, enabled)) or object())

    class YOLO:
        def __init__(self, *args, **kwargs):
            self.callbacks = {}
        def add_callback(self, name, callback):
            self.callbacks[name] = callback
        def train(self, trainer, **kwargs):
            captured.update(kwargs)
            best, last = tmp_path / "native_best.pt", tmp_path / "native_last.pt"
            best.write_bytes(b"best")
            last.write_bytes(b"last")
            self.trainer = state = SimpleNamespace(
                args=SimpleNamespace(**kwargs), device=torch.device("cuda:0"), batch_size=2,
                best=best, last=last, epoch=0, loss=torch.tensor(2.), tloss=torch.tensor([1., 2., .02]),
                optimizer=SimpleNamespace(param_groups=[{"lr": .000084}]),
                metrics={"val/box_loss": 1., "val/cls_loss": 2., "val/dfl_loss": .02,
                         "metrics/mAP50(B)": .25, "metrics/mAP50-95(B)": .15})
            self.callbacks["on_pretrain_routine_end"](state)
            # Diagnostics must be available before any potentially failing step.
            config = json.loads((tmp_path / "effective_training.json").read_text(encoding="utf-8"))
            assert config["epochs_completed"] == 0
            assert config["amp"] is amp
            assert config["ultralytics_args"]["warmup_bias_lr"] == 0
            if failure == "batch":
                state.loss = torch.tensor(float("nan"))
            self.callbacks["on_train_batch_end"](state)
            if failure == "validation":
                state.metrics["val/cls_loss"] = float("nan")
            state.stopper(1, .15)
            self.callbacks["on_fit_epoch_end"](state)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=YOLO))
    monkeypatch.setitem(sys.modules, "ultralytics.models.yolo.detect", SimpleNamespace(DetectionTrainer=object))
    params = training.normalize_training_params("yolo26x", {"optimizer": "AdamW", "lr": .0003, "warmup_epochs": 3.,
                                                 "amp": amp, "batch_size": 2, "accumulate": 4, "device": "cuda:0"})
    events = []
    request = {"model_id": "yolo26x", "dataset": dataset}
    return lambda: training._train_yolo(request, params, tmp_path, events.append), captured, scalers, events


@pytest.mark.parametrize("amp", [True, False])
def test_adapter_passes_safe_warmup_and_persists_actual_amp_config(tmp_path, monkeypatch, amp):
    run, options, scalers, events = fake_yolo_training(tmp_path, monkeypatch, amp=amp)
    result = run()
    assert options["lr0"] == .0003
    assert options["warmup_bias_lr"] == 0.
    assert options["warmup_epochs"] == 3.
    assert options["nbs"] == 8  # Native warmup accumulation remains enabled.
    assert options["amp"] is False  # Native download-dependent AMP probe remains disabled.
    assert scalers == [("cuda", amp)]  # Actual requested AMP is restored by the callback.
    config = json.loads((tmp_path / "effective_training.json").read_text(encoding="utf-8"))
    assert config["epochs_completed"] == 1
    assert config["amp"] is amp
    assert config["warmup"] == {"epochs": 3., "initial_group_lr": 0.}
    assert config["ultralytics_args"]["amp"] is amp
    assert config["ultralytics_args"]["warmup_bias_lr"] == 0.
    assert any("warmup_bias_lr=0.0" in event.get("message", "") for event in events)
    assert Path(result["checkpoint_path"]).is_file()


@pytest.mark.parametrize("failure", ["batch", "validation"])
def test_nonfinite_loss_still_fails_with_effective_warmup_diagnostics(tmp_path, monkeypatch, failure):
    run, options, scalers, events = fake_yolo_training(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="NaN / Inf"):
        run()
    assert json.loads((tmp_path / "effective_training.json").read_text(encoding="utf-8"))["warmup"]["initial_group_lr"] == 0.
    assert not (tmp_path / "best.pt").exists()
