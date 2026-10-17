"""Safety and configuration tests; intentionally do not start expensive training."""
import math
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.training import (
    EarlyStoppingTracker,
    _check_video_samples,
    _coco_to_yolo,
    _install_yolo_stopper,
    _validated_dataset,
    normalize_training_params,
    sample_frame_indices,
    training_capabilities,
)


class EarlyStoppingTests(unittest.TestCase):
    def test_all_zero_stops_at_first_epoch_plus_patience(self):
        events = []
        stopper = EarlyStoppingTracker(5, "max", "val mAP50–95", events.append)
        stopped = [stopper(epoch, 0.0) for epoch in range(1, 7)]
        self.assertEqual(stopped, [False] * 5 + [True])
        self.assertEqual(stopper.best_epoch, 1)
        self.assertEqual(stopper.best_value, 0.0)
        self.assertEqual(stopper.bad_epochs, 5)
        self.assertEqual(stopper.stop_epoch, 6)
        self.assertIn("epoch 6", events[0]["message"])

    def test_positive_ties_and_lower_plateau_count_as_no_improvement(self):
        for values in ([0.5, 0.5, 0.5], [0.5, 0.4, 0.4]):
            with self.subTest(values=values):
                stopper = EarlyStoppingTracker(2, "max", "mAP")
                self.assertEqual([stopper(epoch, value) for epoch, value in enumerate(values, 1)],
                                 [False, False, True])
                self.assertEqual(stopper.best_epoch, 1)
                self.assertEqual(stopper.stop_epoch, 3)

    def test_real_improvement_resets_patience(self):
        stopper = EarlyStoppingTracker(2, "max", "mAP")
        values = [0.0, 0.0, 0.2, 0.2, 0.2]
        self.assertEqual([stopper(epoch, value) for epoch, value in enumerate(values, 1)],
                         [False, False, False, False, True])
        self.assertEqual(stopper.best_epoch, 3)
        self.assertEqual(stopper.stop_epoch, 5)

    def test_zero_patience_disables_stopping_for_both_modes(self):
        for mode in ("min", "max"):
            with self.subTest(mode=mode):
                stopper = EarlyStoppingTracker(0, mode, "metric")
                self.assertFalse(any(stopper(epoch, 0.0) for epoch in range(1, 101)))
                self.assertFalse(stopper.possible_stop)
                self.assertIsNone(stopper.stop_epoch)
                self.assertIsNone(stopper.reason)

    def test_min_mode_keeps_strict_loss_improvement_semantics(self):
        stopper = EarlyStoppingTracker(2, "min", "val_loss")
        values = [1.0, 0.9, 0.9, 0.8, 0.8, 0.8]
        self.assertEqual([stopper(epoch, value) for epoch, value in enumerate(values, 1)],
                         [False, False, False, False, False, True])
        self.assertEqual(stopper.best_epoch, 4)
        self.assertEqual(stopper.best_value, 0.8)
        self.assertEqual(stopper.stop_epoch, 6)

    def test_yolo_hook_does_not_change_checkpoint_selection_state(self):
        trainer = SimpleNamespace(stopper=object(), best_fitness=0.7, fitness=0.5,
                                  best=Path("best.pt"), last=Path("last.pt"), stop=False)
        before = {key: value for key, value in vars(trainer).items() if key != "stopper"}
        _install_yolo_stopper(trainer, 2, lambda event: None)
        self.assertEqual(before, {key: value for key, value in vars(trainer).items() if key != "stopper"})
        self.assertIsInstance(trainer.stopper, EarlyStoppingTracker)
        self.assertFalse(trainer.stopper(1, 0.0))
        self.assertFalse(trainer.stopper.possible_stop)
        self.assertFalse(trainer.stopper(2, 0.0))
        self.assertTrue(trainer.stopper.possible_stop)
        self.assertTrue(trainer.stopper(3, 0.0))

    def test_nonfinite_metric_cannot_be_reported_as_valid_early_stop(self):
        stopper = EarlyStoppingTracker(2, "max", "mAP")
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stopper(1, value)
        self.assertIsNone(stopper.best_value)
        self.assertIsNone(stopper.stop_epoch)


class TrainingConfigTests(unittest.TestCase):
    def test_unsupported_algorithms_and_params_rejected(self):
        for params in [{"optimizer": "auto"}, {"loss": "focal"}, {"gradient_clip": 1}, {"checkpoint": "anything.pt"}]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                normalize_training_params("yolov12x", params)
        with self.assertRaises(ValueError):
            normalize_training_params("dinov3_7b", {})

    def test_yolo26_no_dfl_and_invalid_freeze_not_silent(self):
        self.assertNotIn("dfl", training_capabilities("yolo26x")["defaults"])
        with self.assertRaises(ValueError):
            normalize_training_params("yolo26x", {"dfl": 1.5})
        with self.assertRaises(ValueError):
            normalize_training_params("yolov12x", {"strategy": "full", "freeze": 3})
        self.assertEqual(normalize_training_params("yolov12x", {"strategy": "freeze_backbone", "freeze": 3})["freeze"], 3)

    def test_numeric_validation_and_device(self):
        for params in [{"lr": math.nan}, {"lr": math.inf}, {"epochs": 1.5}, {"epochs": True}, {"amp": "false"}, {"imgsz": 650}]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                normalize_training_params("yolov12x", params)
        self.assertEqual(normalize_training_params("yolov12x", {"device": "0"})["device"], "cuda:0")
        self.assertEqual(normalize_training_params("vivit_base_k400", {})["batch_size"], 1)
        with self.assertRaises(ValueError):
            normalize_training_params("vivit_base_k400", {"lr": 0.0000001})

    def test_all_catalog_defaults_are_valid(self):
        for model in ("yolov12x", "yolo26x", "vivit_base_k400", "x3d_l", "slowfast_r101_8x8", "videomaev2_giant_k710"):
            defaults = training_capabilities(model)["defaults"]
            self.assertEqual(normalize_training_params(model, defaults), defaults)

    def test_temporal_sampling_preserves_order_and_bounds(self):
        self.assertEqual(sample_frame_indices(1, 32), [0] * 32)
        for total in [3, 32, 71, 1000]:
            for training in [True, False]:
                indices = sample_frame_indices(total, 32, training, random.Random(42))
                self.assertEqual(len(indices), 32)
                self.assertEqual(indices, sorted(indices))
                self.assertTrue(all(0 <= value < total for value in indices))
        with self.assertRaises(ValueError):
            sample_frame_indices(0, 32)

    def test_vivit_requires_real_video_dataset(self):
        with self.assertRaises(ValueError):
            _validated_dataset({"dataset": {"format": "imagefolder"}}, "video_csv")

    def test_coco_conversion_maps_boxes_and_never_changes_source_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset"
            source.mkdir()
            output = root / "job"
            output.mkdir()
            for split in ("train", "val"):
                (source / f"{split}.jpg").write_bytes(b"conversion test; image decoder tested independently")
                document = {"images": [{"id": 2, "file_name": f"{split}.jpg", "width": 100, "height": 50}],
                            "annotations": [{"id": 1, "image_id": 2, "category_id": 9, "bbox": [10, 5, 20, 10]},
                                            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [0, 0, 10, 10], "iscrowd": 1}]}
                (source / f"{split}.json").write_text(json.dumps(document), encoding="utf-8")
            dataset = {"classes": ["cat", "dog"], "format": "coco_detection", "details": {
                       "root": str(source), "image_root": str(source), "category_id_to_index": {"5": 0, "9": 1},
                       "train_json": str(source / "train.json"), "val_json": str(source / "val.json")}}
            events = []
            converted = _coco_to_yolo(dataset, output, events.append)
            label = output / "converted_dataset/labels/train/00000000.txt"
            self.assertEqual(label.read_text().strip(), "1 0.2000000000 0.2000000000 0.2000000000 0.2000000000")
            Path(converted["details"]["train_images"][0]).write_bytes(b"loader repair must not touch source")
            self.assertTrue((source / "train.jpg").read_bytes().startswith(b"conversion test"))
            report = json.loads((output / "coco_conversion.json").read_text(encoding="utf-8"))
            self.assertEqual(report["ignored_crowd_annotations"], 2)

    def test_samples_cannot_escape_dataset_or_use_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip.mp4"
            clip.write_bytes(b"fixture")
            sample = {"path": str(clip), "label": 0}
            dataset = {"classes": ["walk"], "details": {"root": str(root), "train_samples": [sample], "val_samples": [sample]}}
            self.assertEqual(_check_video_samples(dataset)[0][0]["label"], 0)
            sample["label"] = -1
            with self.assertRaises(ValueError):
                _check_video_samples(dataset)
            sample["label"] = 0
            sample["path"] = str(root.parent / "outside.mp4")
            with self.assertRaises(ValueError):
                _check_video_samples(dataset)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "PyTorch / safetensors runtime not installed")
class TrainingTorchTests(unittest.TestCase):
    def test_real_optimizers_and_focal_gradients(self):
        import torch
        from backend.training import _build_optimizer, _classification_loss
        for name in training_capabilities("vivit_base_k400")["optimizers"]:
            with self.subTest(optimizer=name):
                params = normalize_training_params("vivit_base_k400", {"optimizer": name, "loss": "focal"})
                logits = torch.nn.Parameter(torch.tensor([[0.3, -0.4], [-0.2, 0.7]]))
                initial = logits.detach().clone()
                optimizer = _build_optimizer([logits], params)
                loss = _classification_loss(logits, torch.tensor([1, 0]), params)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                optimizer.step()
                self.assertFalse(torch.equal(logits.detach(), initial))

    def test_real_head_training_saves_reusable_checkpoint(self):
        import torch
        from safetensors.torch import load_file
        from backend.training import run_training

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = torch.nn.Linear(3, 3)
                self.head = torch.nn.Linear(3, 2)

            def forward(self, tensor):
                return self.head(torch.tanh(self.trunk(tensor)))

        class TinySamples:
            def __init__(self, samples, metadata, augment):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                label = self.samples[index]["label"]
                return torch.tensor([float(label), 1.0 - label, 0.5]), label

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip.mp4"
            clip.write_bytes(b"synthetic test fixture; decoding is isolated in separate tests")
            samples = [{"path": str(clip), "label": i % 2} for i in range(5)]
            dataset = {"id": "unit-fixture", "format": "video_csv", "classes": ["zero", "one"],
                       "train_count": 5, "val_count": 2, "details": {"root": str(root),
                       "train_samples": samples, "val_samples": samples[:2]}}
            model = TinyClassifier()
            trunk_before = model.trunk.weight.detach().clone()
            head_before = model.head.weight.detach().clone()
            request = {"model_id": "x3d_l", "dataset": dataset, "params": {"epochs": 2,
                       "batch_size": 2, "accumulate": 2, "device": "cpu", "amp": False,
                       "lr": 0.05, "scheduler": "none", "strategy": "head_only", "augmentation": False}}
            events = []
            with patch("backend.training._catalog_weight", return_value=clip), \
                 patch("backend.training._load_action_classifier", return_value=(model, model.head, {"num_frames": 2})), \
                 patch("backend.training._ActionSamples", TinySamples):
                result = run_training(request, root / "job", events.append)
            self.assertTrue(torch.equal(model.trunk.weight.detach(), trunk_before))
            self.assertFalse(torch.equal(model.head.weight.detach(), head_before))
            self.assertTrue(Path(result["checkpoint_path"]).is_file())
            restored = TinyClassifier()
            restored.load_state_dict(load_file(str(root / "job/best/model.safetensors")), strict=True)
            rows = json.loads((root / "job/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(math.isfinite(row["val_loss"]) for row in rows))
            self.assertEqual(len([event for event in events if event["event"] == "metric"]), 2)


if __name__ == "__main__":
    unittest.main()
