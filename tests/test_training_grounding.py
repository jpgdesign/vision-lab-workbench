"""Real token/box alignment and native differentiable loss, without GPU loading."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.training_grounding import (
    _caption, _Collator, _DetectionSamples, _detection_samples, _supervised_loss,
    normalize_training_params, training_capabilities,
)


SOURCE = Path(__file__).resolve().parents[2] / "model_weights" / "grounding_dino_base"


class GroundingConfigurationTests(unittest.TestCase):
    def test_actual_loss_and_validation_monitor(self):
        caps = training_capabilities()
        self.assertEqual(caps["losses"], ["grounding_dino_composite"])
        self.assertEqual(caps["early_stopping_monitor"]["metric"], "val_loss")
        self.assertEqual(caps["early_stopping_monitor"]["mode"], "min")
        self.assertEqual(normalize_training_params(None)["imgsz"], 256)
        self.assertEqual(normalize_training_params({"patience": 0})["patience"], 0)

    def test_bad_configs_rejected(self):
        for params in ({"imgsz": 128}, {"imgsz": 225}, {"lr": float("nan")},
                       {"batch_size": 1.2}, {"batch_size": True}, {"amp": 1},
                       {"loss": "cross_entropy"}, {"scheduler": "mystery"},
                       {"optimizer": "unknown"}, {"min_lr": .01, "lr": .001},
                       {"download": True}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                normalize_training_params(params)


class DetectionBoxTests(unittest.TestCase):
    def test_coco_sparse_ids_crowd_and_background(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            info = []
            for index in range(4):
                Image.new("RGB", (100, 50)).save(root / "images" / f"{index}.png")
                info.append({"id": index, "file_name": f"{index}.png", "width": 100, "height": 50})
            details = {"root": str(root), "category_id_to_index": {"7": 0, "93": 1}}
            for split, indexes in (("train", [0, 1]), ("val", [2, 3])):
                doc = {"images": [info[i] for i in indexes], "annotations": [
                    {"id": 1, "image_id": indexes[0], "category_id": 93, "bbox": [10, 5, 20, 10]},
                    {"id": 2, "image_id": indexes[0], "category_id": 7, "bbox": [0, 0, 100, 50], "iscrowd": 1}]}
                path = root / f"{split}.json"
                path.write_text(json.dumps(doc), encoding="utf-8")
                details[f"{split}_json"] = str(path)
                details[f"{split}_images"] = {str(i): str(root / "images" / f"{i}.png") for i in indexes}
            dataset = {"format": "coco_detection", "classes": ["cat", "dog"], "details": details}
            train, val = _detection_samples(dataset)
            self.assertEqual(train[0]["labels"], [1])
            self.assertEqual(train[0]["boxes"], [[.2, .2, .2, .2]])
            self.assertEqual(val[1]["boxes"], [])
            with patch("backend.training_grounding.random.random", return_value=0):
                image, target = _DetectionSamples(train, 224, True)[0]
            self.assertEqual(image.size, (224, 224))
            self.assertAlmostEqual(target["boxes"][0, 0].item(), .8)
            self.assertEqual(train[0]["boxes"][0][0], .2)
            _, empty = _DetectionSamples(val, 224, False)[1]
            self.assertEqual(tuple(empty["boxes"].shape), (0, 4))

    def test_yolo_box_and_root_containment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            details = {"root": str(root)}
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                image = root / "images" / split / "image.png"
                Image.new("RGB", (40, 80)).save(image)
                (root / "labels" / split / "image.txt").write_text("1 .3 .4 .2 .4\n", encoding="utf-8")
                details[f"{split}_images"] = [str(image)]
            dataset = {"format": "yolo_detection", "classes": ["cat", "dog"], "details": details}
            train, _ = _detection_samples(dataset)
            self.assertEqual(train[0]["labels"], [1])
            self.assertEqual(train[0]["boxes"], [[.3, .4, .2, .4]])
            details["val_images"] = [str(root.parent / "escape.png")]
            with self.assertRaises(ValueError):
                _detection_samples(dataset)


@unittest.skipUnless((SOURCE / "tokenizer_config.json").is_file(), "Local tokenizer not present")
class GroundingNativeLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        cls.processor = AutoProcessor.from_pretrained(str(SOURCE), local_files_only=True)

    def test_tokens_multitoken_classes_and_batch(self):
        from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
        import torch
        caption, count = _caption(["Traffic Light", "dog"], self.processor.tokenizer, 256)
        self.assertEqual(caption, "traffic light . dog .")
        sample = (Image.new("RGB", (224, 224)), {"class_labels": torch.tensor([0]), "boxes": torch.tensor([[.5, .5, .2, .2]])})
        inputs = _Collator(self.processor, caption)([sample, sample])
        maps = build_label_maps(torch.empty((2, 4, 256)), inputs["input_ids"])
        self.assertEqual(len(maps), 2)
        self.assertEqual(maps[0].shape[0], 2)
        self.assertGreater(maps[0][0].sum().item(), maps[0][1].sum().item())
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 224, 224))
        self.assertEqual(count, inputs["input_ids"].shape[1])

    def test_ambiguous_and_truncated_labels_fail(self):
        for labels in (["cat. dog"], ["cat?"], ["Dog", "dog"], ["[MASK]"]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                _caption(labels, self.processor.tokenizer, 256)
        with self.assertRaisesRegex(ValueError, "tokens"):
            _caption(["dog " * 300], self.processor.tokenizer, 256)

    def test_native_loss_backpropagates_box_and_token_gradients(self):
        import torch
        from transformers import GroundingDinoConfig
        from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
        from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps, build_text_mask
        caption, _ = _caption(["traffic light", "dog"], self.processor.tokenizer, 256)
        encoded = self.processor.tokenizer([caption, caption], return_tensors="pt")
        logits = torch.zeros((2, 3, 256), requires_grad=True)
        boxes = torch.full((2, 3, 4), .4, requires_grad=True)
        labels = [{"class_labels": torch.tensor([1]), "boxes": torch.tensor([[.5, .5, .2, .2]])},
                  {"class_labels": torch.tensor([], dtype=torch.long), "boxes": torch.empty((0, 4))}]
        config = GroundingDinoConfig(two_stage=False, auxiliary_loss=False)
        loss, components, _ = GroundingDinoForObjectDetectionLoss(
            logits, labels, logits.device, boxes, config, build_label_maps(logits, encoded["input_ids"]),
            build_text_mask(logits, encoded["attention_mask"]))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_bbox", components)
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertGreater(boxes.grad.abs().sum().item(), 0)

    def test_encoder_proposal_regression_has_real_gradient(self):
        import torch
        from transformers import GroundingDinoConfig
        caption, _ = _caption(["cat", "dog"], self.processor.tokenizer, 256)
        encoded = dict(self.processor.tokenizer(caption, return_tensors="pt"))
        encoded["labels"] = [{"class_labels": torch.tensor([1]), "boxes": torch.tensor([[.5, .5, .2, .2]])}]
        encoder_coords = torch.zeros((1, 5, 4), requires_grad=True)
        encoder_logits = torch.zeros((1, 5, 256), requires_grad=True)
        output = SimpleNamespace(logits=torch.zeros((1, 3, 256), requires_grad=True),
                                 pred_boxes=torch.full((1, 3, 4), .4, requires_grad=True),
                                 enc_outputs_class=encoder_logits, enc_outputs_coord_logits=encoder_coords)

        class Model:
            config = GroundingDinoConfig(two_stage=True, auxiliary_loss=False)

            def __call__(self, **inputs):
                if "labels" in inputs:
                    raise AssertionError("Must not use detached convenience encoder loss")
                return output

        loss, components = _supervised_loss(Model(), encoded)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_bbox_enc", components)
        loss.backward()
        self.assertGreater(encoder_coords.grad.abs().sum().item(), 0)
        self.assertGreater(encoder_logits.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
