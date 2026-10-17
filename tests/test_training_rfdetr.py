"""RF-DETR conversion and parameter contracts, without loading GPU weights."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from backend.datasets import validate_dataset
from backend.training_rfdetr import _prepare_coco, normalize_training_params, training_capabilities


class RFDETRConfigTests(unittest.TestCase):
    def setUp(self):
        self.runtime = tempfile.TemporaryDirectory()
        self.addCleanup(self.runtime.cleanup)
        environment = patch.dict(os.environ, {"VISION_WORKBENCH_RUNTIME": self.runtime.name})
        environment.start()
        self.addCleanup(environment.stop)

    def test_license_is_required_and_original_default_is_false(self):
        self.assertFalse(training_capabilities()["defaults"]["accept_platform_license"])
        with self.assertRaisesRegex(ValueError, "License"):
            normalize_training_params({})
        self.assertEqual(normalize_training_params({"accept_platform_license": True})["optimizer"], "AdamW")

    def test_only_native_options_and_patch_window_sizes_are_accepted(self):
        for invalid in ({"imgsz": 336}, {"optimizer": "SGD"}, {"loss": "cross_entropy"},
                        {"group_detr": 1.5}, {"amp": "true"}, {"lr": float("nan")},
                        {"box": 0, "giou": 0, "cls": 0}, {"arbitrary_file": "unsafe.pth"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_training_params({"accept_platform_license": True, **invalid})
        self.assertEqual(normalize_training_params({"accept_platform_license": True, "imgsz": 320})["imgsz"], 320)

    def test_stop_monitor_matches_map_and_schema_contains_no_ignored_switches(self):
        caps = training_capabilities()
        self.assertEqual(caps["early_stopping_monitor"]["mode"], "max")
        self.assertIn("mAP50", caps["early_stopping_monitor"]["metric"])
        self.assertEqual(set(caps["supported_params"]), set(caps["defaults"]))


class RFDETRConversionTests(unittest.TestCase):
    def _validate(self, root, fmt):
        result = validate_dataset(root, fmt)
        self.assertTrue(result["valid"], result["errors"])
        return result["dataset"]

    def test_coco_noncontiguous_categories_become_one_based_and_crowd_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "input"
            root.mkdir()
            (root / "images").mkdir()
            categories = [{"id": 42, "name": "bus"}, {"id": 7, "name": "cat"}]
            for split, image_id in (("train", 100), ("val", 555)):
                Image.new("RGB", (20, 10), "white").save(root / "images" / f"{split}.png")
                document = {"images": [{"id": image_id, "file_name": f"{split}.png", "width": 20, "height": 10}],
                    "categories": categories, "annotations": [
                        {"id": 4, "image_id": image_id, "category_id": 42, "bbox": [2, 1, 10, 5], "iscrowd": 0},
                        {"id": 12, "image_id": image_id, "category_id": 7, "bbox": [1, 1, 8, 4], "iscrowd": 1}]}
                (root / f"{split}.json").write_text(json.dumps(document), encoding="utf-8")
            dataset = self._validate(root, "coco_detection")
            before = (root / "train.json").read_bytes()
            out = Path(temp) / "output"
            out.mkdir()
            converted = _prepare_coco(dataset, out, lambda event: None)
            doc = json.loads((converted / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["categories"], [
                {"id": 1, "name": "cat", "supercategory": "object"},
                {"id": 2, "name": "bus", "supercategory": "object"}])
            self.assertEqual([a["category_id"] for a in doc["annotations"]], [2, 1])
            self.assertEqual([a["iscrowd"] for a in doc["annotations"]], [0, 1])
            self.assertEqual(doc["annotations"][0]["area"], 50)
            self.assertTrue((converted / "valid" / "00000001.png").is_file())
            self.assertEqual((root / "train.json").read_bytes(), before)

    def test_yolo_box_conversion_keeps_background_image_and_class_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "input"
            root.mkdir()
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                Image.new("RGB", (100, 50), "white").save(root / "images" / split / "item.png")
                (root / "labels" / split / "item.txt").write_text("1 0.5 0.5 0.4 0.2\n" if split == "train" else "")
            (root / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [cat, bus]\n", encoding="utf-8")
            dataset = self._validate(root, "yolo_detection")
            out = Path(temp) / "output"
            out.mkdir()
            converted = _prepare_coco(dataset, out, lambda event: None)
            train = json.loads((converted / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
            val = json.loads((converted / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
            self.assertEqual(train["annotations"][0]["category_id"], 2)
            self.assertEqual(train["annotations"][0]["bbox"], [30, 20, 40, 10])
            self.assertEqual(len(val["images"]), 1)
            self.assertEqual(val["annotations"], [])
            (root / "labels" / "train" / "item.txt").write_text("9 0.5 0.5 0.4 0.2\n")
            with self.assertRaisesRegex(ValueError, "資料集已變更"):
                _prepare_coco(dataset, out, lambda event: None)


if __name__ == "__main__":
    unittest.main()
