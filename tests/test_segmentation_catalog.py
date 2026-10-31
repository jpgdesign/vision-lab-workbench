"""Segmentation is a distinct inference task with task-specific weight files."""
import pytest

from backend import catalog
from backend.model_registry import FILES


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "runtime"))
    weight = tmp_path / "fixture.pt"
    weight.write_bytes(b"fixture")
    monkeypatch.setattr(catalog, "WEIGHTS", {model_id: weight for model_id in FILES})
    monkeypatch.setattr(catalog.importlib.util, "find_spec", lambda name: object())
    return {model["id"]: model for model in catalog.get_models()}


@pytest.mark.parametrize("model_id,detect_id,family_id,family_name", [
    ("yolo26x_seg", "yolo26x", "yolo26", "YOLO26x"),
    ("yolov12x_seg", "yolov12x", "yolov12", "YOLOv12x"),
    ("rf_detr_seg_2xl", "rf_detr_2xl", "rf_detr", "RF-DETR"),
])
def test_segmentation_variants_share_family_and_reject_bbox_training(models, model_id, detect_id, family_id, family_name):
    model = models[model_id]
    assert model["task"] == "segmentation"
    assert model["available"] and model["inference_supported"]
    assert model["capabilities"]["input_types"] == ["image", "video"]
    assert (model["family_id"], model["family_name"]) == (family_id, family_name)
    assert (models[detect_id]["family_id"], models[detect_id]["family_name"]) == (family_id, family_name)
    assert not model["training_supported"]
    assert not model["capabilities"]["training"]["dataset_formats"]
    assert "遮罩" in model["training_reason"] and "物件框" in model["training_reason"]
    with pytest.raises(ValueError, match="分割微調尚未接入"):
        catalog.validate_job_request(model_id, "training", {})


def test_open_rf_detr_segmentation_does_not_inherit_detection_pml_gate(models):
    segmentation = models["rf_detr_seg_2xl"]
    assert segmentation["license_url"] is None
    assert "accept_platform_license" not in {p["key"] for p in segmentation["capabilities"]["inference_params"]}
    assert "accept_platform_license" not in catalog.validate_job_request("rf_detr_seg_2xl", "inference", {})
    with pytest.raises(ValueError, match="授權"):
        catalog.validate_job_request("rf_detr_2xl", "inference", {})


@pytest.mark.parametrize("model_id,dependency", [
    ("yolo26x_seg", "ultralytics"), ("yolov12x_seg", "ultralytics"), ("rf_detr_seg_2xl", "rfdetr"),
])
def test_segmentation_availability_requires_its_own_runtime(models, monkeypatch, model_id, dependency):
    monkeypatch.setattr(catalog.importlib.util, "find_spec", lambda name: None if name == dependency else object())
    model = next(model for model in catalog.get_models() if model["id"] == model_id)
    assert not model["environment_available"] and not model["available"]
