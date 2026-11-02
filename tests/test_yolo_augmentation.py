"""Custom augmentation reaches the native trainer and its saved diagnostics."""
import json

import pytest

from backend import training
from test_yolo_warmup import fake_yolo_training


@pytest.mark.parametrize("model_id", ["yolo26x", "yolov12x", "yolo12x"])
def test_default_augmentation_preserves_existing_native_behavior(model_id):
    params = training.normalize_training_params(model_id, {})
    assert training._yolo_augmentation_options(params) == {
        "degrees": 0.0, "hsv_h": .015, "hsv_s": .7, "hsv_v": .4,
    }


@pytest.mark.parametrize("key,limit", [("degrees", 180), ("hsv_h", 1), ("hsv_s", 1), ("hsv_v", 1)])
@pytest.mark.parametrize("invalid", [-.001, "too_large", float("nan"), float("inf"), True, "0.2"])
def test_invalid_augmentation_is_rejected_before_training(key, limit, invalid):
    value = limit + .001 if invalid == "too_large" else invalid
    with pytest.raises(ValueError, match=key):
        training.normalize_training_params("yolo26x", {key: value})


@pytest.mark.parametrize("model_id", ["x3d_l", "vivit_base_k400", "dinov3_7b", "rf_detr_2xl", "grounding_dino_base"])
def test_yolo_controls_are_not_silently_accepted_by_other_trainers(model_id):
    with pytest.raises(ValueError, match="degrees"):
        training.normalize_training_params(model_id, {"degrees": 15})


@pytest.mark.parametrize("enabled", [True, False])
def test_adapter_forwards_custom_settings_or_disables_all_training_augmentation(tmp_path, monkeypatch, enabled):
    normalize = training.normalize_training_params
    custom = {"augmentation": enabled, "degrees": 15., "hsv_h": .01, "hsv_s": .3, "hsv_v": .2}
    monkeypatch.setattr(training, "normalize_training_params",
                        lambda model_id, params: normalize(model_id, {**params, **custom}))
    run, options, _, _ = fake_yolo_training(tmp_path, monkeypatch)
    run()
    actual = json.loads((tmp_path / "effective_training.json").read_text(encoding="utf-8"))
    for key in ("degrees", "hsv_h", "hsv_s", "hsv_v"):
        expected = custom[key] if enabled else 0
        assert options[key] == expected
        assert actual["ultralytics_args"][key] == expected
        assert actual["requested_params"][key] == custom[key]
    if enabled:
        # Untouched native augmentation remains available, including Mosaic.
        assert "mosaic" not in options
    else:
        for key in ("mosaic", "mixup", "copy_paste", "translate", "scale", "shear", "perspective", "flipud", "fliplr"):
            assert options[key] == 0
