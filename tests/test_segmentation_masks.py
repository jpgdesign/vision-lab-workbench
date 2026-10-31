"""Segmentation preserves mask geometry, task identity and compact job state."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend import inference
from backend.segmentation import attach_mask, decode_mask, overlay_masks, prediction_summaries


def object_mask():
    # Non-square input, two disjoint components and an interior hole ensure
    # neither bounding boxes nor flattened polygons can satisfy this contract.
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[8:53, 8:51] = 1
    mask[21:40, 21:38] = 0
    mask[13:45, 69:88] = 1
    return mask


def prediction(mask=None):
    values = object_mask() if mask is None else mask
    return attach_mask({"label": "meal", "score": 0.91, "box": [8, 8, 88, 53]},
                       values, (96, 64))


def test_coco_rle_round_trip_keeps_disconnected_components_holes_and_image_coordinates():
    mask = object_mask()
    value = prediction(mask)
    assert value["mask"]["size"] == [64, 96]
    assert isinstance(value["mask"]["counts"], str)
    assert value["mask_encoding"] == "coco_rle"
    assert value["mask_area"] == int(mask.sum())
    decoded = decode_mask(json.loads(json.dumps(value))["mask"], mask.shape)
    np.testing.assert_array_equal(decoded, mask.astype(bool))
    assert decoded[30, 13] and decoded[30, 76]
    assert not decoded[30, 28] and not decoded[30, 60]


def test_low_resolution_mask_is_mapped_to_the_original_image_size_without_swapping_axes():
    small = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    value = attach_mask({"label": "object", "score": 0.8}, small, (12, 8))
    expected = small.astype(bool).repeat(4, axis=0).repeat(4, axis=1)
    assert value["mask"]["size"] == [8, 12]
    np.testing.assert_array_equal(decode_mask(value["mask"], (8, 12)), expected)
    assert value["mask_area"] == 48


def test_overlay_tints_mask_interiors_but_not_holes_or_background_and_keeps_input_unchanged():
    rgb = np.full((64, 96, 3), 100, dtype=np.uint8)
    original = rgb.copy()
    rendered = overlay_masks(rgb, [prediction()], ["#ff0000"], 1)
    # Sample well away from contours: outlines may occupy boundary pixels.
    assert not np.array_equal(rendered[16, 16], rgb[16, 16])
    assert not np.array_equal(rendered[30, 76], rgb[30, 76])
    np.testing.assert_array_equal(rendered[30, 28], rgb[30, 28])
    np.testing.assert_array_equal(rendered[30, 60], rgb[30, 60])
    np.testing.assert_array_equal(rendered[2, 2], rgb[2, 2])
    np.testing.assert_array_equal(rgb, original)


def test_public_summaries_keep_labels_boxes_and_mask_area_without_mutating_artifact_predictions():
    values = [prediction(), {"label": "plain detection", "score": 0.7, "box": [1, 2, 3, 4]}]
    before = deepcopy(values)
    summaries = prediction_summaries(values)
    assert "mask" not in summaries[0]
    assert summaries[0]["label"] == "meal"
    assert summaries[0]["box"] == [8, 8, 88, 53]
    assert summaries[0]["mask_area"] == int(object_mask().sum())
    assert summaries[0]["mask_encoding"] == "coco_rle"
    assert summaries[1] == values[1]
    assert values == before
    assert summaries[0] is not values[0]


@pytest.mark.parametrize("bad_mask", [
    None,
    {"size": [96, 64], "counts": "not-used"},
    {"size": [64], "counts": "not-used"},
    {"size": [64, 96, 1], "counts": "not-used"},
    {"size": [64, 96], "counts": [1, 2, 3]},
])
def test_malformed_or_mismatched_encoded_mask_dimensions_are_rejected(bad_mask):
    with pytest.raises(ValueError):
        decode_mask(bad_mask, (64, 96))


@pytest.mark.parametrize("shape", [(64,), (64, 96, 1)])
def test_model_mask_must_have_exactly_two_dimensions(shape):
    with pytest.raises(ValueError, match="二維"):
        attach_mask({"label": "object"}, np.zeros(shape), (96, 64))


def yolo_model(family, task, scale="x", *, infer_scale_from_filename=False):
    config = {"backbone": [[-1, 1, "A2C2f" if family == "12" else "C2PSA", []]],
              "head": [[-1, 1, "Segment" if task == "segment" else "Detect", []]],
              "yaml_file": f"yolo{family}{scale}{'-seg' if task == 'segment' else ''}.yaml"}
    if not infer_scale_from_filename:
        config["scale"] = scale
    if family == "26":
        config.update(end2end=True, reg_max=1)
    return SimpleNamespace(task=task, model=SimpleNamespace(yaml=config))


@pytest.mark.parametrize("family,model_id", [("12", "yolov12x"), ("26", "yolo26x")])
def test_yolo_detection_and_segmentation_weights_cannot_be_used_interchangeably(family, model_id):
    with pytest.raises(ValueError, match="不能共用權重"):
        inference.validate_yolo_architecture(yolo_model(family, "detect"), model_id + "_seg")
    with pytest.raises(ValueError, match="不能共用權重"):
        inference.validate_yolo_architecture(yolo_model(family, "segment"), model_id)


@pytest.mark.parametrize("family,model_id", [("12", "yolov12x_seg"), ("26", "yolo26x_seg")])
@pytest.mark.parametrize("infer_scale", [False, True])
def test_matching_yolo_x_segmentation_architecture_is_accepted(family, model_id, infer_scale):
    inference.validate_yolo_architecture(yolo_model(family, "segment", infer_scale_from_filename=infer_scale), model_id)


@pytest.mark.parametrize("loaded_family,scale,selected", [
    ("12", "x", "yolo26x_seg"), ("26", "x", "yolov12x_seg"),
    ("12", "l", "yolov12x_seg"), ("26", "l", "yolo26x_seg"),
])
def test_wrong_yolo_family_or_scale_is_rejected_even_when_both_are_segmenters(loaded_family, scale, selected):
    with pytest.raises(ValueError, match="架構"):
        inference.validate_yolo_architecture(yolo_model(loaded_family, "segment", scale), selected)


def test_image_inference_writes_exact_mask_artifact_and_returns_only_compact_summary(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    Image.new("RGB", (96, 64), (30, 40, 50)).save(source)
    output = tmp_path / "output"
    output.mkdir()
    mask_prediction = prediction()
    events = []
    observed = []
    monkeypatch.setattr(inference, "synchronize", lambda device: None)

    def predict(image, params):
        observed.append((image.size, params))
        return [mask_prediction]

    result = inference.run_detection({"model_id": "yolo26x_seg", "media_path": str(source), "params": {"conf": 0.25}},
                                     output, events.append, "cpu", predict=predict)
    assert observed == [((96, 64), {"conf": 0.25})]
    assert result["task"] == "segmentation"
    assert result["media_kind"] == "image" and result["frames"] == 1
    assert result["predictions"][0]["label"] == "meal"
    assert result["predictions"][0]["mask_area"] == int(object_mask().sum())
    assert "mask" not in result["predictions"][0]
    assert events[0]["event"] == "progress"
    with Image.open(result["media_path"]) as rendered:
        assert rendered.size == (96, 64)
    artifact = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert result["artifacts"]["predictions"] == str(output / "predictions.json")
    assert artifact == [mask_prediction]
    np.testing.assert_array_equal(decode_mask(artifact[0]["mask"], (64, 96)), object_mask().astype(bool))
