"""Image wires really carry pixels; geometry survives branches and transforms."""
import copy

import numpy as np
from PIL import Image
import pytest

from backend.workflow_frames import IDENTITY, reproject_packet, route_inputs


def frame(size=(200, 100), matrix=None, source="camera"):
    return {"image": Image.new("RGB", size), "width": size[0], "height": size[1],
            "timestamp": 1., "frame_index": 1, "source_id": source,
            "to_source": matrix or IDENTITY}


def test_crop_resize_boxes_joints_polygons_and_normalized_overlays_map_to_source_without_mutating_branch():
    source = frame()
    crop = frame((50, 25), [[2, 0, 50], [0, 2, 20], [0, 0, 1]])
    values = {"frame": crop, "items": [{"box": [5, 5, 20, 15], "keypoints": [[10, 10, .9]],
              "polygon": [[5, 5], [20, 5], [20, 15]]}],
              "overlays": [{"kind": "polygon", "points": [[0, 0], [1, 0], [1, 1]]}]}
    before = copy.deepcopy(values["items"])
    result = reproject_packet(values, source)
    assert result["items"][0]["box"] == [60, 30, 90, 50]
    assert result["items"][0]["keypoints"] == [[70, 40, .9]]
    assert result["items"][0]["polygon"] == [[60, 30], [90, 30], [90, 50]]
    assert np.allclose(result["overlays"][0]["points"], [[.25, .2], [.75, .2], [.75, .7]])
    assert values["items"] == before and values["frame"] is crop


def test_source_observations_into_crop_are_clipped_and_invisible_actors_removed():
    source = frame()
    crop = frame((50, 25), [[2, 0, 50], [0, 2, 20], [0, 0, 1]])
    packet = {"frame": source, "fresh": False, "items": [], "display_items": [
        {"box": [20, 10, 100, 40], "keypoints": [[40, 15, .9], [60, 30, .8]]},
        {"box": [0, 0, 10, 10]}]}
    result = reproject_packet(packet, crop)
    assert len(result["display_items"]) == 1
    assert result["display_items"][0]["box"] == [0, 0, 25, 10]
    assert result["display_items"][0]["keypoints"] == [[-5, -2.5, 0], [5, 5, .8]]
    assert result["fresh"] is False and result["items"] == []


def test_segmentation_masks_and_relation_endpoints_are_reprojected():
    from backend.segmentation import attach_mask, decode_mask
    source = frame((100, 80))
    crop = frame((20, 20), [[1, 0, 30], [0, 1, 10], [0, 0, 1]])
    mask = np.zeros((20, 20), np.uint8)
    mask[3:10, 2:8] = 1
    item = attach_mask({"box": [2, 3, 8, 10]}, mask, (20, 20))
    item.update(subject_box=[2, 3, 8, 10], object_box=[10, 10, 15, 15])
    mapped = reproject_packet({"frame": crop, "items": [item]}, source)["items"][0]
    full = decode_mask(mapped["mask"], (80, 100))
    assert full.sum() == 42 and full[13:20, 32:38].all()
    assert mapped["mask_area"] == 42
    assert mapped["subject_box"] == [32, 13, 38, 20]
    assert mapped["object_box"] == [40, 20, 45, 25]


def test_rotation_geometry_uses_four_corners():
    source = frame((100, 80))
    rotated = frame((80, 100), [[0, 1, 0], [-1, 0, 80], [0, 0, 1]])
    mapped = reproject_packet({"frame": rotated, "items": [{"box": [10, 20, 30, 50]}]}, source)
    assert mapped["items"][0]["box"] == [20, 50, 50, 70]


def test_rotated_segmentation_retains_outer_row_and_column():
    from backend.segmentation import attach_mask, decode_mask
    source = frame((5, 3))
    rotated = frame((3, 5), [[0, -1, 5], [1, 0, 0], [0, 0, 1]])
    mask = np.arange(15).reshape(5, 3) % 2
    item = attach_mask({"box": [0, 0, 3, 5]}, mask, (3, 5))
    mapped = reproject_packet({"frame": rotated, "items": [item]}, source)["items"][0]
    assert np.array_equal(decode_mask(mapped["mask"], (3, 5)), np.rot90(mask, -1))


def test_contour_area_changes_with_coordinate_basis():
    source, crop = frame(), frame((50, 25), [[2, 0, 50], [0, 2, 20], [0, 0, 1]])
    mapped = reproject_packet({"frame": crop, "items": [{"box": [0, 0, 10, 10], "area": 100., "area_ratio": .08}]}, source)
    assert mapped["items"][0]["area"] == pytest.approx(400.)
    assert mapped["items"][0]["area_ratio"] == pytest.approx(.02)


def test_invalid_cross_source_and_singular_transforms_rejected():
    with pytest.raises(ValueError, match="來源"):
        reproject_packet({"frame": frame(source="other"), "items": []}, frame())
    with pytest.raises(ValueError, match="轉換"):
        reproject_packet({"frame": frame(matrix=[[0, 0, 0]] * 3), "items": []}, frame())


def test_frame_wire_wins_while_rules_use_source_basis():
    original, cropped = frame(), frame((20, 20), [[1, 0, 30], [0, 1, 10], [0, 0, 1]])
    observations = {"frame": cropped, "items": [{"box": [0, 0, 20, 20]}]}
    routed, selected = route_inputs({"observations": observations}, original)
    assert selected is original and routed["observations"]["items"][0]["box"] == [30, 10, 50, 30]
    routed, selected = route_inputs({"frame": {"frame": cropped}, "observations": observations}, original)
    assert selected is cropped and routed["observations"]["items"][0]["box"] == [0, 0, 20, 20]


def graph(with_detector=True, processed_output=False):
    nodes = [{"id": "source", "type": "source"},
             {"id": "crop", "type": "cv_crop", "params": {"roi": [.25, .2, .75, .8]}},
             {"id": "resize", "type": "cv_resize", "params": {"width": 50, "height": 30, "mode": "stretch"}},
             {"id": "gray", "type": "cv_color"}, {"id": "output", "type": "output"}]
    edges = [("source", "frame", "crop", "frame"), ("crop", "frame", "resize", "frame"),
             ("resize", "frame", "gray", "frame")]
    if with_detector:
        nodes.append({"id": "model", "type": "detector"})
        edges += [("gray", "frame", "model", "frame"), ("model", "detections", "output", "observations")]
    edges.append(("gray" if processed_output else "source", "frame", "output", "frame"))
    return {"version": 1, "nodes": nodes, "edges": [{"source": a, "sourceHandle": b, "target": c, "targetHandle": d} for a, b, c, d in edges]}


def test_engine_feeds_processed_pixels_to_model_and_remaps_boxes_for_original_output(monkeypatch):
    from backend import workflow_runtime as runtime
    seen = {}
    class Detector:
        def process(self, inputs, frame):
            seen["model_image"] = np.asarray(frame["image"]).copy()
            seen["model_frame"] = frame
            return {"detections": runtime.packet([{"box": [0, 0, 50, 30], "label": "test", "score": .9}], frame)}
    def render(frame, inputs, recent):
        seen["output"] = inputs["observations"][0]["items"][0]
        return frame["image"].copy()
    monkeypatch.setattr(runtime, "render_output", render)
    engine = runtime.WorkflowEngine({"workflow_graph": graph()}, model_factory=lambda *a, **kw: Detector())
    engine.device = "cpu"
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    output, _, records = engine.process(image, 0., 0)
    assert seen["model_image"].shape == (30, 50, 3)
    assert np.all(seen["model_image"][:, :, 0] == seen["model_image"][:, :, 1])
    assert seen["output"]["box"] == pytest.approx([50, 20, 150, 80])
    assert output.size == image.size and output.getpixel((100, 50)) == (255, 0, 0)
    assert records["model"]["detections"]["coordinate_space"]["width"] == 50


def test_model_free_engine_outputs_actual_processed_image(monkeypatch):
    from backend import workflow_runtime as runtime
    monkeypatch.setattr(runtime, "render_output", lambda frame, *_: frame["image"])
    engine = runtime.WorkflowEngine({"workflow_graph": graph(False, True)})
    result, _, _ = engine.process(Image.new("RGB", (200, 100), (255, 0, 0)), 0., 0)
    assert result.size == (50, 30)
    assert result.getpixel((10, 10))[0] == result.getpixel((10, 10))[1]
    assert engine.device is None  # OpenCV-only workflows never initialize ML/GPU.
