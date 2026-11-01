"""Real OpenCV evidence: pixel flow, geometry, masks and temporal reset rules."""
import math

import cv2
import numpy as np
from PIL import Image
import pytest

from backend.workflow_opencv import OpenCVNode, OPENCV_KINDS, validate_opencv_params
from backend.workflow_opencv_catalog import opencv_catalog


def frame(pixels=None, t=0., **extra):
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB") if pixels is not None else Image.new("RGB", (120, 80), "black")
    return {"image": image, "width": image.width, "height": image.height, "timestamp": t, "frame_index": round(t * 30), "source_id": "camera", "to_source": np.eye(3).tolist(), **extra}


def run(node, pixels=None, t=0., **extra):
    f = frame(pixels, t, **extra)
    return node.process({"frame": {"frame": f, "timestamp": t, "items": []}}, f)


@pytest.mark.parametrize("kind,params", [
    ("cv_crop", {"roi": [0, 0, 0, 1]}), ("cv_crop", {"roi": [0, 0, math.nan, 1]}),
    ("cv_crop", {"roi": [False, 0, 1, 1]}), ("cv_crop", {"roi": [0, 0, 1]}),
    ("cv_resize", {"width": 16.5}), ("cv_resize", {"height": 100000}),
    ("cv_rotate", {"angle": 181}), ("cv_rotate", {"expand": 1}),
    ("cv_blur", {"kernel_size": 4}), ("cv_blur", {"method": "unsupported"}),
    ("cv_threshold", {"block_size": 10}), ("cv_edges", {"low": 201, "high": 200}),
    ("cv_edges", {"aperture": 4}), ("cv_morphology", {"iterations": True}),
    ("cv_hsv", {"lower": [180, 0, 0]}), ("cv_hsv", {"lower": [0, 255, 1], "upper": [179, 0, 255]}),
    ("cv_hsv", {"lower": [0, math.inf, 0]}), ("cv_hsv", {"upper": [179, 255, 1.5]}),
    ("cv_contours", {"label": " "}), ("cv_contours", {"max_objects": 1000}),
    ("cv_background", {"learning_rate": -.5}), ("cv_optical_flow", {"window_size": 4}),
    ("cv_optical_flow", {"min_speed_px_s": 0}), ("cv_blur", {"unknown": True}),
])
def test_invalid_params_rejected_before_cv_or_model_loading(kind, params):
    with pytest.raises(ValueError): validate_opencv_params(kind, params)


def test_catalog_defaults_and_ports_cover_every_adapter():
    from backend.workflow_graph import node, param, port
    definitions = opencv_catalog(node, param, port)
    assert {d["type"] for d in definitions} == OPENCV_KINDS
    for spec in definitions:
        values = {p["key"]: p["default"] for p in spec["params"]}
        assert validate_opencv_params(spec["type"], values) == values
        result = run(OpenCVNode(spec["type"], {}))
        assert set(result) == {p["name"] for p in spec["outputs"]}
        for packet in result.values():
            assert packet["fresh"] is True
            assert packet["frame"]["image"].mode == "RGB"


def test_crop_and_resize_compose_exact_source_mapping():
    f = frame(np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3))
    original = np.asarray(f["image"]).copy()
    crop = OpenCVNode("cv_crop", {"roi": [.25, .25, .75, .75]}).process({}, f)["frame"]["frame"]
    assert crop["image"].size == (60, 40)
    np.testing.assert_array_equal(np.asarray(crop["image"]), original[20:60, 30:90])
    resized = OpenCVNode("cv_resize", {"width": 120, "height": 120, "mode": "stretch"}).process({}, crop)["frame"]["frame"]
    np.testing.assert_allclose(np.asarray(resized["to_source"]) @ [120, 120, 1], [90, 60, 1])
    np.testing.assert_allclose(np.asarray(resized["to_source"]) @ [0, 0, 1], [30, 20, 1])
    assert f["image"].size == (120, 80)
    np.testing.assert_array_equal(np.asarray(f["image"]), original)


def test_resize_fit_preserves_aspect_without_padding():
    out = run(OpenCVNode("cv_resize", {"width": 60, "height": 60}))["frame"]["frame"]
    assert out["image"].size == (60, 40)
    np.testing.assert_allclose(out["to_source"], np.diag([2, 2, 1]))


def test_tiny_crop_rounds_to_at_least_one_pixel():
    out = run(OpenCVNode("cv_crop", {"roi": [.99999, .99999, 1, 1]}))["frame"]["frame"]
    assert out["image"].size == (1, 1)


def test_rotation_inverse_geometry_and_expand_dimensions():
    f = frame()
    out = OpenCVNode("cv_rotate", {"angle": 90}).process({}, f)["frame"]["frame"]
    assert out["image"].size == (80, 120)
    np.testing.assert_allclose(np.asarray(out["to_source"]) @ [40, 60, 1], [60, 40, 1], atol=1e-10)
    np.testing.assert_allclose(np.asarray(out["to_source"]) @ [0, 0, 1], [120, 0, 1], atol=1e-10)
    noexpand = OpenCVNode("cv_rotate", {"angle": 45, "expand": False}).process({}, f)["frame"]["frame"]
    assert noexpand["image"].size == f["image"].size
    np.testing.assert_allclose(np.asarray(noexpand["to_source"]) @ [60, 40, 1], [60, 40, 1], atol=1e-10)


@pytest.mark.parametrize("angle,turns", [(0, 0), (90, 1), (180, 2), (-90, -1), (-180, -2)])
@pytest.mark.parametrize("shape", [(9, 14, 3), (50, 100, 3), (1, 1, 3)])
def test_right_angle_rotation_preserves_every_border_pixel(angle, turns, shape):
    # Unequal odd/even dimensions expose pixel-center versus edge-coordinate
    # off-by-one errors that plain geometric center assertions cannot detect.
    pixels = np.random.default_rng(128).integers(1, 256, shape, dtype=np.uint8)
    result = run(OpenCVNode("cv_rotate", {"angle": angle}), pixels)
    np.testing.assert_array_equal(np.asarray(result["frame"]["frame"]["image"]), np.rot90(pixels, turns))


@pytest.mark.parametrize("mode", ["gray", "equalize", "clahe"])
def test_gray_contrast_operations_return_rgb_channels_without_geometry_change(mode):
    rgb = np.random.default_rng(12).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    out = run(OpenCVNode("cv_color", {"mode": mode}), rgb)["frame"]["frame"]
    pixels = np.asarray(out["image"])
    assert pixels.shape == rgb.shape
    np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
    np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])
    np.testing.assert_allclose(out["to_source"], np.eye(3))


@pytest.mark.parametrize("method", ["gaussian", "median", "bilateral"])
def test_blurs_remove_impulse_noise(method):
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    # Bilateral filtering intentionally preserves high-contrast edges; test a
    # small impulse within its color-distance bandwidth instead.
    pixels[20, 20] = 30
    out = run(OpenCVNode("cv_blur", {"method": method, "sigma": 100}), pixels)["frame"]["frame"]
    assert np.asarray(out["image"])[20, 20, 0] < 30


@pytest.mark.parametrize("method", ["binary", "otsu", "adaptive"])
def test_threshold_binary_and_inverse(method):
    pixels = np.repeat(np.linspace(0, 255, 64, dtype=np.uint8)[None, :], 32, axis=0)
    a = np.asarray(run(OpenCVNode("cv_threshold", {"method": method}), pixels)["frame"]["frame"]["image"])
    b = np.asarray(run(OpenCVNode("cv_threshold", {"method": method, "invert": True}), pixels)["frame"]["frame"]["image"])
    assert set(np.unique(a)).issubset({0, 255})
    np.testing.assert_array_equal(a.astype(int) + b, 255)


def test_hsv_wrap_includes_both_red_hues_not_green():
    hsv = np.array([[[0, 255, 255], [179, 255, 255], [60, 255, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    out = run(OpenCVNode("cv_hsv", {"lower": [170, 200, 200], "upper": [10, 255, 255]}), rgb)["frame"]["frame"]
    np.testing.assert_array_equal(np.asarray(out["image"])[0, :, 0], [255, 255, 0])


def test_canny_and_morphology_produce_real_edges_and_remove_small_noise():
    pixels = np.zeros((60, 100), dtype=np.uint8)
    pixels[20:40, 30:70] = 255
    edges = np.asarray(run(OpenCVNode("cv_edges", {}), pixels)["frame"]["frame"]["image"])[..., 0]
    assert 80 < np.count_nonzero(edges) < 200
    pixels[2, 2] = 255
    opened = np.asarray(run(OpenCVNode("cv_morphology", {}), pixels)["frame"]["frame"]["image"])[..., 0]
    assert opened[2, 2] == 0 and opened[30, 50] == 255


def test_contours_expose_real_boxes_bounded_polygons_and_nonsemantic_scores():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[10:40, 20:70] = 255
    pixels[80:85, 80:85] = 255
    result = run(OpenCVNode("cv_contours", {"min_area": 50, "label": "region"}), pixels)
    assert len(result["detections"]["items"]) == 1
    item = result["detections"]["items"][0]
    assert item["box"] == [20, 10, 70, 40]
    assert item["label"] == "region" and item["score_type"] == "geometric_rule"
    assert item["area"] == 49 * 29 and len(item["polygon"]) == 4
    assert result["detections"]["diagnostics"]["semantic_classifier"] is False


def test_contours_cap_count_and_filter_large_region():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    for i in range(5): pixels[5 + i * 15:15 + i * 15, 20:80] = 255
    assert len(run(OpenCVNode("cv_contours", {"max_objects": 2}), pixels)["detections"]["items"]) == 2
    assert run(OpenCVNode("cv_contours", {"max_area_ratio": .01}), pixels)["detections"]["items"] == []


@pytest.mark.parametrize("method", ["MOG2", "KNN"])
def test_background_warmup_static_then_foreground_and_gap_reset(method):
    node = OpenCVNode("cv_background", {"method": method, "warmup_frames": 8, "detect_shadows": False, "min_motion_ratio": .02})
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(12):
        result = run(node, pixels, i * .1)
        if i < 8: assert result["actions"]["items"][0]["label"] == "unknown"
    assert result["actions"]["items"][0]["label"] == "still"
    pixels[10:40, 10:40] = 255
    result = run(node, pixels, 1.2)
    action = result["actions"]["items"][0]
    assert action["label"] == "motion" and action["motion_ratio"] > .02
    assert np.count_nonzero(np.asarray(result["frame"]["frame"]["image"])) > 0
    assert run(node, pixels, 5.)["actions"]["items"][0]["label"] == "unknown"
    assert node.warmup_count == 1


def shifted_texture():
    first = np.random.default_rng(1234).integers(0, 256, (96, 128), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (5, 5), 0)
    second = cv2.warpAffine(first, np.array([[1., 0., 3.], [0., 1., 0.]]), (128, 96), borderMode=cv2.BORDER_REFLECT)
    return first, second


def test_flow_uses_source_time_and_source_pixel_scale_not_wall_time():
    first, second = shifted_texture()
    values = []
    for dt, scale in ((.1, 1), (.2, 1), (.1, 2)):
        node = OpenCVNode("cv_optical_flow", {"min_speed_px_s": 1})
        transform = np.diag([scale, scale, 1]).tolist()
        assert run(node, first, 0, to_source=transform)["actions"]["items"][0]["label"] == "unknown"
        result = run(node, second, dt, to_source=transform)
        action = result["actions"]["items"][0]
        assert action["label"] == "motion" and action["dt_seconds"] == dt
        assert np.count_nonzero(np.asarray(result["frame"]["frame"]["image"])) > 0
        values.append(action["mean_speed_px_s"])
    assert 15 < values[0] < 40
    assert values[1] == pytest.approx(values[0] / 2, rel=.01)
    assert values[2] == pytest.approx(values[0] * 2, rel=.01)


@pytest.mark.parametrize("change", ["gap", "backwards", "duplicate_time", "source", "geometry", "size"])
def test_flow_resets_without_false_motion_after_discontinuity(change):
    first, second = shifted_texture()
    node = OpenCVNode("cv_optical_flow", {})
    run(node, first, 0.)
    run(node, second, .1)
    t, kwargs = .2, {}
    if change == "gap": t = 2
    elif change == "backwards": t = .05
    elif change == "duplicate_time": t = .1
    elif change == "source": kwargs["source_id"] = "another"
    elif change == "geometry": kwargs["to_source"] = [[1, 0, 10], [0, 1, 0], [0, 0, 1]]
    elif change == "size": second = second[:60, :60]
    result = run(node, second, t, **kwargs)
    assert result["actions"]["items"][0]["label"] == "unknown"
    assert not np.count_nonzero(np.asarray(result["frame"]["frame"]["image"]))


def test_flow_zero_frame_and_bounded_buffer_cleanup():
    node = OpenCVNode("cv_optical_flow", {"max_side": 64})
    pixels = np.zeros((400, 600), dtype=np.uint8)
    run(node, pixels, 0)
    result = run(node, pixels, .1)
    assert result["actions"]["items"][0]["label"] == "still"
    assert result["frame"]["frame"]["image"].size == (600, 400)
    assert max(node.previous.shape) <= 64
    node.close()
    assert node.previous is None and node.background is None


def test_crop_resize_color_contour_tracker_workflow_maps_real_objects_to_source():
    from backend.workflow_runtime import WorkflowEngine
    types = [("input", "source", {}), ("crop", "cv_crop", {"roi": [.25, .2, .75, .8]}),
             ("resize", "cv_resize", {"width": 200, "height": 120, "mode": "stretch"}),
             ("color", "cv_hsv", {"lower": [35, 100, 100], "upper": [90, 255, 255]}),
             ("contour", "cv_contours", {"min_area": 20}), ("track", "tracker", {"labels": []}), ("out", "output", {})]
    links = [("input", "frame", "crop", "frame"), ("crop", "frame", "resize", "frame"),
             ("resize", "frame", "color", "frame"), ("color", "frame", "contour", "frame"),
             ("contour", "detections", "track", "observations"), ("track", "tracks", "out", "observations"),
             ("input", "frame", "out", "frame")]
    graph = {"version": 1, "nodes": [{"id": ident, "type": kind, "params": params} for ident, kind, params in types],
             "edges": [{"source": a, "sourceHandle": output, "target": b, "targetHandle": target} for a, output, b, target in links]}
    engine = WorkflowEngine({"id": "cv-integration", "workflow_graph": graph, "params": {"device": "cpu"}})
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[30:60, 60:100] = [0, 255, 0]
    rendered, events, observations = engine.process(Image.fromarray(pixels), 0., 0)
    assert rendered.size == (200, 100) and events == []
    detections = observations["track"]["tracks"]["items"]
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0]["box"], [60, 30, 100, 60], atol=.5)
    assert detections[0]["track_id"] == 1
    assert set(engine.instances) == {"crop", "resize", "color", "contour", "track"}
