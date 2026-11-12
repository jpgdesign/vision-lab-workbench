"""Evidence semantics for sink workflows, without model loading or a GPU."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend.segmentation import attach_mask, decode_mask
from backend.workflow_handwash import HandwashNode, validate_handwash_params
from backend.workflow_hands import HandModelNode


@pytest.fixture
def params():
    return {"zones": [{"name": "sink", "points": [[.25, .25], [.75, .25], [.75, .75], [.25, .75]]}],
            "require_masks": True, "min_mask_overlap": .5, "min_video_score": .25,
            "hold_seconds": 1., "max_gap_seconds": 1., "evidence_max_age": .75, "motion_threshold": .005}


def frame(t=0., source="camera", size=100):
    return {"timestamp": t, "source_id": source, "width": size, "height": size,
            "image": Image.new("RGB", (size, size))}


def hand(x, y=50):
    return {"label": "hand", "observed": True, "box": [x-4, y-4, x+4, y+4],
            "keypoints": [[x, y, 1.] for _ in range(21)]}


def masks(t, inside=True):
    values = []
    for x in (40, 60):
        pixels = np.zeros((100, 100), np.uint8)
        pixels[46:54, x-4:x+4] = 1 if inside else 0
        if not inside:
            pixels[:8, x-4:x+4] = 1
        # Box remains inside in either case; only actual mask pixels matter.
        values.append(attach_mask(hand(x), pixels, (100, 100)))
    return {"timestamp": t, "items": values, "fresh": True}


def video(t, label="washing hands", score=.8, scope="scene", track_id=None):
    row = {"scope": scope, "label": label, "candidates": [{"label": label, "score": score}]}
    if track_id is not None:
        row["track_id"] = track_id
    return {"timestamp": t, "items": [row], "fresh": True,
            "diagnostics": {"model_id": "x3d_l" if scope == "scene" else "stgcnpp"}}


def inputs(t=0., *, xs=(40, 60), rgb=True, mask=True):
    result = {"hands": {"timestamp": t, "items": [hand(x) for x in xs], "fresh": True}}
    if mask:
        result["masks"] = masks(t)
    if rgb:
        result["actions"] = [video(t)]
    return result


def action(output):
    return output["actions"]["items"][0]


def kinds(output):
    return [v["kind"] for v in output["events"]["items"]]


def test_roi_and_masks_are_only_spatial_evidence(params):
    result = HandwashNode(params).process(inputs(rgb=False), frame())
    row = action(result)
    assert row["state"] == "hands_in_roi"
    assert row["evidence"]["mask_valid"] is True
    assert row["evidence"]["rgb_washing_score"] is None
    assert not kinds(result)
    assert row["evidence"]["rinsing"] == "unknown"
    assert row["evidence"]["hygiene_compliance"] == "not_evaluated"


def test_moving_hands_without_rgb_never_confirm_washing(params):
    node = HandwashNode(params)
    node.process(inputs(rgb=False), frame())
    result = node.process(inputs(.5, xs=(45, 65), rgb=False), frame(.5))
    assert action(result)["state"] == "hand_motion_candidate"
    assert "washing_candidate" not in kinds(result)


def test_missing_mask_blocks_candidate_unless_explicitly_disabled(params):
    result = HandwashNode(params).process(inputs(mask=False), frame())
    assert action(result)["state"] != "washing_candidate"
    params["require_masks"] = False
    result = HandwashNode(params).process(inputs(mask=False), frame())
    assert action(result)["state"] == "washing_candidate"
    assert action(result)["evidence"]["mask_valid"] is False


def test_mask_overlap_uses_pixels_not_prompt_box(params):
    value = inputs()
    value["masks"] = masks(0., inside=False)
    result = HandwashNode(params).process(value, frame())
    assert action(result)["evidence"]["both_hands_in_roi"] is True
    assert action(result)["evidence"]["mask_evidence"]["overlaps"] == [0., 0.]
    assert action(result)["state"] != "washing_candidate"


def test_body_washing_prediction_cannot_be_fused_into_unassociated_hands(params):
    value = inputs(rgb=False)
    value["actions"] = [video(0., scope="person", track_id=12)]
    result = HandwashNode(params).process(value, frame())
    row = action(result)
    assert row["state"] != "washing_candidate"
    assert row["evidence"]["rgb_washing_score"] is None
    assert row["evidence"]["body_skeleton"][0]["track_id"] == 12
    assert row["evidence"]["body_skeleton_status"] == "available_separate_identity"


def test_low_score_or_non_top_rgb_class_cannot_confirm_washing(params):
    value = inputs()
    value["actions"] = [video(0., score=.1)]
    assert action(HandwashNode(params).process(value, frame()))["state"] != "washing_candidate"
    value["actions"] = [video(0., label="eating")]
    value["actions"][0]["items"][0]["candidates"].append({"label": "washing hands", "score": .7})
    assert action(HandwashNode(params).process(value, frame()))["state"] != "washing_candidate"


def test_mask_and_rgb_ttl_expire_even_with_repeated_stale_packets(params):
    node = HandwashNode(params)
    assert action(node.process(inputs(), frame()))["state"] == "washing_candidate"
    for t, expected in ((.5, "washing_candidate"), (.9, "hands_in_roi")):
        value = inputs(t)
        value["masks"] = {"timestamp": t, "fresh": False, "items": []}
        value["actions"] = [{"timestamp": t, "fresh": False, "items": []}]
        row = action(node.process(value, frame(t)))
        assert row["state"] == expected
    assert row["evidence"]["mask_valid"] is False
    assert row["evidence"]["rgb"] == []


def test_stale_masks_are_not_refreshed_by_timestamp_only(params):
    value = inputs()
    value["masks"]["timestamp"] = -.1
    row = action(HandwashNode(params).process(value, frame()))
    assert row["evidence"]["mask_valid"] is False
    assert row["state"] != "washing_candidate"


def test_disappearance_is_not_leave_and_reappearance_is_not_enter(params):
    node = HandwashNode(params)
    node.process(inputs(), frame())
    lost = node.process(inputs(.2, xs=()), frame(.2))
    assert action(lost)["state"] == "unknown"
    assert not kinds(lost)
    reappeared = node.process(inputs(.4, xs=(5, 10)), frame(.4))
    assert action(reappeared)["state"] == "outside"
    assert not kinds(reappeared)
    entered = node.process(inputs(.6), frame(.6))
    assert kinds(entered) == ["hands_enter_roi"]
    left = node.process(inputs(.8, xs=(5, 10)), frame(.8))
    assert kinds(left) == ["hands_leave_roi"]


def test_unobserved_stale_or_third_hand_is_unknown(params):
    for value in (inputs(xs=(40,)), inputs(xs=(40, 50, 60)), inputs()):
        if len(value["hands"]["items"]) == 2:
            value["hands"]["items"][0]["observed"] = False
        assert action(HandwashNode(params).process(value, frame()))["state"] == "unknown"
    value = inputs()
    value["hands"]["fresh"] = False
    assert action(HandwashNode(params).process(value, frame()))["state"] == "unknown"


def test_continuous_candidate_event_once_and_never_completion(params):
    node = HandwashNode(params)
    events = []
    for t in (0., .5, 1., 1.5):
        output = node.process(inputs(t), frame(t))
        events.extend(output["events"]["items"])
    assert [event["kind"] for event in events] == ["washing_candidate"]
    assert events[0]["timestamp"] == 1.
    assert action(output)["score_kind"] == "rgb_score_not_fused_probability"
    duplicate = node.process(inputs(1.5), frame(1.5))
    assert duplicate["actions"]["fresh"] is False
    assert not duplicate["events"]["items"]


def test_occlusion_breaks_hold_duration(params):
    node = HandwashNode(params)
    for t in (0., .5):
        node.process(inputs(t), frame(t))
    node.process(inputs(.6, xs=()), frame(.6))
    reappeared = node.process(inputs(.7), frame(.7))
    assert action(reappeared)["duration"] == 0.
    assert not kinds(node.process(inputs(1.1), frame(1.1)))


def test_rectangular_boxes_without_encoded_masks_are_not_segmentation(params):
    value = inputs()
    value["masks"]["items"] = [hand(40), hand(60)]
    row = action(HandwashNode(params).process(value, frame()))
    assert row["evidence"]["mask_valid"] is False
    assert row["state"] != "washing_candidate"


@pytest.mark.parametrize("reset_kind", ["explicit", "source", "time_backwards", "gap", "resolution"])
def test_reset_breaks_continuity_and_old_evidence(params, reset_kind):
    node = HandwashNode(params)
    node.process(inputs(0.), frame(0.))
    node.process(inputs(.5), frame(.5))
    t, source, size = .75, "camera", 100
    if reset_kind == "explicit":
        node.reset()
    elif reset_kind == "source":
        source = "other-camera"
    elif reset_kind == "time_backwards":
        t = .1
    elif reset_kind == "gap":
        t = 2.
    else:
        size = 120
    value = inputs(t, rgb=False, mask=False)
    output = node.process(value, frame(t, source, size))
    row = action(output)
    assert row["duration"] == 0
    assert row["evidence"]["rgb"] == []
    assert row["evidence"]["mask_evidence"]["overlaps"] == []
    assert "washing_candidate" not in kinds(output)


@pytest.mark.parametrize("points", [[], [[0, 0], [1, 1]], [[0, 0], [.5, .5], [1, 1]], [[0, 0], [2, 0], [0, 1]], [[0, 0], [float("nan"), .5], [1, 1]]])
def test_invalid_roi_rejected(params, points):
    params["zones"][0]["points"] = points
    with pytest.raises(ValueError):
        validate_handwash_params(params)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_bad_roi_coordinate_is_rejected_during_graph_validation(params, coordinate):
    from backend.workflow_graph import validate_graph
    params["zones"][0]["points"][1][0] = coordinate
    graph = {"version": 1, "nodes": [
        {"id": "source", "type": "source"}, {"id": "hands", "type": "hand_landmarks"},
        {"id": "wash", "type": "handwash", "params": params}, {"id": "output", "type": "output"}],
        "edges": [
            {"source": "source", "sourceHandle": "frame", "target": "hands", "targetHandle": "frame"},
            {"source": "hands", "sourceHandle": "poses", "target": "wash", "targetHandle": "hands"},
            {"source": "wash", "sourceHandle": "actions", "target": "output", "targetHandle": "actions"},
            {"source": "source", "sourceHandle": "frame", "target": "output", "targetHandle": "frame"}]}
    checked = validate_graph(graph)
    assert checked["valid"] is False
    assert any("ROI" in error or "NaN" in error or "Infinity" in error for error in checked["errors"])


def test_corrupt_mask_is_an_explicit_error_not_silent_negative(params):
    value = inputs()
    value["masks"]["items"][0]["mask"]["size"] = [1, 1]
    with pytest.raises(ValueError):
        HandwashNode(params).process(value, frame())


def test_prompt_masks_skip_stale_objects_and_do_not_reuse_pixels(monkeypatch):
    node = HandModelNode("prompt_mask", {"every_seconds": .5}, {"model_id": "sam2.1_tiny"})
    calls = []
    monkeypatch.setattr(node, "_load", lambda: calls.append("load"))
    monkeypatch.setattr(node, "_masks", lambda f, values: masks(f["timestamp"])["items"])
    original = node.process({"objects": {"items": [hand(40)], "timestamp": 0}}, frame())["detections"]
    assert original["items"] and original["fresh"]
    skipped = node.process({"objects": {"items": [hand(40)], "timestamp": .1}}, frame(.1))["detections"]
    assert skipped["items"] == [] and skipped["fresh"] is False
    stale = node.process({"objects": {"items": [hand(40)], "timestamp": .1}}, frame(.6))["detections"]
    assert stale["items"] == []
    assert calls == ["load"]


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()


def test_sam_output_uses_real_mask_geometry_and_rejects_low_quality():
    node = HandModelNode("prompt_mask", {"min_quality": .5}, {})
    pixels = np.zeros((2, 100, 100), np.uint8)
    pixels[0, 40:42, 40:42] = 1
    pixels[1, 60:62, 60:62] = 1
    result = SimpleNamespace(masks=SimpleNamespace(data=Tensor(pixels)), boxes=SimpleNamespace(conf=Tensor([.9, .2])))
    node.model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    values = node._masks(frame(), [hand(40), hand(60)])
    assert len(values) == 1
    assert values[0]["mask_area"] == 4
    assert values[0]["mask_quality"] == .9
    assert np.array_equal(decode_mask(values[0]["mask"], (100, 100)), pixels[0].astype(bool))
    result.masks.data = Tensor(pixels[:1])
    with pytest.raises(ValueError, match="SAM"):
        node._masks(frame(), [hand(40), hand(60)])


def test_sam_rejects_nonfinite_quality():
    node = HandModelNode("prompt_mask", {"min_quality": .5}, {})
    pixels = np.ones((1, 100, 100), np.uint8)
    result = SimpleNamespace(masks=SimpleNamespace(data=Tensor(pixels)),
                             boxes=SimpleNamespace(conf=Tensor([float("nan")])))
    node.model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    assert node._masks(frame(), [hand(40)]) == []


def test_hand_model_close_releases_model_and_reset_allows_replay(monkeypatch):
    node = HandModelNode("hand_landmarks", {}, {})
    closed = []
    node.model = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(node, "_load", lambda: None)
    monkeypatch.setattr(node, "_hands", lambda f: [hand(40)])
    assert node.process({}, frame())["poses"]["fresh"] is True
    assert node.process({}, frame())["poses"]["fresh"] is False
    node.reset()
    assert node.process({}, frame())["poses"]["fresh"] is True
    node.close()
    assert node.model is None and closed == [True]
