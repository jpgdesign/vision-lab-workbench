import copy
import math

from PIL import Image
import pytest

from backend.workflow_cv_rules import CVRuleNode, MAX_HISTORY, MAX_TRACKS, validate_cv_rule_params


def frame(stamp=0., color="red", size=(100, 100), **extra):
    return {"image": Image.new("RGB", size, color), "timestamp": stamp, "frame_index": round(stamp * 30), "source_id": "cam", "width": size[0], "height": size[1], **extra}


def tracked(x=10, identity="one", **extra):
    return {"track_key": identity, "track_id": 1, "box": [x, 10, x + 20, 40], "label": "person", "score": .9, **extra}


def process(node, stamp=0., items=None, **kwargs):
    inputs = {"tracks": {"items": items}} if items is not None else {}
    return node.process(inputs, frame(stamp, **kwargs))


@pytest.mark.parametrize("kind,params", [
    ("cv_roi", {"points": [[0, 0], [1, 1], [0, 1], [1, 0]]}),
    ("cv_roi", {"points": [[0, 0], [.5, .5], [1, 1]]}),
    ("cv_roi", {"points": [[0, 0], [1, 0], [1, 1], [0, 0]]}),
    ("cv_roi", {"points": [[0, 0], [1, 0], [2, 1]]}),
    ("cv_roi", {"points": [[0, 0], [1, 0]]}),
    ("cv_roi", {"alpha": math.nan}),
    ("cv_roi", {"color": [255, 0, .5]}),
    ("cv_roi", {"color": [True, 0, 0]}),
    ("cv_roi", {"mode": "bad"}),
    ("cv_histogram", {"reference": "bad"}),
    ("cv_histogram", {"hue_bins": 12.5}),
    ("cv_histogram", {"similarity_threshold": 2}),
    ("cv_track_motion", {"anchor": "bad"}),
    ("cv_track_motion", {"window_seconds": -1}),
    ("cv_track_motion", {"moving_threshold": math.inf}),
    ("cv_track_motion", {"max_gap_seconds": True}),
])
def test_bad_parameters_rejected_before_execution(kind, params):
    with pytest.raises(ValueError):
        validate_cv_rule_params(kind, params)


def test_roi_mask_inverse_tint_and_no_source_mutation():
    source = frame(color=(100, 100, 100), to_source=[[2, 0, 10], [0, 2, 20], [0, 0, 1]])
    for mode, inner, outer in [("mask_inside", 100, 0), ("mask_outside", 0, 100), ("tint", 150, 100)]:
        node = CVRuleNode("cv_roi", {"mode": mode, "color": [200, 200, 200], "alpha": .5})
        result = node.process({}, source)["frame"]["frame"]
        assert result["image"].getpixel((50, 50)) == (inner,) * 3
        assert result["image"].getpixel((0, 0)) == (outer,) * 3
        assert result["width"] == 100 and result["height"] == 100
        assert result["to_source"] == source["to_source"]
    assert source["image"].getpixel((0, 0)) == (100, 100, 100)


def test_concave_polygon_allowed_but_bowtie_rejected():
    p = validate_cv_rule_params("cv_roi", {"points": [[0, 0], [1, 0], [.5, .5], [1, 1], [0, 1]]})
    assert len(p["points"]) == 5
    with pytest.raises(ValueError, match="自相交"):
        validate_cv_rule_params("cv_roi", {"points": [[0, 0], [1, 1], [.2, .8], [1, 0]]})


def test_histogram_scene_red_to_green_detects_distribution_change():
    node = CVRuleNode("cv_histogram", {})
    assert process(node)["actions"]["items"][0]["label"] == "unknown"
    stable = process(node, .1)["actions"]["items"][0]
    assert stable["label"] == "appearance_stable" and stable["similarity"] == pytest.approx(1)
    changed = process(node, .2, color="green")["actions"]["items"][0]
    assert changed["label"] == "appearance_change"
    assert changed["scope"] == "scene"
    assert process(node, .3, color="green")["actions"]["items"][0]["label"] == "appearance_stable"


def test_histogram_first_reference_retained_until_gap():
    node = CVRuleNode("cv_histogram", {"reference": "first"})
    process(node)
    process(node, .1, color="green")
    assert process(node, .2, color="green")["actions"]["items"][0]["label"] == "appearance_change"
    assert process(node, 2., color="green")["actions"]["items"][0]["label"] == "unknown"
    assert process(node, 2.1, color="green")["actions"]["items"][0]["label"] == "appearance_stable"


def test_histogram_track_crops_are_independent_and_missing_reference_dropped():
    node = CVRuleNode("cv_histogram", {})
    image = Image.new("RGB", (100, 100), "red")
    image.paste("green", (50, 0, 100, 100))
    items = [tracked(), tracked(60, identity="two", track_id=2)]
    source = frame()
    source["image"] = image
    node.process({"tracks": {"items": items}}, source)
    source["timestamp"] = .1
    actions = node.process({"tracks": {"items": list(reversed(items))}}, source)["actions"]["items"]
    assert all(a["label"] == "appearance_stable" for a in actions)
    assert [a["track_key"] for a in actions] == ["two", "one"]
    source["timestamp"] = .2
    assert node.process({"tracks": {"items": []}}, source)["actions"]["items"] == []
    source["timestamp"] = .3
    assert all(a["label"] == "unknown" for a in node.process({"tracks": {"items": items}}, source)["actions"]["items"])


def test_motion_uses_source_time_sums_path_and_keeps_id():
    node = CVRuleNode("cv_track_motion", {"moving_threshold": 20})
    first = process(node, 0, [tracked(10)])["actions"]["items"][0]
    assert first["label"] == "unknown" and first["speed_px_s"] is None
    second = process(node, .5, [tracked(30)])["actions"]["items"][0]
    assert second["label"] == "moving" and second["track_key"] == "one"
    assert second["speed_px_s"] == pytest.approx(40)
    assert second["distance_px"] == pytest.approx(20)
    assert second["normalized_speed"] == pytest.approx(40 / math.hypot(100, 100))
    returned = process(node, 1., [tracked(10)])["actions"]["items"][0]
    assert returned["distance_px"] == pytest.approx(40)  # Not net displacement.
    assert returned["speed_px_s"] == pytest.approx(40)
    assert "speed_m_s" not in returned and "distance_m" not in returned


def test_motion_missing_person_gap_and_zero_dt_never_invent_motion():
    node = CVRuleNode("cv_track_motion", {})
    process(node, 0, [tracked()])
    process(node, .1, [])
    assert process(node, .2, [tracked(30)])["actions"]["items"][0]["label"] == "unknown"
    assert process(node, .2, [tracked(40)])["actions"]["items"][0]["label"] == "unknown"
    assert process(node, 2, [tracked(50)])["actions"]["items"][0]["label"] == "unknown"


@pytest.mark.parametrize("kind", ["cv_histogram", "cv_track_motion"])
def test_skipped_updates_do_not_hide_long_gap(kind):
    node = CVRuleNode(kind, {})
    process(node, 0, [tracked()])
    for stamp in [.5, 1., 1.5]:
        result = node.process({"tracks": {"items": [tracked()], "fresh": False}}, frame(stamp))
        assert result["actions"]["fresh"] is False
        assert result["actions"]["items"] == []
    assert process(node, 2., [tracked()])["actions"]["items"][0]["label"] == "unknown"


@pytest.mark.parametrize("kind", ["cv_histogram", "cv_track_motion"])
@pytest.mark.parametrize("change", [{"source_id": "other"}, {"session_id": "new"}, {"size": (120, 120)}, {"to_source": [[1, 0, 10], [0, 1, 10], [0, 0, 1]]}])
def test_context_change_resets_history(kind, change):
    node = CVRuleNode(kind, {})
    process(node, 0, [tracked()])
    assert process(node, .1, [tracked()], **change)["actions"]["items"][0]["label"] == "unknown"


def test_motion_preserves_upstream_data_without_mutation():
    node = CVRuleNode("cv_track_motion", {})
    item = tracked(keypoints=[[10, 10, .9]])
    untouched = copy.deepcopy(item)
    overlays = [{"kind": "polygon", "points": [[0, 0], [1, 0], [1, 1]]}]
    packet = node.process({"tracks": {"items": [item], "overlays": overlays}}, frame())["tracks"]
    packet["items"][0]["keypoints"][0][0] = 99
    packet["overlays"][0]["points"][0][0] = .2
    assert item == untouched
    assert overlays[0]["points"][0] == [0, 0]


def test_motion_invalid_and_unobserved_boxes_do_not_leave_state():
    node = CVRuleNode("cv_track_motion", {})
    process(node, 0, [tracked()])
    invalid = [tracked(observed=False), tracked(box=[math.nan, 10, 30, 40]), tracked(box=[200, 200, 250, 250]), {"box": [1, 1, 10, 10]}]
    assert process(node, .1, invalid)["actions"]["items"] == []
    assert node._states == {}


@pytest.mark.parametrize("kind", ["cv_histogram", "cv_track_motion"])
def test_active_track_count_is_bounded(kind):
    node = CVRuleNode(kind, {})
    items = [tracked(identity=str(i)) for i in range(MAX_TRACKS + 100)]
    result = process(node, 0, items)
    assert len(node._states) == MAX_TRACKS
    assert len(result["actions"]["items"]) == MAX_TRACKS
    process(node, .1, [])
    assert node._states == {}


def test_motion_high_frame_rate_history_is_bounded():
    node = CVRuleNode("cv_track_motion", {"window_seconds": 30})
    for i in range(500):
        process(node, i / 100, [tracked()])
    assert len(node._states["one"]["history"]) == MAX_HISTORY
    node.close()
    assert node._states == {}
