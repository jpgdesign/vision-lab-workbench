"""Desk occlusion rules: observable posture evidence, never attention claims."""
from copy import deepcopy
import math

import pytest

from backend.workflow_office import (
    OfficeNode,
    partial_seated_pose_evidence,
    validate_office_params,
)


SEAT_ZONE = {"name": "座位肩膀區", "points": [[.15, .32], [.65, .32], [.65, .55], [.15, .55]]}
BROAD_ZONE = {"name": "校正區", "points": [[.1, .1], [.8, .1], [.8, .85], [.1, .85]]}


def pose(seated=True):
    points = [[0., 0., 0.] for _ in range(17)]
    for index, x in enumerate((325, 315, 335, 300, 350)):
        points[index] = [float(x), 140., .9]
    for shoulder, hip, knee, ankle, x in ((5, 11, 13, 15, 300), (6, 12, 14, 16, 350)):
        points[shoulder] = [float(x), 200., .9]
        points[hip] = [float(x), 400., .9]
        points[knee] = [x + (150. if seated else 0.), 420. if seated else 550., .9]
        points[ankle] = [x + (150. if seated else 0.), 650. if seated else 750., .9]
    return points


def without(points, *indices):
    result = deepcopy(points)
    for index in indices:
        result[index][2] = 0.
    return result


def upper_body(y=400., dx=0.):
    result = without(pose(), 11, 12, 13, 14, 15, 16)
    for index in range(7):
        result[index][0] += dx
        result[index][1] += y - 200.
    return result


def person(points, identity=1, box=None):
    return {"label": "person", "score": .95, "track_id": identity,
            "track_key": f"camera/pose/{identity}", "observed": True,
            "box": box or [200., 100., 600., 850.], "keypoints": points}


def node(**params):
    return OfficeNode("office_monitor", {
        "posture_mode": "occlusion_assisted", "sitting_dwell_seconds": .2,
        "occluded_dwell_seconds": .2, "side_dwell_seconds": .1,
        "cooldown_seconds": .5, **params,
    })


def step(rule, timestamp, *items, source="camera", session="one", fresh=True, packet_time=None):
    frame = {"timestamp": timestamp, "width": 1000, "height": 1000,
             "source_id": source, "session_id": session}
    packet = {"items": list(items), "timestamp": timestamp if packet_time is None else packet_time,
              "fresh": fresh}
    return rule.process({"tracks": packet}, frame)


def action(result):
    return result["actions"]["items"][0]


def occlusion_events(result):
    return [event for event in result["events"]["items"] if event["kind"] == "seated_occlusion_candidate"]


def test_strict_mode_remains_default_and_requires_ankles():
    assert validate_office_params({})["posture_mode"] == "strict_full_body"
    rule = node(posture_mode="strict_full_body", zones=[SEAT_ZONE])
    item = person(without(pose(), 15, 16))
    step(rule, 0., item)
    assert action(step(rule, .3, item))["signal"] == "unknown"


def test_visible_thighs_allow_partial_evidence_without_ankles():
    signal, confidence, details = partial_seated_pose_evidence(
        person(without(pose(), 15, 16)), 1000, 1000, .5)
    assert signal == "seated_candidate"
    assert confidence == pytest.approx(.9)
    assert details["method"] == "partial_skeleton"


def test_one_visible_body_side_is_usable_without_inventing_hidden_joints():
    points = without(pose(), 6, 12, 14, 15, 16)
    assert partial_seated_pose_evidence(person(points), 1000, 1000, .5)[0] == "seated_candidate"


def test_visible_conflicting_leg_is_not_discarded_in_favor_of_seated_side():
    points = without(pose(), 15, 16)
    points[14] = [350., 550., .9]  # Right thigh is vertical, unlike the left.
    assert partial_seated_pose_evidence(person(points), 1000, 1000, .5)[0] == "unknown"


def test_partial_standing_geometry_is_not_a_seated_candidate():
    points = without(pose(False), 15, 16)
    signal = partial_seated_pose_evidence(person(points), 1000, 1000, .5)[0]
    assert signal in {"standing", "unknown"}
    rule = node(zones=[BROAD_ZONE])
    step(rule, 0., person(points))
    assert action(step(rule, .3, person(points)))["signal"] not in {
        "seated_occlusion_candidate", "seated_rule"}


def test_partial_candidates_remain_qualified_after_dwell_and_event_is_once():
    rule = node()
    item = person(without(pose(), 15, 16))
    assert action(step(rule, 0., item))["signal"] == "occlusion_warming"
    result = step(rule, .3, item)
    candidate = action(result)
    assert candidate["signal"] == "seated_occlusion_candidate"
    assert candidate["label"] == "認真（坐姿規則） · 遮擋候選"
    assert candidate["infers_attention"] is False
    assert len(occlusion_events(result)) == 1
    assert not occlusion_events(step(rule, .6, item))
    assert all(event["kind"] != "seated_rule" for event in result["events"]["items"])


def test_transition_requirement_only_applies_to_upper_body_roi_fallback():
    rule = node(require_sit_transition=True, zones=[])
    item = person(without(pose(), 15, 16))
    step(rule, 0., item)
    assert action(step(rule, .3, item))["signal"] == "seated_occlusion_candidate"


def test_full_visible_seated_rule_keeps_existing_strict_behavior():
    rule = node()
    item = person(pose())
    assert action(step(rule, 0., item))["signal"] == "seated_candidate"
    assert action(step(rule, .3, item))["signal"] == "seated_rule"


def test_upper_body_alone_requires_explicit_seat_calibration():
    rule = node()
    item = person(upper_body())
    step(rule, 0., item)
    result = action(step(rule, .3, item))
    assert result["signal"] == "unknown"
    assert result["details"]["reason"] == "seat_zones_not_configured"
    assert "尚未校正" in result["label"]


def test_calibrated_upper_body_becomes_occlusion_candidate_after_stability():
    rule = node(zones=[SEAT_ZONE])
    item = person(upper_body())
    assert action(step(rule, 0., item))["signal"] == "occlusion_warming"
    result = step(rule, .3, item)
    assert action(result)["signal"] == "seated_occlusion_candidate"
    assert "遮擋候選" in action(result)["label"]
    assert action(result)["infers_attention"] is False
    assert len(occlusion_events(result)) == 1


def test_person_box_inside_roi_does_not_replace_shoulder_position():
    rule = node(zones=[SEAT_ZONE])
    item = person(upper_body(y=200.), box=[200., 330., 600., 530.])
    step(rule, 0., item)
    assert action(step(rule, .3, item))["signal"] == "unknown"


def test_visible_leaning_torso_rejects_roi_even_when_shoulders_are_in_zone():
    points = without(pose(), 13, 14, 15, 16)
    points[5] = [500., 300., .9]
    points[6] = [550., 300., .9]
    rule = node(zones=[BROAD_ZONE])
    step(rule, 0., person(points))
    assert action(step(rule, .3, person(points)))["signal"] == "unknown"


@pytest.mark.parametrize("bad", ["head", "shoulder", "nan", "offscreen"])
def test_insufficient_or_invalid_upper_body_evidence_remains_unknown(bad):
    points = upper_body()
    if bad == "head":
        points = without(points, 0, 1, 2, 3, 4)
    elif bad == "shoulder":
        points[5][2] = .1
    elif bad == "nan":
        points[5][0] = math.nan
    else:
        points[5][0] = 1200.
    rule = node(zones=[SEAT_ZONE])
    step(rule, 0., person(points))
    assert action(step(rule, .3, person(points)))["signal"] == "unknown"


def test_moving_upper_body_restarts_stability_window():
    rule = node(zones=[SEAT_ZONE], max_pose_drift=.03)
    step(rule, 0., person(upper_body()))
    step(rule, .15, person(upper_body(dx=150.)))
    assert action(step(rule, .31, person(upper_body(dx=150.))))["signal"] == "occlusion_warming"
    assert action(step(rule, .4, person(upper_body(dx=150.))))["signal"] == "seated_occlusion_candidate"


def test_transition_requirement_does_not_accept_initial_seated_appearance():
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True)
    item = person(upper_body())
    step(rule, 0., item)
    result = step(rule, .3, item)
    assert action(result)["signal"] not in {"seated_occlusion_candidate", "seated_rule"}
    assert not occlusion_events(result)


def test_transition_requirement_accepts_same_id_standing_then_lower_shoulders():
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True, sit_drop_ratio=.06)
    assert action(step(rule, 0., person(pose(False))))["signal"] == "standing"
    step(rule, .15, person(upper_body()))
    result = step(rule, .45, person(upper_body()))
    assert action(result)["signal"] == "seated_occlusion_candidate"
    assert len(occlusion_events(result)) == 1


def test_visible_lowering_before_entering_seat_zone_preserves_standing_reference():
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True)
    step(rule, 0., person(pose(False)))
    # The tracked upper body stays visible while it moves down toward the
    # calibrated seated band; it has not reached that band yet.
    assert action(step(rule, .15, person(upper_body(y=300.))))["signal"] == "unknown"
    step(rule, .3, person(upper_body()))
    assert action(step(rule, .6, person(upper_body())))["signal"] == "seated_occlusion_candidate"


def test_standing_baseline_is_not_shared_between_people():
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True)
    step(rule, 0., person(pose(False), identity=1))
    step(rule, .15, person(upper_body(), identity=2))
    assert action(step(rule, .45, person(upper_body(), identity=2)))["signal"] not in {
        "seated_occlusion_candidate", "seated_rule"}


def test_small_shoulder_motion_is_not_a_sit_down_transition():
    rule = node(zones=[BROAD_ZONE], require_sit_transition=True, sit_drop_ratio=.06)
    step(rule, 0., person(pose(False)))
    step(rule, .15, person(upper_body(y=240.)))  # Only 4% of frame height.
    assert action(step(rule, .45, person(upper_body(y=240.))))["signal"] not in {
        "seated_occlusion_candidate", "seated_rule"}


def test_track_disappearance_invalidates_standing_transition_reference():
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True)
    step(rule, 0., person(pose(False)))
    step(rule, .1)
    step(rule, .2, person(upper_body()))
    assert action(step(rule, .5, person(upper_body())))["signal"] not in {
        "seated_occlusion_candidate", "seated_rule"}


def test_old_standing_reference_cannot_certify_a_later_lower_position():
    rule = node(zones=[BROAD_ZONE], require_sit_transition=True)
    step(rule, 0., person(pose(False)))
    # Keep observing the same person without a valid drop, so this tests the
    # transition reference lifetime independently of track expiry/gap resets.
    for index in range(1, 22):
        step(rule, index * .5, person(upper_body(y=240.)))
    step(rule, 10.6, person(upper_body()))
    assert action(step(rule, 10.9, person(upper_body())))["signal"] == "unknown"


@pytest.mark.parametrize("scale", [.5, 1.5])
def test_approaching_or_receding_person_is_not_a_sit_transition(scale):
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True)
    step(rule, 0., person(pose(False)))
    points = upper_body()
    for point in points:
        if point[2]:
            point[0] = 325. + (point[0] - 325.) * scale
    step(rule, .15, person(points))
    assert action(step(rule, .45, person(points)))["signal"] == "unknown"


def test_large_sideways_motion_with_lower_shoulders_is_not_a_sit_transition():
    rule = node(zones=[SEAT_ZONE], require_sit_transition=True)
    step(rule, 0., person(pose(False)))
    item = person(upper_body(dx=150.))
    step(rule, .15, item)
    assert action(step(rule, .45, item))["signal"] == "unknown"


def test_fresh_missing_track_breaks_occlusion_dwell_and_rearms_event():
    rule = node(zones=[SEAT_ZONE])
    item = person(upper_body())
    step(rule, 0., item)
    assert len(occlusion_events(step(rule, .3, item))) == 1
    step(rule, .4)
    assert action(step(rule, .5, item))["signal"] == "occlusion_warming"
    assert len(occlusion_events(step(rule, .8, item))) == 1


@pytest.mark.parametrize("change", ["source", "session", "gap", "rewind"])
def test_source_or_temporal_discontinuity_cannot_reuse_occlusion_dwell(change):
    rule = node(zones=[SEAT_ZONE])
    item = person(upper_body())
    step(rule, 1., item)
    assert action(step(rule, 1.3, item))["signal"] == "seated_occlusion_candidate"
    kwargs = {change: "two"} if change in {"source", "session"} else {}
    t = 3. if change == "gap" else .1 if change == "rewind" else 1.4
    result = step(rule, t, item, **kwargs)
    assert action(result)["signal"] == "occlusion_warming"
    assert not occlusion_events(result)


def test_duplicate_and_stale_packets_do_not_emit_candidate_events():
    rule = node(zones=[SEAT_ZONE])
    item = person(upper_body())
    step(rule, 0., item)
    for result in (step(rule, 0., item), step(rule, .3, item, fresh=False),
                   step(rule, .4, item, packet_time=.1)):
        assert result["actions"]["fresh"] is False
        assert not result["actions"]["items"]
        assert not result["events"]["items"]


@pytest.mark.parametrize("stale", [{"fresh": False}, {"packet_time": .1}])
def test_new_frame_after_stale_packet_starts_a_new_occlusion_window(stale):
    rule = node(zones=[SEAT_ZONE])
    item = person(upper_body())
    step(rule, 0., item)
    step(rule, .3, item, **stale)
    result = step(rule, .4, item)
    assert action(result)["signal"] == "occlusion_warming"
    assert not occlusion_events(result)
    assert action(step(rule, .7, item))["signal"] == "seated_occlusion_candidate"


def test_switching_evidence_method_does_not_reuse_prior_dwell():
    rule = node(zones=[BROAD_ZONE])
    step(rule, 0., person(without(pose(), 15, 16)))
    result = step(rule, .3, person(upper_body(y=200.)))
    assert action(result)["signal"] == "occlusion_warming"
    assert action(step(rule, .6, person(upper_body(y=200.))))["signal"] == "seated_occlusion_candidate"


def test_occluded_posture_mode_does_not_change_crossing_direction_counts():
    rule = node(zones=[SEAT_ZONE])
    for t, x in ((0., .2), (.2, .2), (.4, .8), (.6, .8)):
        result = step(rule, t, person(upper_body(), box=[x * 1000 - 40, 100., x * 1000 + 40, 850.]))
    crossings = [event for event in result["events"]["items"] if event["kind"] == "line_crossing"]
    assert [(event["label"], event["details"]["direction"]) for event in crossings] == [("出去", "out")]
    assert result["events"]["counts"] == {"in": 0, "out": 1}


@pytest.mark.parametrize("params", [
    {"posture_mode": "guess"}, {"occluded_dwell_seconds": -1},
    {"occluded_dwell_seconds": math.nan}, {"require_sit_transition": "yes"},
    {"sit_drop_ratio": math.inf}, {"max_pose_drift": -1},
    {"zones": [{"name": "bad", "points": [[0., 0.], [1., 1.]]}]},
])
def test_occlusion_parameters_reject_invalid_configuration(params):
    with pytest.raises(ValueError):
        validate_office_params(params)


@pytest.mark.parametrize("zones", [
    None, {},
    [{**SEAT_ZONE, "name": str(index)} for index in range(9)],
    [SEAT_ZONE, SEAT_ZONE],
    [{"name": "", "points": SEAT_ZONE["points"]}],
    [{"name": "bad", "points": [[0., 0.], [.5, .5], [1., 1.]]}],
    [{"name": "bad", "points": [[False, 0.], [1., 0.], [0., 1.]]}],
    [{"name": "bad", "points": [[math.nan, 0.], [1., 0.], [0., 1.]]}],
    [{"name": "bad", "points": [[-.1, 0.], [1., 0.], [0., 1.]]}],
    [{"name": "bad", "points": [[0., 0., 1.], [1., 0.], [0., 1.]]}],
])
def test_seat_calibration_rejects_invalid_or_ambiguous_zone_configuration(zones):
    with pytest.raises(ValueError):
        validate_office_params({"zones": zones})
