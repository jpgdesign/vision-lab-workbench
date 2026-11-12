"""Temporal office rules: no real model or camera is required."""
import math

import pytest

from backend.workflow_office import OfficeNode, seated_pose_evidence, validate_office_params


def node(**params):
    return OfficeNode("office_monitor", {"side_dwell_seconds": .1, "cooldown_seconds": .5,
                                         "sitting_dwell_seconds": .2, **params})


def person(x=.25, y=.5, track_id=1, keypoints=None, **extras):
    result = {"label": "person", "score": .95, "track_id": track_id,
              "track_key": f"source/pose/{track_id}", "box": [x * 1000 - 40, y * 1000 - 100, x * 1000 + 40, y * 1000 + 100]}
    if keypoints is not None:
        result["keypoints"] = keypoints
    return {**result, **extras}


def step(rule, t, x=None, *, items=None, source="camera", session="one", fresh=True, packet_time=None):
    frame = {"timestamp": t, "width": 1000, "height": 1000, "source_id": source, "session_id": session}
    if items is None:
        items = [] if x is None else [person(x)]
    return rule.process({"tracks": {"items": items, "timestamp": t if packet_time is None else packet_time, "fresh": fresh}}, frame)


def crossings(result):
    return [e for e in result["events"]["items"] if e["kind"] == "line_crossing"]


def skeleton(seated=True):
    points = [[0., 0., 0.] for _ in range(17)]
    for s, h, k, a, x in ((5, 11, 13, 15, 300), (6, 12, 14, 16, 350)):
        points[s] = [x, 200., .9]
        points[h] = [x, 400., .9]
        points[k] = [x + (150 if seated else 0), 420. if seated else 550., .9]
        points[a] = [x + (150 if seated else 0), 650. if seated else 750., .9]
    return points


def test_direction_counts_are_source_coordinates_and_initial_appearance_is_not_crossing():
    rule = node()
    assert not crossings(step(rule, 0., .25))
    assert not crossings(step(rule, .2, .25))
    assert not crossings(step(rule, .4, .75))
    out = step(rule, .6, .75)
    assert [(e["label"], e["details"]["direction"]) for e in crossings(out)] == [("出去", "out")]
    assert out["events"]["counts"] == {"in": 0, "out": 1}
    assert not crossings(step(rule, .8, .75))
    assert not crossings(step(rule, 1., .25))
    incoming = step(rule, 1.2, .25)
    assert crossings(incoming)[0]["label"] == "進來"
    assert incoming["actions"]["counts"] == {"in": 1, "out": 1}
    assert incoming["events"]["overlays"][0]["points"] == [[.5, 0.], [.5, 1.]]
    assert len({e["event_id"] for e in crossings(out) + crossings(incoming)}) == 2


def test_jitter_line_touch_and_insufficient_travel_do_not_count():
    rule = node(min_travel=.2)
    for t, x in enumerate((.35, .35, .49, .50, .51, .48, .50, .53, .53)):
        assert not crossings(step(rule, t * .2, x))
    assert rule.counts == {"in": 0, "out": 0}
    # Once the center really travels beyond the configured distance it counts.
    assert len(crossings(step(rule, 1.8, .75))) == 1


def test_cooldown_does_not_defer_phantom_count():
    rule = node(cooldown_seconds=2.)
    for t, x in ((0, .2), (.2, .2), (.4, .8), (.6, .8), (.8, .2), (1., .2), (1.5, .2), (2., .2), (2.8, .2)):
        step(rule, t, x)
    assert rule.counts == {"in": 0, "out": 1}


def test_outside_finite_line_is_not_crossing():
    rule = node(line_y_start=.3, line_y_end=.7)
    for t, x in ((0, .2), (.2, .2), (.4, .8), (.6, .8)):
        assert not crossings(step(rule, t, items=[person(x, y=.1)]))
    assert rule.counts == {"in": 0, "out": 0}
    # Moving down on the new side cannot retroactively become a crossing.
    assert not crossings(step(rule, .8, items=[person(.8, y=.5)]))


def test_new_identity_and_long_observation_gap_do_not_create_crossing():
    rule = node(max_gap_seconds=.5, track_ttl_seconds=1.)
    step(rule, 0., .2)
    step(rule, .2, .2)
    step(rule, .4, items=[person(.8, track_id=2)])
    assert not crossings(step(rule, .6, items=[person(.8, track_id=2)]))
    assert not crossings(step(rule, 1.2, .8))
    assert not crossings(step(rule, 1.4, .8))
    assert rule.counts == {"in": 0, "out": 0}
    step(rule, 3.)
    assert not rule._states


def test_stale_duplicate_and_unobserved_samples_cannot_supply_dwell():
    rule = node(max_gap_seconds=.5)
    step(rule, 0., .2)
    step(rule, .2, .2)
    assert not crossings(step(rule, .4, .8, fresh=False))
    assert not crossings(step(rule, .6, .8, packet_time=.2))
    assert not crossings(step(rule, .8, items=[person(.8, observed=False)]))
    assert not crossings(step(rule, 1., .8))
    assert not crossings(step(rule, 1., .8))
    assert not crossings(step(rule, 1.2, .8))  # origin evidence expired
    assert rule.counts == {"in": 0, "out": 0}


@pytest.mark.parametrize("change", ["source", "session", "time"])
def test_each_source_session_and_rewind_resets_state_and_counts(change):
    rule = node()
    for t, x in ((0, .2), (.2, .2), (.4, .8), (.6, .8)):
        step(rule, t, x)
    assert rule.counts["out"] == 1
    kwargs = {"source": "second"} if change == "source" else {"session": "second"} if change == "session" else {}
    result = step(rule, 0. if change == "time" else .8, .2, **kwargs)
    assert not crossings(result)
    assert result["events"]["counts"] == {"in": 0, "out": 0}


def test_nodes_and_track_states_are_isolated_and_bounded():
    first, second = node(max_tracks=2), node()
    for t, x in ((0, .2), (.2, .2), (.4, .8), (.6, .8)):
        step(first, t, x)
    assert second.counts == {"in": 0, "out": 0}
    step(first, .8, items=[person(.2, track_id=i) for i in range(10)])
    assert len(first._states) == 2


def test_seated_rule_requires_both_legs_and_dwell_then_reports_user_rule():
    rule = node()
    item = person(keypoints=skeleton())
    first = step(rule, 0., items=[item])
    assert first["actions"]["items"][0]["signal"] == "seated_candidate"
    result = step(rule, .3, items=[item])
    action = result["actions"]["items"][0]
    assert action["label"] == "認真（坐姿規則）"
    assert action["infers_attention"] is False
    assert action["score_semantics"] == "minimum_keypoint_confidence_not_action_probability"
    assert len(result["events"]["items"]) == 1
    assert not step(rule, .5, items=[item])["events"]["items"]
    occluded = skeleton()
    occluded[15][2] = .1
    missing = step(rule, .6, items=[person(keypoints=occluded)])
    assert missing["actions"]["items"][0]["kind"] == "unknown"
    assert step(rule, .7, items=[item])["actions"]["items"][0]["signal"] == "seated_candidate"


def test_missing_track_breaks_sitting_duration():
    rule = node(sitting_dwell_seconds=.5)
    item = person(keypoints=skeleton())
    step(rule, 0., items=[item])
    step(rule, .3)
    assert step(rule, .6, items=[item])["actions"]["items"][0]["signal"] == "seated_candidate"
    assert not step(rule, .8, items=[item])["events"]["items"]


def test_standing_and_missing_keypoints_are_not_seated():
    assert seated_pose_evidence(person(keypoints=skeleton(False)), 1000, 1000)[0] == "standing"
    assert seated_pose_evidence(person(), 1000, 1000)[0] == "unknown"
    bad = skeleton()
    bad[13] = [math.nan, 100, .99]
    assert seated_pose_evidence(person(keypoints=bad), 1000, 1000)[0] == "unknown"
    outside = skeleton()
    outside[15][1] = 1100
    assert seated_pose_evidence(person(keypoints=outside), 1000, 1000)[0] == "unknown"
    crouch = skeleton()
    for hip, knee in ((11, 13), (12, 14)):
        crouch[hip][1] = crouch[knee][1] + 100
    assert seated_pose_evidence(person(keypoints=crouch), 1000, 1000)[0] == "unknown"


@pytest.mark.parametrize("params", [{"line_x": 0}, {"line_y_start": .8, "line_y_end": .2},
                                    {"hysteresis": math.nan}, {"max_gap_seconds": True},
                                    {"max_tracks": 2.5}, {"track_ttl_seconds": .5}, {"unknown": 1}])
def test_bad_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        validate_office_params(params)


def test_custom_seated_label_remains_explicitly_a_rule():
    assert validate_office_params({"seated_label": "認真"})["seated_label"] == "認真（坐姿規則）"


def test_nonperson_and_invalid_tracks_do_not_acquire_state():
    rule = node()
    result = step(rule, 0, items=[person(label="chair"), person(track_key="", track_id=None),
                                person(score=math.nan), person(box=[1, 1, 0, 0])])
    assert not rule._states and not result["actions"]["items"]
