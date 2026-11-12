"""Recent IoU keeps current observations ahead of temporarily lost boxes."""
import pytest

import backend.workflow_tracking as tracking
from backend.workflow_tracking import TrackerNode


def person(x=0, label="person"):
    return {"box": [x, 0, x + 10, 20], "label": label, "score": .9}


def step(node, t, items, *, fresh=True, packet_time=None, source="cam", session="one"):
    frame = {"timestamp": t, "width": 1000, "height": 1000, "source_id": source, "session_id": session}
    return node.process({"observations": {"items": items, "fresh": fresh,
        "timestamp": t if packet_time is None else packet_time}}, frame)["tracks"]


def make(**params):
    return TrackerNode({"association_mode": "recent_iou", "min_iou": .15, "max_age_seconds": 1, **params})


@pytest.mark.parametrize("mode,expected", [("iou", 2), ("recent_iou", 1)])
def test_active_match_has_priority_over_lost_box_with_higher_overlap(mode, expected):
    tracker = make(association_mode=mode)
    assert [i["track_id"] for i in step(tracker, 0, [person(), person(9)])["items"]] == [1, 2]
    assert step(tracker, .1, [person(4)])["items"][0]["track_id"] == 1
    # Lost ID2 has IoU .818; current ID1 has .429. Only opt-in mode
    # prioritizes continuity. Neither mode claims appearance-based identity.
    assert step(tracker, .2, [person(8)])["items"][0]["track_id"] == expected


def test_default_mode_stays_original_iou():
    assert TrackerNode({}).association_mode == "iou"
    assert TrackerNode({}).method == "hungarian_iou_same_class"


def test_active_assignments_are_one_to_one_and_input_order_independent():
    tracker = make()
    step(tracker, 0, [person(0), person(50), person(80, "chair")])
    result = step(tracker, .1, [person(51), person(80, "chair"), person(1)])
    assert [i["track_id"] for i in result["items"]] == [2, 3, 1]
    assert result["tracking"]["association_mode"] == "recent_iou"
    assert result["tracking"]["method"].endswith("_recent_first")
    assert len({i["track_id"] for i in result["items"]}) == 3


def test_lost_track_recovers_when_active_track_does_not_match_it():
    tracker = make()
    first = step(tracker, 0, [person(), person(50)])["items"]
    step(tracker, .1, [person(1)])
    result = step(tracker, .2, [person(51), person(2)])["items"]
    assert [i["track_key"] for i in result] == [first[1]["track_key"], first[0]["track_key"]]


def test_fresh_empty_frame_removes_active_priority_but_keeps_recovery():
    tracker = make()
    old = step(tracker, 0, [person()])["items"][0]
    assert step(tracker, .2, [])["items"] == []
    assert tracker._last_observed_ids == set()
    assert step(tracker, .8, [person()])["items"][0]["track_key"] == old["track_key"]


@pytest.mark.parametrize("bad", ["stale", "wrong_timestamp"])
def test_nonfresh_packets_do_not_extend_identity_ttl(bad):
    tracker = make()
    old = step(tracker, 0, [person()])["items"][0]
    kwargs = {"fresh": False} if bad == "stale" else {"packet_time": .2}
    ignored = step(tracker, .8, [person()], **kwargs)
    assert ignored["items"] == [] and ignored["fresh"] is False
    assert ignored["tracking"]["stale_packet_ignored"] is True
    new = step(tracker, 1.01, [person()])["items"][0]
    assert new["track_key"] != old["track_key"]


def test_duplicate_timestamp_cannot_create_track_or_replace_current_box():
    tracker = make()
    old = step(tracker, 0, [person()])["items"][0]
    ignored = step(tracker, 0, [person(100)])
    assert ignored["fresh"] is False and ignored["items"] == []
    assert len(tracker._tracks) == 1
    new = step(tracker, .1, [person()])["items"][0]
    assert old["track_key"] == new["track_key"]


def test_cached_frame_does_not_take_away_last_fresh_active_priority():
    tracker = make()
    step(tracker, 0, [person(), person(9)])
    step(tracker, .1, [person(4)])
    step(tracker, .15, [], fresh=False)
    assert step(tracker, .2, [person(8)])["items"][0]["track_id"] == 1


def test_valid_packet_at_timestamp_of_preceding_stale_packet_is_processed():
    tracker = make()
    step(tracker, 0, [person()])
    assert step(tracker, .1, [person()], fresh=False)["items"] == []
    assert step(tracker, .1, [person()])["items"][0]["track_id"] == 1


@pytest.mark.parametrize("kwargs,t", [({"source": "other"}, .2), ({"session": "two"}, .2), ({}, .05)])
def test_source_session_or_rewind_resets_priority_and_scoped_identity(kwargs, t):
    tracker = make()
    old = step(tracker, .1, [person()])["items"][0]
    new = step(tracker, t, [person()], **kwargs)["items"][0]
    assert new["track_id"] == 1 and new["track_key"] != old["track_key"]


def test_no_overlap_or_different_class_never_uses_active_identity():
    tracker = make()
    step(tracker, 0, [person()])
    assert step(tracker, .1, [person(50)])["items"][0]["track_id"] == 2
    assert step(tracker, .2, [person(50, "chair")])["items"][0]["track_id"] == 3


def test_recent_first_greedy_fallback_remains_explicit(monkeypatch):
    monkeypatch.setattr(tracking, "_solver", lambda: None)
    tracker = make()
    step(tracker, 0, [person(), person(9)])
    step(tracker, .1, [person(4)])
    result = step(tracker, .2, [person(8)])
    assert result["items"][0]["track_id"] == 1
    assert result["tracking"]["method"] == "greedy_iou_same_class_recent_first"


@pytest.mark.parametrize("value", [None, True, 1, "nearest", [], {}])
def test_invalid_association_mode(value):
    with pytest.raises(ValueError, match="association_mode"):
        make(association_mode=value)
