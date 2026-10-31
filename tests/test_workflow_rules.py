import math

import pytest

from backend.workflow_rules import RuleNode


def frame(t, source="camera-a", **extra):
    return {"timestamp": t, "frame_index": round(t * 30), "width": 100, "height": 100, "source_id": source, **extra}


def packet(items, t=0):
    return {"items": items, "timestamp": t, "frame": frame(t)}


def track(i=1, box=(20, 10, 40, 80), **extra):
    return {"track_id": i, "track_key": f"person-{i}", "box": list(box), "label": "person", "score": .9, "observed": True, **extra}


def action(label, i=1, score=.9, **extra):
    return {"label": label, "track_id": i, "track_key": f"person-{i}", "scope": "person", "score": score, **extra}


def step(node, t, port="tracks", items=None, **kwargs):
    return node.process({port: packet(items or [], t)}, frame(t, **kwargs))


def labels(result, port="events"):
    return [x["label"] for x in result[port]["items"]]


ZONE = {"name": "check", "points": [[0, .5], [.5, .5], [.5, 1], [0, 1]]}


@pytest.mark.parametrize("kind,params", [
    ("unknown", {}), ("condition", {"min_duration": math.nan}),
    ("condition", {"cooldown": math.inf}), ("condition", {"min_score": 1.01}),
    ("condition", {"min_score": True}), ("zone", {"dwell_seconds": -1}),
    ("zone", {"occupancy_limit": 1.5}), ("zone", {"anchor": "wrong"}),
    ("zone", {"zones": [{"name": "x", "points": [[0, 0], [0, 0], [0, 0]]}]}),
    ("zone", {"zones": [{"name": "x", "points": [[0, 0], [2, 0], [0, 1]]}]}),
    ("zone", {"zones": [ZONE, ZONE]}), ("sequence", {"steps": ["A"]}),
    ("sequence", {"steps": ["A", "A"]}), ("sequence", {"steps": ["A", "B"], "reset_on_complete": "yes"}),
    ("ppe", {"helmet_labels": []}), ("ppe", {"missing_hold_seconds": "2"}),
])
def test_invalid_parameters_are_rejected(kind, params):
    with pytest.raises(ValueError):
        RuleNode(kind, params)


def test_zone_irregular_source_time_and_per_person_dwell():
    node = RuleNode("zone", {"zones": [ZONE], "dwell_seconds": 1, "occupancy_limit": 1}, "z")
    result = step(node, 0, items=[track(1), track(2)])
    assert labels(result) == ["zone_enter", "zone_enter", "zone_occupancy"]
    assert result["events"]["items"][0]["signal"] == "zone_enter:check"
    assert result["tracks"]["overlays"][0]["count"] == 2
    assert labels(step(node, .3, items=[track(1), track(2)])) == []
    assert labels(step(node, .9, items=[track(1)])) == []
    result = step(node, 1.1, items=[track(1), track(2)])
    dwell = [x for x in result["events"]["items"] if x["kind"] == "zone_dwell"]
    assert len(dwell) == 1 and dwell[0]["track_id"] == 1
    assert dwell[0]["details"]["observed_seconds"] == pytest.approx(1.1)
    assert labels(step(node, 1.2, items=[])) == []  # Lost, not observed exit.
    result = step(node, 1.3, items=[track(1, box=(60, 10, 80, 80))])
    assert "zone_exit" not in labels(result)


def test_zone_explicit_exit_gap_and_anchor():
    node = RuleNode("zone", {"zones": [ZONE], "dwell_seconds": .5})
    step(node, 0, items=[track()])
    result = step(node, .3, items=[track(box=(60, 10, 80, 80))])
    assert labels(result) == ["zone_exit"]
    assert result["events"]["items"][0]["signal"] == "zone_exit:check"
    step(node, .5, items=[track()])
    assert labels(step(node, 10, items=[track()])) == ["zone_enter"]
    center = RuleNode("zone", {"zones": [ZONE], "anchor": "center"})
    assert labels(step(center, 0, items=[track()])) == []


@pytest.mark.parametrize("new_frame", [frame(0), frame(2, source="camera-b"), frame(2, session_id="new-session")])
def test_source_session_or_time_reversal_resets_temporal_state(new_frame):
    node = RuleNode("condition", {"labels": ["smoking"], "min_duration": 1})
    step(node, 1, "actions", [action("smoking")])
    assert labels(node.process({"actions": packet([action("smoking")])}, new_frame)) == []


def test_condition_unknown_interleaved_scene_and_person_are_isolated():
    node = RuleNode("condition", {"labels": ["smoking"], "min_duration": 1, "cooldown": 10})
    scene = {"scope": "scene", "label": "smoking", "score": .9}
    step(node, 0, "actions", [action("smoking", 1), scene])
    step(node, .5, "actions", [action("unknown", 1), action("smoking", 2), scene])
    result = step(node, 1, "actions", [action("smoking", 1), action("smoking", 2), scene])
    assert len(result["events"]["items"]) == 1
    assert result["events"]["items"][0]["scope"] == "scene"
    assert "track_id" not in result["events"]["items"][0]
    result = step(node, 1.5, "actions", [action("smoking", 1), action("smoking", 2), scene])
    assert [x.get("track_id") for x in result["events"]["items"]] == [2]
    result = step(node, 2, "actions", [action("smoking", 1), action("smoking", 2), scene])
    assert [x.get("track_id") for x in result["events"]["items"]] == [1]
    assert labels(step(node, 2.5, "actions", [action("smoking", 1), action("smoking", 2), scene])) == []


def test_condition_cached_action_and_missing_frames_do_not_accumulate():
    node = RuleNode("condition", {"min_duration": 1, "max_gap_seconds": 2})
    for t in (0, .5, 1, 1.5, 2):
        assert labels(step(node, t, "actions", [action("smoking", timestamp=0)])) == []
    step(node, 3, "actions", [action("smoking", timestamp=3)])
    step(node, 3.5, "actions", [])
    assert labels(step(node, 4, "actions", [action("smoking", timestamp=4)])) == []
    assert labels(step(node, 5, "actions", [action("smoking", timestamp=5)])) == ["smoking"]


def test_condition_cooldown_is_per_entity_and_requires_fresh_episode():
    node = RuleNode("condition", {"labels": ["lying"], "cooldown": 2})
    assert labels(step(node, 0, "actions", [action("lying", 1)])) == ["lying"]
    step(node, .2, "actions", [action("lying", 1, score=.1)])
    result = step(node, .4, "actions", [action("lying", 1), action("lying", 2)])
    assert [x["track_id"] for x in result["events"]["items"]] == [2]
    assert labels(step(node, 2.1, "actions", [action("lying", 1)])) == ["lying"]


def test_relation_conditions_preserve_pair_scope_and_do_not_merge_other_objects():
    node=RuleNode("condition",{"labels":["holding"],"min_duration":.5})
    first={"scope":"relation","label":"holding","predicate":"holding","score":.8,"relation_key":"person1-holding-object2",
           "track_key":"person1","subject_track_key":"person1","object_track_key":"object2","subject_index":0,"object_index":1}
    second={**first,"relation_key":"person1-holding-object3","object_track_key":"object3","object_index":2}
    step(node,0,"actions",[first])
    result=step(node,.5,"actions",[first,second])["events"]["items"]
    assert len(result)==1 and result[0]["scope"]=="relation"
    assert result[0]["relation_key"]==first["relation_key"] and result[0]["object_track_key"]=="object2"
    assert result[0]["subject_index"]==0 and result[0]["object_index"]==1 and result[0]["track_key"]=="person1"
    result=step(node,1,"actions",[first,second])["events"]["items"]
    assert len(result)==1 and result[0]["object_track_key"]=="object3"


def test_untracked_relation_list_indices_cannot_prove_duration():
    node=RuleNode("condition",{"labels":["holding"],"min_duration":.5})
    relation={"scope":"relation","label":"holding","score":.8,"subject_index":0,"object_index":1}
    for t in (0,.5,1,1.5):
        assert labels(step(node,t,"actions",[relation]))==[]


def skeleton(mode="standing"):
    points = [[50, 10, .9] for _ in range(17)]
    for i, xy in {5:(35, 25),6:(65,25),11:(40,50),12:(60,50),13:(40,70),14:(60,70),15:(40,95),16:(60,95),9:(30,60),10:(70,60)}.items():
        points[i] = [*xy, .9]
    if mode == "lying":
        for i, xy in {5:(20,40),6:(20,60),11:(65,40),12:(65,60)}.items():
            points[i] = [*xy, .9]
    if mode == "squatting":
        for i, xy in {13:(65,60),14:(85,60),15:(40,90),16:(60,90)}.items():
            points[i] = [*xy, .9]
    if mode == "hands":
        points[9], points[10] = [30,10,.9], [70,10,.9]
    return points


def test_posture_is_explicit_geometry_with_missing_visibility_unknown():
    node = RuleNode("posture", {})
    assert labels(step(node, 0, items=[track(keypoints=skeleton())]), "actions") == ["standing"]
    assert "raised_hands" in labels(step(node, .1, items=[track(keypoints=skeleton("hands"))]), "actions")
    assert labels(step(node, .2, items=[track(keypoints=skeleton("squatting"))]), "actions") == ["squatting"]
    points = skeleton(); points[5][2] = .1
    result = step(node, .3, items=[track(keypoints=points)])
    assert labels(result, "actions") == ["unknown"]
    assert result["actions"]["items"][0]["method"] == "geometry_rule"
    assert labels(step(node, .4, items=[track(keypoints=[[1,2,.9]]*33)]), "actions") == ["unknown"]


def test_fall_requires_observed_upright_transition_and_hold():
    node = RuleNode("posture", {"fall_hold_seconds": 1})
    step(node, 0, items=[track(keypoints=skeleton())])
    assert labels(step(node, .2, items=[track(keypoints=skeleton("lying"))]), "actions") == ["lying"]
    step(node, .7, items=[track(keypoints=skeleton("lying"))])
    assert "fall_candidate" in labels(step(node, 1.2, items=[track(keypoints=skeleton("lying"))]), "actions")
    node = RuleNode("posture", {"fall_hold_seconds": .5})
    for t in (0, .5, 1):
        assert labels(step(node, t, items=[track(keypoints=skeleton("lying"))]), "actions") == ["lying"]
    node = RuleNode("posture", {"fall_hold_seconds": .5})
    step(node, 0, items=[track(keypoints=skeleton())])
    step(node, .1, items=[track(keypoints=skeleton("lying"))])
    step(node, .2, items=[track()])
    for t in (.5, 1):
        assert "fall_candidate" not in labels(step(node, t, items=[track(keypoints=skeleton("lying"))]), "actions")


def test_sequence_signal_priority_repeated_labels_and_person_isolation():
    node = RuleNode("sequence", {"steps": ["zone_enter:check", "zone_enter:work", "zone_enter:exit"]})
    def zone(signal, i=1):
        return action("zone_enter", i, signal=f"zone_enter:{signal}")
    step(node, 0, "events", [zone("check",1), zone("check",2)])
    assert labels(step(node, .1, "events", [zone("check",1)])) == []
    step(node, .2, "events", [zone("work",1)])
    result = step(node, .3, "events", [zone("exit",1), zone("exit",2)])
    assert labels(result) == ["sequence_complete", "sequence_out_of_order"]
    assert [x["track_id"] for x in result["events"]["items"]] == [1, 2]


def test_sequence_timeout_unknown_scene_and_simultaneous_steps():
    node = RuleNode("sequence", {"steps": ["A", "B", "C"], "max_gap_seconds": 1})
    scene = lambda label: {"label": label, "scope": "scene"}
    assert labels(step(node, 0, "actions", [scene("B")])) == []
    step(node, .1, "actions", [scene("A"), action("A")])
    step(node, .5, "actions", [action("unknown")])
    result = step(node, 1.2, "actions", [])
    assert labels(result) == ["sequence_timeout"] and result["events"]["items"][0]["scope"] == "scene"
    step(node, 2, "events", [scene("A"), scene("B"), scene("C")])
    assert labels(step(node, 2.1, "events", [scene("C")])) == ["sequence_out_of_order"]


def test_sequence_complete_without_reset_stays_complete_while_observed():
    node = RuleNode("sequence", {"steps": ["A", "B"], "reset_on_complete": False})
    step(node, 0, "actions", [action("A")])
    assert labels(step(node, .1, "actions", [action("B")])) == ["sequence_complete"]
    assert labels(step(node, 100, "actions", [action("A"), action("B")])) == []


def test_live_histories_have_hard_caps_and_expire_without_observations():
    condition = RuleNode("condition", {"max_gap_seconds": 10, "cooldown": 20})
    sequence = RuleNode("sequence", {"steps": ["A", "B"], "max_gap_seconds": 10, "reset_on_complete": False})
    for node in (condition, sequence):
        node._MAX_ENTRIES = 32  # Exercise the real capacity policy with a short fixture.
    for i in range(300):
        t = i * .002
        step(condition, t, "actions", [action("A", i, timestamp=t)])
        step(sequence, t, "signals", [action("A", i, timestamp=t, event_id=f"start-{i}")])
        step(sequence, t + .001, "signals", [action("B", i, timestamp=t + .001, event_id=f"end-{i}")])
        for node in (condition, sequence):
            assert all(len(mapping) <= 32 for mapping in (node._states, node._sample_times, node._cooldowns, node._seen_events))
    for node in (condition, sequence):
        assert labels(step(node, 30, "signals", [])) == []
        assert not any((node._states, node._sample_times, node._cooldowns, node._seen_events))


def test_history_pressure_preserves_active_condition_support_and_cooldown():
    node = RuleNode("condition", {"min_duration": 3, "cooldown": 10, "max_gap_seconds": 1})
    node._MAX_ENTRIES = 16
    emitted = []
    for i in range(401):
        t = i / 100
        result = step(node, t, "actions", [action("A", 1, timestamp=t), action("A", i + 100, timestamp=t)])
        emitted.extend(result["events"]["items"])
        assert len(node._sample_times) <= 16
    assert len(emitted) == 1 and emitted[0]["track_id"] == 1
    assert emitted[0]["details"]["observed_seconds"] == pytest.approx(3)
    # A break resets support, but not an unexpired actor-specific cooldown.
    step(node, 4.01, "actions", [action("unknown", 1)])
    for t in (4.1, 5, 5.9, 6.8, 7.7):
        assert labels(step(node, t, "actions", [action("A", 1, timestamp=t)])) == []


def test_completed_sequence_active_identity_survives_churn_then_expires():
    node = RuleNode("sequence", {"steps": ["A", "B"], "reset_on_complete": False, "max_gap_seconds": 1})
    node._MAX_ENTRIES = 16
    step(node, 0, "signals", [action("A", 1)])
    assert labels(step(node, .1, "signals", [action("B", 1)])) == ["sequence_complete"]
    for i in range(100):
        t = .2 + i / 100
        result = step(node, t, "signals", [action("B", 1, timestamp=t), action("A", i + 100, timestamp=t)])
        assert "sequence_complete" not in labels(result)
        assert ("sequence", ("person", "person-1")) in node._states
    step(node, 3, "signals", [])
    assert ("sequence", ("person", "person-1")) not in node._states
    step(node, 3.1, "signals", [action("A", 1)])
    assert labels(step(node, 3.2, "signals", [action("B", 1)])) == ["sequence_complete"]


def test_expired_sample_dedup_does_not_make_stale_actions_fresh():
    node = RuleNode("condition", {"min_duration": 0, "cooldown": 0, "max_gap_seconds": 1})
    stale = action("A", timestamp=0)
    assert labels(step(node, 0, "actions", [stale])) == ["A"]
    step(node, 2, "actions", [])
    assert not node._sample_times
    assert labels(step(node, 3, "actions", [stale])) == []


def test_sequence_accepts_catalog_signals_port():
    node = RuleNode("sequence", {"steps": ["zone_enter:check", "zone_enter:work"]})
    step(node, 0, "signals", [action("zone_enter", signal="zone_enter:check")])
    assert labels(step(node, .2, "signals", [action("zone_enter", signal="zone_enter:work")])) == ["sequence_complete"]


def test_sequence_unknown_precedes_timeout_and_replayed_events_do_not_restart():
    node = RuleNode("sequence", {"steps": ["A", "B"], "max_gap_seconds": 1})
    step(node, 0, "actions", [action("A")])
    assert labels(step(node, 2, "actions", [action("unknown")])) == []
    first, last = action("A", event_id="first", timestamp=3), action("B", event_id="last", timestamp=3.5)
    step(node, 3, "events", [first])
    assert labels(step(node, 3.5, "events", [last])) == ["sequence_complete"]
    assert labels(step(node, 4, "events", [first])) == []
    assert labels(step(node, 4.5, "events", [action("B")])) == []


def test_time_reversal_does_not_reuse_event_ids():
    node = RuleNode("condition", {"cooldown": 0})
    first = step(node, 1, "actions", [action("A")])["events"]["items"][0]
    second = step(node, 0, "actions", [action("A")])["events"]["items"][0]
    assert first["event_id"] != second["event_id"]


def ppe_step(node, t, people, detections=None):
    return node.process({"tracks": packet(people,t), "detections": packet(detections or [],t)}, frame(t))


def test_ppe_head_occlusion_and_long_gap_are_unknown_not_missing():
    node = RuleNode("ppe", {"missing_hold_seconds": 1})
    for t in (0, .5, 1, 10):
        assert labels(ppe_step(node, t, [track()])) == ["ppe_unknown"]
    assert labels(ppe_step(node, 10.1, [track(keypoints=skeleton())])) == []
    assert labels(ppe_step(node, 10.6, [track(keypoints=skeleton())])) == []
    result = ppe_step(node, 11.1, [track(keypoints=skeleton())])
    assert labels(result) == ["missing_helmet"]
    assert result["events"]["items"][0]["details"]["observed_seconds"] == pytest.approx(1)


def test_ppe_one_helmet_cannot_cover_two_people_and_handheld_is_not_worn():
    node = RuleNode("ppe", {"missing_hold_seconds": .5})
    people = [track(1,(10,0,40,100),head_visible=True),track(2,(60,0,90,100),head_visible=True)]
    helmet = {"box":[15,0,35,20],"label":"helmet","score":.9}
    ppe_step(node,0,people,[helmet])
    result = ppe_step(node,.5,people,[helmet])
    assert labels(result) == ["missing_helmet"] and result["events"]["items"][0]["track_id"] == 2
    node = RuleNode("ppe", {"missing_hold_seconds": 0})
    handheld = dict(helmet,box=[15,60,35,80])
    assert labels(ppe_step(node,0,people[:1],[handheld])) == ["missing_helmet"]
    twins = [track(1,(10,0,40,100),head_visible=True),track(2,(10,0,40,100),head_visible=True)]
    assert labels(ppe_step(node,.1,twins,[helmet])) == ["ppe_unknown","ppe_unknown"]


def test_ppe_vest_requires_visible_torso_and_configurable_labels():
    node = RuleNode("ppe", {"require_helmet":False,"require_vest":True,"vest_labels":["hi_vis"],"missing_hold_seconds":0})
    assert labels(ppe_step(node,0,[track()])) == ["ppe_unknown"]
    vest = {"label":"hi_vis","box":[22,30,38,50],"score":.9}
    assert labels(ppe_step(node,.1,[track(torso_visible=True)],[vest])) == []
    assert labels(ppe_step(node,.2,[track(torso_visible=True)],[])) == ["missing_vest"]


def test_merge_keeps_scope_and_deduplicates_shared_events():
    a = {"kind":"zone_enter","label":"zone_enter","scope":"person","track_id":3,"event_id":"same","timestamp":0}
    b = {"kind":"condition","label":"smoking","scope":"scene","event_id":"scene","timestamp":0}
    node = RuleNode("merge", {})
    result = node.process({"events":[packet([a]),packet([a,b])]},frame(0))
    assert result["events"]["items"] == [a,b]
    assert result["events"]["items"][0] is not a


@pytest.mark.parametrize("value", [math.nan,math.inf,-1,"1",True])
def test_frame_timestamp_must_be_finite_source_seconds(value):
    with pytest.raises(ValueError):
        RuleNode("merge",{}).process({},frame(value) if isinstance(value,(int,float)) and math.isfinite(value) else dict(frame(0),timestamp=value))
