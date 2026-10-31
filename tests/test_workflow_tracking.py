import math

import pytest

import backend.workflow_tracking as tracking
from backend.workflow_tracking import MAX_ACTIVE_TRACKS, TrackerNode


def observation(box=(10, 10, 30, 50), label="person", score=.9, **extra):
    return {"box":list(box),"label":label,"score":score,**extra}


def run(node, t, items, source="cam", **extra):
    frame = {"timestamp":t,"frame_index":int(t*30) if isinstance(t,(int,float)) and math.isfinite(t) else 0,
             "width":1000,"height":1000,"source_id":source,**extra}
    return node.process({"observations":{"items":items,"timestamp":t,"frame":frame}},frame)["tracks"]


@pytest.mark.parametrize("params", [{"min_iou":-1},{"min_iou":1.1},{"min_iou":math.nan},
    {"min_iou":True},{"max_age_seconds":math.inf},{"max_age_seconds":-1},{"labels":"person"},
    {"labels":[""]},{"labels":[1]}])
def test_reject_invalid_parameters(params):
    with pytest.raises(ValueError):
        TrackerNode(params)


def test_same_class_hungarian_association_preserves_ids_despite_input_order():
    node = TrackerNode({})
    first = run(node,0,[observation(),observation((50,10,70,50)),observation(label="helmet")])
    assert first["tracking"]["method"] == "hungarian_iou_same_class"
    second = run(node,.3,[observation((51,10,71,50)),observation(label="helmet"),observation((11,10,31,50))])
    assert [x["track_id"] for x in second["items"]] == [2,3,1]
    assert all(x["age_seconds"]==pytest.approx(.3) for x in second["items"])
    changed_class = run(node,.5,[observation(label="car")])["items"][0]
    assert changed_class["track_id"] == 4


def test_global_assignment_finds_two_matches_where_best_first_greedy_fails(monkeypatch):
    node = TrackerNode({"min_iou":.1})
    run(node,0,[observation((0,0,10,10)),observation((20,0,30,10))])
    # Feasible matrix [[.9,.8],[.7,invalid]]: maximum-cardinality Hungarian
    # pairs old1->new2 and old2->new1 rather than taking .9 and losing old2.
    def fake_iou(old,new):
        return {(0,40):.9,(0,60):.8,(20,40):.7,(20,60):0}[old[0],new[0]]
    monkeypatch.setattr(tracking,"_iou",fake_iou)
    result=run(node,.1,[observation((40,0,50,10)),observation((60,0,70,10))])
    assert [x["track_id"] for x in result["items"]] == [2,1]


def test_dropout_no_ghosts_then_reappearance_and_expiry_on_source_time():
    node=TrackerNode({"max_age_seconds":1})
    original=run(node,0,[observation()])["items"][0]
    assert run(node,.2,[])["items"] == []
    returned=run(node,.8,[observation((11,10,31,50))])["items"][0]
    assert returned["track_key"]==original["track_key"]
    assert returned["age_seconds"]==pytest.approx(.8)
    expired=run(node,1.801,[observation((11,10,31,50))])
    assert expired["items"][0]["track_key"] != original["track_key"]
    assert expired["items"][0]["age_seconds"]==0
    assert expired["tracking"]["expired_tracks"]==1


@pytest.mark.parametrize("kwargs,t", [({"source":"other"},1),({"session_id":"reopened"},1),({},0)])
def test_source_session_and_backward_time_change_track_key(kwargs,t):
    node=TrackerNode({},node_id="person/track")
    old=run(node,.5,[observation()])["items"][0]
    new=run(node,t,[observation()],**kwargs)["items"][0]
    assert old["track_key"]!=new["track_key"]
    assert new["track_id"]==1
    assert "person%2Ftrack" in new["track_key"]


def test_labels_and_matched_pose_data_are_from_current_observation():
    node=TrackerNode({"labels":["person"]})
    points=[[20,20,.8] for _ in range(17)]
    original=observation(keypoints=points)
    first=run(node,0,[original,observation(label="dog")])
    assert first["tracking"]["label_filtered"]==1
    points2=[[22,21,.9] for _ in range(17)]
    second=run(node,.1,[observation((11,10,31,50),keypoints=points2)])["items"][0]
    assert second["track_key"]==first["items"][0]["track_key"]
    assert second["keypoints"]==points2 and second["keypoints"] is not points2
    second["keypoints"][0][0]=999
    assert points2[0][0]==22 and points[0][0]==20
    no_pose=run(node,.2,[observation()])["items"][0]
    assert "keypoints" not in no_pose  # Never carry forward stale joints.


def test_invalid_observations_are_counted_and_filtered():
    node=TrackerNode({})
    bad=[None,observation((10,10,9,20)),observation((math.nan,0,1,1)),observation(score=math.inf),
         observation(score=-.1),observation(score=True),observation(label=""),
         observation(keypoints=[[1,2,math.nan]]),observation(keypoints=[[1,2,2]]),observation(observed=False)]
    result=run(node,0,bad+[observation()])
    assert result["tracking"]["invalid_observations"]==len(bad)
    assert len(result["items"])==1 and result["items"][0]["observed"] is True


def test_capacity_prefers_high_scores_and_evicts_unmatched_old_state():
    node=TrackerNode({"max_age_seconds":100})
    many=[observation((i*2,0,i*2+1,1),score=.2 if i==0 else .9) for i in range(MAX_ACTIVE_TRACKS+1)]
    first=run(node,0,many)
    assert len(first["items"])==MAX_ACTIVE_TRACKS
    assert first["items"][0]["box"][0]==2
    assert first["tracking"]["capacity_filtered_observations"]==1
    assert first["tracking"]["active_tracks"]==MAX_ACTIVE_TRACKS
    second=run(node,.1,[observation((4000+i*2,0,4001+i*2,1)) for i in range(10)])
    assert second["tracking"]["capacity_evictions"]==10
    assert second["tracking"]["active_tracks"]==MAX_ACTIVE_TRACKS
    assert len(second["items"])==10


def test_long_run_state_is_bounded_and_cleanup_uses_elapsed_seconds():
    node=TrackerNode({"max_age_seconds":100})
    for index in range(600):
        result=run(node,index*.01,[observation((index*2,0,index*2+1,1))])
        assert result["tracking"]["active_tracks"] <= MAX_ACTIVE_TRACKS
    assert result["tracking"]["active_tracks"]==MAX_ACTIVE_TRACKS
    clean=run(node,106,[])
    assert clean["tracking"]["active_tracks"]==0 and clean["items"]==[]


def test_greedy_fallback_is_explicit_and_deterministic(monkeypatch):
    monkeypatch.setattr(tracking,"_solver",lambda:None)
    outputs=[]
    for _ in range(2):
        node=TrackerNode({})
        run(node,0,[observation(),observation((50,10,70,50))])
        result=run(node,.1,[observation((51,10,71,50)),observation((11,10,31,50))])
        assert result["tracking"]["method"]=="greedy_iou_same_class"
        outputs.append([x["track_id"] for x in result["items"]])
    assert outputs==[[2,1],[2,1]]


@pytest.mark.parametrize("value", [math.nan,math.inf,-1,True,"1"])
def test_invalid_frame_time_rejected(value):
    with pytest.raises(ValueError):
        run(TrackerNode({}),value,[])
