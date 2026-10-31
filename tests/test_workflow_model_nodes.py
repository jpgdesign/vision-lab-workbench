import numpy as np
import pytest
from PIL import Image

from backend.workflow_model_nodes import ModelNode, MAX_VIDEO_FRAMES, MAX_VIDEO_SIDE


def frame(t=0., *, source="clip", size=(100, 80), image=None):
    return dict(image=image or Image.new("RGB", size, (10, 20, 30)), timestamp=float(t),
                frame_index=round(t*30), width=size[0], height=size[1], source_id=source)


def track(key="a", x=10, *, points=True):
    result = dict(label="person", score=.9, track_key=key, track_id=key, box=[x, 5, x+25, 70], observed=True)
    if points:
        result["keypoints"] = [[x, 20, .9] for _ in range(17)]
    return result


def inputs(t, tracks):
    return {"tracks": {"items": tracks, "timestamp": float(t), "fresh": True}}


class Skeleton:
    def __init__(self, score=.9):
        self.calls, self.score = [], score

    def predict(self, sequence, image_size):
        self.calls.append(sequence.copy())
        return [{"label": "standing up", "score": self.score, "class_id": 8},
                {"label": "sitting down", "score": .05, "class_id": 7}]


def skeleton(every=1., score=.9):
    node = ModelNode("skeleton_action", dict(window_seconds=2., min_frames=30, every_seconds=every, min_score=.5), {})
    node.model = Skeleton(score)
    return node


def test_all_current_people_share_global_cadence_but_not_pose_buffers():
    node = skeleton()
    for index in range(31):
        t = index/30
        packet = node.process(inputs(t, [track("a", 10), track("b", 60)]), frame(t))["actions"]
        if index == 10:
            assert packet["items"] == [] and packet["fresh"] is False
    assert len(node.model.calls) == 2
    assert {row["track_key"] for row in packet["items"]} == {"a", "b"}
    assert all(row["label"] == "standing up" and len(row["candidates"]) == 2 for row in packet["items"])
    assert np.all(node.model.calls[0][..., 0] == 10)
    assert np.all(node.model.calls[1][..., 0] == 60)
    cached = node.process(inputs(1.01, [track("a"), track("b", 60)]), frame(1.01))["actions"]
    assert cached["items"] == [] and cached["fresh"] is False
    assert all(row["stale"] and row["evidence_timestamp"] == 1 for row in cached["display_items"])


def test_missing_keypoints_and_lost_tracks_reset_evidence():
    node = skeleton(every=0)
    for index in range(30):
        t = index/30
        node.process(inputs(t, [track()]), frame(t))
    assert len(node.model.calls) == 1
    bad = node.process(inputs(1., [track(points=False)]), frame(1.))["actions"]["items"]
    assert bad[0]["label"] == "unknown" and bad[0]["reason"] == "missing_keypoints"
    assert "a" not in node.buffers
    node.process(inputs(1.1, []), frame(1.1))
    fresh = node.process(inputs(1.2, [track()]), frame(1.2))["actions"]["items"]
    assert fresh[0]["reason"] == "warming" and fresh[0]["unique_frames"] == 1
    assert len(node.model.calls) == 1


def test_non_due_loss_invalidates_evidence_immediately():
    node = skeleton(every=1)
    for index in range(31):
        t = index/30
        node.process(inputs(t, [track()]), frame(t))
    assert len(node.model.calls) == 1
    missing = node.process(inputs(1.1, []), frame(1.1))["actions"]
    assert missing["fresh"] is True and missing["items"] == [] and missing["display_items"] == []
    assert node.buffers == {}
    node.process(inputs(1.2, [track()]), frame(1.2))
    assert len(node.buffers["a"]) == 1 and len(node.model.calls) == 1


def test_non_due_missing_keypoints_emit_unknown():
    node = skeleton(every=1)
    node.process(inputs(0, [track()]), frame(0))
    packet = node.process(inputs(.1, [track(points=False)]), frame(.1))["actions"]
    assert packet["fresh"] is True and packet["items"][0]["label"] == "unknown"


def test_non_due_long_gap_emits_unknown_and_resets_timer():
    node = skeleton(every=10)
    node.process(inputs(0, [track()]), frame(0))
    packet = node.process(inputs(2, [track()]), frame(2))["actions"]
    assert packet["fresh"] is True and packet["items"][0]["reason"] == "warming_after_gap"


def test_duplicate_frames_are_not_new_training_evidence_and_rewind_resets():
    node = skeleton(every=0)
    node.process(inputs(0., [track()]), frame(0.))
    node.process(inputs(.1, [track()]), frame(.1))
    repeated = node.process(inputs(.1, [track()]), frame(.1))["actions"]
    assert repeated["fresh"] is False and len(node.buffers["a"]) == 2
    node.process(inputs(0., [track()]), frame(0.))
    assert len(node.buffers["a"]) == 1
    node.process(inputs(.1, [track()]), frame(.1, source="other"))
    assert len(node.buffers["a"]) == 1


def test_low_confidence_top1_is_unknown_with_candidates_only():
    node = skeleton(every=0, score=.2)
    for index in range(30):
        t = index/30
        result = node.process(inputs(t, [track()]), frame(t))["actions"]["items"]
    assert len(result) == 1 and result[0]["label"] == "unknown" and result[0]["score"] == 0
    assert result[0]["reason"] == "below_confidence_threshold"
    assert result[0]["candidates"][0]["label"] == "standing up"


def test_duplicate_track_keys_are_rejected():
    node = skeleton()
    with pytest.raises(ValueError, match="duplicate"):
        node.process(inputs(0, [track(), track()]), frame())


def test_stale_track_packet_never_accumulates_poses():
    node = skeleton()
    stale = inputs(0, [track()])
    stale["tracks"]["fresh"] = False
    assert node.process(stale, frame())["actions"]["items"] == []
    assert node.buffers == {}
    node.process(inputs(0, [track()]), frame(1))
    assert node.buffers == {}


def video_node(scope="scene", every=.25):
    node = ModelNode("video_action", dict(scope=scope, window_seconds=1., every_seconds=every, min_score=.5), {})
    node.model = object()
    node.metadata = {"num_frames": 16}
    node.seen = []
    def candidates(history, t):
        node.seen.append([(time, np.array(image)) for time, image in history])
        return [{"class_id": 1, "label": "running", "score": .8}], len(history)
    node._video_candidates = candidates
    return node


def test_video_waits_for_complete_source_time_window():
    node = video_node()
    for index in range(11):
        t = index/10
        packet = node.process({}, frame(t))["actions"]
        if t < 1:
            assert not node.seen
    # Due schedule is global (.0,.3,.6,.9,1.2 on this frame grid).
    packet = node.process({}, frame(1.2))["actions"]
    assert len(node.seen) == 1
    assert packet["items"][0]["scope"] == "scene"
    assert "track_key" not in packet["items"][0]
    assert packet["items"][0]["window_start"] == pytest.approx(.2)
    assert packet["items"][0]["sampled_frames"] == 16


def test_sparse_or_single_image_video_does_not_fabricate_clip():
    node = video_node(every=0)
    for t in [0, .1, .2, 1.2, 1.3]:
        node.process({}, frame(t))
    assert node.seen == []
    assert len(node.buffers["scene"]) == 2  # Gap resets prior motion evidence.


def test_person_action_uses_actual_per_track_crops_and_lost_reset():
    node = video_node(scope="person", every=0)
    image = Image.new("RGB", (100, 80), "red")
    image.paste("blue", (50, 0, 100, 80))
    for index in range(11):
        t = index/10
        packet = node.process(inputs(t, [track("left", 5), track("right", 60)]), frame(t, image=image))["actions"]
    assert len(node.seen) == 2
    assert np.all(node.seen[0][0][1] == [255, 0, 0])
    assert np.all(node.seen[1][0][1] == [0, 0, 255])
    assert {row["track_key"] for row in packet["items"]} == {"left", "right"}
    node.process(inputs(1.1, [track("right", 60)]), frame(1.1, image=image))
    assert "left" not in node.buffers
    packet = node.process(inputs(1.2, [track("left", 5)]), frame(1.2, image=image))["actions"]
    assert packet["items"][0]["reason"] == "warming"


def test_temporal_video_memory_is_bounded_at_high_source_fps():
    node = video_node(every=120)
    large = Image.new("RGB", (1280, 720))
    for index in range(1000):
        node.process({}, frame(index/240, size=large.size, image=large))
    assert len(node.buffers["scene"]) <= MAX_VIDEO_FRAMES
    assert all(max(image.size) <= MAX_VIDEO_SIDE for _, image in node.buffers["scene"])


def test_invalid_person_box_is_unknown_and_not_full_scene():
    node = video_node(scope="person", every=0)
    item = track()
    item["box"] = [200, 200, 220, 250]
    packet = node.process(inputs(0, [item]), frame())["actions"]
    assert packet["items"][0]["reason"] == "invalid_person_crop"
    assert not node.buffers


def test_relation_lazy_load_cadence_and_rgb_to_bgr():
    node = ModelNode("relation", dict(every_seconds=1., min_score=.3, predicates=["holding"]), {})
    calls = []
    class Relations:
        last_diagnostics = {"selected_count": 2}
        def predict(self, pixels, objects, **kwargs):
            calls.append((pixels.copy(), kwargs))
            return [dict(scope="relation", label="holding", predicate="holding", score=.8, track_key="a")]
    node._load = lambda: setattr(node, "model", Relations())
    assert node.process({"objects": {"items": []}}, frame())["relations"]["items"] == []
    assert node.model is None
    packet = node.process({"objects": {"items": [track(), track("b")]}}, frame(1))["relations"]
    assert len(calls) == 1 and np.all(calls[0][0][0, 0] == [30, 20, 10])
    assert packet["items"][0]["evidence_timestamp"] == 1
    assert calls[0][1]["threshold"] == .3
    next_packet = node.process({"objects": {"items": [track(), track("b")]}}, frame(1.1))["relations"]
    assert next_packet["items"] == [] and next_packet["fresh"] is False
    assert len(calls) == 1


def test_non_due_relation_disappearance_clears_cached_positive():
    node = ModelNode("relation", dict(every_seconds=1.), {})
    class Relations:
        last_diagnostics = {}
        def predict(self, *args, **kwargs):
            return [dict(scope="relation", label="holding", score=.8,
                         subject_track_key="a", object_track_key="b")]
    node.model = Relations()
    node.process({"objects": {"items": [track("a"), track("b")]}}, frame(0))
    # There are still two objects, but the previous relation's endpoint is gone.
    packet = node.process({"objects": {"items": [track("a"), track("c")]}}, frame(.1))["relations"]
    assert packet["fresh"] is True and packet["items"] == [] and packet["display_items"] == []
    packet = node.process({"objects": {"items": [track("a")]}}, frame(.2))["relations"]
    assert packet["fresh"] is True and packet["items"] == []


def test_detector_factory_created_once_and_checkpoint_forwarded(monkeypatch):
    from backend import inference
    calls = []
    monkeypatch.setattr(inference, "resolve_device", lambda device: "cpu")
    def factory(*args):
        calls.append(args)
        return lambda image, params: [dict(label="bowl", score=.9, box=[0, 0, 10, 10])]
    monkeypatch.setattr(inference, "create_detector", factory)
    resolved = dict(model_id="yolo26x", params={"conf": .3}, checkpoint_path="selected/model.pt")
    node = ModelNode("detector", {}, resolved)
    for t in [0, 1, 2]:
        assert node.process({}, frame(t))["detections"]["items"][0]["label"] == "bowl"
    assert len(calls) == 1 and calls[0][3] == "selected/model.pt"


def test_actual_video_tensor_path_uses_existing_logits_contract(monkeypatch):
    import torch
    from backend import inference
    seen = []
    node = ModelNode("video_action", {}, {})
    node.metadata = dict(num_frames=4, architecture="pytorchvideo", outputs_probabilities=False,
                         labels={0: "first", 1: "second"})
    node.model = lambda tensor: torch.tensor([[0., 2.]])
    monkeypatch.setattr(inference, "prepare_action_tensor", lambda pixels, meta, device: seen.append(pixels.copy()))
    history = [(0., Image.new("RGB", (10, 10), "red")), (2., Image.new("RGB", (10, 10), "blue"))]
    candidates, unique = node._video_candidates(history, 2.)
    assert candidates[0]["label"] == "second" and candidates[0]["score"] == pytest.approx(.880797, abs=1e-6)
    assert seen[0].shape == (4, 10, 10, 3) and unique == 2


def test_frame_size_mismatch_rejected():
    node = skeleton()
    value = frame()
    value["width"] = 20
    with pytest.raises(ValueError, match="match"):
        node.process(inputs(0, []), value)
