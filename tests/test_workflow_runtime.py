"""Exercise real routing/rules and media loops using deterministic model adapters."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from backend import workflow_runtime as runtime
from backend.workflow_templates import templates


def template(name="rgb-action-rules"):
    return copy.deepcopy(next(item["graph"] for item in templates()["templates"] if item["id"] == name))


def chosen(graph, kind):
    return next(item for item in graph["nodes"] if item["type"] == kind)


def request(graph=None, **extra):
    return {"id": "source-a", "workflow_graph": graph or template(), "params": {"device": "cpu"}, **extra}


def action(label="smoking", **extra):
    return {"label": label, "score": .95, "scope": "scene", **extra}


def person(x=10, **extra):
    return {"label": "person", "score": .9, "box": [x, 10, x + 15, 80], **extra}


class FakeModel:
    def __init__(self, kind, callback):
        self.kind, self.callback, self.reset_count = kind, callback, 0

    def process(self, inputs, frame):
        return self.callback(self.kind, inputs, frame)

    def reset(self):
        self.reset_count += 1


def make_engine(monkeypatch, graph, callback, **extra):
    engine = runtime.WorkflowEngine(request(graph, **extra), model_factory=lambda kind, params, resolved, **kw: FakeModel(kind, callback))
    engine.device = "cpu"  # The adapter is fake; never resolve/load a real model.
    monkeypatch.setattr(runtime, "render_output", lambda frame, inputs, recent: frame["image"].copy())
    return engine


def process(engine, stamp, index=0, size=(100, 100)):
    return engine.process(Image.new("RGB", size, "white"), stamp, index)


def test_condition_uses_irregular_source_time_not_frame_count_or_worker_time(monkeypatch):
    graph = template()
    chosen(graph, "condition")["params"].update(min_duration=1., cooldown=0.)
    seen = []
    def infer(kind, inputs, frame):
        seen.append((frame["timestamp"], frame["frame_index"], inputs["frame"]["timestamp"]))
        return {"actions": runtime.packet([action()], frame)}
    engine = make_engine(monkeypatch, graph, infer)
    assert process(engine, 40., 0)[1] == []
    assert process(engine, 40.2, 300)[1] == []
    events = process(engine, 41., 301)[1]
    assert len(events) == 1
    assert events[0]["timestamp"] == 41.
    assert events[0]["details"]["observed_seconds"] == pytest.approx(1.)
    assert seen == [(40., 0, 40.), (40.2, 300, 40.2), (41., 301, 41.)]


def test_throttled_predictions_never_trigger_until_a_fresh_sample_and_large_gaps_reset(monkeypatch):
    graph = template()
    chosen(graph, "condition")["params"].update(min_duration=.5, cooldown=0.)
    def infer(kind, inputs, frame):
        return {"actions": runtime.packet([action()], frame, fresh=frame["frame_index"] != 1)}
    engine = make_engine(monkeypatch, graph, infer)
    assert process(engine, 0, 0)[1] == []
    assert process(engine, .5, 1)[1] == []
    assert engine.stats["rule"]["updates"] == 1
    assert len(process(engine, .75, 2)[1]) == 1
    engine = make_engine(monkeypatch, graph, infer)
    assert process(engine, 0, 0)[1] == []
    assert process(engine, .5, 1)[1] == []
    assert process(engine, 3, 2)[1] == []
    assert len(process(engine, 3.5, 3)[1]) == 1


def test_unknown_breaks_temporal_support_and_scene_events_do_not_claim_a_person(monkeypatch):
    graph = template()
    chosen(graph, "condition")["params"].update(min_duration=.5, cooldown=0.)
    def infer(kind, inputs, frame):
        return {"actions": runtime.packet([action("unknown" if frame["frame_index"] == 1 else "smoking")], frame)}
    engine = make_engine(monkeypatch, graph, infer)
    for stamp, index in [(0, 0), (.3, 1), (.6, 2)]:
        assert process(engine, stamp, index)[1] == []
    event = process(engine, 1.1, 3)[1][0]
    assert event["scope"] == "scene" and event.get("track_id") is None
    assert event["details"]["start_timestamp"] == .6


def test_person_histories_survive_detection_order_change_without_combining_people(monkeypatch):
    graph = template("zone-guard")
    zone = chosen(graph, "zone")
    zone["params"].update(zones=[{"name": "all", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}], dwell_seconds=.5)
    def infer(kind, inputs, frame):
        people = [person(10), person(60)] if frame["frame_index"] == 0 else [person(60)]
        return {"detections": runtime.packet(people, frame)}
    engine = make_engine(monkeypatch, graph, infer)
    entered = process(engine, 0, 0)[1]
    assert len(entered) == 2 and len({e["track_id"] for e in entered}) == 2
    _, events, observations = process(engine, .5, 1)
    dwell = [e for e in events if e["kind"] == "zone_dwell"]
    assert len(dwell) == 1
    current = observations["track"]["tracks"]["items"][0]
    assert current["box"][0] == 60
    assert dwell[0]["track_id"] == current["track_id"] == entered[1]["track_id"]


@pytest.mark.parametrize("change", ["source", "rewind", "resize"])
def test_source_rewind_or_dimensions_clear_prior_evidence_and_reset_models(monkeypatch, change):
    graph = template()
    chosen(graph, "condition")["params"].update(min_duration=.5, cooldown=0.)
    engine = make_engine(monkeypatch, graph, lambda kind, inputs, frame: {"actions": runtime.packet([action()], frame)})
    assert process(engine, 10, 0)[1] == []
    assert process(engine, 10.5, 1)[1]
    if change == "source": engine.request["id"] = "source-b"
    stamp = 0 if change == "rewind" else 10.6
    _, events, _ = process(engine, stamp, 2, (110, 100) if change == "resize" else (100, 100))
    assert events == [] and not engine.recent_events
    assert engine.instances["action"].reset_count == 1


def test_output_events_follow_wires_instead_of_collecting_every_intermediate_event(monkeypatch):
    graph = template("zone-guard")
    graph["nodes"].append({"id": "filter", "type": "condition", "params": {"labels": ["never-match"]}})
    original = next(e for e in graph["edges"] if e["source"] == "zone" and e["sourceHandle"] == "events")
    original.update(target="filter", targetHandle="actions")
    graph["edges"].append({"id": "filtered", "source": "filter", "sourceHandle": "events", "target": "out", "targetHandle": "events"})
    engine = make_engine(monkeypatch, graph, lambda kind, inputs, frame: {"detections": runtime.packet([person(50)], frame)})
    _, events, observations = process(engine, 0)
    assert observations["zone"]["events"]["items"]  # The upstream rule truly fired.
    assert events == [] and engine.event_count == 0 and not engine.recent_events


def test_public_observations_strip_pixels_masks_credentials_recursively_without_mutating_model_output(monkeypatch):
    graph = template("zone-guard")
    original = person(50, mask=np.ones((100, 100)), mask_area=10000, details={"key_file_path": "private", "frame": object(), "score": float("nan")})
    engine = make_engine(monkeypatch, graph, lambda kind, inputs, frame: {"detections": runtime.packet([original], frame)})
    _, _, observations = process(engine, 0)
    serialized = json.dumps(observations, allow_nan=False)
    assert '"mask"' not in serialized and '"frame"' not in serialized and "private" not in serialized
    item = observations["detect"]["detections"]["items"][0]
    assert item["mask_area"] == 10000 and item["details"]["score"] is None
    assert original["mask"].shape == (100, 100) and original["details"]["key_file_path"] == "private"


def install_fake_models(monkeypatch, callback=None):
    real_engine, engines = runtime.WorkflowEngine, []
    callback = callback or (lambda kind, inputs, frame: {"actions": runtime.packet([action()], frame)})
    def factory(req, emit):
        engine = real_engine(req, emit, model_factory=lambda kind, params, resolved, **kw: FakeModel(kind, callback))
        engine.device = "cpu"
        engines.append(engine)
        return engine
    monkeypatch.setattr(runtime, "WorkflowEngine", factory)
    return engines


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_image_runner_writes_real_overlay_and_portable_artifacts(tmp_path, monkeypatch):
    install_fake_models(monkeypatch)
    source = tmp_path / "input.jpg"
    Image.new("RGB", (160, 120), "white").save(source)
    output = tmp_path / "output"
    result = runtime.run_workflow(request(media_path=str(source)), output, lambda event: None)
    assert result["media_kind"] == "image" and result["frames"] == 1 and result["event_count"] == 1
    rendered = np.asarray(Image.open(result["media_path"]))
    assert rendered.shape == (120, 160, 3) and rendered.min() < 100  # Actual overlay drew on white source.
    rows = json_lines(result["artifacts"]["frames_jsonl"])
    assert rows[0]["timestamp"] == 0 and rows[0]["frame_index"] == 0
    assert json_lines(result["artifacts"]["events_jsonl"])[0]["label"] == "smoking"
    assert all(Path(path).is_file() for path in result["artifacts"].values())


@pytest.mark.parametrize("fps,stride,pts,indices,expected", [
    (10., 2, [0.] * 6, [0, 2, 4], [0., .2, .4]),
    # Real AVI behavior: an initial PTS offset, then missing PTS near the end.
    # Fallback must continue from normalized source time, not jump backwards.
    (30., 1, [200 / 3, 100., 0., 0.], [0, 1, 2, 3], [0., 1 / 30, 2 / 30, 3 / 30]),
])
def test_video_runner_uses_source_fps_fallback_stride_and_closes_capture_writer(tmp_path, monkeypatch, fps, stride, pts, indices, expected):
    import cv2
    from backend import inference
    install_fake_models(monkeypatch)
    class Capture:
        index = 0
        released = False
        def isOpened(self): return True
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: len(pts), cv2.CAP_PROP_POS_MSEC: pts[max(0, self.index - 1)]}.get(prop, 0.)
        def read(self):
            self.index += 1
            return (True, np.full((120, 160, 3), 255, np.uint8)) if self.index <= len(pts) else (False, None)
        def release(self): self.released = True
    class Writer:
        released = False
        frames = []
        def isOpened(self): return True
        def write(self, frame): self.frames.append(frame.copy())
        def release(self): self.released = True
    capture, writer, writer_args = Capture(), Writer(), []
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: (writer_args.append(args), writer)[1])
    monkeypatch.setattr(inference, "h264_video", lambda raw, target: Path(target).write_bytes(b"encoded-by-test-double"))
    (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 12)
    result = runtime.run_workflow(request(media_path=str(tmp_path / "clip.mp4"), params={"device": "cpu", "frame_stride": stride, "max_frames": len(indices)}), tmp_path / "output", lambda event: None)
    rows = json_lines(result["artifacts"]["frames_jsonl"])
    assert [row["frame_index"] for row in rows] == indices
    assert [row["timestamp"] for row in rows] == pytest.approx(expected)
    assert writer_args[0][2] == fps / stride and len(writer.frames) == len(indices)
    assert result["media_kind"] == "video" and result["source_fps"] == fps
    assert capture.released and writer.released


@pytest.mark.parametrize("fail", [False, True])
def test_live_runner_uses_ingress_time_stops_and_cleans_only_its_raw_frames(tmp_path, monkeypatch, fail):
    from backend import live_inference
    def infer(kind, inputs, frame):
        if fail: raise RuntimeError("test model failure")
        return {"actions": runtime.packet([action()], frame)}
    install_fake_models(monkeypatch, infer)
    output = tmp_path / "live"
    inputs = output / "live_input"
    inputs.mkdir(parents=True)
    sibling = tmp_path / "unrelated.jpg"
    sibling.write_bytes(b"retain")
    for sequence in (1, 2, 3): Image.new("RGB", (160, 120), "white").save(inputs / f"frame{sequence}.jpg")
    frames = [{"sequence": i, "filename": f"frame{i}.jpg", "received_at": t} for i, t in [(1, 100.), (2, 100.2), (3, 100.7)]]
    calls = []
    def manifest(directory):
        calls.append(directory)
        assert len(calls) <= 2, "Live test did not honor the stop signal"
        return {"frames": frames[:1] if len(calls) == 1 else frames}
    original_save = live_inference._save_image
    def save(image, target):
        original_save(image, target)
        if len(calls) == 2: (output / "stop.signal").touch()
    monkeypatch.setattr(live_inference, "_manifest", manifest)
    monkeypatch.setattr(live_inference, "_save_image", save)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 101.)
    req = request(kind="live", max_duration_seconds=None)
    emitted = []
    if fail:
        with pytest.raises(RuntimeError, match="test model failure"):
            runtime.run_workflow_live(req, output, emitted.append)
    else:
        result = runtime.run_workflow_live(req, output, emitted.append)
        rows = json_lines(result["artifacts"]["frames_jsonl"])
        assert [row["timestamp"] for row in rows] == pytest.approx([0., .7])
        assert [row["frame_index"] for row in rows] == [1, 3]
        assert result["frames"] == 2 and result["live"]["frames_received"] == 3
        assert result["live"]["raw_frames_retained"] is False
        assert Image.open(result["media_path"]).size == (160, 120)
        assert result["node_stats"] and all(item["calls"] == 2 for item in result["node_stats"])
    assert not any(item.get("event") == "node" for item in emitted)
    state = json.loads((output / "live_state.json").read_text(encoding="utf-8"))
    assert state["status"] == ("failed" if fail else "completed")
    assert not inputs.exists() and sibling.read_bytes() == b"retain"
    # Handles must be closed even after a model exception (also exercises Windows semantics).
    (output / "workflow_events.jsonl").rename(output / "closed_events.jsonl")


@pytest.mark.parametrize("limited", ["frames", "events", "bytes"])
def test_export_caps_do_not_stop_inference_or_recent_events(tmp_path, monkeypatch, limited):
    monkeypatch.setattr(runtime, "MAX_FRAME_RECORDS", 2 if limited == "frames" else 100)
    monkeypatch.setattr(runtime, "MAX_EVENT_RECORDS", 2 if limited == "events" else 100)
    monkeypatch.setattr(runtime, "MAX_EXPORT_BYTES", 300 if limited == "bytes" else 1_000_000)
    graph = template()
    chosen(graph, "condition")["params"].update(cooldown=0.)
    def infer(kind, inputs, frame):
        return {"actions": runtime.packet([action("unknown" if frame["frame_index"] % 2 else "smoking")], frame)}
    engine = make_engine(monkeypatch, graph, infer)
    artifacts = runtime.Artifacts(tmp_path, engine)
    try:
        for index in range(7):
            _, events, observations = process(engine, index * .5, index)
            artifacts.append(index * .5, index, events, observations)
        result = artifacts.result()
    finally:
        artifacts.close()
    assert result["frames"] == 7 and engine.stats["action"]["calls"] == 7
    assert result["event_count"] == 4 and result["events"][-1]["timestamp"] == 3.
    assert result["data_export"]["truncated"] is True
    assert "容量上限" in result["summary"]
    frame_rows = json_lines(result["artifacts"]["frames_jsonl"])
    event_rows = json_lines(result["artifacts"]["events_jsonl"])
    assert len(frame_rows) == result["data_export"]["frame_records_written"] <= runtime.MAX_FRAME_RECORDS
    assert len(event_rows) == result["data_export"]["event_records_written"] <= runtime.MAX_EVENT_RECORDS
    if limited == "frames":
        assert len(frame_rows) == 2 and len(event_rows) == 4
    elif limited == "events":
        assert len(frame_rows) == 7 and len(event_rows) == 2
    else:
        # CSV includes its small fixed header; records share the bounded budget.
        data_bytes = sum(Path(result["artifacts"][key]).stat().st_size for key in ("frames_jsonl", "events_jsonl", "events_csv"))
        assert data_bytes <= runtime.MAX_EXPORT_BYTES + 200


def test_remote_rewind_discards_old_display_without_replenishing_api_budget(monkeypatch):
    from backend import workflow_integrations
    calls = []
    class Client:
        def __init__(self, *args): pass
        def predict(self, image, instruction, events):
            calls.append(instruction)
            return [action("original-scene")]
    monkeypatch.setattr(workflow_integrations, "VisionClient", Client)
    graph = {"version": 1, "nodes": [{"id": "source", "type": "source"}, {"id": "remote", "type": "vlm", "params": {"model_id": "fake", "max_calls": 1}}, {"id": "out", "type": "output"}], "edges": [
        {"id": "source-out", "source": "source", "sourceHandle": "frame", "target": "out", "targetHandle": "frame"},
        {"id": "source-remote", "source": "source", "sourceHandle": "frame", "target": "remote", "targetHandle": "frame"},
        {"id": "remote-out", "source": "remote", "sourceHandle": "actions", "target": "out", "targetHandle": "actions"},
    ]}
    engine = make_engine(monkeypatch, graph, lambda *args: None, runtime_path="unused-by-fake-client")
    assert process(engine, 10)[2]["remote"]["actions"]["items"][0]["label"] == "original-scene"
    assert process(engine, 10.5)[2]["remote"]["actions"]["fresh"] is False
    after_rewind = process(engine, 0)[2]["remote"]["actions"]
    assert after_rewind["fresh"] is True and after_rewind["items"][0]["label"] == "unknown"
    assert len(calls) == 1 and engine.instances["remote"].calls == 1
    assert all(item["label"] != "original-scene" for item in engine.instances["remote"].display)
