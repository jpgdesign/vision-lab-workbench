"""Exercise the persistent worker and temporal buffers without loading a GPU."""
import copy
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from backend import live_inference as live
from backend.live_limits import FRAME_RING_SIZE, MAX_CAPTURE_FPS


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def harness(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    monkeypatch.setattr(live, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(live, "synchronize", lambda device: None)
    monkeypatch.setattr(live, "draw_predictions", lambda pixels, predictions, title=None: Image.fromarray(pixels))
    output = tmp_path / "job"
    output.mkdir()
    inputs = output / "live_input"
    inputs.mkdir()
    snapshots = []
    write = live._write_json

    def record(path, value):
        if path.name == "live_state.json":
            snapshots.append(copy.deepcopy(value))
        write(path, value)

    monkeypatch.setattr(live, "_write_json", record)
    return output, inputs, clock, snapshots


def publish_frames(inputs, values):
    frames = []
    for sequence, timestamp, gray in values:
        filename = f"frame_{sequence:012d}.jpg"
        Image.new("RGB", (32, 24), (gray, gray, gray)).save(inputs / filename)
        frames.append({"sequence": sequence, "received_at": timestamp, "filename": filename, "width": 32, "height": 24})
    (inputs / "input.json").write_text(json.dumps({"sequence": frames[-1]["sequence"], "received_at": frames[-1]["received_at"], "frames": frames}))
    return frames


def request(models=None, **kwargs):
    return {"models": models or [{"key": "0", "model_id": "yolo26x", "params": {"device": "cpu"}}],
            "capture_fps": 8, "max_duration_seconds": 900, **kwargs}


def test_models_load_once_and_compare_same_frame_across_multiple_updates(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    publish_frames(inputs, [(1, clock.now, 20)])
    loaded, seen = [], {"0": [], "1": []}

    def load(spec, device):
        key = spec["key"]
        loaded.append(key)

        def predict(image):
            value = image.getpixel((0, 0))[0]
            seen[key].append(value)
            if key == "1":
                clock.now += 1
                if len(seen[key]) == 3:
                    (output / "stop.signal").write_text("stop")
                else:
                    sequence = len(seen[key]) + 1
                    publish_frames(inputs, [(sequence, clock.now, sequence * 20)])
            return [{"label": str(value), "score": 1.0}]

        return {"task": "detection", "predict": predict, "device": device}

    monkeypatch.setattr(live, "_load_runner", load)
    specs = [{"key": str(index), "model_id": "yolo26x", "params": {"device": "cpu"}} for index in range(2)]
    result = live.run_live(request(specs), output, lambda event: None)
    assert loaded == ["0", "1"]
    assert seen == {"0": [20, 40, 60], "1": [20, 40, 60]}
    assert [row["processed_frames"] for row in result["live"]["models"]] == [3, 3]
    assert result["live"]["frames_received"] == 3
    assert snapshots[-1]["status"] == "completed"
    assert all(Path(path).is_file() for path in result["artifacts"].values())
    assert result["live"]["raw_frames_retained"] is False
    assert not inputs.exists()


def test_action_warms_until_full_window_then_samples_timestamps(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    clock.now = 101
    publish_frames(inputs, [(1, 100, 20), (2, 101, 40)])
    original_manifest = live._manifest
    reads = [0]

    def manifest(folder):
        reads[0] += 1
        if reads[0] == 2:
            clock.now = 104
            publish_frames(inputs, [(1, 100, 20), (2, 101, 40), (3, 103, 60), (4, 104, 80)])
        return original_manifest(folder)

    monkeypatch.setattr(live, "_manifest", manifest)
    monkeypatch.setattr(live, "_load_runner", lambda spec, device: {"task": "action", "model": object(), "device": device,
                                                                   "metadata": {"num_frames": 4}})
    observed = []

    def predict(model, metadata, pixels, params, device):
        observed.append(pixels[:, 0, 0, 0].tolist())
        (output / "stop.signal").write_text("stop")
        return [{"label": "fixture action", "score": .9}]

    monkeypatch.setattr(live, "_action_predict", predict)
    specs = [{"key": "0", "model_id": "x3d_l", "params": {"device": "cpu", "clip_seconds": 4, "top_k": 1}}]
    result = live.run_live(request(specs), output, lambda event: None)
    assert observed == [[20, 40, 60, 80]]
    assert any(s["models"][0].get("available_seconds") == 1 for s in snapshots)
    row = result["live"]["models"][0]
    assert row["sequence"] == 4 and row["window_seconds"] == 4
    assert row["sampled_frames"] == row["unique_frames"] == 4
    assert not inputs.exists()


def test_waiting_camera_times_out_and_always_cleans_input(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    monkeypatch.setattr(live, "_load_runner", lambda spec, device: {"task": "detection", "device": device,
                                                                   "predict": lambda image: pytest.fail("No frame should be inferred")})
    result = live.run_live(request(), output, lambda event: None)
    assert "20 秒" in result["summary"]
    assert result["live"]["frames_received"] == 0
    assert not inputs.exists()
    assert snapshots[-1]["status"] == "completed"


def test_action_uses_captured_window_after_slow_detector_retires_source_files(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    clock.now = 104
    publish_frames(inputs, [(1, 100, 20), (2, 101, 40), (3, 103, 60), (4, 104, 80)])
    observed = []

    def detector(image):
        # Simulate the API retiring this entire manifest during a slow first
        # model. Later comparison models still need the same temporal input.
        clock.now += 18
        for path in inputs.glob("*.jpg"):
            path.unlink()
        return []

    def load(spec, device):
        if spec["model_id"] == "yolo26x":
            return {"task": "detection", "device": device, "predict": detector}
        return {"task": "action", "device": device, "model": object(), "metadata": {"num_frames": 4}}

    def action(model, metadata, pixels, params, device):
        observed.append(pixels[:, 0, 0, 0].tolist())
        (output / "stop.signal").write_text("stop")
        return [{"label": "old coherent window", "score": .8}]

    monkeypatch.setattr(live, "_load_runner", load)
    monkeypatch.setattr(live, "_action_predict", action)
    specs = [{"key": "0", "model_id": "yolo26x", "params": {"device": "cpu"}},
             {"key": "1", "model_id": "x3d_l", "params": {"device": "cpu", "clip_seconds": 4, "top_k": 1}}]
    result = live.run_live(request(specs), output, lambda event: None)
    assert observed == [[20, 40, 60, 80]]
    assert [row["sequence"] for row in result["live"]["models"]] == [4, 4]
    assert not inputs.exists()


def test_duration_limit_prevents_loading_next_model_after_slow_load(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    loaded = []

    def load(spec, device):
        loaded.append(spec["key"])
        clock.now += 31
        return {"task": "detection", "device": device, "predict": lambda image: []}

    monkeypatch.setattr(live, "_load_runner", load)
    specs = [{"key": str(index), "model_id": "yolo26x", "params": {"device": "cpu"}} for index in range(2)]
    result = live.run_live(request(specs, max_duration_seconds=30), output, lambda event: None)
    assert loaded == ["0"]
    assert "時間上限" in result["summary"]
    assert not inputs.exists()


def test_custom_duration_limit_stops_even_with_camera_heartbeat(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    frame = publish_frames(inputs, [(1, clock.now, 20)])[0]
    predictions = []
    monkeypatch.setattr(live, "_manifest", lambda directory: {"frames": [{**frame, "received_at": clock.now}]})
    monkeypatch.setattr(live, "_load_runner", lambda spec, device: {"task": "detection", "device": device,
                                                                   "predict": lambda image: predictions.append(True) or []})
    result = live.run_live(request(max_duration_seconds=7), output, lambda event: None)
    assert "時間上限" in result["summary"]
    assert 7 <= result["live"]["elapsed_seconds"] <= 7.1
    assert predictions == [True]
    assert not inputs.exists()


@pytest.mark.parametrize("duration", [None, 7200, 10**400], ids=["unlimited", "two-hours", "large-integer"])
def test_unlimited_or_long_duration_passes_old_limits_and_manual_stop_cleans_frames(harness, monkeypatch, duration):
    output, inputs, clock, snapshots = harness
    frame = publish_frames(inputs, [(1, clock.now, 20)])[0]
    loaded, predicted = [], []
    monkeypatch.setattr(live, "_manifest", lambda directory: {"frames": [{**frame, "received_at": clock.now}]})

    def load(spec, device):
        loaded.append(spec["key"])
        clock.now += 1000

        def predict(image):
            predicted.append(spec["key"])
            if spec["key"] == "0":
                clock.now += 4000
            else:
                (output / "stop.signal").write_text("stop")
            return []

        return {"task": "detection", "device": device, "predict": predict}

    monkeypatch.setattr(live, "_load_runner", load)
    specs = [{"key": str(index), "model_id": "yolo26x", "params": {"device": "cpu"}} for index in range(2)]
    result = live.run_live(request(specs, max_duration_seconds=duration), output, lambda event: None)
    assert loaded == predicted == ["0", "1"]
    assert result["live"]["elapsed_seconds"] == 6000
    assert "使用者停止" in result["summary"]
    assert not inputs.exists()


def test_unlimited_duration_still_disconnects_after_missing_camera_heartbeat(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    frame = publish_frames(inputs, [(1, clock.now, 20)])[0]
    reads, predicted = [0], []

    def manifest(directory):
        reads[0] += 1
        return {"frames": [{**frame, "received_at": clock.now}]} if reads[0] == 1 else None

    def load(spec, device):
        clock.now += 4000
        return {"task": "detection", "device": device, "predict": lambda image: predicted.append(True) or []}

    monkeypatch.setattr(live, "_manifest", manifest)
    monkeypatch.setattr(live, "_load_runner", load)
    result = live.run_live(request(max_duration_seconds=None), output, lambda event: None)
    assert predicted == [True]
    assert "20 秒" in result["summary"]
    assert 4020 <= result["live"]["elapsed_seconds"] <= 4020.1
    assert not inputs.exists()


def test_omitted_duration_keeps_900_second_default(harness, monkeypatch):
    output, inputs, clock, snapshots = harness

    def load(spec, device):
        clock.now += 901
        return {"task": "detection", "device": device, "predict": lambda image: pytest.fail("Default duration already elapsed")}

    monkeypatch.setattr(live, "_load_runner", load)
    payload = request()
    payload.pop("max_duration_seconds")
    result = live.run_live(payload, output, lambda event: None)
    assert "時間上限" in result["summary"]
    assert not inputs.exists()


def test_stop_before_load_does_not_load_models(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    (output / "stop.signal").write_text("stop")
    monkeypatch.setattr(live, "_load_runner", lambda spec, device: pytest.fail("Stopped session must not load weights"))
    result = live.run_live(request(), output, lambda event: None)
    assert "使用者停止" in result["summary"]
    assert not inputs.exists()


def test_model_failure_publishes_failure_and_discards_raw_frames(harness, monkeypatch):
    output, inputs, clock, snapshots = harness
    publish_frames(inputs, [(1, clock.now, 20)])

    def predict(image):
        raise RuntimeError("test model failed")

    monkeypatch.setattr(live, "_load_runner", lambda spec, device: {"task": "detection", "device": device, "predict": predict})
    with pytest.raises(RuntimeError, match="test model failed"):
        live.run_live(request(), output, lambda event: None)
    assert snapshots[-1]["status"] == snapshots[-1]["models"][0]["status"] == "failed"
    assert not inputs.exists()


def test_clip_selection_requires_history_and_preserves_time_spacing():
    frames = [{"sequence": index + 1, "received_at": timestamp} for index, timestamp in enumerate([0, .3, 1, 2.7, 3.0, 4])]
    assert live._select_clip(frames[-2:], 4, 4) is None
    selected = live._select_clip(frames, 4, 4)
    assert [frame["received_at"] for frame in selected] == [0, 1, 2.7, 4]


@pytest.mark.parametrize("bad", [
    [{"filename": "../escape.jpg", "sequence": 1, "received_at": 1}],
    [{"filename": "one.jpg", "sequence": 2, "received_at": 1}, {"filename": "two.jpg", "sequence": 1, "received_at": 2}],
    [{"filename": "one.jpg", "sequence": 1, "received_at": float("nan")}],
])
def test_manifest_rejects_escape_invalid_order_and_nonfinite_time(tmp_path, bad):
    (tmp_path / "input.json").write_text(json.dumps({"frames": bad}))
    with pytest.raises(ValueError):
        live._manifest(tmp_path)


def test_manifest_windows_read_sharing_violation_is_transient(tmp_path, monkeypatch):
    frames = publish_frames(tmp_path, [(1, 100, 20)])
    real_read = Path.read_text
    failures = [1]

    def shared_read(path, *args, **kwargs):
        if path == tmp_path / "input.json" and failures[0]:
            failures[0] -= 1
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", shared_read)
    assert live._manifest(tmp_path) is None
    assert live._manifest(tmp_path)["frames"] == frames


def test_30_fps_ring_accepts_full_16_second_action_window_and_rejects_overflow(tmp_path):
    frames = [{"sequence": index + 1, "received_at": 100 + index / MAX_CAPTURE_FPS,
               "filename": f"frame_{index + 1:012d}.jpg", "width": 640, "height": 480}
              for index in range(FRAME_RING_SIZE)]
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"frames": frames}))
    loaded = live._manifest(tmp_path)["frames"]
    selected = live._select_clip(loaded, 16, 32)
    assert FRAME_RING_SIZE >= 16 * MAX_CAPTURE_FPS + 1
    assert len(loaded) == 512 and len(selected) == 32
    assert selected[-1]["sequence"] == 512
    assert selected[-1]["received_at"] - selected[0]["received_at"] == pytest.approx(16, abs=1 / MAX_CAPTURE_FPS)
    frames.append({"sequence": FRAME_RING_SIZE + 1, "received_at": 118,
                   "filename": "overflow.jpg", "width": 640, "height": 480})
    path.write_text(json.dumps({"frames": frames}))
    with pytest.raises(ValueError, match="緩衝區"):
        live._manifest(tmp_path)


@pytest.mark.parametrize("recovers", [True, False])
def test_unreadable_manifest_after_long_run_keeps_last_camera_timestamp(harness, monkeypatch, recovers):
    output, inputs, clock, snapshots = harness
    publish_frames(inputs, [(1, 100, 20)])
    real_manifest = live._manifest
    reads, seen = [0], []

    def manifest(directory):
        reads[0] += 1
        if reads[0] >= 3:
            if reads[0] == 3 or not recovers:
                return None
            publish_frames(inputs, [(3, clock.now, 60)])
        return real_manifest(directory)

    def predict(image):
        seen.append(image.getpixel((0, 0))[0])
        if len(seen) == 1:
            clock.now = 115
            publish_frames(inputs, [(2, clock.now, 40)])
        elif len(seen) == 2:
            # Session is 21 seconds old, but the latest received frame is
            # only six seconds old when Windows briefly denies the next read.
            clock.now = 121
        else:
            (output / "stop.signal").write_text("stop")
        return []

    monkeypatch.setattr(live, "_manifest", manifest)
    monkeypatch.setattr(live, "_load_runner", lambda spec, device: {"task": "detection", "device": device, "predict": predict})
    result = live.run_live(request(), output, lambda event: None)
    if recovers:
        assert seen == [20, 40, 60]
        assert "使用者停止" in result["summary"]
    else:
        assert seen == [20, 40]
        assert "20 秒" in result["summary"]
        # Repeated unreadable manifests must not count as new camera input.
        assert 35 <= result["live"]["elapsed_seconds"] <= 35.1
    assert not inputs.exists()


def test_manifest_does_not_touch_unselected_image_files_and_selected_image_is_checked(tmp_path, monkeypatch):
    frames = [{"sequence": index + 1, "received_at": 100 + index / 30,
               "filename": f"frame_{index + 1:012d}.jpg"} for index in range(512)]
    (tmp_path / "input.json").write_text(json.dumps({"frames": frames}))
    selected_path = tmp_path / frames[-1]["filename"]
    Image.new("RGB", (16, 12), "blue").save(selected_path)
    lstat = Path.lstat
    touched = []

    def inspect(path, *args, **kwargs):
        if path.suffix.lower() == ".jpg":
            touched.append(path)
            assert path == selected_path, "Unused ring records must not trigger filesystem lookups"
        return lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", inspect)
    manifest = live._manifest(tmp_path)
    assert len(manifest["frames"]) == 512
    assert touched == []
    assert live._read_image(tmp_path, manifest["frames"][-1]).size == (16, 12)
    assert selected_path in touched


@pytest.mark.parametrize("filename", ["../outside.jpg", "..\\outside.jpg", "/absolute.jpg", "C:outside.jpg", "\\server\\outside.jpg", "nul\x00.jpg"])
def test_selected_image_rejects_traversal_without_opening_file(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Unsafe image must not be opened"))
    with pytest.raises(ValueError, match="路徑"):
        live._read_image(tmp_path, {"filename": filename})


@pytest.mark.parametrize("linked", ["directory", "manifest", "image", "junction"])
def test_manifest_and_selected_image_reject_links_before_open(tmp_path, monkeypatch, linked):
    directory = tmp_path / "input"
    directory.mkdir()
    frames = publish_frames(directory, [(1, 100, 20)])
    image_path = directory / frames[0]["filename"]
    target = directory if linked in {"directory", "junction"} else directory / "input.json" if linked == "manifest" else image_path
    lstat = Path.lstat

    def inspect(path, *args, **kwargs):
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFDIR if linked == "junction" else stat.S_IFLNK,
                                   st_file_attributes=0x400 if linked == "junction" else 0)
        return lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", inspect)
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Linked image must not be opened"))
    if linked != "image":
        with pytest.raises(ValueError, match="連結"):
            live._manifest(directory)
    if linked != "manifest":
        with pytest.raises(ValueError, match="連結"):
            live._read_image(directory, frames[0])


def test_selected_image_checks_resolved_containment(tmp_path, monkeypatch):
    directory = tmp_path / "input"
    directory.mkdir()
    frames = publish_frames(directory, [(1, 100, 20)])
    target = directory / frames[0]["filename"]
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == target:
            return tmp_path / "outside.jpg"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Escaped image must not be opened"))
    with pytest.raises(ValueError, match="路徑"):
        live._read_image(directory, frames[0])


def test_action_clip_resolves_root_once_and_decodes_unique_frames_only(tmp_path, monkeypatch):
    frames = publish_frames(tmp_path, [(1, 100, 20), (2, 101, 40)])
    original_resolve, original_open = Path.resolve, Image.open
    roots, images = [], []

    def resolve(path, *args, **kwargs):
        if path == tmp_path:
            roots.append(path)
        return original_resolve(path, *args, **kwargs)

    def image_open(path, *args, **kwargs):
        images.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Image, "open", image_open)
    pixels = live._clip_pixels(tmp_path, [frames[0], frames[0], frames[1], frames[1]])
    assert pixels[:, 0, 0, 0].tolist() == [20, 20, 40, 40]
    assert len(roots) == 1 and len(images) == 2
