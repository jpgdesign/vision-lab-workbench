import importlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.jobs import JobManager, write_json_atomic
from backend.live import FRAME_RING_SIZE


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    module = importlib.import_module("backend.app")
    models = [
        {"id": "yolo26x", "available": True, "inference_supported": True, "capabilities": {"input_types": ["image", "video"]}},
        {"id": "action", "available": True, "inference_supported": True, "capabilities": {"input_types": ["video"]}},
        {"id": "image_only", "available": True, "inference_supported": True, "capabilities": {"input_types": ["image"]}},
        {"id": "missing", "available": False, "inference_supported": True},
        {"id": "rf_detr_2xl", "available": True, "inference_supported": True, "capabilities": {"input_types": ["image", "video"]}},
    ]
    monkeypatch.setattr(module, "get_models", lambda: models)

    def validate(model_id, kind, params):
        assert kind == "inference"
        if set(params) - {"conf", "accept_platform_license"}:
            raise ValueError("未知參數")
        if model_id == "rf_detr_2xl" and params.get("accept_platform_license") is not True:
            raise ValueError("必須確認授權")
        return {"conf": 0.25, **params}

    monkeypatch.setattr(module, "validate_job_request", validate)
    weight = tmp_path / "yolo26x.pt"
    weight.write_bytes(b"isolated test weight")
    monkeypatch.setattr(module.ModelRegistry, "resolve", lambda self, model_id, **kwargs: weight)
    application = module.create_app(tmp_path / "runtime", tmp_path / "samples", start_worker=False)
    with TestClient(application, base_url="http://127.0.0.1") as client:
        yield client, application


def jpeg(size=(80, 60), format="JPEG"):
    stream = io.BytesIO()
    Image.new("RGB", size, "blue").save(stream, format=format)
    return stream.getvalue()


def start(client, **kwargs):
    response = client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x"}], **kwargs})
    assert response.status_code == 201, response.text
    return response.json()


def frame(client, session_id, content=None):
    return client.post(f"/api/live/sessions/{session_id}/frames", files={"frame": ("camera.jpg", content if content is not None else jpeg(), "image/jpeg")})


def test_live_validation_licenses_and_atomic_gpu_reservation(api):
    client, app = api
    for payload, expected in [
        ({"models": []}, 422),
        ({"models": [{"model_id": "yolo26x"}] * 5}, 422),
        ({"models": [{"model_id": "missing"}]}, 409),
        ({"models": [{"model_id": "unknown"}]}, 404),
        ({"models": [{"model_id": "image_only"}]}, 422),
        ({"models": [{"model_id": "rf_detr_2xl"}]}, 422),
        ({"models": [{"model_id": "yolo26x", "params": {"invalid": 1}}]}, 422),
        ({"models": [{"model_id": "yolo26x", "checkpoint_path": "C:/other.pt"}]}, 422),
        ({"models": [{"model_id": "yolo26x"}], "capture_fps": 31}, 422),
        ({"models": [{"model_id": "yolo26x"}], "max_duration_seconds": 0}, 422),
    ]:
        assert client.post("/api/live/sessions", json=payload).status_code == expected
    job = start(client, models=[{"model_id": "yolo26x"}, {"model_id": "action"}])
    assert [m["key"] for m in job["models"]] == ["0", "1"]
    assert job["capture_fps"] == 30 and job["kind"] == "live"
    assert job["max_duration_seconds"] == 900
    assert client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x"}]}).status_code == 409
    # Ordinary tasks remain serially queued until this persistent worker ends.
    ordinary = app.state.jobs.submit({"kind": "inference", "model_id": "yolo26x", "params": {}})
    assert ordinary["status"] == "queued"
    client.post(f"/api/live/sessions/{job['id']}/stop")
    assert client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x"}]}).status_code == 409


@pytest.mark.parametrize("duration", [None, 1, 7, 3601, 86_400])
def test_live_duration_accepts_unlimited_and_custom_positive_seconds(api, duration):
    client, app = api
    job = start(client, max_duration_seconds=duration)
    assert job["max_duration_seconds"] == duration
    request = json.loads((app.state.jobs.job_root / job["id"] / "request.json").read_text())
    assert request["max_duration_seconds"] == duration


@pytest.mark.parametrize("duration", [0, -1, True, False, 1.5, "900"])
def test_live_duration_rejects_nonpositive_and_noninteger_values(api, duration):
    client, _ = api
    response = client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x"}], "max_duration_seconds": duration})
    assert response.status_code == 422


def test_frames_bounded_rate_limited_and_cleanup_on_cancel(api, monkeypatch):
    client, app = api
    module = importlib.import_module("backend.live")
    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    job = start(client, capture_fps=30)
    session_id = job["id"]
    for _ in range(3):
        assert frame(client, session_id).status_code == 202
    assert frame(client, session_id).status_code == 429
    folder = app.state.jobs.job_root / session_id / "live_input"
    for _ in range(FRAME_RING_SIZE + 5):
        clock[0] += 1 / 30 + 0.000001
        assert frame(client, session_id).status_code == 202
    manifest = json.loads((folder / "input.json").read_text())
    assert len(manifest["frames"]) == FRAME_RING_SIZE
    assert len(list(folder.glob("*.jpg"))) == FRAME_RING_SIZE
    assert manifest["sequence"] == FRAME_RING_SIZE + 8
    assert all(Path(item["filename"]).name == item["filename"] for item in manifest["frames"])
    assert manifest["frames"][-1]["received_at"] == clock[0]
    assert client.post(f"/api/jobs/{session_id}/cancel").json()["status"] == "cancelled"
    assert not folder.exists()
    assert frame(client, session_id).status_code == 409
    assert not folder.exists()


def test_manifest_replacement_retries_windows_sharing_violation(tmp_path, monkeypatch):
    module = importlib.import_module("backend.live")
    original_replace = module.os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError(13, "sharing violation")
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", replace)
    manifest = {"sequence": 1, "frames": [{"sequence": 1, "filename": "frame_1.jpg"}]}
    path = tmp_path / "input.json"
    module._write_manifest(path, manifest)
    assert json.loads(path.read_text()) == manifest
    assert len(attempts) == 3
    assert not path.with_suffix(".json.tmp").exists()


def test_frames_must_be_small_valid_jpeg_and_routes_stay_local(api):
    client, app = api
    job = start(client)
    session_id = job["id"]
    assert frame(client, session_id, b"x" * (1024 * 1024 + 1)).status_code == 413
    for bad in (b"not a JPEG", jpeg(format="PNG"), jpeg((1601, 8)), jpeg((1100, 1000))):
        assert frame(client, session_id, bad).status_code == 422
    assert frame(client, "missing").status_code == 404
    for headers in ({"Origin": "https://external.example"}, {"Host": "192.168.1.5"}, {"Sec-Fetch-Site": "cross-site"}):
        assert client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x"}]}, headers=headers).status_code == 403
    assert not (app.state.jobs.job_root / session_id / "live_input").exists()


def test_stop_is_graceful_and_published_images_are_whitelisted(api):
    client, app = api
    job = start(client)
    session_id = job["id"]
    manager = app.state.jobs
    directory = manager.job_root / session_id
    manager.jobs[session_id]["status"] = "running"
    (directory / "result_0.jpg").write_bytes(jpeg())
    write_json_atomic(directory / "live_state.json", {"status": "running", "models": [
        {"key": "0", "sequence": 9, "image_filename": "../../private.jpg", "image_url": "https://bad.example"},
        {"key": "4", "image_url": "https://bad.example"},
    ]})
    state = client.get(f"/api/live/sessions/{session_id}").json()["live_state"]
    url = state["models"][0]["image_url"]
    assert url == f"/api/live/sessions/{session_id}/images/0?v=9"
    assert "image_url" not in state["models"][1]
    response = client.get(url)
    assert response.content == jpeg() and response.headers["cache-control"].startswith("no-store")
    assert client.get(f"/api/live/sessions/{session_id}/images/1").status_code == 404
    assert client.get(f"/api/live/sessions/{session_id}/images/-1").status_code == 404
    assert client.get(f"/api/live/sessions/{session_id}/images/request.json").status_code == 422
    assert client.post(f"/api/live/sessions/{session_id}/stop").json()["status"] == "running"
    assert (directory / "stop.signal").is_file()
    assert frame(client, session_id).status_code == 409
    assert client.get(f"/api/artifacts/{session_id}/live_input/input.json").status_code == 404
    manager.cancel(session_id)
    cancelled = client.get(f"/api/live/sessions/{session_id}").json()
    assert cancelled["status"] == cancelled["live_state"]["status"] == "cancelled"
    assert all(row["status"] == "cancelled" for row in cancelled["live_state"]["models"])


def test_live_checkpoint_resolves_known_training_job_and_hides_path(api):
    client, app = api
    manager = app.state.jobs
    trained = manager.submit({"kind": "training", "model_id": "yolo26x", "params": {}})
    checkpoint = manager.job_root / trained["id"] / "best.pt"
    checkpoint.write_bytes(b"local trusted checkpoint")
    payload = {"models": [{"model_id": "yolo26x", "checkpoint_job_id": trained["id"]}]}
    assert client.post("/api/live/sessions", json=payload).status_code == 422
    manager.jobs[trained["id"]]["status"] = "completed"
    job = start(client, **payload)
    assert "checkpoint_path" not in job["models"][0]
    request = json.loads((manager.job_root / job["id"] / "request.json").read_text())
    assert request["models"][0]["checkpoint_path"] == str(checkpoint.resolve())


def test_live_worker_holds_queue_until_stop_then_cleans_frames(tmp_path):
    worker = tmp_path / "dummy_worker.py"
    worker.write_text("""import json, pathlib, sys, time
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
output = pathlib.Path(sys.argv[2])
if request['kind'] == 'live':
    while not (output / 'stop.signal').exists():
        time.sleep(.02)
(output / 'result.json').write_text(json.dumps({'status': 'completed', 'result': {'summary': request['kind']}}))
""", encoding="utf-8")
    manager = JobManager(tmp_path / "runtime", tmp_path, command_factory=lambda request, output: [sys.executable, str(worker), str(request), str(output)])
    live = manager.submit({"kind": "live", "model_id": "yolo26x", "params": {}, "models": [{"key": "0", "model_id": "yolo26x"}], "capture_fps": 8, "max_duration_seconds": 30})
    following = manager.submit({"kind": "inference", "model_id": "yolo26x", "params": {}})
    directory = manager.job_root / live["id"]
    (directory / "live_input").mkdir()
    (directory / "live_input" / "frame.jpg").write_bytes(jpeg())
    manager.start()
    try:
        deadline = time.monotonic() + 10
        while manager.get(live["id"])["status"] != "running" and time.monotonic() < deadline:
            time.sleep(.02)
        assert manager.get(live["id"])["status"] == "running"
        assert manager.get(following["id"])["status"] == "queued"
        (directory / "stop.signal").write_text("stop")
        while manager.get(following["id"])["status"] != "completed" and time.monotonic() < deadline:
            time.sleep(.02)
        assert manager.get(live["id"])["status"] == "completed"
        assert manager.get(following["id"])["status"] == "completed"
        assert not (directory / "live_input").exists()
    finally:
        manager.close()


def test_restart_discards_interrupted_raw_camera_input(tmp_path):
    root = tmp_path / "runtime"
    manager = JobManager(root, tmp_path)
    job = manager.submit({"kind": "live", "model_id": "yolo26x", "params": {}, "models": [{"key": "0", "model_id": "yolo26x"}], "capture_fps": 8, "max_duration_seconds": 30})
    folder = manager.job_root / job["id"] / "live_input"
    folder.mkdir()
    (folder / "private.jpg").write_bytes(jpeg())
    restarted = JobManager(root, tmp_path)
    assert restarted.get(job["id"])["status"] == "failed"
    assert not folder.exists()
