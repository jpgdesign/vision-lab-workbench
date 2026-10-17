import importlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.jobs import JobManager, write_json_atomic


MODELS = [
    {"id": "yolo26x", "available": True, "inference_supported": True, "training_supported": True, "capabilities": {"input_types": ["image", "video"], "training": {"dataset_formats": ["yolo_detection"]}}},
    {"id": "detector", "available": True, "inference_supported": True, "training_supported": True, "capabilities": {"input_types": ["image", "video"], "training": {"dataset_formats": ["yolo_detection"]}}},
    {"id": "action", "available": True, "inference_supported": True, "training_supported": False, "capabilities": {"input_types": ["video"]}},
    {"id": "missing", "available": False, "reason": "權重尚未下載", "inference_supported": True},
]


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "get_models", lambda: MODELS)
    def validate(model_id, kind, params):
        if set(params) - {"conf"}:
            raise ValueError("未知參數")
        return {"conf": params.get("conf", 0.25)}
    monkeypatch.setattr(module, "validate_job_request", validate)
    weight = tmp_path / "yolo26x.pt"
    weight.write_bytes(b"isolated test weight")
    monkeypatch.setattr(module.ModelRegistry, "resolve", lambda self, model_id, **kwargs: weight)
    application = module.create_app(tmp_path / "runtime", tmp_path / "samples", start_worker=False)
    with TestClient(application, base_url="http://127.0.0.1") as client:
        yield client, application


def png():
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(stream, format="PNG")
    return stream.getvalue()


def upload_image(client):
    result = client.post("/api/media", files={"file": ("input.png", png(), "image/png")})
    assert result.status_code == 201, result.text
    return result.json()


def test_default_samples_prefer_bundled_media_and_show_ten_images_one_video(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    module = importlib.import_module("backend.app")
    app_root = tmp_path / "clone"
    bundled = app_root / "examples/media"
    (bundled / "images").mkdir(parents=True)
    (bundled / "video").mkdir()
    for index in range(10):
        Image.new("RGB", (16, 12), "blue").save(bundled / "images" / f"coco-{index:02d}.jpg")
    video = bundled / "video" / "teaching.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 24)
    # Generated overview images and metadata are not inputs.
    Image.new("RGB", (16, 12)).save(bundled / "contact_sheet.jpg")
    legacy = tmp_path / "vision_training_starter"
    (legacy / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(legacy / "images" / "legacy.jpg")
    monkeypatch.setattr(module, "APP_ROOT", app_root)
    monkeypatch.setattr(module, "DEFAULT_SAMPLE_ROOT", legacy)
    application = module.create_app(tmp_path / "runtime", start_worker=False)
    with TestClient(application, base_url="http://127.0.0.1") as client:
        items = client.get("/api/media").json()["media"]
        assert len(items) == 11
        assert sum(item["kind"] == "image" for item in items) == 10
        assert sum(item["kind"] == "video" for item in items) == 1
        assert all(item["source"] == "sample" for item in items)
        assert "legacy.jpg" not in {item["name"] for item in items}
        listed_video = next(item for item in items if item["kind"] == "video")
        assert client.get(listed_video["url"]).content == video.read_bytes()
        assert not list((application.state.runtime / "media").iterdir())
        assert client.get("/api/jobs").json()["jobs"] == []
    assert video.exists() and (legacy / "images/legacy.jpg").exists()


def test_default_samples_fall_back_to_legacy_and_explicit_root_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "APP_ROOT", tmp_path / "clone")
    legacy = tmp_path / "vision_training_starter"
    custom = tmp_path / "custom"
    for root, name in ((legacy, "legacy.jpg"), (custom, "custom.jpg")):
        (root / "images").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(root / "images" / name)
    monkeypatch.setattr(module, "DEFAULT_SAMPLE_ROOT", legacy)
    for runtime, source, expected in (("legacy-runtime", None, "legacy.jpg"), ("custom-runtime", custom, "custom.jpg")):
        application = module.create_app(tmp_path / runtime, source, start_worker=False)
        with TestClient(application, base_url="http://127.0.0.1") as client:
            assert [item["name"] for item in client.get("/api/media").json()["media"]] == [expected]


def test_bundled_dataset_template_requires_explicit_validation_and_never_trains(tmp_path, monkeypatch):
    from test_datasets import yolo
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    module = importlib.import_module("backend.app")
    app_root = tmp_path / "clone"
    template_dir = app_root / "examples/finetuning/coco10_yolo"
    config = yolo(template_dir)
    config.rename(template_dir / "dataset.yaml")
    before = {str(path.relative_to(app_root)): path.read_bytes() for path in app_root.rglob("*") if path.is_file()}
    monkeypatch.setattr(module, "APP_ROOT", app_root)
    application = module.create_app(tmp_path / "runtime", tmp_path / "no-samples", start_worker=False)
    with TestClient(application, base_url="http://127.0.0.1") as client:
        template = client.get("/api/examples").json()["datasets"][0]
        assert template["id"] == "coco10_yolo" and template["format"] == "yolo_detection"
        assert template["available"] is True and Path(template["path"]).is_absolute()
        assert client.get("/api/datasets").json()["datasets"] == []
        result = client.post("/api/datasets/validate", json={key: template[key] for key in ("name", "path", "format")})
        assert result.status_code == 200, result.text
        assert result.json()["valid"] is True
        assert result.json()["dataset"]["name"] == template["name"]
        assert result.json()["dataset"]["train_count"] == 1
        assert result.json()["dataset"]["val_count"] == 1
        assert len(client.get("/api/datasets").json()["datasets"]) == 1
        assert client.get("/api/jobs").json()["jobs"] == []
    after = {str(path.relative_to(app_root)): path.read_bytes() for path in app_root.rglob("*") if path.is_file()}
    assert before == after


def test_missing_bundled_dataset_is_not_advertised_as_available(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "APP_ROOT", tmp_path / "empty-clone")
    application = module.create_app(tmp_path / "runtime", tmp_path / "no-samples", start_worker=False)
    with TestClient(application, base_url="http://127.0.0.1") as client:
        template = client.get("/api/examples").json()["datasets"][0]
        assert template["available"] is False
        assert client.get("/api/datasets").json()["datasets"] == []


def test_upload_registered_media_and_traversal(api):
    client, app = api
    media = upload_image(client)
    assert "path" not in media
    assert media["width"] == 16
    assert client.get(media["url"]).content == png()
    assert client.get("/api/media").json()["media"][0]["id"] == media["id"]
    assert client.get("/api/media/not-registered/content").status_code == 404
    assert client.get("/api/media/%2e%2e%2fsecrets/content").status_code == 404
    wrong = client.post("/api/media", files={"file": ("fake.png", b"<script>alert(1)</script>", "image/png")})
    assert wrong.status_code == 422
    assert len(list((app.state.runtime / "media").iterdir())) == 1


def test_upload_size_is_bounded(api, monkeypatch):
    client, app = api
    module = importlib.import_module("backend.app")
    monkeypatch.setattr(module, "MAX_MEDIA_BYTES", 10)
    response = client.post("/api/media", files={"file": ("large.png", png(), "image/png")})
    assert response.status_code == 413
    assert not list((app.state.runtime / "media").iterdir())


def test_streaming_body_limit_without_content_length(api):
    client, _ = api
    response = client.post("/api/jobs", content=iter([b" " * (1100 * 1024)]), headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text


def test_api_rejects_cross_site_driving_local_gpu(api):
    client, _ = api
    for headers in ({"Host": "evil.test"}, {"Origin": "https://evil.test"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
        assert client.get("/api/models", headers=headers).status_code == 403
    assert client.get("/api/models", headers={"Origin": "http://localhost:5173"}).status_code == 200


def test_job_validates_model_kind_params_and_registered_input(api):
    client, _ = api
    media = upload_image(client)
    base = {"kind": "inference", "model_id": "detector", "media_id": media["id"]}
    response = client.post("/api/jobs", json={**base, "params": {"conf": 0.4}})
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["status"] == "queued" and job["params"]["conf"] == 0.4
    assert client.post("/api/jobs", json={**base, "model_id": "unknown"}).status_code == 404
    assert client.post("/api/jobs", json={**base, "model_id": "missing"}).status_code == 409
    assert client.post("/api/jobs", json={**base, "model_id": "action"}).status_code == 422
    assert client.post("/api/jobs", json={**base, "params": {"unsafe": "value"}}).status_code == 422
    assert client.post("/api/jobs", json={**base, "media_path": "C:/secret"}).status_code == 422
    assert client.post("/api/jobs", json={**base, "media_id": "not-registered"}).status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"


def test_dataset_revalidated_before_training(api, tmp_path):
    from test_datasets import yolo
    client, _ = api
    config = yolo(tmp_path / "dataset")
    response = client.post("/api/datasets/validate", json={"path": str(config), "format": "yolo_detection"})
    assert response.json()["valid"]
    dataset = response.json()["dataset"]
    assert "train_images" not in dataset["details"]
    base = {"kind": "training", "model_id": "detector", "dataset_id": dataset["id"]}
    assert client.post("/api/jobs", json=base).status_code == 201
    (config.parent / "labels/train/train.txt").unlink()
    response = client.post("/api/jobs", json=base)
    assert response.status_code == 422 and "重新驗證" in response.text
    assert client.post("/api/jobs", json={**base, "model_id": "action"}).status_code == 422


def test_zip_upload_blocks_escape_and_cleans_partial(api):
    client, app = api
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("../outside.txt", "bad")
    response = client.post("/api/datasets/upload", data={"format": "video_csv"}, files={"file": ("dataset.zip", stream.getvalue(), "application/zip")})
    assert response.status_code == 200 and not response.json()["valid"]
    assert not list((app.state.runtime / "datasets").iterdir())


def test_valid_zip_is_registered_with_extraction_boundary(api):
    client, app = api
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("dataset/train/cat/one.png", png())
        bundle.writestr("dataset/val/cat/two.png", png())
    response = client.post("/api/datasets/upload", data={"format": "imagefolder"}, files={"file": ("dataset.zip", stream.getvalue(), "application/zip")})
    assert response.status_code == 200 and response.json()["valid"], response.text
    dataset = response.json()["dataset"]
    assert dataset["train_count"] == dataset["val_count"] == 1
    assert Path(dataset["path"]).is_relative_to(Path(dataset["allowed_root"]))
    assert app.state.datasets.get(dataset["id"])["details"]["train_samples"]
    assert not list((app.state.runtime / "datasets").glob("*.zip"))


def test_only_published_job_artifacts_are_served(api):
    client, app = api
    item = upload_image(client)
    job = client.post("/api/jobs", json={"kind": "inference", "model_id": "detector", "media_id": item["id"]}).json()
    manager = app.state.jobs
    directory = manager.job_root / job["id"]
    (directory / "output.png").write_bytes(png())
    result, artifacts = manager._public_result(job["id"], {"media_path": "output.png", "media_kind": "image"})
    manager.jobs[job["id"]]["_artifacts"] = artifacts
    assert client.get(result["media_url"]).status_code == 200
    assert client.get(f"/api/artifacts/{job['id']}/request.json").status_code == 404
    assert client.get(f"/api/artifacts/{job['id']}/../jobs.json").status_code == 404
    with pytest.raises(ValueError):
        manager._public_result(job["id"], {"media_path": "../jobs.json"})
    with pytest.raises(ValueError):
        manager._public_result(job["id"], {"media_url": "https://evil.test/result"})


def test_finetuned_checkpoint_requires_completed_compatible_job(api):
    client, app = api
    item = upload_image(client)
    manager = app.state.jobs
    trained = manager.submit({"kind": "training", "model_id": "yolo26x", "params": {}})
    checkpoint = manager.job_root / trained["id"] / "best.pt"
    checkpoint.write_bytes(b"test checkpoint")
    request = {"kind": "inference", "model_id": "yolo26x", "media_id": item["id"], "checkpoint_job_id": trained["id"]}
    assert client.post("/api/jobs", json=request).status_code == 422
    manager.jobs[trained["id"]]["status"] = "completed"
    response = client.post("/api/jobs", json=request)
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["checkpoint_job_id"] == trained["id"]
    saved = json.loads((manager.job_root / job["id"] / "request.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_path"] == str(checkpoint.resolve())
    assert client.post("/api/jobs", json={**request, "model_id": "detector"}).status_code == 422
    assert client.post("/api/jobs", json={**request, "checkpoint_job_id": "../escape"}).status_code == 422
    assert client.post("/api/jobs", json={**request, "kind": "training"}).status_code == 422
    checkpoint.unlink()
    assert client.post("/api/jobs", json=request).status_code == 422


@pytest.mark.parametrize("model_id,config_name", [("vivit_base_k400", "config.json"), ("x3d_l", "model_config.json")])
def test_action_checkpoint_requires_files_and_object_metadata(tmp_path, model_id, config_name):
    manager = JobManager(tmp_path / "runtime", tmp_path)
    trained = manager.submit({"kind": "training", "model_id": model_id, "params": {}})
    manager.jobs[trained["id"]]["status"] = "completed"
    best = manager.job_root / trained["id"] / "best"
    best.mkdir()
    for name in ("model.safetensors", config_name):
        (best / name).write_bytes(b"test")
    metadata = best / "workbench_training.json"
    metadata.write_text("[]")
    with pytest.raises(ValueError, match="中繼資料"):
        manager.checkpoint_for(trained["id"], model_id)
    metadata.write_text(json.dumps({"model_id": model_id}))
    assert manager.checkpoint_for(trained["id"], model_id) == best.resolve()
    (best / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="必要檔案"):
        manager.checkpoint_for(trained["id"], model_id)


@pytest.mark.parametrize("model_id,files", [
    ("grounding_dino_base", ["model.safetensors", "config.json"]),
    ("rf_detr_2xl", ["model.pth"]),
])
def test_detection_checkpoint_requires_completed_job_and_matching_metadata(tmp_path, model_id, files):
    manager = JobManager(tmp_path / "runtime", tmp_path)
    trained = manager.submit({"kind": "training", "model_id": model_id, "params": {}})
    best = manager.job_root / trained["id"] / "best"
    best.mkdir()
    for name in files:
        (best / name).write_bytes(b"test")
    metadata = best / "workbench_training.json"
    metadata.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
    with pytest.raises(ValueError, match="已完成"):
        manager.checkpoint_for(trained["id"], model_id)
    manager.jobs[trained["id"]]["status"] = "completed"
    assert manager.checkpoint_for(trained["id"], model_id) == best.resolve()
    metadata.write_text(json.dumps({"model_id": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="不相容"):
        manager.checkpoint_for(trained["id"], model_id)
    metadata.write_text(json.dumps({"model_id": model_id}), encoding="utf-8")
    (best / files[0]).unlink()
    with pytest.raises(ValueError, match="必要檔案"):
        manager.checkpoint_for(trained["id"], model_id)


WORKER = r'''
import json,sys,time
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out=Path(sys.argv[2])
with (out/"events.jsonl").open("w",encoding="utf-8") as f:
    f.write(json.dumps({"event":"progress","progress":0.5,"message":"halfway"})+"\n");f.flush()
    f.write(json.dumps({"event":"metric","epoch":1,"train_loss":0.25})+"\n");f.flush()
time.sleep(request.get("params",{}).get("delay",0.1))
(out/"output.txt").write_text("actual subprocess output")
(out/"result.json").write_text(json.dumps({"status":"completed","result":{"summary":"done","artifacts":{"report":"output.txt"}}}))
'''


def wait_for(manager, job_id, statuses, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.03)
    raise AssertionError(f"job did not reach {statuses}: {manager.get(job_id)}")


def test_real_subprocess_queue_is_serial_and_cancel_is_terminal(tmp_path):
    manager = JobManager(tmp_path / "runtime", tmp_path, lambda req, out: [sys.executable, "-c", WORKER, str(req), str(out)])
    manager.start()
    try:
        first = manager.submit({"kind": "inference", "model_id": "fake", "params": {"delay": 3}})
        second = manager.submit({"kind": "inference", "model_id": "fake", "params": {"delay": 0.1}})
        wait_for(manager, first["id"], {"running"})
        assert manager.get(second["id"])["status"] == "queued"
        assert manager.counts() == {"running": 1, "queued": 1}
        manager.cancel(first["id"])
        completed = wait_for(manager, second["id"], {"completed", "failed"})
        assert completed["status"] == "completed", completed
        assert completed["metrics"][0]["train_loss"] == 0.25
        assert completed["result"]["artifact_urls"]["report"].endswith("/output.txt")
        assert manager.get(first["id"])["status"] == "cancelled"
        assert manager.artifact(second["id"], "output.txt").read_text() == "actual subprocess output"
    finally:
        manager.close()


def test_restart_marks_interrupted_jobs_failed_without_relaunch(tmp_path):
    manager = JobManager(tmp_path / "runtime", tmp_path)
    job = manager.submit({"kind": "training", "model_id": "fake", "params": {}})
    restored = JobManager(tmp_path / "runtime", tmp_path)
    status = restored.get(job["id"])
    assert status["status"] == "failed"
    assert "沒有自動" in status["error"]
    assert restored.pending.empty()


def test_partial_event_line_waits_for_completion(tmp_path):
    manager = JobManager(tmp_path / "runtime", tmp_path)
    job = manager.submit({"kind": "training", "model_id": "fake", "params": {}})
    manager.jobs[job["id"]]["status"] = "running"
    path = manager.job_root / job["id"] / "events.jsonl"
    path.write_bytes(b'{"event":"metric","epoch":1')
    assert manager._consume_events(job["id"], path, 0) == 0
    assert manager.get(job["id"])["metrics"] == []
    with path.open("ab") as stream:
        stream.write(b',"train_loss":0.2}\n')
    offset = manager._consume_events(job["id"], path, 0)
    assert offset == path.stat().st_size
    assert manager.get(job["id"])["metrics"] == [{"epoch": 1, "train_loss": 0.2}]
    manager._consume_events(job["id"], path, offset)
    assert len(manager.get(job["id"])["metrics"]) == 1
