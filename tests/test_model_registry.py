"""Discovery/registration uses tiny files and metadata, never model execution."""
import importlib
import io
import json
from pathlib import Path
import sys
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import model_registry as registry_module
from backend.model_registry import ModelRegistry, WeightPaths, registry_runtime, validate_checkpoint, validate_weight


def write(path, content=b"fixture weight"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def document(path, value):
    return write(path, json.dumps(value).encode())


def hf_folder(folder, model_type="vivit"):
    document(folder / "config.json", {"model_type": model_type, "id2label": {"0": "walking"}})
    document(folder / "preprocessor_config.json", {})
    write(folder / "model.safetensors")
    if model_type == "grounding-dino":
        document(folder / "tokenizer_config.json", {})
        document(folder / "tokenizer.json", {})
    return folder


def checkpoint(folder, model_id="x3d_l"):
    metadata = {"model_id": model_id, "classes": ["walking", "running"]}
    if model_id in {"yolo26x", "yolov12x"}:
        write(folder / "best.pt")
    elif model_id in {"vivit_base_k400", "grounding_dino_base"}:
        hf_folder(folder, registry_module.HF_TYPES[model_id])
        if model_id == "vivit_base_k400":
            document(folder / "config.json", {"model_type": "vivit", "id2label": {str(i): name for i, name in enumerate(metadata["classes"])}})
    elif model_id == "rf_detr_2xl":
        write(folder / "model.pth")
        metadata.update(class_id_offset=1, resolution=560, group_detr=13, positional_encoding_size=44)
    else:
        write(folder / "model.safetensors")
        document(folder / "model_config.json", {"model_id": model_id, "classes": metadata["classes"]})
    document(folder / "workbench_training.json", metadata)
    return folder / "best.pt" if model_id in {"yolo26x", "yolov12x"} else folder


@pytest.fixture
def registry(tmp_path, monkeypatch):
    app_root = tmp_path / "application"
    app_root.mkdir()
    monkeypatch.setattr(registry_module, "APP_ROOT", app_root)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hf-cache"))
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "runtime"))
    return ModelRegistry(app_root=app_root)


def test_default_layout_and_dynamic_binding_keep_external_files_in_place(registry, tmp_path):
    base = write(registry.app_root / "models/yolo26x/yolo26x.pt")
    assert registry.resolve("yolo26x") == base
    external = write(tmp_path / "elsewhere/custom.pt", b"other file")
    registered = registry.bind("yolo26x", str(external))
    assert registered["valid"]
    assert WeightPaths()["yolo26x"] == external
    assert ModelRegistry(registry.runtime).resolve("yolo26x") == external
    assert base.read_bytes() == b"fixture weight" and external.read_bytes() == b"other file"
    registry.unbind("yolo26x")
    assert WeightPaths()["yolo26x"] == base
    assert external.exists()


def test_scan_never_selects_ambiguous_candidates_and_binding_deletion_does_not_delete_weights(registry, tmp_path):
    first = write(tmp_path / "downloads/a/yolo26x.pt")
    second = write(tmp_path / "downloads/b/yolo26x.pt")
    result = registry.scan([str(tmp_path / "downloads")])
    candidates = [item for item in result["candidates"] if item["model_id"] == "yolo26x"]
    assert len(candidates) == 2 and all(item["valid"] for item in candidates)
    with pytest.raises(ValueError, match="多個"):
        registry.resolve("yolo26x")
    registry.bind("yolo26x", str(first))
    assert registry.resolve("yolo26x") == first
    first.unlink()
    # A broken explicit binding never silently switches to the other model.
    assert registry.snapshot()["bindings"][0]["valid"] is False
    with pytest.raises(ValueError):
        registry.resolve("yolo26x")
    registry.unbind("yolo26x")
    assert registry.resolve("yolo26x") == second


def test_scan_bounds_and_saved_root_semantics(registry, tmp_path, monkeypatch):
    root = tmp_path / "downloads"
    write(root / "deep/a/b/c/yolo26x.pt")
    write(root / "visible/yolo12x.pt")
    monkeypatch.setattr(registry_module, "MAX_DEPTH", 2)
    result = registry.scan([str(root)])
    assert result["truncated"] is True
    assert {item["model_id"] for item in result["candidates"]} == {"yolov12x"}
    assert any(item["path"] == str(root) for item in registry.scan([])["roots"])
    monkeypatch.setattr(registry_module, "MAX_FILES", 1)
    assert registry.scan([])["truncated"] is True
    with pytest.raises(ValueError):
        registry.scan(["relative/path"])
    with pytest.raises(ValueError):
        registry.scan([str(root)] * 13)


def test_hf_complete_companions_and_shards_are_required(registry, tmp_path):
    folder = hf_folder(tmp_path / "hf", "grounding-dino")
    assert validate_weight("grounding_dino_base", folder) == folder / "model.safetensors"
    with pytest.raises(ValueError, match="缺少"):
        validate_weight("grounding_dino_base", folder / "does-not-exist.safetensors")
    write(folder / "unrelated.safetensors")
    with pytest.raises(ValueError, match="不是"):
        validate_weight("grounding_dino_base", folder / "unrelated.safetensors")
    (folder / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        validate_weight("grounding_dino_base", folder)
    write(folder / "tokenizer.json", b"")
    with pytest.raises(ValueError, match="tokenizer"):
        validate_weight("grounding_dino_base", folder)
    write(folder / "vocab.txt", b"token")
    (folder / "model.safetensors").unlink()
    document(folder / "model.safetensors.index.json", {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}})
    write(folder / "one.safetensors")
    with pytest.raises(ValueError, match="two"):
        validate_weight("grounding_dino_base", folder)
    write(folder / "two.safetensors")
    assert validate_weight("grounding_dino_base", folder).name == "one.safetensors"
    document(folder / "model.safetensors.index.json", {"weight_map": {"a": "../outside.safetensors"}})
    with pytest.raises(ValueError, match="路徑"):
        validate_weight("grounding_dino_base", folder)
    with pytest.raises(ValueError, match="架構"):
        validate_weight("vivit_base_k400", folder)


def test_hf_cache_file_symlink_preserves_snapshot_companions(registry, tmp_path):
    folder = hf_folder(tmp_path / "snapshots/revision")
    weight = folder / "model.safetensors"
    blob = write(tmp_path / "blobs/content-hash")
    weight.unlink()
    try:
        weight.symlink_to(blob)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    assert registry.bind("vivit_base_k400", str(folder))["path"] == str(weight)
    assert registry.resolve("vivit_base_k400").parent == folder


def test_hf_default_folder_accepts_complete_alternative_weight_format(registry):
    folder = hf_folder(registry.app_root / "models/vivit_base_k400")
    assert not (folder / "pytorch_model.bin").exists()
    assert registry.resolve("vivit_base_k400") == folder / "model.safetensors"


@pytest.mark.parametrize("model_id", ["yolov12x", "yolo26x", "rf_detr_2xl", "grounding_dino_base", "vivit_base_k400", "x3d_l", "slowfast_r101_8x8", "videomaev2_giant_k710"])
def test_external_checkpoint_metadata_and_companions_revalidated(registry, tmp_path, model_id):
    path = checkpoint(tmp_path / model_id, model_id)
    item = registry.add_checkpoint(model_id, str(path), "My model")
    assert item["source"] == "external" and item["valid"]
    assert registry.checkpoint_for(item["id"], model_id) == path
    assert registry.add_checkpoint(model_id, str(path), "Renamed")["id"] == item["id"]
    with pytest.raises(ValueError):
        registry.checkpoint_for(item["id"], "unknown")
    folder = path.parent if path.is_file() else path
    (folder / "workbench_training.json").unlink()
    assert registry.checkpoints()[0]["valid"] is False
    with pytest.raises((ValueError, OSError)):
        registry.checkpoint_for(item["id"], model_id)
    registry.remove_checkpoint(item["id"])
    assert registry.checkpoints() == [] and path.exists()


def test_external_metadata_cannot_mismatch_model_or_classes(registry, tmp_path):
    path = checkpoint(tmp_path / "x3d")
    document(path / "model_config.json", {"model_id": "x3d_l", "classes": ["wrong"]})
    with pytest.raises(ValueError, match="不符"):
        registry.add_checkpoint("x3d_l", str(path))
    document(path / "workbench_training.json", {"model_id": "other", "classes": ["walking"]})
    with pytest.raises(ValueError, match="model_id"):
        validate_checkpoint("x3d_l", path)
    with pytest.raises(ValueError):
        registry.remove_checkpoint("job:123")


def test_scan_finetuned_hf_is_not_automatically_selected_as_base(registry, tmp_path):
    path = checkpoint(tmp_path / "trained", "vivit_base_k400")
    result = registry.scan([str(path)])
    assert {item["kind"] for item in result["candidates"]} == {"checkpoint"}
    with pytest.raises(ValueError, match="找不到"):
        registry.resolve("vivit_base_k400")
    default = checkpoint(registry.app_root / "models/vivit_base_k400", "vivit_base_k400")
    with pytest.raises(ValueError, match="找不到"):
        registry.resolve("vivit_base_k400")
    # Explicit binding remains an intentional choice, unlike autodiscovery.
    registry.bind("vivit_base_k400", str(default))
    assert registry.resolve("vivit_base_k400") == default / "model.safetensors"


def test_known_weight_filename_and_metadata_cannot_mislabel_architecture(registry, tmp_path):
    wrong = write(tmp_path / "yolo12x.pt")
    with pytest.raises(ValueError, match="其他模型"):
        registry.bind("yolo26x", str(wrong))
    renamed = write(tmp_path / "renamed.pt")
    document(tmp_path / "workbench_training.json", {"model_id": "yolov12x"})
    with pytest.raises(ValueError, match="model_id"):
        registry.bind("yolo26x", str(renamed))


def test_vivit_checkpoint_labels_cannot_disagree_with_metadata(registry, tmp_path):
    folder = checkpoint(tmp_path / "vivit", "vivit_base_k400")
    document(folder / "config.json", {"model_type": "vivit", "id2label": {"0": "wrong", "1": "running"}})
    with pytest.raises(ValueError, match="id2label"):
        validate_checkpoint("vivit_base_k400", folder)


@pytest.mark.parametrize("family", ["12", "26"])
def test_loaded_yolo_architecture_rejects_renamed_family_and_wrong_scale(family):
    from backend.inference import validate_yolo_architecture
    config = {"scale": "x", "backbone": [[-1, 1, "A2C2f" if family == "12" else "C2PSA", []]],
              "end2end": family == "26", "reg_max": 1}
    model = types.SimpleNamespace(task="detect", model=types.SimpleNamespace(yaml=config))
    model_id = "yolov12x" if family == "12" else "yolo26x"
    validate_yolo_architecture(model, model_id)
    with pytest.raises(ValueError, match="不符"):
        validate_yolo_architecture(model, "yolo26x" if family == "12" else "yolov12x")
    config["scale"] = "n"
    with pytest.raises(ValueError, match="不符"):
        validate_yolo_architecture(model, model_id)


def test_legacy_yolo_export_training_config_is_recognized(registry, tmp_path):
    weight = write(tmp_path / "old/best.pt")
    document(weight.parent / "training_config.json", {"model_id": "yolo26x", "dataset_classes": ["cat"]})
    assert validate_checkpoint("yolo26x", weight) == weight


def test_registry_runtime_override_is_context_local(registry, tmp_path):
    weight = write(tmp_path / "external.pt")
    other = ModelRegistry(tmp_path / "other-runtime", registry.app_root)
    other.bind("yolo26x", str(weight))
    with registry_runtime(other.runtime):
        assert WeightPaths()["yolo26x"] == weight
    assert not WeightPaths()["yolo26x"].exists()


def test_worker_receives_its_registry_runtime_not_parent_environment(registry, tmp_path, monkeypatch):
    from backend.jobs import JobManager
    script = tmp_path / "worker.py"
    script.write_text("import json,os,sys\nfrom pathlib import Path\nout=Path(sys.argv[1])\n"
                      "(out/'runtime.txt').write_text(os.environ['VISION_WORKBENCH_RUNTIME'])\n"
                      "(out/'result.json').write_text(json.dumps({'status':'completed','result':{'summary':'ok'}}))\n")
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "wrong-runtime"))
    manager = JobManager(registry.runtime, registry.app_root, lambda request, output: [sys.executable, str(script), str(output)])
    job = manager.submit({"kind": "inference", "model_id": "dummy", "params": {}})
    manager._run(job["id"])
    assert manager.get(job["id"])["status"] == "completed"
    assert (manager.job_root / job["id"] / "runtime.txt").read_text() == str(registry.runtime)


def test_action_full_checkpoint_does_not_read_missing_base(registry, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from backend import inference

    class SmallVideo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            head = torch.nn.Module()
            head.proj = torch.nn.Linear(2, 400)
            head.activation = torch.nn.Softmax(dim=-1)
            self.blocks = torch.nn.ModuleList([head])

    hub = types.ModuleType("pytorchvideo.models.hub")
    hub.x3d_l = lambda pretrained=False: SmallVideo()
    hub.slowfast_r101 = hub.x3d_l
    monkeypatch.setitem(sys.modules, "pytorchvideo.models.hub", hub)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("must not deserialize the base checkpoint"))
    folder = checkpoint(tmp_path / "trained-action", "x3d_l")
    trained = SmallVideo()
    trained.blocks[-1].proj = torch.nn.Linear(2, 2)
    trained.blocks[-1].activation = torch.nn.Identity()
    safetensors.save_file(trained.state_dict(), str(folder / "model.safetensors"))
    model, metadata = inference.create_action_model("x3d_l", "cpu", str(folder))
    assert metadata["labels"] == {0: "walking", 1: "running"}
    assert metadata["outputs_probabilities"] is False
    assert model.blocks[-1].proj.out_features == 2
    assert torch.equal(model.blocks[-1].proj.weight, trained.blocks[-1].proj.weight)


def test_completed_job_checkpoints_are_listed_with_external_entries(registry, tmp_path):
    from backend.jobs import JobManager
    manager = JobManager(registry.runtime, registry.app_root)
    job = manager.submit({"kind": "training", "model_id": "yolo26x", "params": {}})
    manager.jobs[job["id"]]["status"] = "completed"
    write(manager.job_root / job["id"] / "best.pt")
    external = registry.add_checkpoint("x3d_l", str(checkpoint(tmp_path / "custom")))
    entries = registry.checkpoints(manager)
    assert {item["source"] for item in entries} == {"training_job", "external"}
    trained = next(item for item in entries if item["source"] == "training_job")
    assert trained["job_id"] == job["id"] and trained["valid"]
    assert registry.checkpoint_for(trained["id"], "yolo26x", manager).name == "best.pt"
    (manager.job_root / job["id"] / "best.pt").unlink()
    assert next(item for item in registry.checkpoints(manager) if item["source"] == "training_job")["valid"] is False


@pytest.fixture
def api(registry, monkeypatch):
    from backend import catalog, system_info
    monkeypatch.setattr(catalog.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(system_info, "available_device_options", lambda: ["auto", "cpu"])
    # Training modules imported this helper directly.
    for name in ("backend.training", "backend.training_grounding", "backend.training_rfdetr"):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "available_device_options", lambda: ["auto", "cpu"])
    app_module = importlib.import_module("backend.app")
    monkeypatch.setattr(app_module, "APP_ROOT", registry.app_root)
    app = app_module.create_app(registry.runtime, registry.app_root / "samples", start_worker=False)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, app


def upload(client):
    stream = io.BytesIO()
    Image.new("RGB", (12, 10)).save(stream, format="PNG")
    response = client.post("/api/media", files={"file": ("photo.png", stream.getvalue(), "image/png")})
    assert response.status_code == 201
    return response.json()["id"]


def test_api_freezes_bound_weight_and_supports_checkpoint_without_base(api, tmp_path):
    client, app = api
    first = write(tmp_path / "first.pt")
    second = write(tmp_path / "second.pt")
    media_id = upload(client)
    assert client.post("/api/model-files/bind", json={"model_id": "yolo26x", "path": str(first)}).status_code == 200
    job = client.post("/api/jobs", json={"kind": "inference", "model_id": "yolo26x", "media_id": media_id})
    assert job.status_code == 201, job.text
    client.post("/api/model-files/bind", json={"model_id": "yolo26x", "path": str(second)})
    saved = json.loads((app.state.jobs.job_root / job.json()["id"] / "request.json").read_text())
    assert saved["model_weight_path"] == str(first)
    assert "model_weight_path" not in job.json()
    assert client.delete("/api/model-files/bind/yolo26x").status_code == 200
    custom = checkpoint(tmp_path / "custom", "yolo26x")
    registered = client.post("/api/checkpoints", json={"model_id": "yolo26x", "path": str(custom), "name": "External"})
    assert registered.status_code == 201, registered.text
    checkpoint_id = registered.json()["id"]
    models = client.get("/api/models").json()["models"]
    assert next(m for m in models if m["id"] == "yolo26x")["available"] is False
    payload = {"kind": "inference", "model_id": "yolo26x", "media_id": media_id, "checkpoint_id": checkpoint_id}
    inferred = client.post("/api/jobs", json=payload)
    assert inferred.status_code == 201, inferred.text
    assert inferred.json()["checkpoint_id"] == checkpoint_id
    saved = json.loads((app.state.jobs.job_root / inferred.json()["id"] / "request.json").read_text())
    assert saved["checkpoint_path"] == str(custom) and "model_weight_path" not in saved
    assert client.post("/api/jobs", json={**payload, "checkpoint_job_id": "other"}).status_code == 422
    assert client.post("/api/jobs", json={**payload, "kind": "training"}).status_code == 422
    assert client.get("/api/checkpoints").json()["checkpoints"][0]["name"] == "External"
    assert client.delete(f"/api/checkpoints/{checkpoint_id}").status_code == 200
    assert custom.exists()


def test_live_checkpoint_path_and_base_path_snapshots(api, tmp_path):
    client, app = api
    custom = checkpoint(tmp_path / "custom", "yolo26x")
    item = client.post("/api/checkpoints", json={"model_id": "yolo26x", "path": str(custom)}).json()
    payload = {"models": [{"model_id": "yolo26x", "checkpoint_id": item["id"]}]}
    response = client.post("/api/live/sessions", json=payload)
    assert response.status_code == 201, response.text
    assert "checkpoint_path" not in response.json()["models"][0]
    saved = json.loads((app.state.jobs.job_root / response.json()["id"] / "request.json").read_text())
    assert saved["models"][0]["checkpoint_path"] == str(custom)
    client.post(f"/api/jobs/{response.json()['id']}/cancel")
    weight = write(tmp_path / "base.pt")
    client.post("/api/model-files/bind", json={"model_id": "yolo26x", "path": str(weight)})
    response = client.post("/api/live/sessions", json={"models": [{"model_id": "yolo26x"}]})
    assert response.status_code == 201, response.text
    saved = json.loads((app.state.jobs.job_root / response.json()["id"] / "request.json").read_text())
    assert saved["models"][0]["model_weight_path"] == str(weight)


def test_api_license_persists_only_explicit_acceptance_in_this_runtime(api):
    client, app = api
    path = app.state.runtime / "model_license_acceptance.json"
    document(path, {"schema_version": 1, "models": {"other": {"preserve": "yes"}}})
    for value in (False, 1, "true"):
        assert client.post("/api/model-licenses/rf_detr_2xl", json={"accepted": value}).status_code == 422
    accepted = client.post("/api/model-licenses/rf_detr_2xl", json={"accepted": True})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["model"]["license_accepted"] is True
    saved = json.loads(path.read_text())
    assert saved["models"]["other"] == {"preserve": "yes"}
    with registry_runtime(app.state.runtime):
        from backend.model_licenses import is_model_license_accepted
        assert is_model_license_accepted("rf_detr_2xl")
    assert client.post("/api/model-licenses/yolo26x", json={"accepted": True}).status_code == 422


def test_api_scan_and_invalid_registration_have_no_file_side_effects(api, tmp_path):
    client, app = api
    weights = write(tmp_path / "downloads/yolo12x.pt")
    result = client.post("/api/model-files/scan", json={"roots": [str(weights.parent)]})
    assert result.status_code == 200, result.text
    assert any(c["model_id"] == "yolov12x" for c in result.json()["candidates"])
    assert client.get("/api/model-files").status_code == 200
    assert client.post("/api/model-files/bind", json={"model_id": "yolo26x", "path": "relative.pt"}).status_code == 422
    assert client.post("/api/checkpoints", json={"model_id": "yolo26x", "path": str(weights)}).status_code == 422
    assert weights.read_bytes() == b"fixture weight"
