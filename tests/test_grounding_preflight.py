from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import grounding_preflight as module


@pytest.fixture(autouse=True)
def clear_cache():
    module._cached_prompt_plan.cache_clear()
    yield
    module._cached_prompt_plan.cache_clear()


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type":"grounding-dino"}')
    (folder / "tokenizer.json").write_text('{}')
    (folder / "model.safetensors").write_bytes(b"unused weights")
    return folder


def checked(classes=None):
    return {"valid": True, "errors": [], "warnings": ["existing warning"],
            "plan": {"classes": classes} if classes is not None else {}}


def request(source, mode="standard", classes=None):
    return {"model_id": "grounding_dino_base", "training_mode": mode,
            "model_weight_path": str(source / "model.safetensors"),
            "dataset": {"classes": ["new category"] if classes is None else classes}}


def result_plan(classes, split=90):
    chunks = [{"caption": " . ".join(classes[start:start + split]) + " .",
               "global_class_ids": list(range(start, min(len(classes), start + split))),
               "text_tokens": len(classes[start:start + split]) * 4 + 2}
              for start in range(0, len(classes), split)]
    return {"caption_chunks": chunks, "base_max_text_len": 256,
            "max_text_len": max(256, max(chunk["text_tokens"] for chunk in chunks)),
            "text_backbone_max_position_embeddings": 512}


def output(plan=None, error=None, noise=True):
    payload = {"ok": False, "error": error} if error else {"ok": True, "plan": plan}
    return SimpleNamespace(returncode=1 if error else 0, stderr="",
                           stdout=("tokenizer informational output\n" if noise else "")
                           + module.RESULT_PREFIX + json.dumps(payload) + "\n")


def fake_inspector(monkeypatch):
    calls = []

    def run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        calls.append((command, kwargs, payload))
        return output(result_plan(payload["classes"]))

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("mode", ["expand_frozen", "expand_classes"])
def test_expansion_inspects_all_union_classes_and_publishes_untruncated_group_plan(source, monkeypatch, mode):
    classes = [f"class_{index}" for index in range(170)]
    previous = checked(classes)
    before = deepcopy(previous)
    calls = fake_inspector(monkeypatch)
    value = module.preflight(request(source, mode), previous)
    assert value["valid"]
    assert calls[0][2]["classes"] == classes
    plan = value["plan"]["grounding_prompts"]
    assert plan["class_count"] == 170 and plan["chunk_count"] == 2
    assert plan["text_tokens"] == [362, 322]
    assert plan["max_text_len"] == 362
    assert plan["classes_per_chunk"] == [90, 80]
    assert [index for chunk in plan["caption_chunks"] for index in chunk["global_class_ids"]] == list(range(170))
    assert any("170" in warning and "不會截斷" in warning for warning in value["warnings"])
    assert previous == before


@pytest.mark.parametrize("mode", ["standard", "dual_branch"])
def test_standard_and_new_dual_branch_only_inspect_dataset_classes(source, monkeypatch, mode):
    calls = fake_inspector(monkeypatch)
    value = module.preflight(request(source, mode, ["new meal", "new fruit"]), checked(["old", "new meal", "new fruit"]))
    assert value["valid"]
    assert calls[0][2]["classes"] == ["new meal", "new fruit"]
    assert value["plan"]["grounding_prompts"]["class_count"] == 2


def test_subprocess_is_bounded_offline_cpu_and_checkpoint_source_overrides_base(source, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{}')
    calls = fake_inspector(monkeypatch)
    value = module.preflight({**request(source), "checkpoint_path": str(checkpoint)}, checked())
    assert value["valid"]
    command, options, payload = calls[0]
    assert command == [sys.executable, "-m", "backend.grounding_preflight", "--inspect"]
    assert options["timeout"] == 90
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert options["env"]["HF_HUB_OFFLINE"] == "1"
    assert options["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert payload["folder"] == str(checkpoint.resolve())
    assert not options.get("shell")


def test_preview_submit_cache_reuses_same_inputs_but_rechecks_changed_classes_config_or_tokenizer(source, monkeypatch):
    calls = fake_inspector(monkeypatch)
    original = request(source)
    value = module.preflight(original, checked())
    value["plan"]["grounding_prompts"]["caption_chunks"][0]["caption"] = "caller changed this"
    again = module.preflight(original, checked())
    assert len(calls) == 1
    assert again["plan"]["grounding_prompts"]["caption_chunks"][0]["caption"] == "new category ."
    for name in ("config.json", "tokenizer.json"):
        path = source / name
        before = path.stat()
        path.write_text(path.read_text() + " ")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        assert module.preflight(original, checked())["valid"]
    assert len(calls) == 3
    assert module.preflight(request(source, classes=["another category"]), checked())["valid"]
    assert len(calls) == 4


def test_invalid_expansion_and_other_models_do_not_spawn_or_require_grounding_files(monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected tokenizer process"))
    invalid = {"valid": False, "errors": ["missing coverage"], "warnings": [], "plan": {}}
    assert module.preflight({"model_id": "grounding_dino_base"}, invalid) is invalid
    ordinary = checked()
    assert module.preflight({"model_id": "yolo26x"}, ordinary) is ordinary


@pytest.mark.parametrize("failure", ["tokenizer", "timeout", "duplicate", "malformed", "coverage"])
def test_failures_become_invalid_preview_errors_without_partial_plan_or_cached_success(source, monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 90)
        if failure == "tokenizer":
            return output(error="Local tokenizer files are missing")
        success = output(result_plan(["new category"]))
        if failure == "duplicate":
            success.stdout += success.stdout
        elif failure == "malformed":
            success.stdout = module.RESULT_PREFIX + "not json"
        elif failure == "coverage":
            success = output(result_plan(["class one", "unexpected second class"]))
        return success
    monkeypatch.setattr(module.subprocess, "run", run)
    previous = checked()
    value = module.preflight(request(source), previous)
    assert not value["valid"]
    assert value["errors"]
    assert "grounding_prompts" not in value["plan"]
    assert previous["valid"] and not previous["errors"]
    assert module._cached_prompt_plan.cache_info().currsize == 0


def test_worker_uses_only_local_config_and_tokenizer_and_passes_every_class(monkeypatch):
    from backend import grounding_prompts
    calls = []
    config = SimpleNamespace(max_text_len=256)
    tokenizer = object()
    def loader(value):
        def load(path, **kwargs):
            calls.append((path, kwargs))
            return value
        return SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=loader(config), AutoTokenizer=loader(tokenizer)))
    classes = [f"category_{i}" for i in range(170)]
    def prepare(actual, actual_tokenizer, actual_config):
        assert actual == classes
        assert actual_tokenizer is tokenizer and actual_config is config
        return [{"caption": "complete groups", "global_class_ids": list(range(170)), "text_tokens": 490}], {"max_text_len": 490}
    monkeypatch.setattr(grounding_prompts, "prepare_caption_chunks", prepare)
    result = module._inspect({"folder": "already-validated-local-folder", "classes": classes})
    assert len(calls) == 2
    assert all(options == {"local_files_only": True, "trust_remote_code": False} for _, options in calls)
    assert result["caption_chunks"][0]["global_class_ids"] == list(range(170))


def test_importing_preflight_does_not_import_torch_or_transformers():
    # A clean interpreter catches accidental heavy imports even if other tests
    # have already loaded torch into the pytest process.
    result = subprocess.run([sys.executable, "-c", "import sys; import backend.grounding_preflight; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_api_preview_and_enqueue_block_invalid_grounding_text_before_any_job_is_created(tmp_path, source, monkeypatch):
    from fastapi.testclient import TestClient
    from test_datasets import yolo
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    api = importlib.import_module("backend.app")
    monkeypatch.setattr(api, "get_models", lambda: [{"id": "grounding_dino_base", "available": True,
        "training_supported": True, "capabilities": {"training": {"dataset_formats": ["yolo_detection"]}}}])
    monkeypatch.setattr(api, "validate_job_request", lambda model_id, kind, params: params)
    monkeypatch.setattr(api.ModelRegistry, "resolve", lambda self, model_id: source / "model.safetensors")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: output(error="A single phrase exceeds the text limit"))
    application = api.create_app(tmp_path / "api-runtime", tmp_path / "samples", start_worker=False)
    with TestClient(application, base_url="http://127.0.0.1") as client:
        dataset = client.post("/api/datasets/validate", json={"path": str(yolo(tmp_path / "dataset")), "format": "yolo_detection"}).json()["dataset"]
        payload = {"kind": "training", "model_id": "grounding_dino_base", "dataset_id": dataset["id"]}
        preview = client.post("/api/training/preview", json=payload)
        assert preview.status_code == 200 and not preview.json()["valid"]
        assert "single phrase" in preview.text
        submitted = client.post("/api/jobs", json=payload)
        assert submitted.status_code == 422 and "single phrase" in submitted.text
        assert not application.state.jobs.list()
        calls = fake_inspector(monkeypatch)
        ready = client.post("/api/training/preview", json=payload)
        assert ready.status_code == 200 and ready.json()["valid"]
        assert ready.json()["plan"]["grounding_prompts"]["class_count"] == 1
        queued = client.post("/api/jobs", json=payload)
        assert queued.status_code == 201, queued.text
        assert len(calls) == 1  # Valid preview and submit share metadata cache.
        assert len(application.state.jobs.list()) == 1
