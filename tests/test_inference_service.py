"""CPU subprocess tests for GPU ownership and persistent-worker IPC."""
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import pytest

from backend.jobs import JobManager
from backend import worker


SERVICE = r'''
import json, os, sys, time
from pathlib import Path
for line in sys.stdin:
    command = json.loads(line)
    out = Path(command["output_dir"])
    request = json.loads(Path(command["request_path"]).read_text(encoding="utf-8"))
    params = request.get("params", {})
    (out / "started").write_text(str(os.getpid()))
    with (out / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event":"log", "message":request["id"]}) + "\n")
    if params.get("crash"):
        os._exit(7)
    payload = {"job_id": command["job_id"], "dispatch_token": command["dispatch_token"],
               "status": "completed", "result": {"worker_pid": os.getpid(),
               "runtime": os.environ["VISION_WORKBENCH_RUNTIME"], "artifacts": {"report":"output.txt"}}}
    if params.get("wrong_identity"):
        payload[params["wrong_identity"]] = "0" * 32
    if params.get("fail"):
        payload.update(status="failed", error="ordinary model failure")
    if params.get("malformed"):
        (out / "result.json").write_text("{")
        time.sleep(10)
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(params.get("delay", 0.02))
    (out / "output.txt").write_text(request["id"], encoding="utf-8")
    (out / "worker.log").write_text("log-" + request["id"], encoding="utf-8")
    tmp.replace(out / "result.json")
'''

ISOLATED = r'''
import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[2])
(out / "started").write_text(str(os.getpid()))
time.sleep(0.02)
(out / "result.json").write_text(json.dumps({"status":"completed", "result":{"worker_pid":os.getpid()}}))
'''


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.015)
    raise AssertionError("condition was not reached before timeout")


def finished(manager, job):
    result = wait_until(lambda: manager.get(job["id"]) if manager.get(job["id"])["status"] in {"completed", "failed", "cancelled"} else None)
    return result


def submit(manager, kind="inference", **params):
    request = {"kind": kind, "model_id": "fake", "params": params}
    if kind == "live":
        request.update(models=[{"key": "0", "model_id": "fake", "params": {}}], capture_fps=30, max_duration_seconds=None)
    return manager.submit(request)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager = JobManager(tmp_path / "runtime", tmp_path)
    monkeypatch.setattr(manager, "_inference_command", lambda: [sys.executable, "-u", "-c", SERVICE])
    # Keep production service selection while replacing only isolated workers.
    manager.command_factory = lambda req, out: [sys.executable, "-u", "-c", ISOLATED, str(req), str(out)]
    manager.start()
    try:
        yield manager
    finally:
        manager.close()


def test_reuses_one_process_across_jobs_with_independent_outputs(manager):
    first = submit(manager, delay=0.15)
    second = submit(manager)
    first_result = finished(manager, first)
    second_result = finished(manager, second)
    assert first_result["status"] == second_result["status"] == "completed"
    assert first_result["result"]["worker_pid"] == second_result["result"]["worker_pid"]
    assert first_result["result"]["runtime"] == str(manager.runtime)
    for job, result in ((first, first_result), (second, second_result)):
        assert manager.artifact(job["id"], "output.txt").read_text() == job["id"]
        assert result["result"]["artifact_urls"]["report"] == f'/api/artifacts/{job["id"]}/output.txt'
        assert result["logs"] == [job["id"], "log-" + job["id"]]
    process = manager._inference_process
    assert process.poll() is None
    assert manager.release_inference_cache() is True
    assert process.poll() is not None
    assert manager.release_inference_cache() is False


def test_temporary_result_does_not_finish_or_run_next_job(manager):
    first = submit(manager, delay=0.6)
    second = submit(manager)
    output = manager.job_root / first["id"]
    wait_until(lambda: (output / "result.json.tmp").exists())
    assert not (output / "output.txt").exists()
    assert manager.get(first["id"])["status"] == "running"
    assert manager.get(second["id"])["status"] == "queued"
    assert not (manager.job_root / second["id"] / "started").exists()
    with pytest.raises(ValueError, match="工作"):
        manager.release_inference_cache()
    assert finished(manager, second)["status"] == "completed"


def test_cancel_waits_for_process_exit_and_next_inference_restarts(manager):
    first = submit(manager, delay=10)
    wait_until(lambda: (manager.job_root / first["id"] / "started").exists())
    process = manager._inference_process
    original_pid = int((manager.job_root / first["id"] / "started").read_text())
    second = submit(manager)
    assert manager.cancel(first["id"])["status"] == "cancelled"
    assert process.poll() is not None
    result = finished(manager, second)
    assert result["status"] == "completed", result
    assert result["result"]["worker_pid"] != original_pid
    assert manager._inference_process is not process
    assert manager.get(first["id"])["status"] == "cancelled"
    assert not (manager.job_root / first["id"] / "output.txt").exists()


def test_cancel_queued_job_preserves_active_service(manager):
    first = submit(manager, delay=0.3)
    wait_until(lambda: (manager.job_root / first["id"] / "started").exists())
    process = manager._inference_process
    second = submit(manager)
    manager.cancel(second["id"])
    assert process.poll() is None
    first_result = finished(manager, first)
    assert first_result["status"] == "completed"
    third = finished(manager, submit(manager))
    assert third["result"]["worker_pid"] == first_result["result"]["worker_pid"]
    assert manager._inference_process is process
    assert manager.get(second["id"])["status"] == "cancelled"


@pytest.mark.parametrize("kind", ["training", "live"])
def test_isolated_jobs_release_cached_process_before_spawn(manager, monkeypatch, kind):
    first = finished(manager, submit(manager))
    assert first["status"] == "completed"
    cached = manager._inference_process
    original = manager.command_factory
    checked = []

    def isolated_command(req, out):
        assert cached.poll() is not None
        assert manager._inference_process is None
        checked.append(kind)
        return original(req, out)

    monkeypatch.setattr(manager, "command_factory", isolated_command)
    result = finished(manager, submit(manager, kind))
    assert result["status"] == "completed", result
    assert checked == [kind]
    next_result = finished(manager, submit(manager))
    assert next_result["status"] == "completed"
    assert next_result["result"]["worker_pid"] != first["result"]["worker_pid"]
    assert manager._inference_process is not cached


def test_worker_crash_fails_only_current_job_and_next_job_recovers(manager):
    first = submit(manager, crash=True)
    second = submit(manager)
    failure = finished(manager, first)
    assert failure["status"] == "failed"
    assert "7" in failure["error"]
    assert finished(manager, second)["status"] == "completed"


@pytest.mark.parametrize("identity", ["job_id", "dispatch_token"])
def test_mismatched_completion_identity_is_rejected_and_worker_replaced(manager, identity):
    first = submit(manager, wrong_identity=identity, delay=0.2)
    wait_until(lambda: (manager.job_root / first["id"] / "started").exists())
    process = manager._inference_process
    original_pid = int((manager.job_root / first["id"] / "started").read_text())
    second = submit(manager)
    failure = finished(manager, first)
    assert failure["status"] == "failed"
    assert "識別不符" in failure["error"]
    assert process.poll() is not None
    result = finished(manager, second)
    assert result["status"] == "completed"
    assert result["result"]["worker_pid"] != original_pid
    assert manager._inference_process is not process


def test_bad_final_result_kills_service_and_does_not_stall_queue(manager):
    first = submit(manager, malformed=True)
    second = submit(manager)
    assert finished(manager, first)["status"] == "failed"
    assert finished(manager, second)["status"] == "completed"


def test_ordinary_inference_failure_preserves_other_cached_models(manager):
    first = finished(manager, submit(manager))
    failure = finished(manager, submit(manager, fail=True))
    assert failure["status"] == "failed"
    assert failure["error"] == "ordinary model failure"
    third = finished(manager, submit(manager))
    assert first["result"]["worker_pid"] == third["result"]["worker_pid"]


def test_close_terminates_idle_service(manager):
    finished(manager, submit(manager))
    process = manager._inference_process
    assert process.poll() is None
    manager.close()
    assert process.poll() is not None
    assert not manager.thread.is_alive()


def test_close_cancels_active_and_queued_jobs_and_terminates_service(manager):
    first = submit(manager, delay=10)
    second = submit(manager)
    wait_until(lambda: (manager.job_root / first["id"] / "started").exists())
    process = manager._inference_process
    manager.close()
    assert process.poll() is not None
    assert manager.get(first["id"])["status"] == "cancelled"
    assert manager.get(second["id"])["status"] == "cancelled"
    assert not manager.thread.is_alive()


def test_late_completion_cannot_overwrite_cancellation(manager, monkeypatch):
    entered = threading.Event()
    continue_finish = threading.Event()
    original = manager._finish_job

    def blocked_finish(*args, **kwargs):
        entered.set()
        assert continue_finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "_finish_job", blocked_finish)
    first = submit(manager)
    assert entered.wait(5)
    process = manager._inference_process
    try:
        assert manager.cancel(first["id"])["status"] == "cancelled"
        assert process.poll() is not None
    finally:
        continue_finish.set()
    wait_until(lambda: manager.active_id is None)
    assert manager.get(first["id"])["status"] == "cancelled"
    assert manager.artifact(first["id"], "output.txt") is None


def test_custom_command_factory_keeps_isolated_worker_contract(tmp_path):
    manager = JobManager(tmp_path / "runtime", tmp_path,
                         lambda req, out: [sys.executable, "-c", ISOLATED, str(req), str(out)])
    manager.start()
    try:
        first = finished(manager, submit(manager))
        second = finished(manager, submit(manager))
        assert first["status"] == second["status"] == "completed"
        assert first["result"]["worker_pid"] != second["result"]["worker_pid"]
        assert manager._inference_process is None
    finally:
        manager.close()


def make_service_command(runtime, *, kind="inference", **params):
    job_id = uuid.uuid4().hex
    output = runtime / "jobs" / job_id
    output.mkdir(parents=True)
    request = output / "request.json"
    request.write_text(json.dumps({"id": job_id, "kind": kind, "model_id": "fake", "params": params}), encoding="utf-8")
    return {"job_id": job_id, "dispatch_token": uuid.uuid4().hex,
            "request_path": str(request), "output_dir": str(output)}


def test_actual_service_reuses_cache_and_separates_python_native_logs(tmp_path, monkeypatch, capfd):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(runtime))
    monkeypatch.setattr(worker, "configure_torch", lambda: None)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_AUTOINSTALL", "HF_HUB_DISABLE_TELEMETRY", "WANDB_DISABLED", "YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    commands = [make_service_command(runtime), make_service_command(runtime, fail=True), make_service_command(runtime)]
    instances = []
    seen = []
    logger = logging.Logger("persistent-test-logger")

    class Cache:
        def __init__(self):
            instances.append(self)
            self.closed = 0

        def close(self):
            self.closed += 1

    def infer(request, output, emit, *, model_cache):
        seen.append(model_cache)
        if not logger.handlers:
            # Match libraries such as Ultralytics that capture sys.stdout once
            # when first imported and reuse that same handler on later jobs.
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.addHandler(logging.StreamHandler(sys.stderr))
        assert all(not handler.stream.closed for handler in logger.handlers)
        logger.warning("persistent-logger-" + request["id"])
        print("python-" + request["id"])
        print("stderr-" + request["id"], file=sys.stderr)
        os.write(1, ("native-" + request["id"] + "\n").encode())
        emit({"event": "log", "message": request["id"]})
        if request["params"].get("fail"):
            raise ValueError("expected failure")
        (output / "artifact.txt").write_text(request["id"])
        return {"artifacts": {"report": "artifact.txt"}}

    stream = io.StringIO("".join(json.dumps(command) + "\n" for command in commands))
    assert worker.run_inference_service(stream, cache_factory=Cache, inference_runner=infer) == 0
    assert len(instances) == 1
    assert seen == instances * 3
    assert instances[0].closed == 1
    for index, command in enumerate(commands):
        output = Path(command["output_dir"])
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == ("failed" if index == 1 else "completed")
        assert result["job_id"] == command["job_id"]
        assert result["dispatch_token"] == command["dispatch_token"]
        log = (output / "worker.log").read_text(encoding="utf-8")
        for prefix in ("python-", "stderr-", "native-", "persistent-logger-"):
            assert prefix + command["job_id"] in log
        assert all(other["job_id"] not in log for other in commands if other is not command)
        assert "Logging error" not in log
        assert not (output / "result.json.tmp").exists()
    assert "python-" not in capfd.readouterr().out


@pytest.mark.parametrize("mutation", ["outside", "identity", "training", "relative", "token"])
def test_service_rejects_unowned_or_mismatched_job_commands(tmp_path, monkeypatch, mutation):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(runtime))
    command = make_service_command(runtime, kind="training" if mutation == "training" else "inference")
    if mutation == "outside":
        command["output_dir"] = str(tmp_path)
    elif mutation == "identity":
        command["job_id"] = uuid.uuid4().hex
    elif mutation == "relative":
        command["output_dir"] = "jobs/" + command["job_id"]
    elif mutation == "token":
        command["dispatch_token"] = "not-a-dispatch-token"
    with pytest.raises(ValueError):
        worker.run_inference_service(io.StringIO(json.dumps(command) + "\n"))


def test_result_read_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"status":"completed"}', encoding="utf-8")
    original = Path.read_text

    def denied_once(self, *args, **kwargs):
        monkeypatch.setattr(Path, "read_text", original)
        raise PermissionError("Windows sharing violation")

    monkeypatch.setattr(Path, "read_text", denied_once)
    assert JobManager._read_result(path) is None
    assert JobManager._read_result(path) == {"status": "completed"}
