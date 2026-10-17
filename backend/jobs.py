"""Persistent, one-at-a-time worker process queue for a single local GPU."""
from __future__ import annotations

import copy
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

TERMINAL = {"completed", "failed", "cancelled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class JobManager:
    def __init__(self, runtime: Path, app_root: Path, command_factory: Callable[[Path, Path], list[str]] | None = None):
        self.runtime = runtime.resolve()
        self.app_root = app_root.resolve()
        self.job_root = self.runtime / "jobs"
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.runtime / "jobs.json"
        self.lock = threading.RLock()
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.active_id: str | None = None
        # A custom command factory retains the original one-process-per-job
        # contract. Production inference alone uses the reusable service.
        self._persistent_inference = command_factory is None
        self._inference_process: subprocess.Popen | None = None
        self._service_lock = threading.RLock()
        self.command_factory = command_factory or (lambda request, output: [sys.executable, "-u", "-m", "backend.worker", "--request", str(request), "--output", str(output)])
        self.jobs: dict[str, dict[str, Any]] = {}
        if self.registry_path.exists():
            try:
                loaded = json.loads(self.registry_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("invalid jobs registry")
                self.jobs = {key: value for key, value in loaded.items() if isinstance(value, dict) and value.get("id") == key}
            except (OSError, ValueError):
                # Keep damaged data for diagnosis rather than silently erase it.
                backup = self.registry_path.with_name(f"jobs.corrupt-{int(time.time())}.json")
                self.registry_path.replace(backup)
        interrupted = False
        for job in self.jobs.values():
            if job.get("status") in {"running", "queued"}:
                job.update(status="failed", error="服務曾中斷，工作沒有自動重新執行。請確認設定後重新開始。", message="前次執行已中斷", finished_at=now_iso())
                interrupted = True
            if job.get("kind") == "live":
                self._cleanup_live_inputs(job["id"])
        if interrupted:
            self._save()

    def _save(self) -> None:
        write_json_atomic(self.registry_path, self.jobs)

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy({key: value for key, value in job.items() if not key.startswith("_")})

    def start(self) -> None:
        with self.lock:
            if not self.thread or not self.thread.is_alive():
                self.stopping.clear()
                self.thread = threading.Thread(target=self._loop, name="vision-job-queue", daemon=True)
                self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        with self.lock:
            cancellable = [j["id"] for j in self.jobs.values() if j.get("status") in {"queued", "running"}]
        for job_id in cancellable:
            self.cancel(job_id, "服務關閉，工作已停止。")
        self.pending.put(None)
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)
        self._stop_inference_service()

    def release_inference_cache(self) -> bool:
        """Release idle GPU models, waiting for their owning process to exit."""
        with self._service_lock:
            with self.lock:
                if any(job.get("status") in {"queued", "running"} for job in self.jobs.values()):
                    raise ValueError("仍有執行中或等待中的工作，請完成或取消後再釋放模型快取。")
            return self._stop_inference_service()

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.stopping.is_set():
                raise ValueError("服務正在關閉，無法新增工作。")
            if request.get("kind") == "live" and any(j.get("status") in {"queued", "running"} for j in self.jobs.values()):
                raise ValueError("GPU 目前有執行中或等待中的工作；請先完成或取消，再啟動即時攝影機。")
            if sum(j.get("status") == "queued" for j in self.jobs.values()) >= 32:
                raise ValueError("佇列已滿（32 個待執行工作），請等待或取消部分工作。")
            job_id = uuid.uuid4().hex
            directory = self.job_root / job_id
            directory.mkdir()
            request = copy.deepcopy(request)
            request["id"] = job_id
            write_json_atomic(directory / "request.json", request)
            job = {"id": job_id, "kind": request["kind"], "model_id": request["model_id"], "status": "queued", "progress": 0.0, "message": "等待 GPU 執行", "created_at": now_iso(), "params": request.get("params", {}), "logs": [], "metrics": [], "_artifacts": []}
            for field in ("media_id", "dataset_id", "checkpoint_job_id", "checkpoint_id"):
                if field in request:
                    job[field] = request[field]
            if request["kind"] == "live":
                job.update(capture_fps=request["capture_fps"], max_duration_seconds=request["max_duration_seconds"],
                           models=[{key: value for key, value in model.items() if key not in {"checkpoint_path", "model_weight_path"}} for model in request["models"]])
            self.jobs[job_id] = job
            self._save()
            self.pending.put(job_id)
            return self.public(job)

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return self.public(job) if job else None

    def counts(self) -> dict[str, int]:
        with self.lock:
            return {key: sum(j.get("status") == key for j in self.jobs.values()) for key in ("running", "queued")}

    def cancel(self, job_id: str, message: str = "使用者已取消工作。") -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            if job.get("status") in TERMINAL:
                return self.public(job)
            job.update(status="cancelled", message=message, finished_at=now_iso())
            process = self.process if self.active_id == job_id else None
            self._save()
        if process and process.poll() is None:
            self._stop_process(process)
        if job.get("kind") == "live":
            self._cleanup_live_inputs(job_id)
        return self.get(job_id)

    def _cleanup_live_inputs(self, job_id: str) -> None:
        from .live import cleanup_live_inputs
        cleanup_live_inputs(self.job_root, job_id)

    def _worker_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment["VISION_WORKBENCH_RUNTIME"] = str(self.runtime)
        environment.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        return environment

    @staticmethod
    def _process_options() -> dict[str, Any]:
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}

    def _inference_command(self) -> list[str]:
        return [sys.executable, "-u", "-m", "backend.worker", "--inference-service"]

    def _stop_process(self, process: subprocess.Popen, *, graceful: bool = False) -> None:
        # Serialize disposal and creation: a cancelled worker must have actually
        # exited before the next job can allocate another set of GPU models.
        with self._service_lock:
            with self.lock:
                if self._inference_process is process:
                    self._inference_process = None
            if graceful and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.close()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._terminate(process)
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _stop_inference_service(self) -> bool:
        with self._service_lock:
            with self.lock:
                process = self._inference_process
            if process is None:
                return False
            was_running = process.poll() is None
            self._stop_process(process, graceful=True)
            return was_running

    def _get_inference_service(self, job_id: str) -> subprocess.Popen | None:
        with self._service_lock:
            with self.lock:
                if self.stopping.is_set() or self.jobs[job_id].get("status") != "running":
                    return None
                process = self._inference_process
                if process is not None and process.poll() is not None:
                    self._stop_process(process)
                    process = None
                if process is None:
                    process = subprocess.Popen(self._inference_command(), cwd=self.app_root,
                                               env=self._worker_environment(), stdin=subprocess.PIPE,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               **self._process_options())
                    self._inference_process = process
                self.process = process
                return process

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            # Only this worker PID and its children, never a name-wide kill.
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _loop(self) -> None:
        while not self.stopping.is_set():
            job_id = self.pending.get()
            if job_id is None:
                break
            try:
                self._run(job_id)
            except Exception as exc:
                with self.lock:
                    job = self.jobs.get(job_id)
                    if job and job.get("status") not in TERMINAL:
                        job.update(status="failed", message="工作執行失敗", error=f"{type(exc).__name__}: {str(exc)[:2000]}", finished_at=now_iso())
                        self._save()
            finally:
                # An unexpected monitor/persistence error must not leave a
                # detached GPU worker running while the next job starts.
                with self.lock:
                    remaining = self.process
                if remaining is not None and remaining.poll() is None:
                    self._stop_process(remaining)
                with self.lock:
                    self.process = None
                    self.active_id = None
                    job = self.jobs.get(job_id)
                    if job and job.get("kind") == "live":
                        self._cleanup_live_inputs(job_id)
                self.pending.task_done()

    def _run(self, job_id: str) -> None:
        directory = self.job_root / job_id
        with self.lock:
            if self.jobs[job_id].get("status") != "queued" or self.stopping.is_set():
                return
            use_service = self._persistent_inference and self.jobs[job_id].get("kind") == "inference"
        if not use_service:
            # Training and live sessions own their process and GPU exclusively.
            self._stop_inference_service()
        with self.lock:
            job = self.jobs[job_id]
            if job.get("status") != "queued" or self.stopping.is_set():
                return
            job.update(status="running", started_at=now_iso(), message="載入本機模型")
            self.active_id = job_id
            self._save()
        if use_service:
            self._run_persistent_inference(job_id, directory)
            return
        log_path = directory / "worker.log"
        environment = self._worker_environment()
        options = self._process_options()
        with log_path.open("wb") as output:
            with self.lock:
                # Cancellation could happen between queued->running and spawn.
                if self.jobs[job_id].get("status") == "cancelled":
                    return
                self.process = subprocess.Popen(self.command_factory(directory / "request.json", directory), cwd=self.app_root, env=environment, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, **options)
                process = self.process
            offset = 0
            while process.poll() is None:
                offset = self._consume_events(job_id, directory / "events.jsonl", offset)
                if self.stopping.wait(0.2):
                    self._terminate(process)
                    break
            self._consume_events(job_id, directory / "events.jsonl", offset)
        payload = self._read_result(directory / "result.json") or {}
        self._finish_job(job_id, payload, process.returncode == 0,
                         f"工作程序結束（代碼 {process.returncode}）。")

    @staticmethod
    def _read_result(path: Path) -> dict[str, Any] | None:
        try:
            if not path.is_file():
                return None
            if path.stat().st_size > 20 * 1024**2:
                raise ValueError("工作結果超過大小限制。")
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, PermissionError):
            # Windows can briefly deny reads while the final atomic rename is
            # completing. A missing file is never interpreted as completion.
            return None
        if not isinstance(payload, dict):
            raise ValueError("工作結果必須是 JSON 物件。")
        return payload

    def _run_persistent_inference(self, job_id: str, directory: Path) -> None:
        process = None
        reusable = False
        try:
            for name in ("result.json", "result.json.tmp"):
                (directory / name).unlink(missing_ok=True)
            token = uuid.uuid4().hex
            process = self._get_inference_service(job_id)
            if process is None:
                return
            command = {"job_id": job_id, "dispatch_token": token,
                       "request_path": str(directory / "request.json"), "output_dir": str(directory)}
            # Do not hold the state lock while writing a pipe: cancellation must
            # be able to kill a worker that stops reading its input.
            process.stdin.write((json.dumps(command, ensure_ascii=True) + "\n").encode("utf-8"))
            process.stdin.flush()
            offset = 0
            while True:
                offset = self._consume_events(job_id, directory / "events.jsonl", offset)
                with self.lock:
                    cancelled = self.jobs[job_id].get("status") != "running"
                if cancelled or self.stopping.is_set():
                    return
                payload = self._read_result(directory / "result.json")
                if payload is not None:
                    if payload.get("job_id") != job_id or payload.get("dispatch_token") != token:
                        raise ValueError("推論程序回傳的工作識別不符，已停止程序。")
                    if payload.get("status") not in {"completed", "failed"}:
                        raise ValueError("推論程序回傳無效的完成狀態。")
                    self._consume_events(job_id, directory / "events.jsonl", offset)
                    reusable = self._finish_job(job_id, payload, True, "推論工作失敗。")
                    return
                if process.poll() is not None:
                    self._finish_job(job_id, {}, False, f"推論程序意外結束（代碼 {process.returncode}）。下一次推論將重新載入模型。")
                    return
                self.stopping.wait(0.1)
        finally:
            if process is not None and not reusable:
                self._stop_process(process)
            with self.lock:
                if self.process is process:
                    self.process = None

    def _finish_job(self, job_id: str, payload: dict[str, Any], success: bool, fallback_error: str) -> bool:
        with self.lock:
            job = self.jobs[job_id]
            if job.get("status") != "running":
                return False
            if success and payload.get("status") == "completed":
                try:
                    result, artifacts = self._public_result(job_id, payload.get("result", {}))
                    job.update(status="completed", progress=1.0, message="執行完成", result=result, _artifacts=artifacts)
                except (ValueError, OSError) as exc:
                    job.update(status="failed", message="輸出檔案驗證失敗", error=str(exc))
            else:
                error = payload.get("error") or fallback_error
                job.update(status="failed", message="工作執行失敗", error=str(error)[:4000])
            try:
                log_path = self.job_root / job_id / "worker.log"
                with log_path.open("rb") as log_file:
                    log_file.seek(max(0, log_path.stat().st_size - 12_000))
                    tail = log_file.read().decode("utf-8", errors="replace")
                if tail.strip():
                    job["logs"] = (job["logs"] + tail.splitlines())[-300:]
            except OSError:
                pass
            job["finished_at"] = now_iso()
            self._save()
            return True

    def _consume_events(self, job_id: str, path: Path, offset: int) -> int:
        if not path.is_file():
            return offset
        changed = False
        with path.open("rb") as stream:
            stream.seek(offset)
            for _ in range(500):
                start = stream.tell()
                line = stream.readline(64 * 1024)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # A writer may still be writing a line; leave it unread.
                    if len(line) >= 64 * 1024:
                        while line and not line.endswith(b"\n"):
                            line = stream.readline(64 * 1024)
                        offset = stream.tell()
                    else:
                        offset = start
                    break
                offset = stream.tell()
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                with self.lock:
                    job = self.jobs[job_id]
                    if job.get("status") != "running":
                        return offset
                    kind = event.get("event")
                    if kind == "progress":
                        value = event.get("progress")
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            job["progress"] = max(0.0, min(0.999, float(value)))
                        if isinstance(event.get("message"), str):
                            job["message"] = event["message"][:1000]
                        changed = True
                    elif kind == "metric":
                        metric = {key: value for key, value in event.items() if key != "event" and isinstance(value, (int, float)) and math.isfinite(value)}
                        if metric:
                            job["metrics"] = (job["metrics"] + [metric])[-10000:]
                            changed = True
                    elif kind == "log" and isinstance(event.get("message"), str):
                        job["logs"] = (job["logs"] + [event["message"][:4000]])[-300:]
                        changed = True
        if changed:
            with self.lock:
                self._save()
        return offset

    def _public_result(self, job_id: str, result: Any) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(result, dict):
            raise ValueError("工作輸出必須是 JSON 物件。")
        directory = (self.job_root / job_id).resolve()
        allowed: list[str] = []

        def url(value: Any) -> str:
            if not isinstance(value, str):
                raise ValueError("工作輸出路徑必須是字串。")
            path = Path(value)
            path = (path if path.is_absolute() else directory / path).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or path.is_symlink():
                raise ValueError("工作輸出路徑無效或超出工作資料夾。")
            relative = path.relative_to(directory).as_posix()
            if relative in {"request.json", "events.jsonl", "worker.log", "result.json"}:
                raise ValueError("工作內部檔案不能當成結果發布。")
            allowed.append(relative)
            return f"/api/artifacts/{job_id}/{quote(relative, safe='/')}"

        public = copy.deepcopy(result)
        if "artifact_urls" in public:
            raise ValueError("工作應回傳 artifacts 路徑，由服務建立公開 URL。")
        for source, destination in (("media_path", "media_url"), ("checkpoint_path", "checkpoint_url")):
            if source in public:
                public[destination] = url(public.pop(source))
        artifacts = public.pop("artifacts", {})
        if artifacts:
            if not isinstance(artifacts, dict):
                raise ValueError("artifacts 必須是名稱與路徑對照表。")
            public["artifact_urls"] = {str(key): url(value) for key, value in artifacts.items()}
        # Worker must emit paths; accepting arbitrary remote URLs bypasses the
        # trusted artifact route and could make a result leak local data.
        for key in ("media_url", "checkpoint_url"):
            if key in public and not str(public[key]).startswith(f"/api/artifacts/{job_id}/"):
                raise ValueError("工作輸出不得包含外部媒體 URL。")
        try:
            json.dumps(public, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("工作輸出含有無效 JSON 值或非有限數字。") from exc
        return public, sorted(set(allowed))

    def artifact(self, job_id: str, filename: str) -> Path | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or filename not in job.get("_artifacts", []):
                return None
        directory = (self.job_root / job_id).resolve()
        path = (directory / filename).resolve()
        return path if path.is_relative_to(directory) and path.is_file() and not path.is_symlink() else None

    def checkpoint_for(self, job_id: str, model_id: str) -> Path:
        """Resolve only known, completed training outputs, never caller paths."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.get("kind") != "training" or job.get("status") != "completed":
                raise ValueError("只能使用已完成訓練工作的最佳權重。")
            if job.get("model_id") != model_id:
                raise ValueError("訓練權重與所選模型不相容。")
        directory = (self.job_root / job_id).resolve()
        if model_id in {"yolov12x", "yolo26x"}:
            checkpoint = directory / "best.pt"
            if checkpoint.is_symlink() or not checkpoint.is_file() or not checkpoint.resolve().is_relative_to(directory):
                raise ValueError("找不到此訓練工作的 best.pt。")
            return checkpoint.resolve()
        if model_id in {"vivit_base_k400", "x3d_l", "slowfast_r101_8x8", "videomaev2_giant_k710",
                        "grounding_dino_base", "rf_detr_2xl"}:
            checkpoint = directory / "best"
            if model_id == "rf_detr_2xl":
                required = ["model.pth", "workbench_training.json"]
            else:
                required = ["model.safetensors", "workbench_training.json",
                            "config.json" if model_id in {"vivit_base_k400", "grounding_dino_base"} else "model_config.json"]
            if checkpoint.is_symlink() or not checkpoint.is_dir() or not checkpoint.resolve().is_relative_to(directory):
                raise ValueError("找不到此訓練工作的 best 模型資料夾。")
            for name in required:
                path = checkpoint / name
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(checkpoint.resolve()):
                    raise ValueError(f"訓練模型缺少必要檔案：{name}")
            metadata = json.loads((checkpoint / "workbench_training.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("model_id") != model_id:
                raise ValueError("訓練權重中繼資料與所選模型不相容。")
            return checkpoint.resolve()
        raise ValueError("此模型不支援載入工作台的微調權重。")
