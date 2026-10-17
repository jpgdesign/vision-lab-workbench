"""Local HTTP API for the vision workbench; all model work runs in workers."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .datasets import DatasetError, FORMATS, extract_zip_safely, media_kind, validate_dataset
from .jobs import JobManager, write_json_atomic
from .model_registry import FILES as MODEL_FILES, ModelRegistry, registry_runtime

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_ROOT = APP_ROOT.parent / "vision_training_starter"
MAX_MEDIA_BYTES = 512 * 1024**2
MAX_ARCHIVE_BYTES = 2 * 1024**3
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class RequestSizeLimitMiddleware:
    """Bound multipart bodies before Starlette spools the entire file to disk."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        maximum = (MAX_MEDIA_BYTES if path == "/api/media" else MAX_ARCHIVE_BYTES if path == "/api/datasets/upload" else 1024**2) + 64 * 1024
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = maximum + 1
        if declared < 0 or declared > maximum:
            return await JSONResponse({"detail": "請求內容超過上傳容量限制。"}, status_code=413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > maximum:
                    raise HTTPException(413, "請求內容超過上傳容量限制。")
            return message

        await self.app(scope, limited_receive, send)


def get_models() -> list[dict[str, Any]]:
    from .catalog import get_models as load
    return load()


def validate_job_request(model_id: str, kind: str, params: dict[str, Any], *, allow_missing_weights=False) -> dict[str, Any]:
    from .catalog import validate_job_request as validate
    return validate(model_id, kind, params, allow_missing_weights=allow_missing_weights)


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["inference", "training"]
    model_id: str = Field(min_length=1, max_length=100)
    media_id: str | None = Field(default=None, max_length=100)
    dataset_id: str | None = Field(default=None, max_length=100)
    checkpoint_job_id: str | None = Field(default=None, max_length=100)
    checkpoint_id: str | None = Field(default=None, max_length=100)
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)
    format: Literal["yolo_detection", "imagefolder", "video_csv", "coco_detection"]
    name: str | None = Field(default=None, max_length=160)


class ModelPathInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=100)
    path: str = Field(min_length=1, max_length=4096)


class CheckpointInput(ModelPathInput):
    name: str | None = Field(default=None, min_length=1, max_length=160)


class ModelScanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roots: list[str] = Field(default_factory=list, max_length=12)


class LicenseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool = Field(strict=True)


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.items: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.items = loaded
            except (OSError, ValueError):
                path.replace(path.with_name(f"{path.stem}.corrupt-{int(time.time())}.json"))

    def add(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.items[entry["id"]] = copy.deepcopy(entry)
            write_json_atomic(self.path, self.items)
            return copy.deepcopy(entry)

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            return copy.deepcopy(self.items.get(key))

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return copy.deepcopy(list(self.items.values()))


def _media_public(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "path"}


def _dataset_public(entry: dict[str, Any]) -> dict[str, Any]:
    # A list response should not transfer tens of thousands of absolute samples.
    result = copy.deepcopy(entry)
    result["details"] = {k: v for k, v in result.get("details", {}).items() if k not in {"train_samples", "val_samples", "train_images", "val_images"}}
    return result


def _new_media(path: Path, source: str, name: str | None = None, media_id: str | None = None) -> dict[str, Any]:
    kind = media_kind(path)
    media_id = media_id or uuid.uuid4().hex
    entry = {"id": media_id, "name": name or path.name, "kind": kind, "url": f"/api/media/{media_id}/content", "source": source, "path": str(path.resolve()), "bytes": path.stat().st_size}
    if kind == "image":
        from PIL import Image
        with Image.open(path) as image:
            entry.update(width=image.width, height=image.height)
    return entry


def _discover_samples(root: Path, registry: Registry) -> None:
    if not root.is_dir():
        return
    # Only the original sample folders; metadata and generated contact sheets
    # are intentionally excluded from the default input picker.
    for folder in (root / "images", root / "video"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                continue
            media_id = "sample-" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
            try:
                registry.add(_new_media(path, "sample", media_id=media_id))
            except (OSError, DatasetError):
                continue


def _default_sample_root() -> Path:
    bundled = APP_ROOT / "examples" / "media"
    return bundled if bundled.is_dir() else DEFAULT_SAMPLE_ROOT


def _bundled_examples() -> list[dict[str, Any]]:
    path = APP_ROOT / "examples" / "finetuning" / "coco10_yolo" / "dataset.yaml"
    available = path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(APP_ROOT.resolve())
    return [{"id": "coco10_yolo", "name": "COCO 10 圖教學資料集", "format": "yolo_detection",
             "path": str(path.resolve()), "available": available,
             "description": "10 張 COCO 圖片與物件框標註，用於練習資料驗證及微調流程；不適合評估泛化能力。",
             "recommended_params": {"epochs": 1, "batch_size": 1, "workers": 0}}]


def _gpu_info() -> dict[str, Any]:
    try:
        options = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2, **options)
        if result.returncode == 0 and result.stdout.strip():
            name, memory = result.stdout.splitlines()[0].rsplit(",", 1)
            return {"name": name.strip(), "vram_mb": int(memory.strip()), "available": True}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"name": "CPU", "vram_mb": 0, "available": False}


def create_app(runtime_dir: Path | None = None, sample_root: Path | None = None, *, start_worker: bool = True, command_factory=None) -> FastAPI:
    runtime = (runtime_dir or Path(os.environ.get("VISION_WORKBENCH_RUNTIME", str(APP_ROOT / "runtime")))).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "media").mkdir(exist_ok=True)
    (runtime / "datasets").mkdir(exist_ok=True)
    media = Registry(runtime / "media.json")
    datasets = Registry(runtime / "datasets.json")
    manager = JobManager(runtime, APP_ROOT, command_factory=command_factory)
    model_registry = ModelRegistry(runtime, APP_ROOT)

    def load_models():
        with registry_runtime(runtime):
            return get_models()

    def normalize_params(model_id, kind, params, *, allow_missing_weights=False):
        with registry_runtime(runtime):
            if allow_missing_weights:
                return validate_job_request(model_id, kind, params, allow_missing_weights=True)
            return validate_job_request(model_id, kind, params)

    def request_weights(model_id, checkpoint_id=None, checkpoint_job_id=None):
        if checkpoint_id and checkpoint_job_id:
            raise ValueError("checkpoint_id 與 checkpoint_job_id 只能指定一個。")
        if checkpoint_job_id:
            return {"checkpoint_job_id": checkpoint_job_id, "checkpoint_path": str(manager.checkpoint_for(checkpoint_job_id, model_id))}
        if checkpoint_id:
            return {"checkpoint_id": checkpoint_id, "checkpoint_path": str(model_registry.checkpoint_for(checkpoint_id, model_id, manager))}
        if model_id in MODEL_FILES:
            return {"model_weight_path": str(model_registry.resolve(model_id))}
        return {}
    sample_root = sample_root if sample_root is not None else _default_sample_root()
    _discover_samples(sample_root, media)
    gpu_cache: dict[str, Any] = {"at": 0.0, "value": {"name": "CPU", "vram_mb": 0, "available": False}}

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if start_worker:
            manager.start()
        yield
        await run_in_threadpool(manager.close)

    application = FastAPI(title="Vision Workbench", version="0.1.0", lifespan=lifespan)
    application.add_middleware(RequestSizeLimitMiddleware)
    application.state.jobs = manager
    application.state.media = media
    application.state.datasets = datasets
    application.state.runtime = runtime
    application.state.model_registry = model_registry

    @application.middleware("http")
    async def local_only(request: Request, call_next):
        try:
            host = urlparse("//" + request.headers.get("host", "")).hostname
        except ValueError:
            host = None
        if host not in LOOPBACK_HOSTS:
            return JSONResponse({"detail": "此服務僅接受 localhost 存取。"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            try:
                parsed = urlparse(origin)
                allowed = parsed.scheme in {"http", "https"} and parsed.hostname in LOOPBACK_HOSTS and not parsed.username and not parsed.password
            except ValueError:
                allowed = False
            if not allowed:
                return JSONResponse({"detail": "禁止外部網站操作本機模型服務。"}, status_code=403)
        # Browser no-CORS requests can omit Origin. Fetch metadata closes that
        # channel while retaining normal CLI clients and loopback development.
        if request.headers.get("sec-fetch-site") == "cross-site" and not origin:
            return JSONResponse({"detail": "禁止跨網站存取本機模型服務。"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.get("/api/health")
    def health():
        if time.monotonic() - gpu_cache["at"] > 60:
            gpu_cache.update(at=time.monotonic(), value=_gpu_info())
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        return {"status": "ok", "gpu": gpu_cache["value"], "queue": manager.counts(), "runtime": {"python": sys.version.split()[0], "torch": torch_version}}

    @application.get("/api/models")
    def models():
        return {"models": load_models()}

    @application.get("/api/system")
    def system_info():
        from .system_info import collect_system_info
        return collect_system_info(APP_ROOT)

    @application.post("/api/model-licenses/{model_id}")
    def accept_license(model_id: str, payload: LicenseInput):
        if payload.accepted is not True:
            raise HTTPException(422, "必須明確確認 accepted: true 才能儲存授權確認。")
        try:
            from .model_licenses import accept_model_license
            with registry_runtime(runtime):
                record = accept_model_license(model_id)
            return {"model_id": model_id, "license_accepted": True, "license": record,
                    "model": next(model for model in load_models() if model["id"] == model_id)}
        except (OSError, ValueError, StopIteration) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/model-files")
    def model_files():
        return model_registry.snapshot(manager)

    @application.post("/api/model-files/scan")
    def scan_model_files(payload: ModelScanInput):
        try:
            return model_registry.scan(payload.roots, manager)
        except (ValueError, OSError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/model-files/bind")
    def bind_model_file(payload: ModelPathInput):
        try:
            return model_registry.bind(payload.model_id, payload.path)
        except (ValueError, OSError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.delete("/api/model-files/bind/{model_id}")
    def unbind_model_file(model_id: str):
        try:
            return model_registry.unbind(model_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/checkpoints")
    def checkpoints():
        return {"checkpoints": model_registry.checkpoints(manager)}

    @application.post("/api/checkpoints", status_code=201)
    def register_checkpoint(payload: CheckpointInput):
        try:
            return model_registry.add_checkpoint(payload.model_id, payload.path, payload.name)
        except (ValueError, OSError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.delete("/api/checkpoints/{checkpoint_id}")
    def unregister_checkpoint(checkpoint_id: str):
        try:
            return model_registry.remove_checkpoint(checkpoint_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/media")
    def list_media():
        return {"media": [_media_public(item) for item in media.list() if Path(item.get("path", "")).is_file()]}

    async def save_upload(file: UploadFile, path: Path, max_bytes: int) -> None:
        total = 0
        try:
            with path.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(413, f"上傳檔案超過 {max_bytes // 1024**2} MB 限制。")
                    await run_in_threadpool(output.write, chunk)
            if total == 0:
                raise HTTPException(422, "上傳檔案不可為空。")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @application.post("/api/media", status_code=201)
    async def upload_media(file: UploadFile = File(...)):
        filename = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1][:200]
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            await file.close()
            raise HTTPException(422, "不支援的媒體副檔名。")
        if file.content_type and not (file.content_type.startswith("image/") or file.content_type.startswith("video/") or file.content_type == "application/octet-stream"):
            await file.close()
            raise HTTPException(422, "上傳內容 MIME 必須是影像或影片。")
        media_id = uuid.uuid4().hex
        path = runtime / "media" / f"{media_id}{suffix}"
        await save_upload(file, path, MAX_MEDIA_BYTES)
        try:
            entry = await run_in_threadpool(_new_media, path, "upload", filename, media_id)
            await run_in_threadpool(media.add, entry)
            return _media_public(entry)
        except (DatasetError, OSError) as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/media/{media_id}/content")
    def media_content(media_id: str):
        entry = media.get(media_id)
        if not entry:
            raise HTTPException(404, "找不到媒體。")
        path = Path(entry["path"])
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "媒體檔案已不存在。")
        # Only paths registered after validation are exposed; IDs are opaque.
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream", filename=entry["name"], content_disposition_type="inline")

    @application.get("/api/datasets")
    def list_datasets():
        return {"datasets": [_dataset_public(item) for item in datasets.list()]}

    @application.get("/api/examples")
    def examples():
        # Templates describe local inputs only. Registration and training
        # remain explicit actions through the usual validation/job endpoints.
        return {"datasets": _bundled_examples()}

    def register_dataset(result: dict[str, Any]) -> dict[str, Any]:
        if result["valid"]:
            dataset = result["dataset"]
            # Stable ID permits revalidation without proliferating duplicates.
            dataset["id"] = hashlib.sha256(f"{dataset['format']}:{dataset['path']}".encode()).hexdigest()[:32]
            datasets.add(dataset)
            result = {**result, "dataset": _dataset_public(dataset)}
        return result

    @application.post("/api/datasets/validate")
    def validate_data(payload: DatasetInput):
        return register_dataset(validate_dataset(payload.path, payload.format, payload.name))

    @application.post("/api/datasets/upload")
    async def upload_dataset(file: UploadFile = File(...), format: str = Form(...)):
        if format not in FORMATS:
            await file.close()
            raise HTTPException(422, "不支援的資料集格式。")
        if not (file.filename or "").lower().endswith(".zip"):
            await file.close()
            raise HTTPException(422, "資料集上傳目前接受 ZIP。")
        upload_id = uuid.uuid4().hex
        archive = runtime / "datasets" / f"{upload_id}.zip"
        destination = runtime / "datasets" / upload_id
        await save_upload(file, archive, MAX_ARCHIVE_BYTES)
        try:
            extracted = await run_in_threadpool(extract_zip_safely, archive, destination)
            result = await run_in_threadpool(validate_dataset, extracted, format, Path(file.filename or "dataset").stem, destination)
            if not result["valid"]:
                await run_in_threadpool(shutil.rmtree, destination)
                return result
            result["dataset"]["allowed_root"] = str(destination.resolve())
            return await run_in_threadpool(register_dataset, result)
        except (DatasetError, OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
            if destination.exists():
                await run_in_threadpool(shutil.rmtree, destination)
            return {"valid": False, "errors": [str(exc)], "warnings": []}
        finally:
            archive.unlink(missing_ok=True)

    @application.post("/api/jobs", status_code=201)
    def create_job(payload: JobInput):
        model = next((model for model in load_models() if model.get("id") == payload.model_id), None)
        if not model:
            raise HTTPException(404, "找不到模型。")
        has_checkpoint = bool(payload.checkpoint_id or payload.checkpoint_job_id)
        if has_checkpoint and payload.kind != "inference":
            raise HTTPException(422, "微調權重僅供推論比較使用。")
        if not model.get("available") and not (has_checkpoint and model.get("environment_available")):
            raise HTTPException(409, model.get("reason") or "本機模型尚未準備完成。")
        if not model.get(f"{payload.kind}_supported"):
            raise HTTPException(422, model.get(f"{payload.kind}_reason") or "此模型目前未提供這項功能。")
        try:
            weight_fields = request_weights(payload.model_id, payload.checkpoint_id, payload.checkpoint_job_id)
            params = normalize_params(payload.model_id, payload.kind, payload.params, allow_missing_weights=has_checkpoint and not model.get("available"))
        except (ValueError, TypeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        request: dict[str, Any] = {"kind": payload.kind, "model_id": payload.model_id, "params": params, **weight_fields}
        if payload.kind == "inference":
            if not payload.media_id or payload.dataset_id:
                raise HTTPException(422, "推論需要 media_id，且不能同時指定 dataset_id。")
            item = media.get(payload.media_id)
            if not item or not Path(item["path"]).is_file():
                raise HTTPException(404, "請先選擇或上傳存在的媒體。")
            if item["kind"] not in model.get("capabilities", {}).get("input_types", []):
                raise HTTPException(422, "這個模型不支援所選媒體類型。")
            request.update(media_id=payload.media_id, media_path=item["path"])
        else:
            if payload.checkpoint_job_id:
                raise HTTPException(422, "checkpoint_job_id 僅供推論比較微調結果使用。")
            if not payload.dataset_id or payload.media_id:
                raise HTTPException(422, "訓練需要 dataset_id，且不能同時指定 media_id。")
            dataset = datasets.get(payload.dataset_id)
            if not dataset:
                raise HTTPException(404, "請先驗證或上傳資料集。")
            supported = model.get("capabilities", {}).get("training", {}).get("dataset_formats", [])
            if dataset["format"] not in supported:
                raise HTTPException(422, f"此模型訓練支援的格式：{', '.join(supported)}。")
            # Revalidate immediately before enqueuing; files may have changed
            # after a previous validation or after a server restart.
            allowed_root = Path(dataset["allowed_root"]) if dataset.get("allowed_root") else None
            validated = validate_dataset(dataset["path"], dataset["format"], dataset["name"], allowed_root)
            if not validated["valid"]:
                raise HTTPException(422, {"message": "資料集重新驗證失敗", "errors": validated["errors"]})
            dataset = {**validated["dataset"], "id": payload.dataset_id}
            if allowed_root:
                dataset["allowed_root"] = str(allowed_root)
            datasets.add(dataset)
            request.update(dataset_id=payload.dataset_id, dataset=dataset)
        try:
            return manager.submit(request)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get("/api/jobs")
    def list_jobs():
        return {"jobs": manager.list()}

    @application.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = manager.get(job_id)
        if not job:
            raise HTTPException(404, "找不到工作。")
        return job

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = manager.cancel(job_id)
        if not job:
            raise HTTPException(404, "找不到工作。")
        return job

    @application.get("/api/artifacts/{job_id}/{filename:path}")
    def artifact(job_id: str, filename: str):
        path = manager.artifact(job_id, filename)
        if not path:
            raise HTTPException(404, "找不到此工作的輸出檔案。")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        inline = media_type.startswith("image/") or media_type.startswith("video/")
        return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type="inline" if inline else "attachment")

    from .live import register_live_routes
    register_live_routes(application, manager, load_models, normalize_params, request_weights)

    # Mount last so static frontend cannot swallow /api routes. SPA fallback is
    # unnecessary because the canvas uses in-page tabs instead of path routing.
    distribution = APP_ROOT / "frontend" / "dist"
    if distribution.is_dir():
        application.mount("/", StaticFiles(directory=distribution, html=True), name="frontend")
    else:
        @application.get("/")
        def frontend_pending():
            return {"message": "前端尚未建置，請依 README 啟動開發模式。", "api": "/docs"}
    return application


app = create_app()
