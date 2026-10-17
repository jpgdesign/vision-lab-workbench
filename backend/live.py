"""Bounded, local-only camera ingress for one persistent GPU queue worker."""
from __future__ import annotations

import copy
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .jobs import TERMINAL
from .live_limits import DEFAULT_CAPTURE_FPS, FRAME_RING_SIZE, MAX_CAPTURE_FPS

MAX_FRAME_BYTES = 1024 * 1024
MAX_FRAME_PIXELS = 1_000_000
MAX_FRAME_SIDE = 1600


class LiveModelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=100)
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint_job_id: str | None = Field(default=None, max_length=100)
    checkpoint_id: str | None = Field(default=None, max_length=100)


class LiveSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[LiveModelInput] = Field(min_length=1, max_length=4)
    capture_fps: int = Field(default=DEFAULT_CAPTURE_FPS, ge=1, le=MAX_CAPTURE_FPS, strict=True)
    max_duration_seconds: int | None = Field(default=900, ge=1, strict=True)


def cleanup_live_inputs(job_root: Path, job_id: str) -> None:
    """Remove only this registered job's ephemeral camera frames."""
    root = job_root.resolve()
    directory = root / job_id
    target = directory / "live_input"
    if directory.parent != root or directory.is_symlink() or target.is_symlink():
        return
    if not target.resolve().is_relative_to(root):
        return
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


def inspect_frame(content: bytes) -> tuple[int, int]:
    if not content or len(content) > MAX_FRAME_BYTES:
        raise HTTPException(413, "攝影機單張 JPEG 不可超過 1 MB。")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "JPEG" or getattr(image, "is_animated", False):
                raise HTTPException(422, "即時影格只接受單張 JPEG。")
            width, height = image.size
            if width < 1 or height < 1 or max(width, height) > MAX_FRAME_SIDE or width * height > MAX_FRAME_PIXELS:
                raise HTTPException(422, "影格最多 1600 像素長邊、100 萬像素；建議 640×480 或 1280×720。")
            image.load()
            return width, height
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "無法讀取 JPEG 影格。") from exc


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    # This bounded manifest is updated for every camera frame. Compact JSON
    # avoids repeatedly serializing/writing indentation at 30 FPS.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    # Windows readers briefly hold a file handle during read_text. A short
    # bounded retry avoids dropping an upload when replacement overlaps it.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.005)


def _retire_frames(folder: Path, filenames: list[str]) -> list[str]:
    pending = []
    for filename in filenames:
        try:
            (folder / filename).unlink(missing_ok=True)
        except OSError:
            # A decoder can hold an old JPEG briefly on Windows. Retry next
            # upload, with a strict cap so failed deletions cannot grow disk.
            pending.append(filename)
    return pending


def register_live_routes(application, manager, get_models, validate_job_request, request_weights=None) -> None:
    # Protected with manager.lock so terminal-state changes and disk writes
    # cannot race: a cancelled session cannot recreate its input directory.
    ingress: dict[str, dict[str, Any]] = {}

    def live_job(session_id: str) -> dict[str, Any]:
        job = manager.get(session_id)
        if not job or job.get("kind") != "live":
            raise HTTPException(404, "找不到即時攝影機工作。")
        return job

    def image_path(session_id: str, index: int) -> Path:
        job = live_job(session_id)
        if index < 0 or index >= len(job.get("models", [])) or index > 3:
            raise HTTPException(404, "找不到此模型的即時畫面。")
        directory = manager.job_root / session_id
        path = directory / f"result_{index}.jpg"
        if directory.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(manager.job_root.resolve()) or not path.is_file():
            raise HTTPException(404, "模型尚未產生即時畫面。")
        return path

    @application.post("/api/live/sessions", status_code=201)
    def create_session(payload: LiveSessionInput):
        with manager.lock:
            for session_id in list(ingress):
                if manager.jobs.get(session_id, {}).get("status") in TERMINAL:
                    ingress.pop(session_id, None)
        catalog = {model["id"]: model for model in get_models()}
        models = []
        for index, selected in enumerate(payload.models):
            model = catalog.get(selected.model_id)
            if not model:
                raise HTTPException(404, "找不到模型。")
            has_checkpoint = bool(selected.checkpoint_id or selected.checkpoint_job_id)
            if not model.get("available") and not (has_checkpoint and model.get("environment_available")):
                raise HTTPException(409, model.get("reason") or "本機模型尚未準備完成。")
            if not model.get("inference_supported") or "video" not in model.get("capabilities", {}).get("input_types", []):
                raise HTTPException(422, "此模型不支援即時攝影機輸入。")
            try:
                if selected.checkpoint_id and selected.checkpoint_job_id:
                    raise ValueError("checkpoint_id 與 checkpoint_job_id 只能指定一個。")
                params = (validate_job_request(selected.model_id, "inference", selected.params, allow_missing_weights=True)
                          if has_checkpoint and not model.get("available") else validate_job_request(selected.model_id, "inference", selected.params))
                item = {"key": str(index), "model_id": selected.model_id, "params": params}
                if request_weights:
                    item.update(request_weights(selected.model_id, selected.checkpoint_id, selected.checkpoint_job_id))
                elif selected.checkpoint_job_id:
                    checkpoint = manager.checkpoint_for(selected.checkpoint_job_id, selected.model_id)
                    item.update(checkpoint_job_id=selected.checkpoint_job_id, checkpoint_path=str(checkpoint))
            except (ValueError, TypeError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc
            models.append(item)
        request = {"kind": "live", "model_id": models[0]["model_id"], "models": models,
                   "params": {}, "capture_fps": payload.capture_fps,
                   "max_duration_seconds": payload.max_duration_seconds}
        try:
            return manager.submit(request)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.post("/api/live/sessions/{session_id}/frames", status_code=202)
    async def upload_frame(session_id: str, frame: UploadFile = File(...)):
        try:
            live_job(session_id)
            content = await frame.read(MAX_FRAME_BYTES + 1)
        finally:
            await frame.close()
        width, height = inspect_frame(content)
        with manager.lock:
            job = live_job(session_id)
            if job["status"] in TERMINAL:
                ingress.pop(session_id, None)
                raise HTTPException(409, "即時工作已結束，請重新開始。")
            directory = manager.job_root / session_id
            if (directory / "stop.signal").exists():
                raise HTTPException(409, "即時工作正在停止。")
            now = time.monotonic()
            fps = job["capture_fps"]
            state = ingress.setdefault(session_id, {"sequence": 0, "frames": [], "retired": [], "at": now, "tokens": 3.0})
            # Three-frame burst tolerates normal browser scheduling jitter.
            state["tokens"] = min(3.0, state["tokens"] + (now - state["at"]) * fps)
            state["at"] = now
            if state["tokens"] < 1:
                raise HTTPException(429, "影格上傳過快，請依設定 FPS 傳送。", headers={"Retry-After": "1"})
            state["tokens"] -= 1
            sequence = state["sequence"] + 1
            folder = directory / "live_input"
            folder.mkdir(exist_ok=True)
            state["retired"] = _retire_frames(folder, state["retired"])
            if len(state["retired"]) >= 8:
                raise HTTPException(503, "影格暫存檔正在使用中，請稍後重試。")
            filename = f"frame_{sequence:012d}.jpg"
            target = folder / filename
            temporary = target.with_suffix(".jpg.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, target)
            frames = state["frames"] + [{"sequence": sequence, "received_at": now, "filename": filename,
                                         "width": width, "height": height}]
            expired = [item["filename"] for item in frames[:-FRAME_RING_SIZE]]
            frames = frames[-FRAME_RING_SIZE:]
            try:
                _write_manifest(folder / "input.json", {"sequence": sequence, "received_at": now, "frames": frames})
            except OSError as exc:
                state["retired"].append(filename)
                raise HTTPException(503, "影格暫存更新失敗，請稍後重試。") from exc
            state.update(sequence=sequence, frames=frames)
            state["retired"] = _retire_frames(folder, state["retired"] + expired)
        return {"accepted": True, "sequence": sequence, "received_at": now}

    @application.get("/api/live/sessions/{session_id}")
    def session_status(session_id: str):
        job = live_job(session_id)
        path = manager.job_root / session_id / "live_state.json"
        live_state = {"status": job["status"], "models": []}
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024**2:
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    live_state = parsed
            except (OSError, ValueError):
                pass
        # Only fixed job-owned annotated JPEGs are published; file paths in
        # worker state are never accepted as a download source.
        enriched = []
        for state in live_state.get("models", []):
            if not isinstance(state, dict):
                continue
            state = copy.deepcopy(state)
            state.pop("image_url", None)
            key = state.get("key")
            if str(key) in {str(i) for i in range(len(job.get("models", [])))}:
                try:
                    image_path(session_id, int(key))
                    sequence = state.get("sequence", 0)
                    sequence = sequence if isinstance(sequence, int) and sequence >= 0 else 0
                    state["image_url"] = f"/api/live/sessions/{session_id}/images/{key}?v={sequence}"
                except HTTPException:
                    pass
            enriched.append(state)
        live_state["models"] = enriched
        if job["status"] in TERMINAL:
            # A killed/crashed worker cannot publish its final state. The
            # queue's terminal status must override its last running snapshot.
            live_state["status"] = job["status"]
            if job["status"] in {"failed", "cancelled"}:
                live_state["message"] = job.get("error") or job.get("message", "即時工作已停止。")
            for state in enriched:
                state["status"] = job["status"]
                if job["status"] in {"failed", "cancelled"}:
                    state["message"] = live_state["message"]
            with manager.lock:
                ingress.pop(session_id, None)
        return {**job, "live_state": live_state}

    @application.get("/api/live/sessions/{session_id}/images/{index}")
    def session_image(session_id: str, index: int):
        try:
            content = image_path(session_id, index).read_bytes()
        except OSError as exc:
            raise HTTPException(404, "模型畫面正在更新，請稍後重試。") from exc
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})

    @application.post("/api/live/sessions/{session_id}/stop")
    def stop_session(session_id: str):
        with manager.lock:
            job = live_job(session_id)
            ingress.pop(session_id, None)
            if job["status"] in TERMINAL:
                return job
            if job["status"] == "queued":
                return manager.cancel(session_id, "即時攝影機工作已停止。")
            (manager.job_root / session_id / "stop.signal").write_text("stop", encoding="utf-8")
            manager.jobs[session_id]["message"] = "正在停止即時攝影機，完成目前推論後釋放 GPU。"
            manager._save()
            return manager.get(session_id)
