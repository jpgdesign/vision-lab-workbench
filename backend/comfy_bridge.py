"""ComfyUI graph frontend integration, sharing this service's model environment."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
from typing import Any, Literal

from fastapi import Body, HTTPException, Request
from pydantic import Field

from .jobs import write_json_atomic
from .workflow_api import _Input, _parse, _linked, WorkflowOptions
from .workflow_graph import validate_graph
from .workflow_samples import with_samples
from .workflow_templates import templates


class LaunchInput(_Input):
    graph: dict[str, Any]
    source_mode: Literal["camera", "vdo"]
    name: str = Field(default="ComfyUI 即時視覺工作流", min_length=1, max_length=160)
    media_id: str | None = Field(default=None, max_length=100)
    options: WorkflowOptions = Field(default_factory=WorkflowOptions)


def register_comfy_routes(application, media, runtime, load_models=lambda: [], load_checkpoints=lambda: []):
    runtime = Path(runtime).resolve()
    directory = runtime / "comfy_workflows"

    def folder():
        if directory.exists() and _linked(directory):
            raise HTTPException(409, "ComfyUI 工作流目錄不能是連結。")
        directory.mkdir(exist_ok=True)
        return directory

    @application.get("/api/comfy/status")
    def status():
        import requests
        from comfy_manager import load_comfy_config
        installation = Path.home() / "ComfyUI_windows_portable" / "ComfyUI"
        configuration_error = None
        try:
            config = load_comfy_config(Path(__file__).resolve().parents[1])
            if config:
                installation = Path(config["root"])
        except (ValueError, OSError):
            configuration_error = "ComfyUI 安裝設定無效，請重新執行安裝工具或指定既有路徑。"
        online, version = False, None
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get("http://127.0.0.1:8188/system_stats", timeout=1, allow_redirects=False)
                if response.status_code == 200:
                    online = True
                    version = response.json().get("system", {}).get("comfyui_version")
        except (requests.RequestException, ValueError, TypeError):
            pass
        return {"installed": (installation / "main.py").is_file(), "online": online,
                "version": version, "url": "http://127.0.0.1:8188/",
                "plugin_installed": (installation / "custom_nodes/vision_workbench/__init__.py").is_file(),
                "configuration_error": configuration_error}

    @application.get("/api/comfy/templates")
    def get_templates(source_mode: Literal["media", "camera", "vdo"] = "media"):
        from .comfy_workflows import convert_workflow
        entries = media.list()
        values = with_samples(templates(), entries)["templates"]
        resources = {"media": entries, "models": load_models(), "checkpoints": load_checkpoints()}
        return {"templates": [{**entry, "workflow": convert_workflow(
            entry["graph"], name=entry["name"], source_mode=source_mode,
            media_id=(entry.get("sample") or {}).get("media_id") or "",
            options=entry.get("recommended_options") or {}, resources=resources)} for entry in values]}

    @application.post("/api/comfy/workflows/launch", status_code=201)
    def launch(request: Request, payload: Any = Body(...)):
        value = _parse(LaunchInput, payload)
        checked = validate_graph(value.graph)
        if not checked["valid"]:
            raise HTTPException(422, {"message": "ComfyUI 工作流連線或參數無效。", "errors": checked["errors"]})
        # The camera is never opened by this endpoint. The embedded browser
        # viewer performs normal live preflight and requires a connect/start click.
        path = folder()
        if len(list(path.glob("*.json"))) >= 4096:
            raise HTTPException(409, "ComfyUI 即時工作流設定已達上限，請清理不再使用的設定。")
        ident = uuid.uuid4().hex
        entry = {"id": ident, "name": value.name, "graph": checked["graph"],
                 "source_mode": value.source_mode, "options": value.options.model_dump(),
                 "created_at": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(path / f"{ident}.json", entry)
        return {"workflow_id": ident, "viewer_url": str(request.base_url).rstrip("/") +
                f"/?embedded_workflow={ident}&source={value.source_mode}"}

    @application.get("/api/comfy/workflows/{workflow_id}")
    def get_workflow(workflow_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", workflow_id):
            raise HTTPException(404, "找不到 ComfyUI 工作流。")
        path = folder() / f"{workflow_id}.json"
        try:
            if not path.is_file() or _linked(path) or path.stat().st_size > 600_000:
                raise ValueError("invalid file")
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry["id"] != workflow_id or entry["source_mode"] not in {"camera", "vdo"}:
                raise ValueError("invalid workflow")
            return entry
        except (OSError, ValueError, TypeError, KeyError):
            raise HTTPException(404, "找不到 ComfyUI 工作流。") from None
