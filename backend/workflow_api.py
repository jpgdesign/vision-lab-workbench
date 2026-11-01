"""Local workflow CRUD/preflight/queue endpoints; never load model runtimes."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import stat
import threading
from typing import Any, Literal
import uuid

from fastapi import Body, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import workflow_graph, workflow_integrations
from .dual_branch import is_bundle
from .jobs import write_json_atomic
from .live_limits import DEFAULT_CAPTURE_FPS, MAX_CAPTURE_FPS
from .workflow_rules import KINDS as RULE_KINDS, RuleNode
from .workflow_templates import templates
from .workflow_samples import with_samples


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WorkflowOptions(_Input):
    device: Literal["auto", "cpu", "cuda:0"] = "auto"
    frame_stride: int = Field(default=3, ge=1, le=60)
    max_frames: int = Field(default=300, ge=1, le=3000)
    max_side: int = Field(default=1280, ge=256, le=1600)


class SaveInput(_Input):
    name: str = Field(min_length=1, max_length=160)
    graph: dict[str, Any]


class ValidateInput(_Input):
    graph: dict[str, Any]
    media_id: str | None = Field(default=None, min_length=1, max_length=100)
    options: WorkflowOptions = Field(default_factory=WorkflowOptions)


class RunInput(_Input):
    graph: dict[str, Any]
    media_id: str = Field(min_length=1, max_length=100)
    options: WorkflowOptions = Field(default_factory=WorkflowOptions)


class LiveInput(_Input):
    graph: dict[str, Any]
    options: WorkflowOptions = Field(default_factory=WorkflowOptions)
    capture_fps: int = Field(default=DEFAULT_CAPTURE_FPS, ge=1, le=MAX_CAPTURE_FPS)
    max_duration_seconds: int | None = Field(default=900, ge=1)


class IntegrationInput(_Input):
    key_file_path: str = Field(min_length=1, max_length=4096)
    model_id: str = Field(default="", max_length=200)


def _parse(schema, payload):
    # Do not echo invalid input values: callers may accidentally send a secret
    # in a forbidden field. FastAPI's default validation includes those values.
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(422, {"message": "請求格式不正確。", "errors": [
            {"field": ".".join(str(x) for x in item["loc"]), "message": item["msg"]}
            for item in exc.errors(include_url=False, include_input=False)]}) from None


def _linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _dependency_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _integration_view(value):
    item = value.get("munolith", {}) if isinstance(value, dict) else {}
    allowed = ("configured", "key_file_configured", "model_id", "models", "available", "error", "checked_at")
    return {"munolith": {**{key: item[key] for key in allowed if key in item}, "base_url": workflow_integrations.BASE_URL}}


def register_workflow_routes(application, manager, media, load_models, normalize_params, request_weights, runtime):
    runtime = Path(runtime).resolve()
    saved_directory = runtime / "workflows"
    lock = threading.RLock()

    def directory():
        if saved_directory.exists() and _linked(saved_directory):
            raise HTTPException(409, "工作流儲存目錄不能是連結。")
        saved_directory.mkdir(exist_ok=True)
        if saved_directory.resolve().parent != runtime:
            raise HTTPException(409, "工作流儲存目錄無效。")
        return saved_directory

    def saved_entry(path):
        try:
            if not path.is_file() or _linked(path) or path.stat().st_size > 600000:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or value.get("id") != path.stem
                    or not re.fullmatch(r"[0-9a-f]{32}", path.stem)
                    or not isinstance(value.get("name"), str) or not value["name"].strip() or len(value["name"]) > 160):
                return None
            graph = workflow_graph.normalize_graph(value.get("graph"))
            return {"id": value["id"], "name": value["name"], "graph": graph,
                    "created_at": value.get("created_at", "")}
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def integrations():
        try:
            return _integration_view(workflow_integrations.public_config(runtime))
        except (OSError, ValueError, TypeError, KeyError):
            return {"munolith": {"configured": False, "key_file_configured": False, "available": False,
                                  "base_url": workflow_integrations.BASE_URL, "model_id": "", "models": [],
                                  "error": "連線設定無法讀取，請重新設定。", "checked_at": None}}

    def preflight(graph, options, media_id=None, *, live=False):
        checked = workflow_graph.validate_graph(graph)
        if not checked["valid"]:
            # Invalid normalization can leave the caller's raw graph in the
            # graph validator result; never echo forbidden fields from it.
            checked["graph"] = None
            return checked, {}, None
        graph = checked["graph"]
        errors, warnings = checked["errors"], checked["warnings"]
        resolved, node_plan = {}, []
        item = None
        if not live and media_id is not None:
            item = media.get(media_id)
            if (not isinstance(item, dict) or item.get("kind") not in {"image", "video"}
                    or not isinstance(item.get("path"), str) or not Path(item["path"]).is_file()):
                errors.append("請選擇已登錄且仍存在的圖片或影片。")
                item = None
        elif not live:
            warnings.append("尚未選擇素材；目前只預檢圖與模型，執行時仍需有效的 media_id。")
        try:
            models = {model["id"]: model for model in load_models()}
        except (OSError, ValueError, TypeError, KeyError):
            models = {}
            errors.append("無法讀取本機模型清單。")
        for node in graph["nodes"]:
            ident, kind, params = node["id"], node["type"], node["params"]
            public_node = {"id": ident, "type": kind}
            node_plan.append(public_node)
            if kind.startswith("cv_"):
                public_node["device"] = "cpu"
                if not _dependency_available("cv2"):
                    errors.append(f"{ident} 尚未安裝 OpenCV；請安裝 requirements-base.txt 的依賴套件。")
                if item and item["kind"] == "image" and kind in {"cv_background", "cv_optical_flow", "cv_histogram", "cv_track_motion"}:
                    warnings.append(f"{ident} 需要連續影格；單張圖片只能初始化，動態判定會保持未知。")
            if kind in RULE_KINDS:
                try:
                    RuleNode(kind, params, ident)
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{ident} 規則參數不正確：{exc}")
            if kind == "tracker":
                labels = params.get("labels")
                if not isinstance(labels, list) or any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in labels):
                    errors.append(f"{ident} 追蹤類別必須是非空字串清單。")
            if kind in {"detector", "video_action", "embedding"}:
                model_id, checkpoint_id = params["model_id"], params.get("checkpoint_id") or None
                public_node.update(model_id=model_id, checkpoint_id=checkpoint_id)
                model = models.get(model_id)
                tasks = {"detection", "segmentation"} if kind == "detector" else {"embedding"} if kind == "embedding" else {"action"}
                if not model or model.get("task") not in tasks or not model.get("inference_supported"):
                    errors.append(f"{ident} 所選模型不支援這個節點的工作類型。")
                    continue
                if not model.get("available") and not (checkpoint_id and model.get("environment_available")):
                    errors.append(f"{ident} 模型權重或執行環境尚未準備完成。")
                    continue
                try:
                    weights = request_weights(model_id, checkpoint_id)
                    path = Path(weights.get("checkpoint_path") or weights.get("model_weight_path") or "")
                    if not (weights.get("checkpoint_path") or weights.get("model_weight_path")) or not path.exists():
                        raise ValueError("missing weight")
                    if is_bundle(weights.get("checkpoint_path")):
                        if kind != "detector":
                            errors.append(f"{ident} 此節點的雙分支模型包尚未支援；請選擇單一權重。")
                            continue
                        from .dual_branch import validate_bundle
                        validate_bundle(weights["checkpoint_path"], model_id)
                        warnings.append(f"{ident} 會同時載入原始與微調兩個偵測分支，分數各自保留；需要較多記憶體。")
                    allowed = {p["key"] for p in model.get("capabilities", {}).get("inference_params", [])}
                    selected = {"device": options["device"]}
                    if kind == "detector":
                        selected.update({key: params[key] for key in ("conf", "imgsz", "max_det", "prompt", "use_training_classes") if key in params and key in allowed})
                    else:
                        selected["top_k"] = params.get("top_k", 5)
                    selected = {key: value for key, value in selected.items() if key in allowed}
                    clean = normalize_params(model_id, "inference", selected, allow_missing_weights=bool(checkpoint_id and not model.get("available")))
                    if model_id == "grounding_dino_base" and clean.get("use_training_classes"):
                        meta = path / "workbench_training.json" if path.is_dir() else path.parent / "workbench_training.json"
                        metadata = json.loads(meta.read_text(encoding="utf-8")) if checkpoint_id and meta.is_file() else {}
                        if not metadata.get("classes"):
                            errors.append(f"{ident} 使用完整訓練類別需要選擇保存類別清單的單模型微調權重。")
                            continue
                    resolved[ident] = {"model_id": model_id, "params": clean, **weights}
                except (OSError, ValueError, TypeError, KeyError):
                    errors.append(f"{ident} 權重無法解析或推論參數無效，請檢查所選 checkpoint 與模型參數。")
            elif kind in {"pose", "skeleton_action", "relation"}:
                asset_id = {"pose": "yolo26n_pose", "skeleton_action": "stgcnpp_ntu120", "relation": "relateanything_convnext_large"}[kind]
                public_node["model_id"] = asset_id
                path = workflow_graph.asset_path(asset_id, manager.app_root)
                required = [path]
                dependencies = ["torch", "ultralytics"] if kind == "pose" else ["torch"] if kind == "skeleton_action" else ["onnxruntime"]
                if kind == "relation":
                    from .workflow_relations import REQUIRED_FILES
                    required = [path.parent / name for name in REQUIRED_FILES]
                    predicates = params.get("predicates")
                    if not predicates or any(not isinstance(v, str) or not v.strip() for v in predicates) or len(set(predicates)) != len(predicates):
                        errors.append(f"{ident} 關係詞彙必須為不重複的非空字串清單。")
                    warnings.append(f"{ident} RelateAnything 使用 CPU ONNX；不佔用所選 GPU。")
                missing = [p.name for p in required if not p.is_file() or p.stat().st_size == 0]
                absent = [name for name in dependencies if not _dependency_available(name)]
                if missing:
                    errors.append(f"{ident} 缺少模型資產：{', '.join(missing)}。")
                if absent:
                    errors.append(f"{ident} 尚未安裝必要套件：{', '.join(absent)}。")
                if not missing and not absent:
                    if kind == "relation":
                        try:
                            vocabulary_file = path.parent / "thresholds.json"
                            if vocabulary_file.stat().st_size > 2 * 1024**2:
                                raise ValueError("oversized predicate metadata")
                            vocabulary = json.loads(vocabulary_file.read_text(encoding="utf-8"))
                            labels = vocabulary["predicates"]
                            if (not isinstance(labels, list) or not labels
                                    or any(not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"] for row in labels)):
                                raise ValueError("invalid predicate metadata")
                            names = {row["name"] for row in labels}
                            if any(value not in names for value in params["predicates"]):
                                errors.append(f"{ident} 所選關係詞不在模型的預載詞庫；請選擇相容詞彙，不會自動替換。")
                                continue
                        except (OSError, ValueError, TypeError, KeyError):
                            errors.append(f"{ident} 無法讀取模型的關係詞庫清單，請重新驗證下載資產。")
                            continue
                    resolved[ident] = {"model_id": asset_id, "weight_path": str(path.resolve()), "model_dir": str(path.parent.resolve())}
            elif kind == "vlm":
                setting = integrations().get("munolith", {})
                chosen = params.get("model_id") or setting.get("model_id")
                if not setting.get("available") or not chosen or chosen not in setting.get("models", []):
                    errors.append(f"{ident} 雲端視覺服務尚未連通或模型 ID 無效；請先設定連線並選擇模型。")
                else:
                    params["model_id"] = chosen
                    public_node["model_id"] = chosen
                    resolved[ident] = {"model_id": chosen}
            if item and item["kind"] == "image" and kind in {"video_action", "skeleton_action"}:
                warnings.append(f"{ident} 單張圖片沒有足夠時間序列，動作節點會保持未知；請使用影片或即時串流。")
        checked["valid"] = not errors
        checked["plan"].update(nodes=node_plan, mode="live" if live else "media", media_id=media_id,
                               validation_scope="graph_models_media" if item else "graph_models", options=options)
        return checked, resolved, item

    @application.get("/api/workflows/catalog")
    def get_catalog():
        return workflow_graph.catalog()

    @application.get("/api/workflows/templates")
    def get_templates():
        return with_samples(templates(), media.list())

    @application.get("/api/workflows")
    def list_workflows():
        with lock:
            values = [saved_entry(path) for path in directory().glob("*.json")]
        return {"workflows": sorted((v for v in values if v), key=lambda v: str(v["created_at"]), reverse=True)}

    @application.post("/api/workflows", status_code=201)
    def save_workflow(payload: Any = Body(...)):
        value = _parse(SaveInput, payload)
        if not value.name.strip():
            raise HTTPException(422, "工作流名稱不能空白。")
        try:
            graph = workflow_graph.normalize_graph(value.graph)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None
        entry = {"id": uuid.uuid4().hex, "name": value.name.strip(), "graph": graph,
                 "created_at": datetime.now(timezone.utc).isoformat()}
        with lock:
            write_json_atomic(directory() / (entry["id"] + ".json"), entry)
        return entry

    @application.delete("/api/workflows/{workflow_id}")
    def delete_workflow(workflow_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", workflow_id):
            raise HTTPException(404, "找不到已儲存的工作流。")
        with lock:
            path = directory() / (workflow_id + ".json")
            if saved_entry(path) is None:
                raise HTTPException(404, "找不到已儲存的工作流。")
            path.unlink()
        return {"deleted": True, "id": workflow_id}

    @application.post("/api/workflows/validate")
    def validate_workflow(payload: Any = Body(...)):
        value = _parse(ValidateInput, payload)
        checked, _, _ = preflight(value.graph, value.options.model_dump(), value.media_id)
        return checked

    def submit(request):
        try:
            return manager.submit(request)
        except ValueError:
            raise HTTPException(409, "工作無法排入佇列；請確認即時工作狀態或等待佇列空位。") from None

    @application.post("/api/workflows/run", status_code=201)
    def run_workflow(payload: Any = Body(...)):
        value = _parse(RunInput, payload)
        options = value.options.model_dump()
        checked, resolved, item = preflight(value.graph, options, value.media_id)
        if not checked["valid"]:
            raise HTTPException(422, {"message": "工作流預檢未通過。", **checked})
        return submit({"kind": "workflow", "model_id": "workflow", "workflow_graph": checked["graph"],
                       "resolved_nodes": resolved, "media_id": value.media_id, "media_path": item["path"],
                       "runtime_path": str(runtime), "params": options})

    @application.post("/api/workflows/live", status_code=201)
    def live_workflow(payload: Any = Body(...)):
        value = _parse(LiveInput, payload)
        options = value.options.model_dump()
        checked, resolved, _ = preflight(value.graph, options, live=True)
        if not checked["valid"]:
            raise HTTPException(422, {"message": "即時工作流預檢未通過。", **checked})
        return submit({"kind": "live", "model_id": "workflow", "workflow_graph": checked["graph"],
                       "resolved_nodes": resolved, "runtime_path": str(runtime), "params": options,
                       "models": [{"key": "0", "model_id": "workflow", "params": {}}],
                       "capture_fps": value.capture_fps, "max_duration_seconds": value.max_duration_seconds})

    @application.get("/api/workflows/integrations")
    def get_integrations():
        return integrations()

    @application.post("/api/workflows/integrations/munolith")
    def configure_integration(payload: Any = Body(...)):
        value = _parse(IntegrationInput, payload)
        try:
            return _integration_view(workflow_integrations.configure(runtime, value.key_file_path, value.model_id))
        except (OSError, ValueError, TypeError, KeyError):
            raise HTTPException(422, "無法設定連線；請確認金鑰檔案可讀、格式正確，以及模型 ID。") from None
