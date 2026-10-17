"""Portable, metadata-only discovery of local model files. Never loads tensors."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import threading
import uuid

APP_ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "yolov12x": ("model_weights", "yolo12x.pt"),
    "yolo26x": ("model_weights", "yolo26x.pt"),
    "grounding_dino_base": ("model_weights", "model.safetensors"),
    "rf_detr_2xl": ("model_weights", "rf-detr-xxlarge.pth"),
    "vivit_base_k400": ("action_model_weights", "pytorch_model.bin"),
    "videomaev2_giant_k710": ("action_model_weights", "vit_g_hybrid_pt_1200e_k710_ft.pth"),
    "slowfast_r101_8x8": ("action_model_weights", "SLOWFAST_8x8_R101.pyth"),
    "x3d_l": ("action_model_weights", "X3D_L.pyth"),
    "dinov3_7b": ("model_weights", "model-00001-of-00006.safetensors"),
}
HF_TYPES = {"grounding_dino_base": "grounding-dino", "vivit_base_k400": "vivit", "dinov3_7b": "dinov3_vit"}
MAX_ROOTS, MAX_FILES, MAX_DEPTH = 12, 10000, 6
_runtime = ContextVar("model_registry_runtime", default=None)
_lock = threading.RLock()


def runtime_path(default_app_root=None) -> Path:
    return Path(_runtime.get() or os.environ.get("VISION_WORKBENCH_RUNTIME", Path(default_app_root or APP_ROOT) / "runtime")).resolve()


@contextmanager
def registry_runtime(runtime):
    token = _runtime.set(str(runtime))
    try:
        yield
    finally:
        _runtime.reset(token)


def _json(path: Path, maximum=1024 * 1024):
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"中繼資料過大：{path.name}")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"中繼資料必須是物件：{path.name}")
    return value


def _absolute(value) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("請提供本機絕對路徑。")
    # Keep an HF snapshot's file symlink: resolving it would lose config and
    # tokenizer companions in the snapshot directory.
    return path.parent.resolve() / path.name


def _file(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"缺少完整檔案：{path.name}")
    return path


def _directory_link(path):
    try:
        info = path.lstat()
        return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True


def _hf_weight(folder: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (folder / name).is_file():
            return _file(folder / name)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (folder / name).is_file():
            mapping = _json(folder / name).get("weight_map")
            if not isinstance(mapping, dict) or not mapping or len(mapping) > 100000:
                raise ValueError("權重分片索引不正確。")
            shards = sorted(set(mapping.values()))
            for shard in shards:
                if not isinstance(shard, str) or Path(shard).name != shard or any(c in shard for c in "/\\:"):
                    raise ValueError("權重分片索引不可包含路徑。")
                _file(folder / shard)
            return folder / shards[0]
    raise ValueError("資料夾缺少完整權重或權重分片索引。")


def validate_weight(model_id: str, value) -> Path:
    if model_id not in FILES:
        raise ValueError("未知模型。")
    path = _absolute(value)
    if model_id in HF_TYPES:
        selected_file = None if path.is_dir() else _file(path)
        folder = path if selected_file is None else path.parent
        config = _json(folder / "config.json")
        if config.get("model_type") != HF_TYPES[model_id]:
            raise ValueError("config.json 的模型架構與選定模型不符。")
        _file(folder / "preprocessor_config.json")
        if model_id == "grounding_dino_base":
            _file(folder / "tokenizer_config.json")
            if not any((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in ("tokenizer.json", "vocab.txt")):
                raise ValueError("Grounding DINO 缺少 tokenizer.json 或 vocab.txt。")
        weight = _hf_weight(folder)
        if selected_file is not None and selected_file != weight:
            allowed = set()
            for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                if (folder / index_name).is_file():
                    allowed.update(_json(folder / index_name).get("weight_map", {}).values())
            if selected_file.name not in allowed:
                raise ValueError("選定檔案不是此資料夾使用的權重；請選擇完整模型資料夾或對應權重檔案。")
        return weight
    if path.is_dir():
        path = path / FILES[model_id][1]
    if path.suffix.lower() != Path(FILES[model_id][1]).suffix.lower():
        raise ValueError(f"此模型需要 {Path(FILES[model_id][1]).suffix} 權重檔案。")
    known_owners = {mid for mid, (_, filename) in FILES.items() if filename.lower() == path.name.lower()}
    if known_owners and model_id not in known_owners:
        raise ValueError("選定檔名屬於其他模型，不能綁定為目前模型。")
    for metadata_name in ("workbench_training.json", "training_config.json", "model_config.json"):
        if (path.parent / metadata_name).is_file():
            identity = _json(path.parent / metadata_name).get("model_id")
            if identity is not None and identity != model_id:
                raise ValueError("權重中繼資料的 model_id 與選定模型不符。")
    return _file(path)


def validate_checkpoint(model_id: str, value) -> Path:
    if model_id not in FILES or model_id == "dinov3_7b":
        raise ValueError("此模型尚未提供微調權重接頭。")
    path = _absolute(value)
    if model_id in {"yolov12x", "yolo26x"}:
        if path.is_dir():
            path = path / "best.pt"
        _file(path)
        if path.suffix.lower() != ".pt":
            raise ValueError("YOLO 微調權重必須是 .pt。")
        folder = path.parent
        metadata_path = folder / "workbench_training.json"
        if not metadata_path.is_file():
            metadata_path = folder / "training_config.json"
        metadata = _json(metadata_path)
    else:
        folder = path if path.is_dir() else path.parent
        metadata = _json(folder / "workbench_training.json")
        path = folder
    classes = metadata.get("classes", metadata.get("dataset_classes"))
    if metadata.get("model_id") != model_id or not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes):
        raise ValueError("微調中繼資料的 model_id 或 classes 不正確。")
    if model_id in HF_TYPES:
        validate_weight(model_id, folder)
        if model_id == "vivit_base_k400":
            labels = _json(folder / "config.json").get("id2label")
            if not isinstance(labels, dict) or labels != {str(index): name for index, name in enumerate(classes)}:
                raise ValueError("ViViT config.json 的 id2label 與微調 classes 不符。")
    elif model_id == "rf_detr_2xl":
        _file(folder / "model.pth")
        if type(metadata.get("class_id_offset")) is not int or metadata.get("class_id_offset") != 1 or any(type(metadata.get(k)) is not int or metadata[k] < 1 for k in ("resolution", "group_detr", "positional_encoding_size")):
            raise ValueError("RF-DETR 微調權重缺少有效的架構中繼資料。")
    elif model_id not in {"yolov12x", "yolo26x"}:
        _file(folder / "model.safetensors")
        config = _json(folder / "model_config.json")
        if config.get("model_id") != model_id or config.get("classes") != classes:
            raise ValueError("model_config.json 與微調中繼資料不符。")
    return path


def checkpoint_metadata(path):
    folder = path.parent if path.is_file() else path
    metadata_path = folder / "workbench_training.json"
    if not metadata_path.is_file():
        metadata_path = folder / "training_config.json"
    metadata = _json(metadata_path)
    return {"model_id": metadata.get("model_id"), "classes": metadata.get("classes", metadata.get("dataset_classes", [])),
            **{key: metadata[key] for key in ("caption", "checkpoint_format") if key in metadata}}


class ModelRegistry:
    def __init__(self, runtime=None, app_root=None):
        self.runtime = Path(runtime or runtime_path()).resolve()
        self.app_root = Path(app_root or APP_ROOT).resolve()
        self.path = self.runtime / "model_registry.json"

    def _read(self):
        if not self.path.exists():
            return {"schema_version": 1, "roots": [], "bindings": {}, "candidates": [], "checkpoints": []}
        value = _json(self.path, 8 * 1024 * 1024)
        if value.get("schema_version") != 1:
            raise ValueError("模型登錄檔版本不支援。")
        return value

    def _write(self, value):
        from .jobs import write_json_atomic
        self.runtime.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, value)

    def roots(self, saved=None):
        saved = saved or self._read()
        roots = [(self.app_root / "models", "application"),
                 (self.app_root.parent / "model_weights", "legacy"),
                 (self.app_root.parent / "action_model_weights", "legacy")]
        hf = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
        if hf.is_dir():
            roots.append((hf, "huggingface_cache"))
        roots += [(Path(path), "configured") for path in saved.get("roots", [])]
        result, seen = [], set()
        for path, source in roots:
            key = os.path.normcase(str(path.resolve()))
            if key not in seen:
                result.append({"path": str(path.resolve()), "source": source, "exists": path.is_dir()})
                seen.add(key)
        return result

    def _defaults(self, model_id):
        group, filename = FILES[model_id]
        if model_id in HF_TYPES:
            return [self.app_root / "models" / model_id, self.app_root / "models", self.app_root.parent / group / model_id]
        return [self.app_root / "models" / model_id / filename,
                self.app_root / "models" / filename,
                self.app_root.parent / group / model_id / filename]

    def resolve(self, model_id, *, required=True):
        if model_id not in FILES:
            raise ValueError("未知模型。")
        saved = self._read()
        binding = saved.get("bindings", {}).get(model_id)
        if binding:
            return validate_weight(model_id, binding)
        candidates = self._defaults(model_id)
        candidates += [Path(item["path"]) for item in saved.get("candidates", []) if item.get("model_id") == model_id and item.get("kind") == "base"]
        valid = {}
        for path in candidates:
            try:
                folder = path if path.is_dir() else path.parent
                if model_id in HF_TYPES and (folder / "workbench_training.json").is_file():
                    continue
                found = validate_weight(model_id, path)
                valid[os.path.normcase(str(found))] = found
            except (OSError, ValueError, TypeError):
                continue
        if len(valid) > 1:
            raise ValueError("找到多個相容權重，請在模型檔案管理中明確綁定一個。")
        if valid:
            return next(iter(valid.values()))
        if required:
            raise ValueError("找不到完整本機權重；請掃描或綁定已下載的模型路徑。")
        return self._defaults(model_id)[0]

    def bind(self, model_id, path):
        validated = validate_weight(model_id, path)
        with _lock:
            saved = self._read()
            saved.setdefault("bindings", {})[model_id] = str(validated)
            self._write(saved)
        return {"model_id": model_id, "path": str(validated), "valid": True}

    def unbind(self, model_id):
        if model_id not in FILES:
            raise ValueError("未知模型。")
        with _lock:
            saved = self._read()
            saved.setdefault("bindings", {}).pop(model_id, None)
            self._write(saved)
        return {"model_id": model_id, "removed": True}

    def add_checkpoint(self, model_id, path, name=None):
        validated = validate_checkpoint(model_id, path)
        with _lock:
            saved = self._read()
            entries = saved.setdefault("checkpoints", [])
            existing = next((item for item in entries if item["model_id"] == model_id and item["path"] == str(validated)), None)
            item = {"id": existing["id"] if existing else "external-" + uuid.uuid4().hex,
                    "model_id": model_id, "path": str(validated), "name": name or validated.name,
                    "source": "external", "registered_at": datetime.now(timezone.utc).isoformat()}
            if existing:
                entries.remove(existing)
            entries.append(item)
            self._write(saved)
        return {**item, "valid": True, "training": checkpoint_metadata(validated)}

    def checkpoint_for(self, checkpoint_id, model_id, manager=None):
        if checkpoint_id.startswith("job:") and manager is not None:
            return manager.checkpoint_for(checkpoint_id[4:], model_id)
        item = next((item for item in self._read().get("checkpoints", []) if item.get("id") == checkpoint_id), None)
        if item is None or item.get("model_id") != model_id:
            raise ValueError("找不到與此模型相容的已登錄微調權重。")
        return validate_checkpoint(model_id, item["path"])

    def remove_checkpoint(self, checkpoint_id):
        if not checkpoint_id.startswith("external-"):
            raise ValueError("只能移除外部微調權重的登錄；訓練成果由工作紀錄管理。")
        with _lock:
            saved = self._read()
            entries = saved.get("checkpoints", [])
            if not any(item.get("id") == checkpoint_id for item in entries):
                raise ValueError("找不到外部微調權重登錄。")
            saved["checkpoints"] = [item for item in entries if item.get("id") != checkpoint_id]
            self._write(saved)
        return {"id": checkpoint_id, "removed": True}

    def checkpoints(self, manager=None):
        result = []
        for item in self._read().get("checkpoints", []):
            item = dict(item)
            try:
                checked = validate_checkpoint(item["model_id"], item["path"])
                item["valid"] = True
                item["training"] = checkpoint_metadata(checked)
            except (OSError, ValueError, TypeError) as exc:
                item.update(valid=False, error=str(exc))
            result.append(item)
        if manager is not None:
            for job in manager.list():
                if job.get("kind") == "training" and job.get("status") == "completed":
                    item = {"id": "job:" + job["id"], "job_id": job["id"], "model_id": job["model_id"],
                            "source": "training_job", "name": f"{job['model_id']} · {job['id'][:8]}", "path": None}
                    try:
                        item.update(path=str(manager.checkpoint_for(job["id"], job["model_id"])), valid=True)
                        if isinstance(job.get("result", {}).get("training"), dict):
                            item["training"] = job["result"]["training"]
                    except (OSError, ValueError, TypeError) as exc:
                        item.update(valid=False, error=str(exc))
                    result.append(item)
        return result

    def snapshot(self, manager=None):
        saved = self._read()
        bindings = []
        for model_id, path in saved.get("bindings", {}).items():
            item = {"model_id": model_id, "path": path}
            try:
                validate_weight(model_id, path)
                item["valid"] = True
            except (OSError, ValueError, TypeError) as exc:
                item.update(valid=False, error=str(exc))
            bindings.append(item)
        candidates = []
        for item in saved.get("candidates", []):
            item = dict(item)
            try:
                fn = validate_checkpoint if item["kind"] == "checkpoint" else validate_weight
                fn(item["model_id"], item["path"])
                item["valid"] = True
            except (OSError, ValueError, TypeError) as exc:
                item.update(valid=False, error=str(exc))
            candidates.append(item)
        return {"roots": self.roots(saved), "bindings": bindings, "candidates": candidates,
                "checkpoints": self.checkpoints(manager), "scan_errors": saved.get("scan_errors", []),
                "truncated": saved.get("truncated", False)}

    def scan(self, roots=None, manager=None):
        if roots and len(roots) > MAX_ROOTS:
            raise ValueError(f"最多指定 {MAX_ROOTS} 個掃描資料夾。")
        with _lock:
            saved = self._read()
            if roots:
                saved["roots"] = [str(_absolute(root)) for root in roots]
            found, errors, seen, count, truncated = [], [], set(), 0, False
            for root in self.roots(saved):
                folder = Path(root["path"])
                if not folder.is_dir():
                    if root["source"] == "configured":
                        errors.append({"path": str(folder), "error": "資料夾不存在。"})
                    continue
                def onerror(exc):
                    errors.append({"path": str(exc.filename), "error": str(exc)})
                for directory, dirs, files in os.walk(folder, followlinks=False, onerror=onerror):
                    current = Path(directory)
                    depth = len(current.relative_to(folder).parts)
                    dirs[:] = [name for name in dirs if not _directory_link(current / name) and name not in {".git", ".venv", "node_modules", "blobs"}]
                    if depth >= MAX_DEPTH:
                        if dirs:
                            truncated = True
                        dirs[:] = []
                    count += len(files) + len(dirs)
                    if count > MAX_FILES:
                        truncated = True
                        break
                    matches = []
                    if "config.json" in files:
                        try:
                            model_type = _json(current / "config.json").get("model_type")
                            matches += [(mid, current, "base") for mid, kind in HF_TYPES.items() if kind == model_type]
                        except (OSError, ValueError):
                            pass
                    for mid, (_, filename) in FILES.items():
                        if mid not in HF_TYPES and filename in files:
                            matches.append((mid, current / filename, "base"))
                    for metadata_name in ("workbench_training.json", "training_config.json"):
                        if metadata_name in files:
                            try:
                                mid = _json(current / metadata_name).get("model_id")
                                if mid in FILES:
                                    matches.append((mid, current, "checkpoint"))
                            except (OSError, ValueError):
                                pass
                    for mid, candidate, kind in matches:
                        key = (mid, str(candidate), kind)
                        if key in seen:
                            continue
                        seen.add(key)
                        item = {"model_id": mid, "path": str(candidate), "kind": kind, "bytes": 0}
                        try:
                            resolved = (validate_checkpoint if kind == "checkpoint" else validate_weight)(mid, candidate)
                            item.update(path=str(resolved), valid=True, bytes=resolved.stat().st_size if resolved.is_file() else 0)
                        except (OSError, ValueError, TypeError) as exc:
                            item.update(valid=False, error=str(exc))
                        found.append(item)
                if count > MAX_FILES:
                    break
            # A fine-tuned HF directory must not silently replace base weights.
            checkpoint_dirs = {str(Path(item["path"]).parent if Path(item["path"]).suffix == ".pt" else Path(item["path"])) for item in found if item["kind"] == "checkpoint"}
            found = [item for item in found if item["kind"] != "base" or str(Path(item["path"]).parent) not in checkpoint_dirs]
            saved.update(candidates=found, scan_errors=errors, truncated=truncated)
            self._write(saved)
        return self.snapshot(manager)


class WeightPaths(Mapping):
    """Legacy WEIGHTS[id] callers see current bindings without global caches."""
    def __getitem__(self, key):
        return ModelRegistry().resolve(key, required=False)

    def __iter__(self):
        return iter(FILES)

    def __len__(self):
        return len(FILES)
