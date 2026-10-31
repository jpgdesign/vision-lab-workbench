"""Portable frozen-original + independently trained branch bundles.

The original branch is copied and SHA256 checked, never sent to an optimizer.
Branch scores remain independent; no cross-branch suppression or softmax.
All imports of ML runtimes are deferred until training/evaluation/inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import stat

FORMAT = "frozen_branch_bundle"
MANIFEST = "branch_bundle.json"
DETECTORS = {"yolov12x", "yolo26x", "grounding_dino_base", "rf_detr_2xl"}
LARGE_MODELS = {"videomaev2_giant_k710", "dinov3_7b"}
_DIGESTS = {}


def _json(path):
    from .model_registry import _json as read
    return read(path, 8 * 1024 * 1024)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def is_bundle(path):
    if not path:
        return False
    path = Path(path)
    return ((path if path.is_dir() else path.parent) / MANIFEST).is_file()


def _link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _inside(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or "\x00" in relative or PurePosixPath(relative).is_absolute()
            or any(x in {".", ".."} for x in relative.split("/"))):
        raise ValueError("雙分支模型包含不安全的相對路徑。")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("雙分支權重超出模型包範圍。")
    for parent in [path, *path.parents]:
        if parent == root.parent:
            break
        if parent.exists() and _link(parent):
            raise ValueError("雙分支模型包不可包含連結或 junction。")
    return path


def _names(value):
    if (not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ValueError("雙分支類別必須是非空且不重複的名稱清單。")
    return value


def _sha(path, *, cached=False):
    path = Path(path)
    info = path.stat()
    key = (str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
    if cached and key in _DIGESTS:
        return _DIGESTS[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if len(_DIGESTS) > 10000:
        _DIGESTS.clear()
    _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if _link(path):
            raise ValueError("雙分支快照不可包含連結。")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    return result


def _copy_model(model_id, source, target, *, checkpoint, move=False):
    """Copy only complete model inputs, resolving HF file links into real files."""
    from .model_registry import validate_checkpoint, validate_weight, HF_TYPES
    if is_bundle(source):
        raise ValueError("雙分支模型包不能再次作為訓練來源或巢狀分支。")
    source = (validate_checkpoint if checkpoint else validate_weight)(model_id, str(source))
    if target.exists():
        if source.resolve().is_relative_to(target.resolve()):
            return source
        raise ValueError("模型快照目錄已存在，不能覆寫。")
    folder_model = source.is_dir() or model_id in HF_TYPES
    source_root = source if source.is_dir() else source.parent
    if move and source.is_dir():
        shutil.move(str(source), str(target))
        return target
    target.mkdir(parents=True)
    if folder_model:
        # HF models may have shard files and tokenizer companions. The only
        # nested model layout currently supported is a DINO classifier backbone.
        members = [path for path in source_root.iterdir() if path.is_file() and
                   path.suffix.lower() in {".json", ".safetensors", ".bin", ".pth", ".pt", ".pyth", ".txt", ".model"}]
        for path in members:
            if path.name == MANIFEST:
                raise ValueError("不允許巢狀雙分支模型。")
            shutil.copy2(path, target / path.name, follow_symlinks=True)
        if (source_root / "backbone").exists():
            if _link(source_root / "backbone"):
                raise ValueError("backbone 目錄不可使用連結。")
            shutil.copytree(source_root / "backbone", target / "backbone", symlinks=False)
        return target
    destination = target / source.name
    (shutil.move if move else shutil.copy2)(str(source), str(destination))
    for name in ("workbench_training.json", "training_config.json", "model_config.json", "retention.json"):
        if (source_root / name).is_file():
            shutil.copy2(source_root / name, target / name)
    return destination


def create_bundle(model_id, original_path, new_checkpoint_path, plan, destination, *,
                  original_is_checkpoint=False, move_new=False):
    """Package existing weights without training; ``move_new`` is job-owned only."""
    from .model_registry import checkpoint_metadata, validate_checkpoint
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / MANIFEST).exists():
        raise ValueError("雙分支模型包已存在，不能覆寫。")
    base, classes = _names(plan["base_classes"]), _names(plan["classes"])
    new_source = validate_checkpoint(model_id, str(new_checkpoint_path))
    new_classes = _names(checkpoint_metadata(new_source)["classes"])
    mapping = plan.get("new_label_map")
    if (classes[:len(base)] != base or not isinstance(mapping, list) or len(mapping) != len(new_classes)
            or any(type(i) is not int or not 0 <= i < len(classes) for i in mapping)):
        raise ValueError("雙分支類別聯集與新分支標籤映射不一致。")
    aliases = plan.get("class_aliases", {})
    if any(classes[mapped].strip().casefold() != aliases.get(name, name).strip().casefold()
           for name, mapped in zip(new_classes, mapping)):
        raise ValueError("新分支標籤映射與明確指定的名稱／別名不一致。")
    original = _copy_model(model_id, original_path, destination / "original", checkpoint=original_is_checkpoint)
    if original_is_checkpoint and checkpoint_metadata(original)["classes"] != base:
        raise ValueError("原分支 checkpoint 類別與 base_classes 不一致。")
    new = _copy_model(model_id, new_source, destination / "new", checkpoint=True, move=move_new)
    manifest = {"schema_version": 1, "checkpoint_format": FORMAT, "model_id": model_id,
        "classes": classes, "base_classes": base, "new_classes": new_classes, "class_expansion": plan,
        "branches": {
            "original": {"model_id": model_id, "kind": "checkpoint" if original_is_checkpoint else "pretrained",
                         "path": original.relative_to(destination).as_posix(), "classes": base, "label_map": list(range(len(base)))},
            "new": {"model_id": model_id, "kind": "checkpoint", "path": new.relative_to(destination).as_posix(),
                    "classes": new_classes, "label_map": mapping}},
        "score_semantics": "independent_branch_scores_no_joint_normalization", "cross_branch_suppression": False,
        "original_inventory": _inventory(destination / "original")}
    _write(destination / MANIFEST, manifest)
    _write(destination / "workbench_training.json", {"model_id": model_id, "classes": classes,
        "base_classes": base, "class_expansion": plan, "training_mode": "dual_branch", "checkpoint_format": FORMAT,
        "inference_input": "image" if model_id in DETECTORS or model_id == "dinov3_7b" else "video",
        "score_semantics": manifest["score_semantics"], "old_branch_unchanged": True})
    return validate_bundle(destination, model_id)


def validate_bundle(path, model_id):
    """Metadata/file validation only; no model imports, pickle loading or recursion."""
    from .model_registry import validate_checkpoint, validate_weight, checkpoint_metadata, FILES
    root = Path(path)
    root = root if root.is_dir() else root.parent
    if model_id not in FILES or not root.is_dir() or _link(root):
        raise ValueError("雙分支模型目錄無效。")
    root = root.resolve()
    manifest = _json(_inside(root, MANIFEST))
    metadata = _json(_inside(root, "workbench_training.json"))
    classes = _names(manifest.get("classes"))
    if (manifest.get("schema_version") != 1 or manifest.get("checkpoint_format") != FORMAT
            or manifest.get("model_id") != model_id or metadata.get("model_id") != model_id
            or metadata.get("checkpoint_format") != FORMAT or metadata.get("classes") != classes
            or metadata.get("training_mode") != "dual_branch"
            or manifest.get("cross_branch_suppression") is not False
            or manifest.get("score_semantics") != "independent_branch_scores_no_joint_normalization"):
        raise ValueError("雙分支模型包的格式、模型或類別不一致。")
    branches = manifest.get("branches")
    if not isinstance(branches, dict) or set(branches) != {"original", "new"}:
        raise ValueError("雙分支模型必須包含 original 和 new。")
    for name, branch in branches.items():
        if not isinstance(branch, dict):
            raise ValueError("分支資訊必須是 JSON 物件。")
        names = _names(branch.get("classes"))
        mapping = branch.get("label_map")
        if (branch.get("model_id") != model_id or branch.get("kind") not in {"pretrained", "checkpoint"}
                or (name == "new" and branch["kind"] != "checkpoint")
                or not isinstance(mapping, list) or len(mapping) != len(names)
                or any(type(i) is not int or not 0 <= i < len(classes) for i in mapping)):
            raise ValueError("分支模型類別映射不合法。")
        source = _inside(root, branch.get("path"))
        if source.relative_to(root).parts[0] != name or not source.exists():
            raise ValueError("分支權重必須位於指定的獨立目錄。")
        branch_root = root / name
        count = 0
        for current, dirs, files in os.walk(branch_root, followlinks=False):
            for member in [*dirs, *files]:
                child = Path(current) / member
                count += 1
                if count > 10000 or _link(child) or not child.resolve().is_relative_to(branch_root.resolve()):
                    raise ValueError("分支包含連結、過量檔案或不安全路徑。")
                if child.name == MANIFEST:
                    raise ValueError("不允許遞迴或巢狀雙分支模型。")
        if branch["kind"] == "checkpoint":
            checked = validate_checkpoint(model_id, str(source))
            if checkpoint_metadata(checked)["classes"] != names:
                raise ValueError("分支 checkpoint 類別與模型包映射不一致。")
        else:
            validate_weight(model_id, str(source))
    base = branches["original"]["classes"]
    if (manifest.get("base_classes") != base or classes[:len(base)] != base
            or branches["original"]["label_map"] != list(range(len(base)))
            or manifest.get("new_classes") != branches["new"]["classes"]):
        raise ValueError("雙分支原類別順序或新類別資訊不一致。")
    inventory = manifest.get("original_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("雙分支模型缺少原權重完整性稽核。")
    for relative, information in inventory.items():
        source = _inside(root / "original", relative)
        digest = information.get("sha256") if isinstance(information, dict) else None
        if (not isinstance(information, dict) or not source.is_file() or source.stat().st_size != information.get("bytes")
                or not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest)):
            raise ValueError("原模型快照檔案或完整性資訊不符。")
        if _sha(source, cached=True) != digest:
            raise ValueError("原模型快照 SHA256 不符，不能宣稱原分支未變更。")
    actual_files = {path.relative_to(root / "original").as_posix() for path in (root / "original").rglob("*") if path.is_file()}
    if set(inventory) != actual_files:
        raise ValueError("原模型快照完整性清單未涵蓋全部檔案。")
    plan = manifest.get("class_expansion", {})
    if not isinstance(plan, dict):
        raise ValueError("類別映射計畫必須是 JSON 物件。")
    aliases = plan.get("class_aliases", {})
    new = branches["new"]
    if (plan.get("classes") != classes or plan.get("base_classes") != base
            or plan.get("new_label_map") != new["label_map"] or not isinstance(aliases, dict)
            or any(not isinstance(value, str) for value in aliases.values())
            or any(classes[mapped].strip().casefold() != aliases.get(name, name).strip().casefold()
                   for name, mapped in zip(new["classes"], new["label_map"]))):
        raise ValueError("雙分支 manifest 與類別名稱／別名計畫不一致。")
    return root


def branch_request(request, bundle, branch_name):
    manifest = _json(Path(bundle) / MANIFEST)
    branch = manifest["branches"][branch_name]
    result = {**request, "params": copy.deepcopy(request["params"])}
    result.pop("checkpoint_path", None)
    result.pop("model_weight_path", None)
    result["checkpoint_path" if branch["kind"] == "checkpoint" else "model_weight_path"] = str(Path(bundle) / branch["path"])
    # Original branch always receives exactly the user's inference settings.
    # For Grounding DINO, its new task uses the saved supervised caption.
    if request["model_id"] == "grounding_dino_base" and branch["kind"] != "checkpoint":
        result["params"]["use_training_classes"] = False
    if branch_name == "new" and request["model_id"] == "grounding_dino_base":
        from .model_registry import checkpoint_metadata
        metadata = checkpoint_metadata(Path(result["checkpoint_path"]))
        if metadata.get("caption_mode") == "chunked":
            result["params"]["use_training_classes"] = True
        elif metadata.get("caption"):
            result["params"]["use_training_classes"] = False
            result["params"]["prompt"] = metadata["caption"]
    return result


def mapped_predictions(predictions, manifest, branch_name):
    branch = manifest["branches"][branch_name]
    def normalize(value):
        words = " ".join(str(value).strip().lower().replace("_", " ").split())
        # Tokenizers may decode kai-lan as "kai - lan". Exact labels still
        # take priority, and normalized collisions remain explicitly unmapped.
        return re.sub(r"\s*([^\w\s])\s*", r"\1", words)
    exact = {name: index for index, name in enumerate(branch["classes"])}
    lookup = {}
    for index, name in enumerate(branch["classes"]):
        lookup.setdefault(normalize(name), []).append(index)
    result = []
    for original in predictions:
        local = exact.get(original["label"])
        candidates = lookup.get(normalize(original["label"]), [])
        if local is None and len(candidates) == 1:
            local = candidates[0]
        union = branch["label_map"][local] if local is not None else None
        result.append({**original, "branch": branch_name, "source_label": original["label"],
                       "label": manifest["classes"][union] if union is not None else original["label"],
                       "union_class_id": union, "score_group": branch_name,
                       "mapping_status": "mapped" if local is not None else "ambiguous_label" if candidates else "outside_scope"})
    return result


def _evaluate_branch(request, bundle, manifest, branch_name, split):
    """Measure the named independent branch on the held-out split, never train."""
    dataset = request.get("evaluation_datasets", {}).get(split)
    if dataset is None:
        return None
    from .inference import create_detector, create_action_model, resolve_device, classification_probabilities
    branch = manifest["branches"][branch_name]
    p = request["params"]
    device = resolve_device(p.get("device", "auto"))
    inference_params = {"device": device, "conf": .001, "imgsz": p.get("imgsz", 640),
        "iou": .7, "max_det": 100, "accept_platform_license": p.get("accept_platform_license", True),
        "text_threshold": .25, "prompt": " . ".join(name.replace("_", " ") for name in branch["classes"]) + " ."}
    spec = branch_request({**request, "params": inference_params}, bundle, branch_name)
    if request["model_id"] in DETECTORS:
        from .detection_expansion import DetectionRetention
        evaluation_request = {**request, "training_mode": "expand_frozen", "class_expansion": copy.deepcopy(request["class_expansion"]),
                              "dataset": {**request["dataset"], "classes": manifest["classes"]}}
        evaluator = DetectionRetention(evaluation_request, bundle.parent, {**p, "imgsz": inference_params["imgsz"]})
        if split == "new":
            evaluator.spec["new_class_ids"] = sorted(set(branch["label_map"]))
        if request["model_id"] == "grounding_dino_base":
            from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
            from .grounding_prompts import build_caption_chunks, validate_caption_chunks
            from .detection_expansion import grounding_predictor
            source = Path(spec.get("checkpoint_path") or spec["model_weight_path"])
            folder = source if source.is_dir() else source.parent
            processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
            config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
            # Validate BEFORE model allocation/training. Never enlarge the
            # frozen source config or silently truncate the measured old scope.
            metadata = _json(folder / "workbench_training.json") if (folder / "workbench_training.json").is_file() else {}
            chunks = metadata.get("caption_chunks")
            if chunks is None:
                chunks = build_caption_chunks(branch["classes"], processor.tokenizer, int(config.max_text_len))
            chunks = validate_caption_chunks(chunks, branch["classes"], processor.tokenizer, int(config.max_text_len))
            caption = chunks[0]["caption"] if len(chunks) == 1 else ""
            model = AutoModelForZeroShotObjectDetection.from_pretrained(str(folder), local_files_only=True).to(device).eval()
            native = grounding_predictor(model, processor, caption, evaluator.settings, device, caption_chunks=chunks)
            def numeric_predict(image):
                return [{**row, "class_id": branch["label_map"][row["class_id"]]} for row in native(image)]
            metrics = evaluator.evaluate(split, numeric_predict)
            return {**metrics, "caption": caption, "caption_chunks": chunks,
                    "caption_tokens": max(chunk["text_tokens"] for chunk in chunks), "source_max_text_len": int(config.max_text_len)}
        native = create_detector(request["model_id"], spec["params"], device, spec.get("checkpoint_path"), spec.get("model_weight_path"))
        def predict(image):
            rows = mapped_predictions(native(image, spec["params"]), manifest, branch_name)
            return [{**row, "class_id": row["union_class_id"]} for row in rows if row["union_class_id"] is not None]
        return evaluator.evaluate(split, predict)
    import torch
    from PIL import Image
    from .training import _decode_video
    from .inference import prepare_action_tensor
    samples = dataset["details"]["val_samples"]
    if request["model_id"] == "dinov3_7b":
        from .training_dinov3 import load_classifier, prepare_image_pixels
        model, processor, metadata = load_classifier(spec["checkpoint_path"], device)
    else:
        model, metadata = create_action_model(request["model_id"], device, spec.get("checkpoint_path"), spec.get("model_weight_path"))
    correct, loss, per_class = 0, 0., {}
    with torch.inference_mode():
        for sample in samples:
            label = sample["label"]
            rows = [index for index, union_id in enumerate(branch["label_map"]) if union_id == label]
            if not rows:
                raise ValueError("驗證集類別無法對應到此獨立分支。")
            if request["model_id"] == "dinov3_7b":
                with Image.open(sample["path"]) as image:
                    pixels = prepare_image_pixels(image, processor, device)
                logits = model(pixels)
            else:
                frames = _decode_video(sample["path"], int(metadata["num_frames"]), False)
                pixels = prepare_action_tensor(frames, metadata, device)
                with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                    dtype=torch.float16, enabled=str(device).startswith("cuda")):
                    logits = model(pixel_values=pixels).logits if metadata["architecture"] == "vivit" else model(pixels)
            probabilities = classification_probabilities(logits, metadata.get("outputs_probabilities", False))[0]
            prediction = branch["label_map"][int(probabilities.argmax())]
            correct += int(prediction == label)
            loss -= float(probabilities[rows].sum().clamp_min(1e-12).log())
            name = manifest["classes"][label]
            item = per_class.setdefault(name, {"samples": 0, "correct": 0})
            item["samples"] += 1
            item["correct"] += int(prediction == label)
    if not samples:
        raise ValueError("獨立分支驗證集不能為空。")
    return {"accuracy": correct / len(samples), "loss": loss / len(samples), "sample_count": len(samples),
            "class_count": len(per_class), "class_accuracy": per_class, "status": "evaluated"}


def run_training(request, output, emit):
    from .model_registry import validate_checkpoint, validate_weight, checkpoint_metadata
    from .training import run_training as train_native, _catalog_weight
    from .inference_cache import release_unused_memory
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = request.get("class_expansion")
    if not isinstance(plan, dict):
        raise ValueError("雙分支訓練需要預先驗證的類別映射。")
    checkpoint = bool(request.get("checkpoint_path"))
    source = (validate_checkpoint(request["model_id"], request["checkpoint_path"]) if checkpoint else
              _catalog_weight(request["model_id"], request.get("model_weight_path")))
    if is_bundle(source):
        raise ValueError("雙分支模型包不能再作為訓練來源，請選擇單一模型 checkpoint。")
    if request["model_id"] == "dinov3_7b" and not checkpoint:
        raise ValueError("DINOv3 雙分支需要已訓練的分類 checkpoint。")
    bundle = output / "best"
    bundle.mkdir()
    emit({"event": "log", "message": "建立獨立原模型快照並計算 SHA256；原分支不會送入 optimizer。"})
    original = _copy_model(request["model_id"], source, bundle / "original", checkpoint=checkpoint)
    before_files = _inventory(bundle / "original")
    temporary_manifest = {"classes": plan["classes"], "branches": {
        "original": {"kind": "checkpoint" if checkpoint else "pretrained", "path": original.relative_to(bundle).as_posix(),
                     "classes": plan["base_classes"], "label_map": list(range(len(plan["base_classes"])))}}}
    # The evaluation helper needs branch routing before the final package exists.
    _write(bundle / MANIFEST, temporary_manifest)
    try:
        old_before = _evaluate_branch(request, bundle, temporary_manifest, "original", "old")
    finally:
        (bundle / MANIFEST).unlink()
        release_unused_memory()
    new_request = {**request, "training_mode": "standard", "_skip_checkpoint_archives": True}
    for key in ("checkpoint_path", "model_weight_path", "class_expansion", "class_aliases", "replay_dataset", "replay_dataset_id", "evaluation_datasets", "base_class_names"):
        new_request.pop(key, None)
    new_request["checkpoint_path" if checkpoint else "model_weight_path"] = str(original)
    def relay(event):
        event = dict(event)
        if event.get("event") == "progress":
            event["progress"] = .1 + .7 * float(event.get("progress", 0))
        emit(event)
    native = train_native(new_request, output / "new_branch", relay)
    release_unused_memory()
    after_files = _inventory(bundle / "original")
    if before_files != after_files:
        raise RuntimeError("原模型快照 SHA256 在新分支訓練期間改變，停止發布模型包。")
    new_best = native.get("training", {}).get("checkpoint_directory") or native["checkpoint_path"]
    created = create_bundle(request["model_id"], original, new_best, plan, bundle,
                            original_is_checkpoint=checkpoint, move_new=True)
    manifest = _json(created / MANIFEST)
    old_after = _evaluate_branch(request, created, manifest, "original", "old")
    release_unused_memory()
    new_after = _evaluate_branch(request, created, manifest, "new", "new")
    release_unused_memory()
    metric = "map50_95" if request["model_id"] in DETECTORS else "accuracy"
    old_before_score = None if old_before is None else old_before.get(metric)
    old_after_score = None if old_after is None else old_after.get(metric)
    warnings = list(plan.get("warnings", []))
    if old_before is None:
        warnings.append("舊任務準確率未量測；原權重 SHA256 相同不代表已完成準確率驗證。")
    retention = {"metric": metric, "training_mode": "dual_branch", "before": {"old": old_before},
        "after": {"old": old_after, "new": new_after},
        "delta": {"old": None if old_before_score is None or old_after_score is None else old_after_score - old_before_score},
        "old_branch_unchanged": True, "scope": "獨立分支各自的固定驗證集；未宣稱合併輸出的整體準確率。",
        "warnings": warnings, "coverage": plan.get("coverage", {})}
    audit = {"old_branch_unchanged": True, "verification": "SHA256 of every copied original file before and after training",
             "files": before_files, "original_parameters_sent_to_optimizer": False,
             "old_accuracy_status": "not_measured" if old_before is None else "evaluated"}
    _write(output / "frozen_source_audit.json", audit)
    _write(output / "retention.json", retention)
    metadata = _json(bundle / "workbench_training.json")
    metadata.update(retention=retention, integrity_audit="frozen_source_audit.json")
    _write(bundle / "workbench_training.json", metadata)
    _write(bundle / "frozen_source_audit.json", audit)
    _write(bundle / "retention.json", retention)
    artifacts = {"best": str(bundle), "bundle_manifest": str(bundle / MANIFEST),
                 "retention": str(output / "retention.json"), "integrity_audit": str(output / "frozen_source_audit.json")}
    for name, path in native.get("artifacts", {}).items():
        candidate = Path(path)
        if name not in {"best", "last"} and candidate.is_file() and candidate.resolve().is_relative_to(output):
            artifacts[name] = str(candidate)
    emit({"event": "progress", "progress": 1., "message": "雙分支完成；原權重 SHA256 核對一致，新分支可獨立推論。"})
    return {"checkpoint_path": str(bundle), "artifacts": artifacts,
        "summary": "已保留完整原模型並獨立訓練新分支；推論保留兩分支結果，不做跨分支抑制或共同機率正規化。",
        "training": {"model_id": request["model_id"], "checkpoint_format": FORMAT, "checkpoint_directory": str(bundle),
                     "training_mode": "dual_branch", "classes": plan["classes"], "class_expansion": plan,
                     "retention": retention, "old_branch_unchanged": True,
                     "new_branch": {**native.get("training", {}), "checkpoint_directory": str(bundle / "new")}}}


def run_inference(request, output, emit, *, model_cache=None):
    import time
    from .inference import run_inference as run_native, inference_model_key, resolve_device
    from .inference_cache import InferenceModelCache
    start = time.perf_counter()
    bundle = validate_bundle(request["checkpoint_path"], request["model_id"])
    manifest = _json(bundle / MANIFEST)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = model_cache if model_cache is not None else InferenceModelCache(max_entries=3)
    sequential_eviction = request["model_id"] in LARGE_MODELS
    results, specs = {}, {}
    try:
        bundle_key = inference_model_key(request, resolve_device(request["params"].get("device", "auto")))
        _, bundle_hit = cache.acquire(bundle_key, lambda: {"kind": FORMAT})
        for index, name in enumerate(("original", "new")):
            if sequential_eviction:
                cache.clear()
                emit({"event": "log", "message": "大型雙分支依序載入並釋放快取，避免兩份大型權重同時駐留；換圖需要重新載入。"})
            directory = output / name
            directory.mkdir(exist_ok=True)
            specs[name] = branch_request(request, bundle, name)
            def relay(event):
                event = dict(event)
                if event.get("event") == "progress":
                    event["progress"] = (index + float(event.get("progress", 0))) * .45
                if "message" in event:
                    event["message"] = f"[{name}] {event['message']}"
                emit(event)
            results[name] = run_native(specs[name], directory, relay, model_cache=cache)
        predictions = {name: mapped_predictions(result.get("predictions", []), manifest, name) for name, result in results.items()}
        groups = [{"branch": name, "predictions": predictions[name], "score_semantics": "within_branch_only",
                   "input_params": specs[name]["params"]} for name in ("original", "new")]
        combined = predictions["original"] + predictions["new"]
        media, artifacts = _render_combined(request, output, manifest, results, combined)
        document = {"checkpoint_format": FORMAT, "score_semantics": manifest["score_semantics"],
                    "cross_branch_suppression": False, "prediction_groups": groups,
                    "raw_branch_predictions": {name: result.get("predictions", []) for name, result in results.items()}}
        _write(output / "predictions.json", document)
        artifacts["predictions"] = str(output / "predictions.json")
        for name, result in results.items():
            for artifact, path in result.get("artifacts", {}).items():
                artifacts[f"{name}_{artifact}"] = path
        runtimes = {name: result.get("runtime", {}) for name, result in results.items()}
        return {"media_path": str(media), "media_kind": results["original"]["media_kind"],
            "predictions": combined, "prediction_groups": groups,
            "inference_ms": round(sum(result.get("inference_ms", 0) for result in results.values()), 2),
            "frames": results["original"].get("frames", 1), "artifacts": artifacts,
            "summary": "原／新分支結果分別保留；重疊框不互相刪除，分類分數只在各分支內正規化。",
            "runtime": {"model_cache_hit": bool(bundle_hit and not sequential_eviction and all(row.get("model_cache_hit") for row in runtimes.values())),
                "model_load_seconds": sum(row.get("model_load_seconds", 0) for row in runtimes.values()),
                "total_seconds": round(time.perf_counter() - start, 6), "branches": runtimes,
                "memory_policy": "sequential_eviction" if sequential_eviction else "shared_cache_with_oom_eviction",
                "worker_pid": os.getpid()}}
    finally:
        if model_cache is None or sequential_eviction:
            cache.clear()


def _render_combined(request, output, manifest, results, combined):
    from .inference import draw_predictions, h264_video
    import numpy as np
    from PIL import Image
    source = Path(request["media_path"])
    if results["original"]["media_kind"] == "image":
        with Image.open(source) as opened:
            image = np.asarray(opened.convert("RGB"))
        display = [{**item, "label": f"[{item['branch']}] {item['label']}"} for item in combined]
        title = " | ".join(f"{name}: {next((x['label'] for x in combined if x['branch'] == name), '-') }" for name in ("original", "new"))
        destination = output / "result.png"
        draw_predictions(image, display, title if request["model_id"] not in DETECTORS else None).save(destination)
        return destination, {}
    # Action models classify clips rather than locating boxes in individual
    # frames. Preserve the source timeline and expose both full clip records.
    if request["model_id"] not in DETECTORS and request["model_id"] != "dinov3_7b":
        return Path(results["original"]["media_path"]), {}
    records = {}
    for name, result in results.items():
        path = Path(result["artifacts"]["predictions"])
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = _json(path)["frames"]
        records[name] = {row["frame"]: row for row in rows}
    if set(records["original"]) != set(records["new"]):
        raise RuntimeError("雙分支影片取樣影格不同，不能假裝是相同影格的比較。")
    import cv2
    capture, writer = cv2.VideoCapture(str(source)), None
    merged = []
    raw = output / "combined_raw.mp4"
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not 0 < fps <= 1000:
            raise ValueError("無法解碼雙分支影片。")
        for frame_index in sorted(records["original"]):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("雙分支影片合併時無法讀取原影格。")
            rows = sum((mapped_predictions(records[name][frame_index]["predictions"], manifest, name) for name in ("original", "new")), [])
            groups = [{"branch": name, "predictions": [row for row in rows if row["branch"] == name]} for name in ("original", "new")]
            merged.append({"frame": frame_index, "seconds": frame_index / fps, "prediction_groups": groups})
            title = None if request["model_id"] in DETECTORS else " | ".join(f"{g['branch']}: {g['predictions'][0]['label']}" for g in groups if g["predictions"])
            annotated = draw_predictions(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                [{**row, "label": f"[{row['branch']}] {row['label']}"} for row in rows], title)
            if writer is None:
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / int(request["params"].get("frame_stride", 2)), annotated.size)
                if not writer.isOpened():
                    raise RuntimeError("無法建立雙分支比較影片。")
                annotated.save(output / "preview.jpg")
            writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
        if writer is None:
            raise ValueError("雙分支影片沒有取樣影格。")
        writer.release()
        writer = None
        destination = output / "result.mp4"
        h264_video(raw, destination)
        _write(output / "frames.json", {"frames": merged, "score_semantics": "independent_branch_scores_no_joint_normalization"})
        return destination, {"frames": str(output / "frames.json"), "preview": str(output / "preview.jpg")}
    finally:
        capture.release()
        if writer is not None:
            writer.release()
