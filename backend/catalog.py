"""Explicit model capabilities and validated parameters; no ML imports at startup."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

from .model_licenses import RF_DETR_LICENSE_URL, is_model_license_accepted
from .model_registry import WeightPaths

APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = APP_ROOT.parent
WEIGHTS = WeightPaths()


def param(key, label, default, *, options=None, minimum=None, maximum=None, step=None):
    kind = "select" if options is not None else ("boolean" if isinstance(default, bool) else "number" if isinstance(default, (int, float)) else "string")
    result = {"key": key, "label": label, "type": kind, "default": default}
    if options is not None:
        result["options"] = options
    if minimum is not None:
        result["min"] = minimum
    if maximum is not None:
        result["max"] = maximum
    if step is not None:
        result["step"] = step
    return result


DEVICE = param("device", "運算裝置", "auto", options=["auto", "cuda:0", "cpu"])
VIDEO_PARAMS = [param("frame_stride", "影片每幾幀取樣一次", 2, minimum=1, maximum=60, step=1),
                param("max_frames", "影片最多處理幀數", 300, minimum=1, maximum=3000, step=1)]


def get_models():
    from .training import training_capabilities
    from .system_info import available_device_options
    device = deepcopy(DEVICE)
    device["options"] = available_device_options()
    models = []
    definitions = [
        ("yolov12x", "YOLOv12x", "detection", "ultralytics", "Attention 物件偵測"),
        ("yolo26x", "YOLO26x", "detection", "ultralytics", "即時物件偵測"),
        ("grounding_dino_base", "Grounding DINO Base", "detection", "transformers", "以文字指定目標的開放詞彙偵測"),
        ("rf_detr_2xl", "RF-DETR 2XL", "detection", "rfdetr_plus", "需確認 PML 授權後使用"),
        ("vivit_base_k400", "ViViT Base", "action", "transformers", "Kinetics-400 · 32 幀"),
        ("x3d_l", "X3D-L", "action", "pytorchvideo", "Kinetics-400 · 16 幀"),
        ("slowfast_r101_8x8", "SlowFast R101", "action", "pytorchvideo", "Kinetics-400 · 快慢雙路徑"),
        ("videomaev2_giant_k710", "VideoMAE V2 Giant", "action", "timm", "Kinetics-710 · FP16 · 16 幀"),
        ("dinov3_7b", "DINOv3 ViT-7B", "embedding", "transformers", "視覺表徵模型；推論與微調接頭尚未提供"),
    ]
    for model_id, name, task, dependency, description in definitions:
        license_accepted = is_model_license_accepted(model_id)
        if model_id == "rf_detr_2xl" and license_accepted:
            description = "已記住 PML 1.0 使用授權確認"
        weight_error = ""
        try:
            path = WEIGHTS[model_id]
        except (OSError, ValueError, TypeError) as exc:
            path = APP_ROOT / "models" / model_id / "unavailable"
            weight_error = str(exc)
        environment_available = importlib.util.find_spec(dependency) is not None and importlib.util.find_spec("torch") is not None
        available = path.is_file() and environment_available and not weight_error
        reason = "" if available else (weight_error or "找不到完整本機權重；請掃描或綁定模型路徑。" if not path.is_file() else "執行環境尚未安裝完成，請執行 setup.ps1")
        params = [deepcopy(device)]
        if model_id in ("yolov12x", "yolo26x"):
            params += [param("conf", "信心閾值", 0.25, minimum=0.01, maximum=0.99, step=0.01),
                       param("imgsz", "輸入邊長", 640, options=[320, 480, 640, 960, 1280]),
                       param("max_det", "最多偵測物件", 100, minimum=1, maximum=300, step=1)]
            if model_id == "yolov12x":
                params += [param("iou", "NMS IoU 閾值", 0.7, minimum=0.1, maximum=0.95, step=0.05)]
            params += deepcopy(VIDEO_PARAMS)
        elif model_id == "grounding_dino_base":
            params += [param("prompt", "偵測目標（英文，以句點分隔）", "person . car . bicycle . dog ."),
                       param("conf", "物件閾值", 0.3, minimum=0.01, maximum=0.99, step=0.01),
                       param("text_threshold", "文字閾值", 0.25, minimum=0.01, maximum=0.99, step=0.01)] + deepcopy(VIDEO_PARAMS)
        elif model_id == "rf_detr_2xl":
            params += [param("conf", "信心閾值", 0.3, minimum=0.01, maximum=0.99, step=0.01),
                       param("accept_platform_license", "我已閱讀 PML 1.0 並確認具備所需使用授權", license_accepted)] + deepcopy(VIDEO_PARAMS)
        elif task == "action":
            params += [param("num_clips", "均勻取樣片段數", 3, minimum=1, maximum=16, step=1),
                       param("clip_seconds", "每段秒數", 4.0, minimum=0.5, maximum=16.0, step=0.5),
                       param("start_seconds", "開始時間（秒）", 0.0, minimum=0, maximum=86400, step=0.5),
                       param("top_k", "顯示前幾名", 5, minimum=1, maximum=20, step=1)]
        training = training_capabilities(model_id)
        training = deepcopy(training or {})
        training.setdefault("supported_params", list(training.get("defaults", {})))
        training_supported = bool(training.get("dataset_formats"))
        if not training_supported:
            training["unavailable_reason"] = {
                "rf_detr_2xl": "工作台尚未接入 RF-DETR 的資料轉換、訓練驗證與微調權重回測流程；模型本身可以微調。",
                "grounding_dino_base": "工作台尚未接入 Grounding DINO 的文字與框標註配對、訓練驗證及微調權重回測流程；模型本身可以微調。",
                "dinov3_7b": "工作台尚未建立 DINOv3 的微調流程。",
            }.get(model_id, "工作台尚未接入此模型的微調流程。")
        if model_id == "dinov3_7b":
            available = False
            reason = "工作台尚未提供 DINOv3 推論與微調接頭；可登錄完整模型檔案，不會自動下載。"
        models.append({"id": model_id, "name": name, "task": task, "description": description,
                       "weight_path": str(path), "weight_bytes": path.stat().st_size if path.is_file() else 0,
                       "available": available, "environment_available": environment_available, "reason": reason,
                       "inference_supported": task != "embedding", "training_supported": training_supported,
                       "capabilities": {"input_types": ["video"] if task == "action" else ["image", "video"],
                                        "inference_params": params, "training": training},
                       "license_accepted": license_accepted,
                       "license_url": RF_DETR_LICENSE_URL if model_id == "rf_detr_2xl" else None})
    return models


def validate_job_request(model_id, kind, params, *, allow_missing_weights=False):
    if not isinstance(params, dict):
        raise ValueError("params 必須是物件")
    model = next((m for m in get_models() if m["id"] == model_id), None)
    if model is None:
        raise ValueError("未知模型")
    if not model["available"] and not (allow_missing_weights and kind == "inference" and model.get("environment_available")):
        raise ValueError(model["reason"])
    if kind == "training":
        if not model["training_supported"]:
            raise ValueError(model["capabilities"]["training"]["unavailable_reason"])
        from .training import normalize_training_params
        return normalize_training_params(model_id, params)
    if kind != "inference" or not model["inference_supported"]:
        raise ValueError("模型不支援此工作類型")
    definitions = {p["key"]: p for p in model["capabilities"]["inference_params"]}
    unknown = set(params) - set(definitions)
    if unknown:
        raise ValueError("模型不支援的參數：" + ", ".join(sorted(unknown)))
    clean = {}
    for key, spec in definitions.items():
        value = params.get(key, spec["default"])
        if spec["type"] == "number":
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{spec['label']} 必須是有限數字")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{spec['label']} 必須介於 {spec['min']} 與 {spec['max']}")
            if isinstance(spec["default"], int) and not float(value).is_integer():
                raise ValueError(f"{spec['label']} 必須是整數")
        elif spec["type"] == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{spec['label']} 必須是布林值")
        elif spec["type"] == "select" and value not in spec["options"]:
            raise ValueError(f"{spec['label']} 的選項無效")
        elif spec["type"] == "string":
            if not isinstance(value, str) or not value.strip() or len(value) > 2000:
                raise ValueError(f"{spec['label']} 需要 1–2000 字元")
            value = value.strip()
        clean[key] = value
    if model_id == "rf_detr_2xl":
        # Validate the caller's types first, then refresh saved cards whose
        # old checkbox value predates this workspace's explicit acceptance.
        if model["license_accepted"]:
            clean["accept_platform_license"] = True
        if not clean["accept_platform_license"]:
            raise ValueError("RF-DETR 2XL 必須先閱讀 PML 1.0 並確認具備所需授權，再勾選確認。")
    return clean
