"""Shared field ordering for native ComfyUI nodes and workflow conversion."""
from __future__ import annotations

import json

NONE_MEDIA = "(未選擇素材)"
PRETRAINED = "原始預訓練權重"
AUTO_MODEL = "使用已設定的 API 模型"
SOURCE_WIDGETS = ("source_mode", "media_id", "device", "frame_stride", "max_frames", "max_side")
TYPE_MAP = {"frame": "VWB_FRAME", "detections": "VWB_OBSERVATIONS", "poses": "VWB_OBSERVATIONS",
            "tracks": "VWB_OBSERVATIONS", "observations": "VWB_OBSERVATIONS",
            "actions": "VWB_SIGNAL", "events": "VWB_SIGNAL", "relations": "VWB_SIGNAL", "embeddings": "VWB_SIGNAL"}


def resource_id(value):
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError("資源選項格式不正確。")
    if value in {NONE_MEDIA, PRETRAINED, AUTO_MODEL, ""}:
        return ""
    return value.split(" | ", 1)[0].strip()


def resource_label(item, category=""):
    prefix = f"{item.get('model_id', '')} · " if category == "checkpoint" else ""
    return f"{item['id']} | {prefix}{item.get('name', item['id'])}"


def expanded_inputs(spec):
    result = []
    for port in spec["inputs"]:
        for index in range(8 if port.get("multiple") else 1):
            name = port["name"] if index == 0 else f"{port['name']}_{index + 1}"
            result.append({"name": name, "port": port["name"], "type": TYPE_MAP[port["type"]],
                           "required": bool(port.get("required") and index == 0)})
    return result


def input_types(spec, resources):
    if spec["type"] == "source":
        media = [resource_label(item) for item in resources["media"] if item.get("kind") in {"image", "video"}]
        return {"required": {
            "source_mode": (["media", "camera", "vdo"], {"default": "media"}),
            "media_id": ([NONE_MEDIA, *media], {"default": media[0] if media else NONE_MEDIA}),
            "device": (["auto", "cpu", "cuda:0"], {"default": "auto"}),
            "frame_stride": ("INT", {"default": 1, "min": 1, "max": 60}),
            "max_frames": ("INT", {"default": 300, "min": 1, "max": 3000}),
            "max_side": ("INT", {"default": 1280, "min": 256, "max": 1600}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}
    required, optional = {}, {}
    for port in expanded_inputs(spec):
        (required if port["required"] else optional)[port["name"]] = (port["type"], {"forceInput": True})
    for param in spec.get("params", []):
        key, default, kind = param["key"], param["default"], param["type"]
        if key == "model_id":
            if spec["type"] == "vlm":
                choices = [AUTO_MODEL, *resources["integration"].get("models", [])]
                default = default or AUTO_MODEL
            else:
                tasks = {param.get("model_task")}
                if "detection" in tasks:
                    tasks.add("segmentation")
                models = [item for item in resources["models"] if item.get("inference_supported") and item.get("task") in tasks]
                choices = [resource_label(item) for item in models]
                default = next((resource_label(item) for item in models if item["id"] == default), default)
                if default not in choices:
                    choices.insert(0, default)
            field = (choices, {"default": default})
        elif key == "checkpoint_id":
            choices = [PRETRAINED, *[resource_label(item, "checkpoint") for item in resources["checkpoints"] if item.get("valid")]]
            field = (choices, {"default": PRETRAINED})
        elif kind == "boolean":
            field = ("BOOLEAN", {"default": default})
        elif kind == "number":
            integer = type(default) is int and type(param.get("step", 1)) is int
            options = {key: value for key, value in param.items() if key in {"default", "min", "max", "step"}}
            if integer:
                options = {key: int(value) for key, value in options.items()}
            field = ("INT" if integer else "FLOAT", options)
        elif kind == "select":
            field = (param.get("options", [default]), {"default": default})
        else:
            field = ("STRING", {"default": json.dumps(default, ensure_ascii=False) if kind == "json" else default,
                                 "multiline": kind == "json" or key in {"instruction", "prompt"}})
        # Older saved/API prompts predate this appended tracker widget. Comfy
        # validates required fields before build_values can supply defaults.
        # Keep the widget optional so those prompts retain the legacy IoU mode.
        target = optional if (spec["type"], key) == ("tracker", "association_mode") else required
        target[key] = field
    return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}
