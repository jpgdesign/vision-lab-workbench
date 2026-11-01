"""Native ComfyUI graph nodes backed by the existing Vision Workbench engine."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from . import client
from .graph import as_graph, build_values, source_value
from .schema import TYPE_MAP, input_types, resource_id


def _interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _progress(unique_id, row):
    from server import PromptServer
    PromptServer.instance.send_sync("vision_workbench.progress", {"node": str(unique_id), "status": row.get("status"),
                                                               "progress": row.get("progress"), "message": row.get("message", "")})


def _image_tensor(image):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0).unsqueeze(0)


def _output_paths():
    import folder_paths
    return Path(folder_paths.get_temp_directory()), Path(folder_paths.get_output_directory())


def _preview_video(path):
    import av
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return frame.to_image().convert("RGB")
    raise RuntimeError("工作檯結果影片沒有可讀取的影格。")


def execute_output(value, unique_id):
    from PIL import Image
    graph, source = as_graph(value), value.source
    mode = source["source_mode"]
    payload = {"graph": graph, "options": source["options"], "media_id": source["media_id"] or None}
    if mode in {"camera", "vdo"}:
        launch = client.request_json("POST", "/api/comfy/workflows/launch", {**payload, "source_mode": mode, "name": "ComfyUI 視覺應用流程"})
        viewer = launch.get("viewer_url", "")
        if not viewer.startswith(client.BASE_URL + "/?embedded_workflow="):
            raise RuntimeError("工作檯沒有回傳有效的即時預覽入口。")
        message = "即時流程已準備。請在輸出節點的預覽中連接攝影機／VDO.Ninja，再按開始；目前尚未取得影格。"
        return {"ui": {"vision_workbench": [{"mode": mode, "viewer_url": viewer, "message": message}]},
                "result": (_image_tensor(Image.new("RGB", (64, 64), (18, 30, 23))), message)}
    if not source["media_id"]:
        raise ValueError("請在影像來源節點選擇圖片／影片，或改選 camera／vdo。")
    checked = client.request_json("POST", "/api/workflows/validate", payload)
    if not checked.get("valid"):
        raise ValueError("；".join(checked.get("errors", ["工作流預檢失敗。"])))
    job = client.request_json("POST", "/api/workflows/run", payload)
    row = client.wait_for_job(job["id"], _interrupt, lambda current: _progress(unique_id, current))
    result = row.get("result", {})
    media_url = result.get("media_url")
    if not media_url:
        raise RuntimeError("工作檯未回傳輸出圖片或影片。")
    temporary, output = _output_paths()
    token = uuid.uuid4().hex
    preview_file = temporary / "vision-workbench" / f"preview_{token}.png"
    preview_file.parent.mkdir(parents=True, exist_ok=True)
    details = {"mode": "media", "job_id": job["id"], "message": result.get("summary", "工作流已完成"),
               "workbench_url": client.BASE_URL, "artifact_urls": result.get("artifact_urls", {})}
    if result.get("media_kind") == "video":
        filename = f"workflow_{token}.mp4"
        video_path = output / "vision-workbench" / filename
        client.download_artifact(media_url, video_path)
        image = _preview_video(video_path)
        details["video"] = {"filename": filename, "subfolder": "vision-workbench", "type": "output"}
        image.save(preview_file)
    else:
        downloaded = temporary / "vision-workbench" / f"result_{token}.jpg"
        client.download_artifact(media_url, downloaded)
        with Image.open(downloaded) as opened:
            image = opened.convert("RGB")
        image.save(preview_file)
        downloaded.unlink()
    report = json.dumps({"summary": result.get("summary"), "frames": result.get("frames"), "event_count": result.get("event_count"),
                         "node_stats": result.get("node_stats"), "artifact_urls": result.get("artifact_urls", {})}, ensure_ascii=False, indent=2)
    return {"ui": {"images": [{"filename": preview_file.name, "subfolder": "vision-workbench", "type": "temp"}],
                   "vision_workbench": [details]}, "result": (_image_tensor(image), report)}


class _Node:
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return input_types(cls.SPEC, client.resources())

    @classmethod
    def VALIDATE_INPUTS(cls, model_id=None, checkpoint_id=None, media_id=None):
        # API workflows may use stable raw IDs; GUI combos add readable labels.
        # Only these three combo fields bypass Comfy's literal-membership check.
        # Actual model/checkpoint/task compatibility is checked by Workbench.
        try:
            for value in (model_id, checkpoint_id, media_id):
                if value is not None:
                    resource_id(value)
        except ValueError as exc:
            return str(exc)
        return True

    def build(self, unique_id, **values):
        if self.SPEC["type"] == "source":
            return (source_value(unique_id, **values),)
        built = build_values(self.SPEC, unique_id, values)
        if self.SPEC["type"] == "output":
            return execute_output(built[0], unique_id)
        return built


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _spec in client.catalogue():
    _type = _spec["type"]
    _name = "VisionWB_" + _type
    _attributes = {"SPEC": _spec, "CATEGORY": "Vision Workbench/" + _spec["category"],
                   "DESCRIPTION": _spec["description"] + " 此節點在 Vision Workbench 本機環境執行；模型權重沿用已登錄路徑。",
                   "RETURN_TYPES": tuple(TYPE_MAP[port["type"]] for port in _spec["outputs"]),
                   "RETURN_NAMES": tuple(port["name"] for port in _spec["outputs"])}
    if _type == "output":
        _attributes.update(OUTPUT_NODE=True, RETURN_TYPES=("IMAGE", "STRING"), RETURN_NAMES=("preview", "report"),
                           IS_CHANGED=classmethod(lambda cls, **kwargs: float("nan")))
    NODE_CLASS_MAPPINGS[_name] = type(_name, (_Node,), _attributes)
    NODE_DISPLAY_NAME_MAPPINGS[_name] = "Vision Workbench · " + _spec["label"]
