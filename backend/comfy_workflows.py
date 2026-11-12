"""Export validated Vision Workbench graphs as native ComfyUI workflows.

The 0.4 document format is still supported by the installed ComfyUI frontend and
its own blueprints. Graph packets remain the plugin's VWB_* custom types; they
are not mislabeled as ComfyUI IMAGE tensors. Only the terminal renderer emits an
IMAGE preview and a STRING report.
"""
from __future__ import annotations

import json
import uuid

from .workflow_graph import CATALOG, validate_graph


PREFIX = "VisionWB_"
MAX_INPUTS = 8
WIRE_TYPES = {"frame": "VWB_FRAME", "detections": "VWB_OBSERVATIONS", "poses": "VWB_OBSERVATIONS",
              "tracks": "VWB_OBSERVATIONS", "observations": "VWB_OBSERVATIONS",
              "actions": "VWB_SIGNAL", "events": "VWB_SIGNAL", "relations": "VWB_SIGNAL", "embeddings": "VWB_SIGNAL"}
SOURCE_WIDGETS = ("source_mode", "media_id", "device", "frame_stride", "max_frames", "max_side")
SOURCE_DEFAULTS = {"device": "auto", "frame_stride": 1, "max_frames": 300, "max_side": 1280}


def input_slots(kind):
    """Plugin socket order: original port, then _2 ... _8 for multi-inputs."""
    return [{"name": port["name"] if index == 1 else f"{port['name']}_{index}",
             "port": port["name"], "type": WIRE_TYPES[port["type"]],
             "required": port["required"] and index == 1}
            for port in CATALOG[kind]["inputs"] for index in range(1, MAX_INPUTS + 1 if port["multiple"] else 2)]


def output_slots(kind):
    if kind == "output":
        return [{"name": "preview", "type": "IMAGE"}, {"name": "report", "type": "STRING"}]
    return [{"name": port["name"], "type": WIRE_TYPES[port["type"]]} for port in CATALOG[kind]["outputs"]]


def _source(source_mode, media_id, options):
    source_mode = "camera" if source_mode == "live" else source_mode
    if not isinstance(source_mode, str) or source_mode not in {"media", "camera", "vdo"}:
        raise ValueError("ComfyUI 來源模式需為 media、camera 或 vdo。")
    if not isinstance(media_id, str) or len(media_id) > 512:
        raise ValueError("ComfyUI 素材 ID 無效。")
    options = {} if options is None else options
    if not isinstance(options, dict) or set(options) - set(SOURCE_DEFAULTS):
        raise ValueError("ComfyUI 來源包含未支援的選項。")
    values = {**SOURCE_DEFAULTS, **options}
    if not isinstance(values["device"], str) or values["device"] not in {"auto", "cpu", "cuda:0"}:
        raise ValueError("ComfyUI 執行裝置無效。")
    for key, lower, upper in (("frame_stride", 1, 60), ("max_frames", 1, 3000), ("max_side", 256, 1600)):
        value = values[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"ComfyUI {key} 需為 {lower}–{upper} 的整數。")
    return {"source_mode": source_mode, "media_id": media_id or "(未選擇素材)", **values}


def _label(key, value, resources, model_id=""):
    if key == "checkpoint_id" and not value:
        return "原始預訓練權重"
    if not resources or not value or " | " in str(value):
        return value
    resource_key = {"model_id": "models", "checkpoint_id": "checkpoints", "media_id": "media"}.get(key)
    if not resource_key:
        return value
    entries = resources.get(resource_key, [])
    if isinstance(entries, dict):
        entries = entries.get(resource_key, [])
    entry = next((item for item in entries if isinstance(item, dict) and item.get("id") == value), None)
    if entry:
        name = entry.get("name", value)
        if key == "checkpoint_id":
            name = f"{entry.get('model_id') or model_id} · {name}"
        return f"{value} | {name}"
    return value


def _widgets(node, source, resources=None):
    if node["type"] == "source":
        values = dict(source)
        values["media_id"] = _label("media_id", values["media_id"], resources)
        return values
    values = {}
    for definition in CATALOG[node["type"]]["params"]:
        key = definition["key"]
        value = node["params"][key]
        if definition["type"] == "json":
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        elif node["type"] == "vlm" and key == "model_id" and not value:
            value = "使用已設定的 API 模型"
        else:
            value = _label(key, value, resources, node["params"].get("model_id", ""))
        values[key] = value
    return values


def _prepare(graph, name, source_mode, media_id, options, resources):
    if isinstance(graph, dict) and "graph" in graph:
        name = name or graph.get("name", "")
        graph = graph["graph"]
    if not isinstance(name, str) or len(name) > 160:
        raise ValueError("ComfyUI 工作流名稱無效。")
    checked = validate_graph(graph)
    if not checked["valid"]:
        raise ValueError("；".join(checked["errors"]))
    source = _source(source_mode, media_id, options)
    graph = checked["graph"]
    identifiers = {node["id"]: index + 1 for index, node in enumerate(graph["nodes"])}
    nodes = {node["id"]: node for node in graph["nodes"]}
    used, links = {}, []
    for index, edge in enumerate(graph["edges"]):
        a, b = nodes[edge["source"]], nodes[edge["target"]]
        key = (b["id"], edge["targetHandle"])
        used[key] = used.get(key, 0) + 1
        if used[key] > MAX_INPUTS:
            raise ValueError(f"{b['id']}.{edge['targetHandle']} 最多可轉換 {MAX_INPUTS} 路 ComfyUI 輸入。")
        slot_name = edge["targetHandle"] + (f"_{used[key]}" if used[key] > 1 else "")
        outputs = output_slots(a["type"])
        source_slot = next(i for i, item in enumerate(outputs) if item["name"] == edge["sourceHandle"])
        target_slot = next(i for i, item in enumerate(input_slots(b["type"])) if item["name"] == slot_name)
        links.append({"id": index + 1, "source": identifiers[a["id"]], "source_slot": source_slot,
                      "target": identifiers[b["id"]], "target_slot": target_slot, "target_name": slot_name,
                      "type": outputs[source_slot]["type"]})
    widget_values = {node["id"]: _widgets(node, source, resources) for node in graph["nodes"]}
    return graph, identifiers, links, widget_values, checked["plan"]["order"], source, name or "視覺應用工作流"


def to_api_prompt(graph, name="", source_mode="media", media_id="", options=None, *, resources=None):
    """Build the dict sent as /prompt's ``prompt`` value (never execute it)."""
    graph, identifiers, links, widgets, _, _, _ = _prepare(graph, name, source_mode, media_id, options, resources)
    result = {}
    for node in graph["nodes"]:
        node_id = identifiers[node["id"]]
        inputs = dict(widgets[node["id"]])
        for edge in links:
            if edge["target"] == node_id:
                inputs[edge["target_name"]] = [str(edge["source"]), edge["source_slot"]]
        result[str(node_id)] = {"class_type": PREFIX + node["type"], "inputs": inputs,
                                "_meta": {"title": CATALOG[node["type"]]["label"]}}
    return result


def convert_workflow(graph, name="", source_mode="media", media_id="", options=None, *, resources=None):
    """Create a loadable ComfyUI canvas with visible nodes, wires and widgets."""
    graph, identifiers, links, widgets, order, source, name = _prepare(graph, name, source_mode, media_id, options, resources)
    canvas_nodes = []
    for node in graph["nodes"]:
        kind, node_id = node["type"], identifiers[node["id"]]
        inputs, outputs = input_slots(kind), output_slots(kind)
        for slot, entry in enumerate(inputs):
            entry.pop("port", None)
            entry.pop("required", None)
            entry["link"] = next((edge["id"] for edge in links if edge["target"] == node_id and edge["target_slot"] == slot), None)
        for slot, entry in enumerate(outputs):
            entry["links"] = [edge["id"] for edge in links if edge["source"] == node_id and edge["source_slot"] == slot]
            entry["slot_index"] = slot
        params = CATALOG[kind]["params"]
        widget_height = sum(100 if param["type"] == "json" else 30 for param in params)
        if kind == "source": widget_height = len(SOURCE_WIDGETS) * 30
        height = max(130, 70 + max(len(inputs), len(outputs)) * 20 + widget_height)
        canvas_nodes.append({"id": node_id, "type": PREFIX + kind, "title": CATALOG[kind]["label"],
                             "pos": [node["position"]["x"] * 1.35, node["position"]["y"] * 1.5],
                             "size": [350, height], "flags": {}, "order": order.index(node["id"]), "mode": 0,
                             "inputs": inputs, "outputs": outputs,
                             "properties": {"Node name for S&R": PREFIX + kind, "vision_workbench_id": node["id"]},
                             "widgets_values": list(widgets[node["id"]].values()),
                             "color": "#263c32", "bgcolor": "#1b2922"})
    stable = json.dumps({"name": name, "graph": graph, "source": source}, ensure_ascii=False, sort_keys=True)
    left = min(node["pos"][0] for node in canvas_nodes) - 40
    top = min(node["pos"][1] for node in canvas_nodes) - 90
    right = max(node["pos"][0] + node["size"][0] for node in canvas_nodes) + 40
    bottom = max(node["pos"][1] + node["size"][1] for node in canvas_nodes) + 40
    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "vision-workbench:" + stable)), "revision": 0,
            "last_node_id": max(identifiers.values()), "last_link_id": len(links),
            "nodes": canvas_nodes,
            "links": [[edge["id"], edge["source"], edge["source_slot"], edge["target"], edge["target_slot"], edge["type"]] for edge in links],
            "groups": [{"id": 1, "title": name, "bounding": [left, top, right - left, bottom - top],
                        "color": "#355b49", "font_size": 26, "flags": {}}], "config": {}, "version": .4,
            "extra": {"ds": {"scale": .65, "offset": [40, 40]}, "vision_workbench": {
                "name": name, "source_mode": source["source_mode"], "graph_version": 1,
                "node_ids": {str(value): key for key, value in identifiers.items()}}}}


api_prompt = to_api_prompt
