"""Immutable graph fragments passed through native ComfyUI sockets."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import re

from .schema import expanded_inputs, resource_id


@dataclass(frozen=True)
class GraphValue:
    nodes: dict
    edges: tuple
    source: dict
    node_id: str
    port_name: str
    port_type: str


def node_id(unique_id):
    value = "comfy_" + str(unique_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("ComfyUI 節點 ID 不符合工作檯格式。")
    return value


def source_value(unique_id, source_mode="media", media_id="", device="auto", frame_stride=1, max_frames=300, max_side=1280):
    if source_mode not in {"media", "camera", "vdo"}:
        raise ValueError("請選擇圖片／影片、攝影機或 VDO.Ninja。")
    ident = node_id(unique_id)
    source = {"node_id": ident, "source_mode": source_mode, "media_id": resource_id(media_id),
              "options": {"device": device, "frame_stride": frame_stride, "max_frames": max_frames, "max_side": max_side}}
    return GraphValue({ident: {"id": ident, "type": "source", "params": {}, "position": {"x": 0, "y": 0}}}, (), source, ident, "frame", "frame")


def build_values(spec, unique_id, values):
    ident = node_id(unique_id)
    nodes, edges, source, connected = {}, {}, None, set()
    ports = {port["name"]: port for port in spec["inputs"]}
    for entry in expanded_inputs(spec):
        value = values.get(entry["name"])
        if value is None:
            if entry["required"]:
                raise ValueError(f"{spec['label']} 缺少 {entry['name']} 連線。")
            continue
        if not isinstance(value, GraphValue):
            raise ValueError("請連接 Vision Workbench 的相容節點。")
        port = ports[entry["port"]]
        if value.port_type not in port.get("accepts", [port["type"]]):
            raise ValueError(f"{value.port_type} 無法連到 {entry['name']}；請檢查資料類型。")
        if source is not None and source != value.source:
            raise ValueError("同一流程的所有分支必須使用同一個影像來源節點。")
        source = copy.deepcopy(value.source)
        for key, node in value.nodes.items():
            if key in nodes and nodes[key] != node:
                raise ValueError("同一節點在不同分支包含不一致的參數。")
            nodes[key] = copy.deepcopy(node)
        for edge in value.edges:
            edges[tuple(edge[k] for k in ("source", "sourceHandle", "target", "targetHandle"))] = copy.deepcopy(edge)
        key = (value.node_id, value.port_name, ident, entry["port"])
        if key in connected:
            raise ValueError("相同來源不能重複接入同一個插槽。")
        connected.add(key)
        edges[key] = {"id": "edge_" + str(len(edges)), "source": value.node_id, "sourceHandle": value.port_name,
                      "target": ident, "targetHandle": entry["port"]}
    if source is None:
        raise ValueError("流程缺少影像來源。")
    params = {}
    for param in spec.get("params", []):
        key, kind = param["key"], param["type"]
        value = values.get(key, copy.deepcopy(param["default"]))
        if key in {"model_id", "checkpoint_id"}:
            value = resource_id(value)
        elif kind == "json" and isinstance(value, str):
            if len(value) > 100_000:
                raise ValueError(f"{key} 參數過大。")
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError(f"{key} 需為有效 JSON。") from exc
        params[key] = value
    nodes[ident] = {"id": ident, "type": spec["type"], "params": params, "position": {"x": len(nodes) * 300, "y": 0}}
    if len(nodes) > 48 or len(edges) > 128:
        raise ValueError("工作檯流程最多 48 個節點與 128 條連線。")
    return tuple(GraphValue(copy.deepcopy(nodes), tuple(copy.deepcopy(list(edges.values()))), copy.deepcopy(source), ident, port["name"], port["type"])
                 for port in spec["outputs"])


def as_graph(value):
    graph = {"version": 1, "nodes": list(value.nodes.values()), "edges": list(value.edges)}
    # Edge IDs are cosmetic, but must be unique when exported back to a canvas.
    graph["edges"] = [{**edge, "id": f"comfy_edge_{index}"} for index, edge in enumerate(graph["edges"])]
    if len(json.dumps(graph, ensure_ascii=False, allow_nan=False)) > 500_000:
        raise ValueError("工作檯流程超過 500 KB。")
    return graph
