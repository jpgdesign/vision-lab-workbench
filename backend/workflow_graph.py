"""Portable, typed visual graphs. Validation never imports an ML framework."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re


def port(name, kind, *, required=True, multiple=False, accepts=None):
    return {"name": name, "label": name, "type": kind, "required": required,
            "multiple": multiple, **({"accepts": accepts} if accepts else {})}


def param(key, label, default, kind=None, **extra):
    kind = kind or ("boolean" if isinstance(default, bool) else "number" if isinstance(default, (int, float)) else "json" if isinstance(default, (list, dict)) else "string")
    return {"key": key, "label": label, "default": default, "type": kind, **extra}


def node(kind, label, category, description, inputs, outputs, params=(), **extra):
    return {"type": kind, "label": label, "category": category, "description": description,
            "inputs": inputs, "outputs": outputs, "params": list(params), **extra}


NODE_TYPES = [
    node("source", "影像／影片來源", "來源", "使用左側選取的素材；所有節點共用原始時間軸。", [], [port("frame", "frame")]),
    node("detector", "物件偵測／分割", "模型", "可選既有模型與微調權重；分割模型同時回傳區域外框。", [port("frame", "frame")], [port("detections", "detections")], [
        param("model_id", "模型", "yolo26x", "select", model_task="detection"), param("checkpoint_id", "微調權重", "", "select"),
        param("conf", "信心閾值", .25, min=.01, max=.99, step=.01), param("imgsz", "輸入邊長", 640, min=320, max=1280, step=32),
        param("max_det", "最多物件", 40, min=1, max=100, step=1), param("prompt", "Grounding DINO 文字目標", "person . helmet . safety vest ."),
        param("use_training_classes", "使用完整微調類別", False)]),
    node("pose", "人體骨架 · YOLO26n Pose", "模型", "本機 COCO 17 關節與信心分數；不是動作分類。", [port("frame", "frame")], [port("poses", "poses")], [
        param("conf", "人體信心閾值", .35, min=.01, max=.99, step=.01), param("imgsz", "輸入邊長", 640, min=320, max=1280, step=32)]),
    node("embedding", "視覺表徵／影像分類 · DINOv3", "模型", "輸出真正的 CLS 向量；若選擇已訓練的分類 checkpoint，另輸出影像類別。原始表徵不會自動產生物件標籤。", [port("frame", "frame")], [port("embeddings", "embeddings"), port("actions", "actions")], [
        param("model_id", "表徵模型", "dinov3_7b", "select", model_task="embedding"), param("checkpoint_id", "分類微調權重", "", "select"),
        param("every_seconds", "推論間隔（秒）", 1., min=.1, max=120, step=.1), param("normalize", "向量 L2 正規化", True),
        param("top_k", "分類候選數", 5, min=1, max=20, step=1), param("min_score", "分類信心下限", .25, min=0, max=1, step=.01),
        param("max_gap_seconds", "觀察最大間隔（秒）", 5., min=.1, max=120, step=.1)]),
    node("tracker", "物件追蹤", "處理", "以 IoU 與位置配對維持工作內的追蹤 ID；遮擋後重新出現可能取得新 ID。", [port("observations", "observations", accepts=["detections", "poses"])], [port("tracks", "tracks")], [
        param("labels", "追蹤類別（空陣列＝全部）", ["person"]), param("max_age_seconds", "追蹤失聯上限（秒）", 1., min=.1, max=10, step=.1),
        param("min_iou", "配對 IoU 下限", .15, min=.01, max=.95, step=.01)]),
    node("zone", "區域／停留規則", "規則", "可畫多個區域，輸出進出、停留及人數事件；失聯不當作已離開。", [port("tracks", "tracks")], [port("tracks", "tracks"), port("events", "events")], [
        param("zones", "區域（名稱與相對座標）", [{"name": "restricted", "points": [[.35,.2],[.8,.2],[.8,.95],[.35,.95]]}]),
        param("anchor", "判定位置", "foot", "select", options=["foot", "center"]), param("dwell_seconds", "停留事件門檻（秒）", 3., min=0, max=3600, step=.1),
        param("occupancy_limit", "人數上限（0＝不判定）", 0, min=0, max=1000, step=1), param("max_gap_seconds", "連續觀察最大間隔（秒）", 1., min=.1, max=30, step=.1)]),
    node("posture", "姿態／跌倒候選規則", "規則", "依幾何關節與姿態轉換判定，與學習式動作分類分開；遮擋時為 unknown。", [port("tracks", "tracks")], [port("actions", "actions")], [
        param("keypoint_confidence", "關節信心下限", .5, min=.05, max=.99, step=.05),
        param("fall_transition_seconds", "姿態快速轉換（秒）", 1.2, min=.1, max=10, step=.1), param("fall_hold_seconds", "躺姿持續（秒）", 1.5, min=.1, max=30, step=.1)]),
    node("skeleton_action", "骨架動作 · ST-GCN++", "模型", "NTU120 預訓練時序模型；每個追蹤 ID 保持獨立關節序列。短序列／缺關節不作可靠動作判定。", [port("tracks", "tracks")], [port("actions", "actions")], [
        param("window_seconds", "骨架時間窗（秒）", 4., min=1, max=15, step=.5), param("min_frames", "最少有效影格", 32, min=30, max=100, step=1),
        param("every_seconds", "分類間隔（秒）", .75, min=.1, max=10, step=.1), param("min_score", "分類信心下限", .35, min=.01, max=.99, step=.01)]),
    node("video_action", "影片動作分類", "模型", "連接 X3D／SlowFast／ViViT／VideoMAE；整幅影像只代表場景，個人模式需追蹤框裁切。", [port("frame", "frame"), port("tracks", "tracks", required=False)], [port("actions", "actions")], [
        param("model_id", "行為模型", "x3d_l", "select", model_task="action"), param("checkpoint_id", "微調權重", "", "select"),
        param("scope", "分類範圍", "scene", "select", options=["scene", "person"]), param("window_seconds", "片段長度（秒）", 2., min=.5, max=10, step=.5),
        param("every_seconds", "分類間隔（秒）", 1., min=.25, max=10, step=.25), param("min_score", "分類信心下限", .25, min=.01, max=.99, step=.01),
        param("max_people", "最多同時分類人數", 3, min=1, max=10, step=1)]),
    node("condition", "狀態／違規條件", "規則", "依標籤、分數與持續時間產生事件；unknown 不累計，場景事件不冒稱屬於特定人。", [port("actions", "actions", accepts=["actions", "relations", "events"])], [port("events", "events")], [
        param("labels", "觸發標籤", ["falling", "fall_candidate"]), param("min_score", "分數下限", .35, min=0, max=1, step=.01),
        param("min_duration", "持續時間（秒）", 0., min=0, max=3600, step=.1), param("cooldown", "重複事件間隔（秒）", 5., min=0, max=3600, step=.5),
        param("max_gap_seconds", "兩次有效觀察最大間隔（秒）", 1., min=.1, max=120, step=.1)]),
    node("sequence", "流程順序約束", "規則", "逐人檢查事件／動作順序，產生順序錯誤、完成與逾時事件；不混合不同追蹤 ID。", [port("signals", "events", accepts=["events", "actions"])], [port("events", "events")], [
        param("steps", "順序步驟", ["zone_enter:check", "zone_enter:work", "zone_enter:exit"]),
        param("max_gap_seconds", "步驟最大間隔（秒）", 30., min=1, max=3600, step=1), param("reset_on_complete", "完成後重新開始", True)]),
    node("ppe", "PPE 穿戴規則", "規則", "合併人體追蹤與裝備偵測；安全帽判定需要可見頭部關節，遮擋回傳未知。", [port("tracks", "tracks"), port("detections", "detections")], [port("events", "events")], [
        param("helmet_labels", "安全帽標籤", ["helmet", "hard hat"]), param("vest_labels", "反光背心標籤", ["safety vest", "vest"]),
        param("require_helmet", "檢查安全帽", True), param("require_vest", "檢查反光背心", False),
        param("keypoint_confidence", "頭部關節信心下限", .5, min=.05, max=.99, step=.05), param("min_score", "裝備信心下限", .3, min=0, max=1, step=.05),
        param("missing_hold_seconds", "持續缺少（秒）", 2., min=0, max=120, step=.5), param("cooldown", "事件間隔（秒）", 5., min=0, max=3600, step=.5)]),
    node("relation", "關係辨識 · RelateAnything", "模型", "本機 ConvNeXt-L 關係模型，以偵測框為輸入；物件標籤只供顯示，不送入關係模型。", [port("frame", "frame"), port("objects", "detections", accepts=["detections", "tracks"])], [port("relations", "relations")], [
        param("predicates", "關係詞彙（預載詞庫）", ["holding", "wearing", "riding", "next to", "on"]),
        param("min_score", "關係分數下限", .3, min=0, max=1, step=.01), param("top_k", "最多關係", 10, min=1, max=50, step=1),
        param("every_seconds", "推論間隔（秒）", 1., min=.1, max=30, step=.1)]),
    node("vlm", "雲端視覺判讀", "模型", "透過已設定的 OpenAI 相容 API 傳送影像；須先通過金鑰與模型連線檢查。", [port("frame", "frame"), port("events", "events", required=False)], [port("actions", "actions")], [
        param("model_id", "API 模型 ID", ""), param("instruction", "判讀目標", "Describe visible actions and possible safety-rule events. Return only observed evidence; use unknown when uncertain."),
        param("every_seconds", "呼叫間隔（秒）", 5., min=1, max=120, step=1), param("max_calls", "本次最多 API 呼叫", 10, min=1, max=120, step=1)], remote=True),
    node("merge", "合併事件", "處理", "合併多路事件，不改變來源與追蹤身分。", [port("events", "events", multiple=True)], [port("events", "events")]),
    node("output", "畫面與事件輸出", "輸出", "輸出標註圖片／影片、事件時間軸與可下載 JSON／CSV。", [port("frame", "frame"),
        port("observations", "observations", required=False, multiple=True, accepts=["detections", "poses", "tracks"]),
        port("actions", "actions", required=False, multiple=True), port("events", "events", required=False, multiple=True),
        port("relations", "relations", required=False, multiple=True), port("embeddings", "embeddings", required=False, multiple=True)], [port("frame", "frame")]),
]
from .workflow_opencv_catalog import opencv_catalog
from .workflow_cv_rules import cv_rules_catalog
from .workflow_office import office_catalog, validate_office_params
from .workflow_handwash import handwash_catalog, validate_handwash_params

NODE_TYPES.extend(opencv_catalog(node, param, port))
NODE_TYPES.extend(cv_rules_catalog(node, param, port))
NODE_TYPES.extend(office_catalog(node, param, port))
NODE_TYPES.extend(handwash_catalog(node, param, port))
CATALOG = {item["type"]: item for item in NODE_TYPES}
ASSETS = {"yolo26n_pose": ("yolo26n_pose", "yolo26n-pose.pt"), "stgcnpp_ntu120": ("stgcnpp_ntu120", "stgcnpp_ntu120_hrnet_j.pth"),
          "relateanything_convnext_large": ("relateanything_convnext_large", "relateanything.onnx"),
          "mediapipe_hand_landmarker": ("mediapipe_hand_landmarker", "hand_landmarker.task"),
          "sam2_1_tiny": ("sam2_1_tiny", "sam2.1_t.pt")}


def asset_path(model_id, app_root=None):
    app_root = Path(app_root or Path(__file__).resolve().parents[1])
    folder, filename = ASSETS[model_id]
    for base in (app_root / "models", app_root / "model_weights", app_root.parent / "model_weights"):
        candidate = base / folder / filename
        if candidate.is_file():
            return candidate.resolve()
    return app_root / "models" / folder / filename


def catalog():
    return {"node_types": copy.deepcopy(NODE_TYPES), "assets": [
        {"model_id": key, "available": asset_path(key).is_file(), "filename": ASSETS[key][1]} for key in ASSETS]}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("工作流數值不可使用 NaN 或 Infinity。")
    if isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, list):
        for item in value: _finite(item)


def normalize_graph(graph):
    if not isinstance(graph, dict) or graph.get("version") != 1:
        raise ValueError("工作流版本需為 1。")
    _finite(graph)
    if len(json.dumps(graph, ensure_ascii=False)) > 500_000:
        raise ValueError("工作流超過 500 KB。")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 48 or not isinstance(edges, list) or len(edges) > 128:
        raise ValueError("工作流需有 1–48 個節點、最多 128 條連線。")
    normalized, identifiers = [], set()
    for item in nodes:
        if not isinstance(item, dict) or item.get("type") not in CATALOG:
            raise ValueError("工作流含未知節點類型。")
        ident = item.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident) or ident in identifiers:
            raise ValueError("節點 ID 重複或格式不正確。")
        identifiers.add(ident)
        definitions = {p["key"]: p for p in CATALOG[item["type"]]["params"]}
        values = item.get("params", {})
        if not isinstance(values, dict) or set(values) - set(definitions):
            raise ValueError(f"{ident} 包含未支援的參數；權重路徑與金鑰不得寫入工作流。")
        values = {k: copy.deepcopy(values.get(k, p["default"])) for k, p in definitions.items()}
        for key, value in values.items():
            p = definitions[key]
            if p["type"] == "boolean" and type(value) is not bool:
                raise ValueError(f"{ident}.{key} 必須是勾選值。")
            if p["type"] == "number":
                if type(value) not in (int, float) or not math.isfinite(value) or value < p.get("min", -math.inf) or value > p.get("max", math.inf):
                    raise ValueError(f"{ident}.{key} 超出數值範圍。")
                if p.get("step") == 1 and int(value) != value:
                    raise ValueError(f"{ident}.{key} 必須為整數。")
            if p["type"] in {"string", "select"}:
                if not isinstance(value, str) or len(value) > 4096:
                    raise ValueError(f"{ident}.{key} 文字格式不正確。")
                if p.get("options") and value not in p["options"]:
                    raise ValueError(f"{ident}.{key} 選項無效。")
            if p["type"] == "json" and not isinstance(value, type(p["default"])):
                raise ValueError(f"{ident}.{key} JSON 型別不正確。")
        position = item.get("position", {"x": 0, "y": 0})
        if not isinstance(position, dict) or any(type(position.get(axis)) not in (int, float) for axis in ("x", "y")):
            raise ValueError("節點位置不正確。")
        normalized.append({"id": ident, "type": item["type"], "params": values, "position": {axis: position[axis] for axis in ("x", "y")}})
    clean_edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict): raise ValueError("連線格式不正確。")
        clean_edges.append({key: edge.get(key) for key in ("source", "sourceHandle", "target", "targetHandle")})
        clean_edges[-1]["id"] = str(edge.get("id", f"edge-{index}"))[:80]
    return {"version": 1, "nodes": normalized, "edges": clean_edges}


def validate_graph(graph):
    errors, warnings, order = [], [], []
    try:
        graph = normalize_graph(graph)
        nodes = {n["id"]: n for n in graph["nodes"]}
        sources = [n["id"] for n in nodes.values() if n["type"] == "source"]
        outputs = [n["id"] for n in nodes.values() if n["type"] == "output"]
        if len(sources) != 1 or len(outputs) != 1:
            raise ValueError("工作流必須包含恰好一個來源與一個畫面輸出節點。")
        incoming, outgoing = {k: [] for k in nodes}, {k: [] for k in nodes}
        for edge in graph["edges"]:
            a, b = edge["source"], edge["target"]
            if a not in nodes or b not in nodes or a == b: raise ValueError("連線端點不存在或連回自己。")
            left = {p["name"]: p for p in CATALOG[nodes[a]["type"]]["outputs"]}.get(edge["sourceHandle"])
            right = {p["name"]: p for p in CATALOG[nodes[b]["type"]]["inputs"]}.get(edge["targetHandle"])
            if not left or not right or left["type"] not in right.get("accepts", [right["type"]]):
                raise ValueError(f"{a} → {b} 的資料類型不相容或插槽不存在。")
            if not right["multiple"] and any(e["targetHandle"] == edge["targetHandle"] for e in incoming[b]):
                raise ValueError(f"{b}.{edge['targetHandle']} 只能連接一個輸入。")
            if any(e == {**edge, "id": e["id"]} for e in incoming[b]): raise ValueError("不能重複連接相同插槽。")
            incoming[b].append(edge); outgoing[a].append(edge)
        for ident, item in nodes.items():
            if item["type"] in {"office_monitor", "handwash"}:
                try:
                    (validate_office_params if item["type"] == "office_monitor" else validate_handwash_params)(item["params"])
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{ident} 規則參數不正確：{exc}")
            if item["type"].startswith("cv_"):
                from .workflow_opencv import OPENCV_KINDS, validate_opencv_params
                from .workflow_cv_rules import CV_RULE_KINDS, validate_cv_rule_params
                try:
                    if item["type"] in OPENCV_KINDS:
                        validate_opencv_params(item["type"], item["params"])
                    elif item["type"] in CV_RULE_KINDS:
                        validate_cv_rule_params(item["type"], item["params"])
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{ident} OpenCV 參數不正確：{exc}")
            for p in CATALOG[item["type"]]["inputs"]:
                if p["required"] and not any(e["targetHandle"] == p["name"] for e in incoming[ident]):
                    errors.append(f"{ident} 缺少 {p['name']} 輸入。")
            if item["type"] == "video_action" and item["params"]["scope"] == "person" and not any(e["targetHandle"] == "tracks" for e in incoming[ident]):
                errors.append(f"{ident} 個人動作分類必須連接追蹤框。")
        counts = {k: len(v) for k, v in incoming.items()}
        pending = [k for k in nodes if counts[k] == 0]
        while pending:
            ident = pending.pop(0); order.append(ident)
            for edge in outgoing[ident]:
                counts[edge["target"]] -= 1
                if counts[edge["target"]] == 0: pending.append(edge["target"])
        if len(order) != len(nodes): errors.append("工作流有循環連線；時間記憶由追蹤／動作／規則節點內部處理。")
        reachable = set(sources)
        for ident in order:
            if ident in reachable: reachable.update(e["target"] for e in outgoing[ident])
        if set(nodes) - reachable: errors.append("所有節點都必須連到影像來源。")
        useful = set(outputs)
        for ident in reversed(order):
            if ident in useful: useful.update(e["source"] for e in incoming[ident])
        if set(nodes) - useful: errors.append("有節點沒有通往畫面輸出，請接入輸出或移除。")
        if any(n["type"] == "vlm" for n in nodes.values()): warnings.append("此工作流含雲端節點，執行時會將取樣影像送到已設定的 API 服務。")
        if any(n["type"] in {"posture", "ppe", "condition", "sequence"} for n in nodes.values()):
            warnings.append("事件是模型輸出加上所設定規則的結果；請用實際場景的標註影片驗證誤報與漏報。")
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "warnings": warnings, "plan": {"order": order}, "graph": graph}
