"""Execute typed graphs on source-time frames, in one isolated queue worker."""
from __future__ import annotations

from collections import deque
import csv
import io
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

from .workflow_graph import CATALOG, validate_graph
from .workflow_rules import KINDS as RULE_KINDS, RuleNode
from .workflow_tracking import TrackerNode

MAX_FRAME_RECORDS = 20_000
MAX_EVENT_RECORDS = 100_000
MAX_EXPORT_BYTES = 256 * 1024**2


def packet(items, frame, **extra):
    return {"items": items, "timestamp": frame["timestamp"], "frame": frame, "fresh": True, **extra}


def packets(value):
    return value if isinstance(value, list) else [value] if isinstance(value, dict) else []


def json_value(value):
    """Artifacts contain observations, never image tensors or credential paths."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()
                if k not in {"image", "frame", "mask", "mask_data", "key_file_path"}}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    raise TypeError(f"工作流輸出包含不可序列化的 {type(value).__name__}")


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class RemoteNode:
    def __init__(self, params, runtime):
        from .workflow_integrations import VisionClient
        self.params = params
        self.client = VisionClient(runtime, params["model_id"])
        self.last_time, self.calls = None, 0
        self.display = []
        self.pending_events = deque(maxlen=100)

    def reset(self):
        # Keep this job's billing limit even when the input stream changes.
        self.last_time, self.display = None, []
        self.pending_events.clear()

    def process(self, inputs, frame):
        stamp = frame["timestamp"]
        self.pending_events.extend(inputs.get("events", {}).get("items", []))
        if self.last_time is not None and stamp - self.last_time < self.params["every_seconds"]:
            return {"actions": packet([], frame, fresh=False, display_items=self.display)}
        self.last_time = stamp
        if self.calls >= self.params["max_calls"]:
            values = [{"label": "unknown", "score": 0., "scope": "scene", "reason": "API 呼叫額度已用完"}]
        else:
            values = self.client.predict(frame["image"], self.params["instruction"], list(self.pending_events))
            self.calls += 1
            self.pending_events.clear()
        self.display = [{**v, "timestamp": stamp} for v in values]
        return {"actions": packet(self.display, frame)}


def render_output(frame, inputs, recent_events):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from .inference import draw_predictions
    observations, actions, relations, overlays = [], [], [], []
    for name, destination in (("observations", observations), ("actions", actions), ("relations", relations)):
        for p in packets(inputs.get(name)):
            destination.extend(p.get("display_items", p.get("items", [])))
            overlays.extend(p.get("overlays", []))
    for p in packets(inputs.get("events")):
        overlays.extend(p.get("overlays", []))
    # Multiple branches may carry exactly the same tracked boxes into output.
    seen, boxes = set(), []
    for item in observations:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        identity = (tuple(box), item.get("label"), item.get("track_key"))
        if identity in seen: continue
        seen.add(identity)
        label = str(item.get("label", "object"))
        if item.get("track_id") is not None: label += f" #{item['track_id']}"
        boxes.append({**item, "label": label, "score": float(item.get("score", 0))})
    image = draw_predictions(np.asarray(frame["image"]), boxes)
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    size = max(12, round(min(image.size) / 40))
    font = ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
    bones = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for item in observations:
        joints = item.get("keypoints", [])
        if len(joints) != 17: continue
        valid = lambda j: len(j) >= 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in j[:3]) and j[2] >= .3
        for a, b in bones:
            if valid(joints[a]) and valid(joints[b]):
                draw.line([tuple(joints[a][:2]), tuple(joints[b][:2])], fill=(240,207,112,230), width=2)
        for joint in joints:
            if valid(joint):
                x,y = joint[:2]; draw.ellipse((x-2,y-2,x+2,y+2), fill=(140,246,190,255))
    seen_polygons = set()
    for overlay in overlays:
        if overlay.get("kind") != "polygon": continue
        points = [(float(x) * image.width, float(y) * image.height) for x,y in overlay["points"]]
        key = (str(overlay.get("label")), tuple(points))
        if key in seen_polygons: continue
        seen_polygons.add(key)
        draw.polygon(points, fill=(111,180,160,25), outline=(145,232,193,220), width=2)
        draw.text(points[0], f"{overlay.get('label', '')} · {overlay.get('count', 0)}", font=font, fill=(201,255,221,255), stroke_width=1, stroke_fill=(0,0,0,255))
    for relation in relations[:20]:
        a, b = relation.get("subject_box"), relation.get("object_box")
        if a is None or b is None: continue
        start,end = ((a[0]+a[2])/2,(a[1]+a[3])/2), ((b[0]+b[2])/2,(b[1]+b[3])/2)
        draw.line([start,end], fill=(235,192,250,240), width=2)
        theta = math.atan2(end[1]-start[1], end[0]-start[0])
        for offset in (-.5,.5):
            draw.line([end,(end[0]-12*math.cos(theta+offset),end[1]-12*math.sin(theta+offset))], fill=(235,192,250,240), width=2)
        draw.text(((start[0]+end[0])/2,(start[1]+end[1])/2), str(relation.get("predicate", relation.get("label", ""))), font=font, fill=(255,223,255,255), stroke_width=1, stroke_fill=(0,0,0,255))
    lines = [f"{frame['timestamp']:.2f}s · frame {frame['frame_index']}"]
    for item in actions[:6]:
        scope = f"#{item['track_id']}" if item.get("track_id") is not None else item.get("scope", "scene")
        age = max(0., frame["timestamp"] - float(item.get("timestamp", frame["timestamp"])))
        suffix = f" (last {age:.1f}s)" if age > .1 else ""
        lines.append(f"{scope}: {item.get('label', 'unknown')} {item.get('score', 0):.2f}{suffix}")
    for event in list(recent_events)[-3:]:
        if 0 <= frame["timestamp"] - event.get("timestamp", 0) <= 4:
            lines.append(f"{event['timestamp']:.2f}s {event.get('label', 'event')}")
    for i, line in enumerate(lines):
        y = 5+i*(size+7)
        if y+size > image.height: break
        bounds = draw.textbbox((8,y), line, font=font)
        draw.rectangle((4,y-2,min(image.width,bounds[2]+5),bounds[3]+3), fill=(12,26,21,220))
        draw.text((8,y), line, font=font, fill=(207,249,225,255))
    return image


class WorkflowEngine:
    def __init__(self, request, emit=lambda event: None, *, model_factory=None):
        checked = validate_graph(request["workflow_graph"])
        if not checked["valid"]: raise ValueError("；".join(checked["errors"]))
        self.graph, self.order = checked["graph"], checked["plan"]["order"]
        self.request, self.emit = request, emit
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}
        self.instances, self.stats, self.recent_events = {}, {}, deque(maxlen=1000)
        self.event_count = 0
        self.incoming = {ident: [] for ident in self.nodes}
        for edge in self.graph["edges"]: self.incoming[edge["target"]].append(edge)
        self.model_factory = model_factory
        self.device = None
        self.last_context = None

    def _instance(self, node):
        ident, kind, params = node["id"], node["type"], node["params"]
        if ident not in self.instances:
            if kind in RULE_KINDS: instance = RuleNode(kind, params, ident)
            elif kind == "tracker": instance = TrackerNode(params, ident)
            elif kind == "vlm": instance = RemoteNode(params, self.request["runtime_path"])
            else:
                if self.model_factory is None:
                    from .workflow_model_nodes import ModelNode
                    self.model_factory = ModelNode
                if self.device is None:
                    from .inference import resolve_device
                    self.device = resolve_device(self.request.get("params", {}).get("device", "auto"))
                instance = self.model_factory(kind, params, self.request.get("resolved_nodes", {}).get(ident, {}), device=self.device)
            self.instances[ident] = instance
        return self.instances[ident]

    def process(self, image, timestamp, frame_index):
        if not math.isfinite(timestamp) or timestamp < 0: raise ValueError("來源時間戳無效。")
        source = str(self.request.get("id", self.request.get("media_id", "workflow")))
        context = (source, image.size)
        # Resizing a stream changes every tracked coordinate; start new history.
        if self.last_context is not None and (context != self.last_context[0] or timestamp < self.last_context[1]):
            for instance in self.instances.values():
                if hasattr(instance, "reset"): instance.reset()
            self.recent_events.clear()
        self.last_context = (context, timestamp)
        frame = {"image": image, "timestamp": float(timestamp), "frame_index": int(frame_index), "width": image.width, "height": image.height, "source_id": source}
        values, final, frame_events = {}, image, []
        for ident in self.order:
            node = self.nodes[ident]; kind = node["type"]
            inputs = {}
            definitions = {p["name"]: p for p in CATALOG[kind]["inputs"]}
            for edge in self.incoming[ident]:
                incoming = values[edge["source"]][edge["sourceHandle"]]
                name = edge["targetHandle"]
                if definitions[name]["multiple"]: inputs.setdefault(name, []).append(incoming)
                else: inputs[name] = incoming
            started = time.perf_counter()
            try:
                if kind == "source": result = {"frame": packet([], frame)}
                elif kind == "output":
                    seen = set()
                    for p in packets(inputs.get("events")):
                        for event in p.get("items", []):
                            identity = event.get("event_id") or json.dumps(json_value(event), sort_keys=True)
                            if identity not in seen:
                                seen.add(identity); frame_events.append(json_value(event))
                    self.recent_events.extend(frame_events); self.event_count += len(frame_events)
                    final = render_output(frame, inputs, self.recent_events)
                    result = {"frame": packet([], frame)}
                elif kind == "condition" and not inputs.get("actions", {}).get("fresh", True):
                    result = {"events": packet([], frame, fresh=False)}
                else:
                    result = self._instance(node).process(inputs, frame)
                    for value in result.values():
                        value.setdefault("fresh", True)
                        for event in value.get("items", []):
                            if "event_id" in event: event.setdefault("node_id", ident)
                missing = {p["name"] for p in CATALOG[kind]["outputs"]} - set(result)
                if missing: raise ValueError("節點缺少輸出：" + ", ".join(sorted(missing)))
            except Exception as exc:
                raise RuntimeError(f"節點 {ident}（{CATALOG[kind]['label']}）失敗：{exc}") from exc
            elapsed = (time.perf_counter()-started)*1000
            stats = self.stats.setdefault(ident, {"id": ident, "type": kind, "calls": 0, "updates": 0, "total_ms": 0., "last_ms": 0., "items": 0})
            stats["calls"] += 1; stats["updates"] += int(any(p.get("fresh", True) for p in result.values()))
            stats["last_ms"] = round(elapsed, 2); stats["total_ms"] = round(stats["total_ms"]+elapsed, 2)
            stats["items"] = sum(len(p.get("items", [])) for p in result.values())
            stats["mean_ms"] = round(stats["total_ms"]/stats["calls"], 2)
            instance = self.instances.get(ident)
            if instance is not None and hasattr(instance, "actual_device"): stats["device"] = instance.actual_device
            values[ident] = result
        # Live status is atomically replaced in live_state.json. Appending node
        # telemetry per live frame would grow the worker's internal log forever.
        if self.request.get("kind") != "live":
            self.emit({"event": "node", "node_stats": list(self.stats.values())})
        observations = {ident: {port: json_value({k:v for k,v in p.items() if k not in {"frame", "display_items"}})
                               for port,p in ports.items() if port != "frame"}
                        for ident,ports in values.items() if self.nodes[ident]["type"] != "source"}
        return final, frame_events, observations


class Artifacts:
    def __init__(self, output, engine):
        self.output, self.engine = Path(output), engine
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = (self.output / "workflow_events.jsonl").open("w", encoding="utf-8")
        self.frames = (self.output / "workflow_frames.jsonl").open("w", encoding="utf-8")
        self.csv_handle = (self.output / "workflow_events.csv").open("w", encoding="utf-8-sig", newline="")
        self.fields = ["timestamp", "node_id", "kind", "label", "severity", "scope", "track_id", "details"]
        self.csv = csv.DictWriter(self.csv_handle, fieldnames=self.fields, extrasaction="ignore")
        self.csv.writeheader()
        write_json(self.output / "workflow.json", engine.graph)
        self.count = 0
        self.frame_records = self.event_records = self.written_bytes = 0
        self.truncated = False

    def export_status(self):
        return {"truncated": self.truncated, "frame_records_written": self.frame_records,
                "event_records_written": self.event_records, "max_frame_records": MAX_FRAME_RECORDS,
                "max_event_records": MAX_EVENT_RECORDS, "max_combined_bytes": MAX_EXPORT_BYTES}

    def _reserve(self, length):
        if self.written_bytes + length <= MAX_EXPORT_BYTES:
            self.written_bytes += length
            return True
        self.truncated = True
        return False

    def append(self, timestamp, index, events, observations):
        for event in events:
            if self.event_records >= MAX_EVENT_RECORDS:
                self.truncated = True; break
            line = json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n"
            row = {**event, "details": json.dumps(event.get("details", {}), ensure_ascii=False)}
            # Spreadsheet formulas must remain plain data if opened in Excel.
            buffer = io.StringIO(newline="")
            csv.DictWriter(buffer,fieldnames=self.fields,extrasaction="ignore").writerow({k: ("'"+v if isinstance(v,str) and v.startswith(("=","+","-","@","\t","\r")) else v) for k,v in row.items()})
            csv_line = buffer.getvalue()
            if self._reserve(len(line.encode("utf-8"))+len(csv_line.encode("utf-8"))):
                self.events.write(line); self.csv_handle.write(csv_line); self.event_records += 1
        if self.frame_records < MAX_FRAME_RECORDS:
            line = json.dumps({"timestamp": timestamp, "frame_index": index, "nodes": observations}, ensure_ascii=False, allow_nan=False)+"\n"
            if self._reserve(len(line.encode("utf-8"))):
                self.frames.write(line); self.frame_records += 1
        else: self.truncated = True
        self.count += 1
        if self.count % 10 == 0:
            self.events.flush(); self.frames.flush(); self.csv_handle.flush()

    def close(self):
        self.events.close(); self.frames.close(); self.csv_handle.close()

    def result(self, media_path=None, media_kind="image"):
        write_json(self.output / "node_stats.json", list(self.engine.stats.values()))
        files = {"workflow": "workflow.json", "events_jsonl": "workflow_events.jsonl", "events_csv": "workflow_events.csv", "frames_jsonl": "workflow_frames.jsonl", "node_stats": "node_stats.json"}
        return {"summary": f"已處理 {self.count} 幀，產生 {self.engine.event_count} 筆規則事件。" + (" 匯出紀錄已達容量上限，後續僅更新即時畫面與最近事件。" if self.truncated else ""), "event_count": self.engine.event_count,
                "events": list(self.engine.recent_events), "events_truncated": self.engine.event_count > len(self.engine.recent_events),
                "node_stats": list(self.engine.stats.values()), "frames": self.count, "data_export": self.export_status(),
                "artifacts": {k:str(self.output/v) for k,v in files.items()},
                **({"media_path": str(media_path), "media_kind": media_kind} if media_path else {})}


def fit_image(image, maximum):
    image = image.convert("RGB")
    if max(image.size) > maximum:
        image.thumbnail((maximum,maximum))
    return image


def run_workflow(request, output, emit):
    from PIL import Image, ImageOps
    import cv2
    import numpy as np
    from .datasets import media_kind
    from .inference import h264_video
    output = Path(output)
    engine = WorkflowEngine(request, emit)
    artifacts = Artifacts(output, engine)
    options = request.get("params", {})
    maximum, stride, limit = int(options.get("max_side",1280)), int(options.get("frame_stride",3)), int(options.get("max_frames",300))
    path = Path(request["media_path"])
    capture, writer = None, None
    try:
        if media_kind(path) == "image":
            with Image.open(path) as source: image = fit_image(ImageOps.exif_transpose(source), maximum)
            result, events, observations = engine.process(image, 0., 0)
            artifacts.append(0.,0,events,observations)
            target = output / "workflow_result.jpg"; result.save(target, quality=92)
            return artifacts.result(target)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened(): raise ValueError("無法開啟來源影片。")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0: raise ValueError("影片沒有有效 FPS，無法建立可靠的動作時間軸。")
        total = max(0,int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        target, raw = output / "workflow_result.mp4", output / "workflow_raw.avi"
        index, previous, previous_index, pts_origin = 0, -1., 0, None
        while artifacts.count < limit:
            ok, bgr = capture.read()
            if not ok: break
            if index % stride:
                index += 1; continue
            # Frame PTS can be absent or repeated in some codecs; use source FPS.
            raw_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            if pts_origin is None and math.isfinite(raw_pts) and raw_pts >= 0:
                pts_origin = raw_pts-index/fps
            timestamp = raw_pts-(pts_origin or 0.)
            if not math.isfinite(timestamp) or timestamp < 0 or (index > 0 and timestamp <= previous):
                timestamp = previous+(index-previous_index)/fps if previous >= 0 else index/fps
            previous, previous_index = timestamp, index
            image = fit_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), maximum)
            result, events, observations = engine.process(image,timestamp,index)
            if writer is None:
                writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"MJPG"),fps/stride,result.size)
                if not writer.isOpened(): raise ValueError("無法建立結果影片。")
            writer.write(cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2BGR))
            artifacts.append(timestamp,index,events,observations)
            emit({"event":"progress", "progress": min(.98,artifacts.count/max(1,min(limit,math.ceil(total/stride)) if total else limit)), "message":f"工作流已處理 {artifacts.count} 幀 · 來源 {timestamp:.2f} 秒"})
            index += 1
        if not artifacts.count: raise ValueError("影片沒有可解碼的影格。")
        writer.release(); writer = None
        h264_video(raw,target)
        result = artifacts.result(target,"video")
        result.update(source_fps=fps,frame_stride=stride,timestamp_basis="source_pts_with_fps_fallback",output_timing="constant_source_fps_divided_by_stride")
        return result
    finally:
        if capture is not None: capture.release()
        if writer is not None: writer.release()
        artifacts.close()


def run_workflow_live(request, output, emit):
    from .live_inference import _manifest, _read_image, _save_image, _write_json, _is_link, INPUT_TIMEOUT
    output = Path(output).resolve()
    engine = WorkflowEngine(request,emit)
    artifacts = Artifacts(output,engine)
    inputs = output / "live_input"
    started = last_received = time.monotonic()
    maximum, processed, first_received = request.get("max_duration_seconds",900), 0, None
    maximum_side = int(request.get("params",{}).get("max_side",1280))
    row = {"key":"0","model_id":"workflow","status":"warming","sequence":0,"predictions":[],"processed_frames":0,"message":"等待攝影機影像"}
    state = {"status":"running","frames_received":0,"models":[row],"events":[],"node_stats":[]}
    reason = "使用者停止即時工作流"
    target = output / "result_0.jpg"
    def publish():
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,2))
        _write_json(output/"live_state.json",state)
    publish()
    try:
        while not (output/"stop.signal").exists():
            now = time.monotonic()
            if maximum is not None and now-started >= maximum:
                reason = "已達設定的即時測試時間上限"; break
            manifest = _manifest(inputs)
            frames = manifest.get("frames",[]) if manifest else []
            if frames: last_received = max(last_received,frames[-1]["received_at"])
            if now-last_received > INPUT_TIMEOUT:
                reason = "超過 20 秒未收到攝影機影像，已停止工作流"; break
            if not frames or frames[-1]["sequence"] <= processed:
                time.sleep(.025); continue
            latest = frames[-1]; processed = latest["sequence"]
            if first_received is None: first_received = frames[0]["received_at"]
            try: image = fit_image(_read_image(inputs,latest),maximum_side)
            except FileNotFoundError: continue
            stamp = max(0.,latest["received_at"]-first_received)
            tick = time.perf_counter()
            row.update(status="running",message="執行工作流節點",sequence=processed)
            state["frames_received"] = processed; publish()
            result, events, observations = engine.process(image,stamp,processed)
            elapsed = (time.perf_counter()-tick)*1000
            _save_image(result,target)
            artifacts.append(stamp,processed,events,observations)
            labels = [v for node in observations.values() for name,p in node.items() if name in {"actions","relations"} for v in p.get("items",[])][:20]
            row.update(processed_frames=artifacts.count,predictions=labels,inference_ms=round(elapsed,2),result_fps=round(artifacts.count/max(.001,time.monotonic()-started),2),
                       input_age_ms=round(max(0.,time.monotonic()-latest["received_at"])*1000,2),image_filename=target.name,source_width=image.width,source_height=image.height)
            state.update(events=list(engine.recent_events)[-100:],node_stats=list(engine.stats.values()),data_export=artifacts.export_status())
            if artifacts.truncated: row["message"] = "即時執行中；匯出紀錄已達容量上限，僅更新畫面與最近事件"
            publish()
        state.update(status="completed",message=reason); row.update(status="completed",message=reason); publish()
        result = artifacts.result(target if target.exists() else None)
        result.update(summary=reason + ("；匯出紀錄已達容量上限。" if artifacts.truncated else ""),live={"stop_reason":reason,"elapsed_seconds":state["elapsed_seconds"],"frames_received":processed,"models":[row],"recording":False,"raw_frames_retained":False})
        return result
    except Exception as exc:
        state.update(status="failed",message=str(exc)); row.update(status="failed",message=str(exc)); publish()
        raise
    finally:
        artifacts.close()
        if inputs.exists() and not _is_link(inputs) and inputs.resolve().parent == output:
            shutil.rmtree(inputs,ignore_errors=True)
