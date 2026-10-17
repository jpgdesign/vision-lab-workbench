"""Persistent, serial GPU inference on bounded browser-camera frames.

One normal queue worker owns the session and its models. Camera access remains
in the browser; this process never opens a device or an arbitrary network URL.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .inference import (classification_probabilities, create_action_model, create_detector,
                        draw_predictions, prepare_action_tensor, resolve_device, synchronize)
from .live_limits import FRAME_RING_SIZE

DETECTORS = {"yolov12x", "yolo26x", "grounding_dino_base", "rf_detr_2xl"}
ACTIONS = {"vivit_base_k400", "videomaev2_giant_k710", "slowfast_r101_8x8", "x3d_l"}
INPUT_TIMEOUT = 20.0


def _write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # Status polling briefly holds a read handle on Windows.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def _is_link(path: Path) -> bool:
    """Reject symlinks and Windows junction/reparse points without following."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _input_root(directory: Path) -> Path:
    if _is_link(directory):
        raise ValueError("即時影格資料夾不可使用連結。")
    return directory.resolve()


def _valid_frame_filename(filename) -> bool:
    return (isinstance(filename, str) and bool(filename) and not any(c in filename for c in ("/", "\\", ":", "\x00"))
            and filename.lower().endswith(".jpg"))


def _manifest(directory: Path):
    try:
        root = _input_root(directory)
        manifest_path = root / "input.json"
        if _is_link(manifest_path):
            raise ValueError("即時影格清單不可使用連結。")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        # Windows may briefly deny a new reader while ingress atomically
        # replaces the manifest. Retry on the next loop, retaining heartbeat.
        return None
    if not isinstance(data, dict):
        raise ValueError("即時影像緩衝區格式不正確。")
    frames = data.get("frames", [])
    if not isinstance(frames, list) or len(frames) > FRAME_RING_SIZE:
        raise ValueError("即時影像緩衝區格式不正確。")
    previous = 0
    for item in frames:
        if not isinstance(item, dict):
            raise ValueError("即時影格資訊無效。")
        filename = item.get("filename", "")
        sequence, received = item.get("sequence"), item.get("received_at")
        if (not _valid_frame_filename(filename)
                or not isinstance(sequence, int) or sequence <= previous
                or not isinstance(received, (int, float)) or not math.isfinite(received)):
            raise ValueError("即時影格資訊無效。")
        # Validate every record cheaply, then check filesystem containment only
        # for images actually decoded. Resolving all 512 paths per cycle costs
        # hundreds of milliseconds on Windows even for single-frame detection.
        previous = sequence
    return data


def _select_clip(frames: list[dict], seconds: float, num_frames: int):
    """Uniform timestamp sampling of a complete trailing window, never future frames."""
    if len(frames) < 2:
        return None
    end = float(frames[-1]["received_at"])
    start = end - seconds
    if float(frames[0]["received_at"]) > start + 0.05:
        return None
    targets = [start + seconds * i / max(1, num_frames - 1) for i in range(num_frames)]
    return [min(frames, key=lambda row: abs(row["received_at"] - target)) for target in targets]


def _read_image(directory: Path, frame: dict, *, root: Path | None = None):
    from PIL import Image
    filename = frame.get("filename")
    if not _valid_frame_filename(filename):
        raise ValueError("即時影格路徑超出工作資料夾。")
    if root is None:
        root = _input_root(directory)
    elif _is_link(directory):
        raise ValueError("即時影格資料夾不可使用連結。")
    path = directory / filename
    if _is_link(path):
        raise ValueError("即時影格檔案不可使用連結。")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("即時影格路徑超出工作資料夾。")
    with Image.open(resolved) as source:
        return source.convert("RGB")


def _clip_pixels(directory: Path, selected: list[dict]):
    import numpy as np
    from PIL import Image
    cache = {}
    root = _input_root(directory)
    final = _read_image(directory, selected[-1], root=root)
    target_size = final.size
    cache[selected[-1]["sequence"]] = np.asarray(final)
    for frame in selected:
        if frame["sequence"] not in cache:
            source = _read_image(directory, frame, root=root)
            if source.size != target_size:
                source = source.resize(target_size, Image.Resampling.BILINEAR)
            cache[frame["sequence"]] = np.asarray(source)
    return np.stack([cache[frame["sequence"]] for frame in selected])


def _save_image(image, path: Path):
    temporary = path.with_suffix(".tmp.jpg")
    image.save(temporary, format="JPEG", quality=85)
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def _action_predict(model, metadata, pixels, params, device):
    import torch
    tensor = prepare_action_tensor(pixels, metadata, device)
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), context:
        logits = model(pixel_values=tensor).logits if metadata["architecture"] == "vivit" else model(tensor)
        scores = classification_probabilities(logits, metadata["outputs_probabilities"])[0].cpu()
    values, indices = scores.topk(min(int(params["top_k"]), scores.numel()))
    labels = metadata["labels"]
    return [{"label": labels.get(int(index), str(int(index))), "score": float(value)}
            for value, index in zip(values, indices)]


def _load_runner(spec, device):
    model_id = spec["model_id"]
    if model_id in DETECTORS:
        return {"task": "detection", "predict": create_detector(model_id, spec["params"], device,
                                                                   spec.get("checkpoint_path"), spec.get("model_weight_path")), "device": device}
    if model_id in ACTIONS:
        model, metadata = create_action_model(model_id, device, spec.get("checkpoint_path"), spec.get("model_weight_path"))
        return {"task": "action", "model": model, "metadata": metadata, "device": device}
    raise ValueError("此模型尚未提供即時影像辨識接頭。")


def run_live(request, output_dir, emit):
    """Keep models loaded until stop, lost camera, duration limit, or an error."""
    import numpy as np
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "live_input"
    stop_path = output / "stop.signal"
    specs = request["models"]
    if not 1 <= len(specs) <= 4:
        raise ValueError("即時比較需要 1 至 4 個模型。")
    started = time.monotonic()
    # Explicit null means unlimited; an omitted field retains the old default.
    # Keep positive integer durations intact instead of narrowing them to float.
    maximum = request.get("max_duration_seconds", 900)
    rows = [{"key": str(index), "model_id": spec["model_id"], "status": "loading", "sequence": 0,
             "predictions": [], "processed_frames": 0, "message": "等待載入模型"}
            for index, spec in enumerate(specs)]
    state = {"status": "loading", "frames_received": 0, "models": rows, "message": "載入即時辨識模型"}
    runners, last_outputs = [], {}
    processed_sequence = 0
    last_received = started
    stop_reason = "使用者停止即時辨識"

    def publish():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["elapsed_seconds"] = round(time.monotonic() - started, 2)
        _write_json(output / "live_state.json", state)

    publish()
    try:
        for index, spec in enumerate(specs):
            if stop_path.exists():
                break
            if maximum is not None and time.monotonic() - started >= maximum:
                stop_reason = "已達設定的即時測試時間上限"
                break
            device = resolve_device(spec["params"].get("device", "auto"))
            rows[index].update(message=f"正在載入 {spec['model_id']}（{device}）")
            publish()
            emit({"event": "log", "message": rows[index]["message"]})
            runners.append(_load_runner(spec, device))
            rows[index].update(status="warming", message="模型已載入，等待攝影機影像")
            publish()
        state.update(status="running", message="即時辨識中；模型依序處理相同時間的輸入")
        publish()
        while len(runners) == len(specs) and not stop_path.exists():
            now = time.monotonic()
            if maximum is not None and now - started >= maximum:
                stop_reason = "已達設定的即時測試時間上限"
                break
            manifest = _manifest(inputs)
            frames = manifest.get("frames", []) if manifest else []
            if frames:
                last_received = max(last_received, frames[-1]["received_at"])
            if now - last_received > INPUT_TIMEOUT:
                stop_reason = "超過 20 秒未收到攝影機影像，已釋放 GPU"
                break
            if not frames or frames[-1]["sequence"] <= processed_sequence:
                time.sleep(0.025)
                continue
            latest = frames[-1]
            processed_sequence = latest["sequence"]
            state["frames_received"] = processed_sequence
            try:
                image = _read_image(inputs, latest)
            except FileNotFoundError:
                continue  # Ring retired this frame while a very slow model was running.
            # Copy every action window before inference starts. A slow detector
            # may otherwise let ingress retire frames needed by a later model.
            clips = {}
            for index, (spec, runner) in enumerate(zip(specs, runners)):
                if runner["task"] != "action":
                    continue
                window = float(spec["params"].get("clip_seconds", 4))
                sampled = _select_clip(frames, window, runner["metadata"]["num_frames"])
                if sampled is not None:
                    try:
                        clips[index] = (sampled, _clip_pixels(inputs, sampled))
                    except FileNotFoundError:
                        pass
            for index, (spec, runner) in enumerate(zip(specs, runners)):
                if stop_path.exists():
                    break
                if maximum is not None and time.monotonic() - started >= maximum:
                    stop_reason = "已達設定的即時測試時間上限"
                    break
                row = rows[index]
                row.update(status="running", message="即時物件偵測" if runner["task"] == "detection" else "最近片段的行為分類")
                pixels, sampled = None, None
                window = float(spec["params"].get("clip_seconds", 4)) if runner["task"] == "action" else 0
                if runner["task"] == "action":
                    if index not in clips:
                        available = max(0, latest["received_at"] - frames[0]["received_at"])
                        row.update(status="warming", window_seconds=window, available_seconds=round(available, 2),
                                   message=(f"收集行為片段：{min(available, window):.1f} / {window:g} 秒"
                                            if available < window else "影格已更新，正在取得下一個完整片段"))
                        publish()
                        continue
                    sampled, pixels = clips[index]
                device = runner["device"]
                synchronize(device)
                before = time.perf_counter()
                predictions = (runner["predict"](image) if runner["task"] == "detection" else
                               _action_predict(runner["model"], runner["metadata"], pixels, spec["params"], device))
                synchronize(device)
                duration_ms = (time.perf_counter() - before) * 1000
                title = None if runner["task"] == "detection" else (
                    f"{predictions[0]['label']} {predictions[0]['score']:.1%}" if predictions else "No prediction")
                annotated = draw_predictions(np.asarray(image), predictions if runner["task"] == "detection" else [], title)
                filename = f"result_{index}.jpg"
                _save_image(annotated, output / filename)
                finished = time.monotonic()
                previous = last_outputs.get(index)
                last_outputs[index] = finished
                row.update(sequence=processed_sequence, predictions=predictions, inference_ms=round(duration_ms, 2),
                           processed_frames=row["processed_frames"] + 1, window_seconds=window,
                           sampled_frames=len(sampled) if sampled else 1,
                           unique_frames=len({f["sequence"] for f in sampled}) if sampled else 1,
                           result_fps=round(1 / (finished - previous), 2) if previous and finished > previous else None,
                           input_age_ms=round(max(0, finished - latest["received_at"]) * 1000, 1),
                           image_filename=filename, source_width=image.width, source_height=image.height)
                publish()
        state.update(status="completed", message=stop_reason)
        for row in rows:
            row["status"] = "completed"
        publish()
        summary = {"stop_reason": stop_reason, "elapsed_seconds": state["elapsed_seconds"],
                   "frames_received": state["frames_received"], "models": rows,
                   "recording": False, "raw_frames_retained": False}
        _write_json(output / "session_summary.json", summary)
        artifacts = {f"model_{index}": str(output / f"result_{index}.jpg") for index in range(len(rows))
                     if (output / f"result_{index}.jpg").is_file()}
        artifacts["summary"] = str(output / "session_summary.json")
        emit({"event": "log", "message": stop_reason})
        return {"summary": stop_reason, "live": summary, "artifacts": artifacts}
    except Exception as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "即時比較的模型占用過多顯存。請減少同時比較的模型數或降低輸入尺寸。"
        state.update(status="failed", message=message)
        for row in rows:
            row.update(status="failed", message=message)
        publish()
        raise RuntimeError(message) from exc
    finally:
        # The directory is fixed beneath this worker's job; no arbitrary paths.
        if inputs.exists() and inputs.resolve().is_relative_to(output) and not inputs.is_symlink():
            shutil.rmtree(inputs, ignore_errors=True)
