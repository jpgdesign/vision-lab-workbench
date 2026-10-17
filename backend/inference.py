"""Local adapters for image detection and temporal video classification."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .catalog import WEIGHTS

ASSETS = Path(__file__).parent / "assets"


def resolve_device(value="auto"):
    import torch
    if value in ("auto", "cuda", "cuda:0", "0"):
        if torch.cuda.is_available():
            return "cuda:0"
        if value != "auto":
            raise ValueError("指定 CUDA，但目前無法使用 GPU；請改用 CPU 或檢查執行環境。")
        return "cpu"
    if value == "cpu":
        return value
    raise ValueError("無效的運算裝置")


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def classification_probabilities(logits, already_normalized):
    """Keep raw logits distinct from normalized class scores across backends."""
    import torch
    values = logits.float() if already_normalized else logits.float().softmax(-1)
    if (not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not torch.allclose(values.sum(-1), torch.ones_like(values.sum(-1)), atol=0.02)):
        raise ValueError("模型輸出的分類分數不是有效機率，請檢查模型接頭與權重。")
    return values


def kinetics400_labels():
    data = json.loads((ASSETS / "kinetics400.json").read_text(encoding="utf-8"))
    return {int(index): label.strip('"') for label, index in data.items()}


def create_action_model(model_id, device="cpu", checkpoint_path=None, model_weight_path=None):
    import torch
    from .model_registry import validate_checkpoint, validate_weight
    path = (validate_weight(model_id, model_weight_path) if model_weight_path else WEIGHTS[model_id]) if not checkpoint_path else None
    if checkpoint_path:
        checkpoint_path = validate_checkpoint(model_id, checkpoint_path)
    if path is not None and not path.is_file():
        raise ValueError("本機權重尚未下載完成")
    if model_id == "vivit_base_k400":
        from transformers import VivitForVideoClassification, VivitImageProcessor
        source_dir = Path(checkpoint_path) if checkpoint_path else path.parent
        model = VivitForVideoClassification.from_pretrained(str(source_dir), local_files_only=True)
        processor = VivitImageProcessor.from_pretrained(str(source_dir), local_files_only=True)
        labels = {int(k): v for k, v in model.config.id2label.items()}
        # The source checkpoint has generic LABEL_i metadata. The official
        # conversion uses the standard Kinetics-400 map without reordering.
        if not checkpoint_path and len(labels) == 400 and all(v == f"LABEL_{k}" for k, v in labels.items()):
            labels = kinetics400_labels()
        metadata = {"num_frames": 32, "side": 224, "processor": processor,
                    "labels": labels, "outputs_probabilities": False,
                    "mean": [0.5] * 3, "std": [0.5] * 3, "architecture": "vivit"}
    elif model_id in ("x3d_l", "slowfast_r101_8x8"):
        from pytorchvideo.models.hub import x3d_l, slowfast_r101
        model = (x3d_l if model_id == "x3d_l" else slowfast_r101)(pretrained=False)
        if not checkpoint_path:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state"], strict=True)
        metadata = {"num_frames": 16 if model_id == "x3d_l" else 32,
                    "side": 312 if model_id == "x3d_l" else 256,
                    "short_side": 356 if model_id == "x3d_l" else 256,
                    "mean": [0.45] * 3, "std": [0.225] * 3,
                    "labels": kinetics400_labels(), "outputs_probabilities": model_id == "x3d_l",
                    "architecture": "pytorchvideo"}
        if model_id == "slowfast_r101_8x8":
            metadata["slowfast_alpha"] = 4
    elif model_id == "videomaev2_giant_k710":
        from accelerate import init_empty_weights
        from .vendor.videomaev2.modeling_finetune import vit_giant_patch14_224
        # Meta parameters avoid initializing four GB of random weights before loading.
        with init_empty_weights():
            model = vit_giant_patch14_224(num_classes=710, all_frames=16)
        if not checkpoint_path:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint.get("model", checkpoint.get("module", checkpoint))
            model.load_state_dict(state, strict=True, assign=True)
        labels = (ASSETS / "kinetics710.txt").read_text(encoding="utf-8").splitlines()
        metadata = {"num_frames": 16, "side": 224, "short_side": 224,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                    "labels": dict(enumerate(labels)), "outputs_probabilities": False,
                    "architecture": "videomaev2"}
    else:
        raise ValueError("沒有此行為辨識模型的接頭")
    if checkpoint_path and model_id != "vivit_base_k400":
        from safetensors.torch import load_file
        checkpoint_dir = Path(checkpoint_path)
        config = json.loads((checkpoint_dir / "model_config.json").read_text(encoding="utf-8"))
        if config["model_id"] != model_id:
            raise ValueError("微調權重與選定模型不符")
        classes = config["classes"]
        if metadata["architecture"] == "pytorchvideo":
            model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, len(classes))
            model.blocks[-1].activation = torch.nn.Identity()
        else:
            model.head = torch.nn.Linear(model.head.in_features, len(classes))
        model.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")), strict=True, assign=True)
        metadata["labels"] = dict(enumerate(classes))
        metadata["outputs_probabilities"] = False
    # Assign a complete external state before converting a meta-initialized
    # VideoMAE model. Its base checkpoint is not required for fine-tuned use.
    if model_id == "videomaev2_giant_k710" and str(device) == "cpu":
        model.float()
    return model.to(device).eval(), metadata


def prepare_action_tensor(frames, metadata, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    if "processor" in metadata:
        return metadata["processor"](images=list(frames), return_tensors="pt")["pixel_values"].to(device)
    frames = np.asarray(frames)
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    height, width = tensor.shape[-2:]
    side = metadata["side"]
    ratio = metadata.get("short_side", side) / min(height, width)
    tensor = F.interpolate(tensor, size=(round(height * ratio), round(width * ratio)), mode="bilinear", align_corners=False)
    y = (tensor.shape[-2] - side) // 2
    x = (tensor.shape[-1] - side) // 2
    tensor = tensor[:, :, y:y + side, x:x + side]
    mean = torch.tensor(metadata["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(metadata["std"]).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    if metadata.get("slowfast_alpha"):
        index = torch.linspace(0, tensor.shape[2] - 1, tensor.shape[2] // metadata["slowfast_alpha"]).long().to(device)
        return [torch.index_select(tensor, 2, index), tensor]
    return tensor


def read_clip(path, num_frames, start_seconds, duration):
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not 0 < fps <= 1000 or total < 1:
            raise ValueError("影片無法解碼或缺少有效的影格資訊")
        first = min(total - 1, max(0, round(start_seconds * fps)))
        last = min(total - 1, max(first, round((start_seconds + duration) * fps) - 1))
        indices = np.linspace(first, last, num_frames).round().astype(int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"無法解碼影片第 {index} 幀")
            # Limit decoding memory before batching, preserving aspect ratio.
            height, width = frame.shape[:2]
            if max(height, width) > 640:
                factor = 640 / max(height, width)
                frame = cv2.resize(frame, (round(width * factor), round(height * factor)))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames), total / fps
    finally:
        capture.release()


def draw_predictions(rgb, predictions, title=None):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    size = max(14, round(min(image.size) / 38))
    font = ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()
    colors = ["#63e5b3", "#83b8ff", "#f2c66c", "#e2a5f9", "#ff9a91"]
    for index, prediction in enumerate(predictions):
        if "box" not in prediction:
            continue
        box = prediction["box"]
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=max(2, size // 8))
        label = f"{prediction['label']} {prediction['score']:.2f}"
        x, y = max(0, box[0]), max(0, box[1] - size - 6)
        text_box = draw.textbbox((x + 4, y + 2), label, font=font)
        draw.rectangle((x, y, text_box[2] + 4, text_box[3] + 3), fill="#10221c")
        draw.text((x + 4, y + 2), label, font=font, fill=color)
    if title:
        draw.rectangle((0, 0, image.width, size * 2 + 12), fill="#10221c")
        draw.text((12, 8), title, font=font, fill="#a4f0d0")
    return image


def h264_video(raw_path, final_path):
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path),
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(final_path)]
    subprocess.run(command, check=True, capture_output=True, timeout=600)
    raw_path.unlink()


def validate_yolo_architecture(model, model_id):
    """Check loaded architecture, including renamed .pt files, before use."""
    import re
    config = getattr(getattr(model, "model", None), "yaml", None)
    if not isinstance(config, dict) or getattr(model, "task", "detect") != "detect":
        raise ValueError("權重沒有可驗證的 YOLO 物件偵測架構。")
    modules = {layer[2] if isinstance(layer[2], str) else getattr(layer[2], "__name__", "")
               for layer in config.get("backbone", []) + config.get("head", []) if isinstance(layer, (list, tuple)) and len(layer) >= 3}
    family = "12" if "A2C2f" in modules else "26" if config.get("end2end") is True and config.get("reg_max") == 1 and "C2PSA" in modules else None
    expected = "12" if model_id in {"yolov12x", "yolo12x"} else "26"
    scale = config.get("scale")
    if not scale:
        match = re.search(r"yolo(?:v)?(?:12|26)([nslmx])(?:\.|$)", str(config.get("yaml_file", "")).lower())
        scale = match.group(1) if match else None
    if not scale and config.get("depth_multiple") == 1.0 and config.get("width_multiple") == 1.5:
        scale = "x"
    if family != expected or scale != "x":
        raise ValueError(f"載入的 YOLO 架構（{family or '未知'}/{scale or '未知'}）與所選 {model_id} 不符。")


def create_detector(model_id, params, device, checkpoint_path=None, model_weight_path=None):
    import torch
    from .model_registry import validate_checkpoint, validate_weight
    path = (validate_checkpoint(model_id, checkpoint_path) if checkpoint_path else
            validate_weight(model_id, model_weight_path) if model_weight_path else WEIGHTS[model_id])
    if model_id in ("yolov12x", "yolo26x"):
        from ultralytics import YOLO
        model = YOLO(str(path))
        validate_yolo_architecture(model, model_id)
        def predict(image, inference_params=None):
            current = params if inference_params is None else inference_params
            kwargs = {key: current[key] for key in ("conf", "imgsz", "max_det", "iou") if key in current}
            result = model.predict(image, device=device, verbose=False, **kwargs)[0]
            return [{"label": result.names[int(cls)], "score": float(score), "box": box.tolist()}
                    for box, score, cls in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu())]
        return predict
    if model_id == "grounding_dino_base":
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        source_dir = path if checkpoint_path else path.parent
        processor = AutoProcessor.from_pretrained(str(source_dir), local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(str(source_dir), local_files_only=True).to(device).eval()
        def predict(image, inference_params=None):
            current = params if inference_params is None else inference_params
            prompt = current["prompt"].strip().lower()
            if not prompt.endswith("."):
                prompt += " ."
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
                threshold=current["conf"], text_threshold=current["text_threshold"], target_sizes=[(image.height, image.width)])[0]
            labels = results.get("text_labels", results.get("labels", []))
            return [{"label": str(label), "score": float(score), "box": box.cpu().tolist()}
                    for label, score, box in zip(labels, results["scores"], results["boxes"])]
        return predict
    if model_id == "rf_detr_2xl":
        if not params.get("accept_platform_license"):
            raise ValueError("尚未確認 RF-DETR 2XL 所需的 PML 授權")
        from rfdetr_plus import RFDETR2XLarge
        from unittest.mock import patch
        model_options = {}
        if checkpoint_path:
            metadata = json.loads((path / "workbench_training.json").read_text(encoding="utf-8"))
            if metadata.get("model_id") != model_id or metadata.get("class_id_offset") != 1:
                raise ValueError("RF-DETR 微調權重的類別對應資訊無效。")
            model_options = {"resolution": metadata["resolution"], "num_classes": len(metadata["classes"]) + 1,
                             "group_detr": metadata["group_detr"],
                             "positional_encoding_size": metadata["positional_encoding_size"]}
            path = path / "model.pth"
        class LocalRFDETR2XLarge(RFDETR2XLarge):
            def maybe_download_pretrain_weights(self):
                if not Path(self.model_config.pretrain_weights).is_file():
                    raise ValueError("本機 RF-DETR 權重不存在；工作台不會自動下載。")
        # RF-DETR normally retries corrupted weights by downloading them. Keep
        # that path offline too, without bypassing its explicit license gate.
        with patch("rfdetr.main.download_pretrain_weights", side_effect=ValueError("本機 RF-DETR 權重無法讀取；請重新檢查檔案。")):
            rf_device = "cuda" if str(device).startswith("cuda") else device
            model = LocalRFDETR2XLarge(pretrain_weights=str(path), device=rf_device,
                                     accept_platform_model_license=True, **model_options)
        classes = model.class_names
        def predict(image, inference_params=None):
            current = params if inference_params is None else inference_params
            if not current.get("accept_platform_license"):
                raise ValueError("尚未確認 RF-DETR 2XL 所需的 PML 授權")
            detections = model.predict(image, threshold=current["conf"])
            return [{"label": classes.get(int(cls), str(cls)) if isinstance(classes, dict) else classes[int(cls)],
                     "score": float(score), "box": box.tolist()}
                    for box, score, cls in zip(detections.xyxy, detections.confidence, detections.class_id)
                    if not checkpoint_path or int(cls) in classes]
        return predict
    raise ValueError("模型沒有偵測接頭")


def run_detection(request, output, emit, device, *, predict=None):
    import cv2
    import numpy as np
    from PIL import Image
    source = Path(request["media_path"])
    params = request["params"]
    if predict is None:
        predict = create_detector(request["model_id"], params, device, request.get("checkpoint_path"), request.get("model_weight_path"))
    emit({"event": "progress", "progress": 0.15, "message": "模型已載入，開始辨識"})
    is_video = source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if not is_video:
        image = Image.open(source).convert("RGB")
        synchronize(device)
        start = time.perf_counter()
        predictions = predict(image, params)
        synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000
        result_path = output / "result.png"
        draw_predictions(np.asarray(image), predictions).save(result_path)
        (output / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"media_path": str(result_path), "media_kind": "image", "predictions": predictions,
                "inference_ms": round(elapsed, 2), "frames": 1, "summary": f"偵測到 {len(predictions)} 個物件",
                "artifacts": {"predictions": str(output / "predictions.json")}}
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not 0 < fps <= 1000:
            raise ValueError("無法解碼影片")
        stride, maximum = int(params["frame_stride"]), int(params["max_frames"])
        target = min(maximum, (total + stride - 1) // stride)
        raw_path = output / "annotated_raw.mp4"
        frame_index = count = 0
        times, preview = [], []
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as log:
            while count < maximum:
                ok, frame = capture.read()
                if not ok:
                    break
                current = frame_index
                frame_index += 1
                if current % stride:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                synchronize(device)
                start = time.perf_counter()
                predictions = predict(image, params)
                synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
                annotated = draw_predictions(np.asarray(image), predictions)
                if writer is None:
                    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, annotated.size)
                    if not writer.isOpened():
                        raise RuntimeError("無法建立影片輸出檔")
                    preview = predictions
                    annotated.save(output / "preview.jpg")
                writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
                log.write(json.dumps({"frame": current, "seconds": current / fps, "predictions": predictions}, ensure_ascii=False) + "\n")
                count += 1
                if count % 5 == 0:
                    emit({"event": "progress", "progress": 0.15 + 0.75 * count / max(1, target), "message": f"已處理 {count}/{target} 個取樣影格"})
        if writer:
            writer.release()
            writer = None
        if not count:
            raise ValueError("影片中沒有可讀取的影格")
        emit({"event": "progress", "progress": 0.92, "message": "轉換為瀏覽器可播放的 H.264 影片"})
        final_path = output / "result.mp4"
        h264_video(raw_path, final_path)
        truncated = frame_index < total
        return {"media_path": str(final_path), "media_kind": "video", "predictions": preview,
                "inference_ms": round(sum(times) / len(times), 2), "frames": count,
                "summary": f"完成 {count} 個影格；每 {stride} 幀取樣一次" + ("（已達設定上限，未處理全片）" if truncated else ""),
                "artifacts": {"predictions": str(output / "predictions.jsonl"), "preview": str(output / "preview.jpg")}}
    finally:
        capture.release()
        if writer:
            writer.release()


def run_action(request, output, emit, device, *, loaded_model=None):
    import cv2
    import numpy as np
    import torch
    source = Path(request["media_path"])
    params = request["params"]
    model, metadata = loaded_model if loaded_model is not None else create_action_model(
        request["model_id"], device, request.get("checkpoint_path"), request.get("model_weight_path"))
    emit({"event": "progress", "progress": 0.2, "message": "行為模型已載入，讀取影片片段"})
    first_frames, duration = read_clip(source, metadata["num_frames"], params["start_seconds"], params["clip_seconds"])
    if params["start_seconds"] >= duration:
        raise ValueError("開始時間超過影片長度")
    last_start = max(params["start_seconds"], duration - params["clip_seconds"])
    starts = np.linspace(params["start_seconds"], last_start, int(params["num_clips"]))
    scores, clips, elapsed = [], [], []
    for index, start_seconds in enumerate(starts):
        frames = first_frames if index == 0 else read_clip(source, metadata["num_frames"], float(start_seconds), params["clip_seconds"])[0]
        tensor = prepare_action_tensor(frames, metadata, device)
        synchronize(device)
        before = time.perf_counter()
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
        with torch.inference_mode(), context:
            if metadata["architecture"] == "vivit":
                logits = model(pixel_values=tensor).logits
            else:
                logits = model(tensor)
            probabilities = classification_probabilities(logits, metadata["outputs_probabilities"])
        synchronize(device)
        elapsed.append((time.perf_counter() - before) * 1000)
        probabilities = probabilities[0].cpu()
        scores.append(probabilities)
        values, indices = probabilities.topk(min(int(params["top_k"]), probabilities.numel()))
        predictions = [{"label": metadata["labels"].get(int(i), str(int(i))), "score": float(v)} for v, i in zip(values, indices)]
        clips.append({"start_seconds": float(start_seconds), "end_seconds": min(duration, float(start_seconds) + params["clip_seconds"]), "predictions": predictions})
        emit({"event": "progress", "progress": 0.2 + 0.65 * (index + 1) / len(starts), "message": f"片段 {index + 1}/{len(starts)} 辨識完成"})
    average = torch.stack(scores).mean(0)
    values, indices = average.topk(min(int(params["top_k"]), average.numel()))
    predictions = [{"label": metadata["labels"].get(int(i), str(int(i))), "score": float(v)} for v, i in zip(values, indices)]
    title = f"{predictions[0]['label']}  {predictions[0]['score']:.1%}"
    draw_predictions(first_frames[len(first_frames) // 2], [], title).save(output / "preview.jpg")
    # Copy/remux the original video with its own timeline, no fabricated localization.
    import imageio_ffmpeg
    result_path = output / "result.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
                    "-vf", "scale=min(1280\\,iw):-2", "-movflags", "+faststart", str(result_path)],
                   check=True, capture_output=True, timeout=900)
    (output / "predictions.json").write_text(json.dumps({"average_predictions": predictions, "clips": clips}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"media_path": str(result_path), "media_kind": "video", "predictions": predictions,
            "inference_ms": round(sum(elapsed) / len(elapsed), 2), "frames": int(metadata["num_frames"]) * len(clips),
            "clips": clips, "summary": f"{len(clips)} 個片段的平均分類分數；不是人物追蹤或時間區間定位",
            "artifacts": {"predictions": str(output / "predictions.json"), "preview": str(output / "preview.jpg")}}


DETECTOR_IDS = {"yolov12x", "yolo26x", "grounding_dino_base", "rf_detr_2xl"}


def inference_model_key(request, device):
    from .inference_cache import model_cache_key
    from .model_registry import validate_checkpoint, validate_weight
    model_id = request["model_id"]
    # These checks must run on hits too: cache ownership does not grant a
    # license, and a missing/rebound/replaced weight must not reuse old tensors.
    if model_id == "rf_detr_2xl" and not request["params"].get("accept_platform_license"):
        raise ValueError("尚未確認 RF-DETR 2XL 所需的 PML 授權")
    checkpoint = request.get("checkpoint_path")
    source = validate_checkpoint(model_id, checkpoint) if checkpoint else validate_weight(
        model_id, request.get("model_weight_path") or WEIGHTS[model_id])
    extras = []
    if model_id in {"vivit_base_k400", "x3d_l", "slowfast_r101_8x8"}:
        extras.append(ASSETS / "kinetics400.json")
    elif model_id == "videomaev2_giant_k710":
        extras.append(ASSETS / "kinetics710.txt")
    return model_cache_key(model_id, device, source, checkpoint=bool(checkpoint),
                           folder_model=model_id in {"grounding_dino_base", "vivit_base_k400"},
                           extra_metadata=extras)


def _is_out_of_memory(error):
    # PyTorch OutOfMemoryError derives from RuntimeError. Include CPU allocator
    # failures too, since evicting other models can free RAM for the retry.
    torch = sys.modules.get("torch")
    oom_type = getattr(torch, "OutOfMemoryError", ())
    return isinstance(error, MemoryError) or isinstance(error, oom_type) or (isinstance(error, RuntimeError) and
        any(text in str(error).lower() for text in ("out of memory", "can't allocate memory", "cannot allocate memory", "cudnn_status_alloc_failed")))


def _clear_exception_tensors(error, *, detach=False):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if error.__traceback__ is not None:
            traceback.clear_frames(error.__traceback__)
            if detach:
                error.__traceback__ = None
        error = error.__cause__ or error.__context__


def run_inference(request, output_dir, emit, *, model_cache=None):
    """Run one job, optionally borrowing models from a serial worker's cache."""
    from .inference_cache import InferenceModelCache, release_unused_memory
    started = time.perf_counter()
    output = Path(output_dir)
    device = resolve_device(request["params"].get("device", "auto"))
    key = inference_model_key(request, device)
    cache = model_cache if model_cache is not None else InferenceModelCache(max_entries=1)
    detection = request["model_id"] in DETECTOR_IDS
    load_seconds = 0.0
    resource = None
    first_oom = None

    def load():
        nonlocal load_seconds
        before = time.perf_counter()
        try:
            if detection:
                return create_detector(request["model_id"], request["params"], device,
                                       request.get("checkpoint_path"), request.get("model_weight_path"))
            return create_action_model(request["model_id"], device,
                                       request.get("checkpoint_path"), request.get("model_weight_path"))
        finally:
            load_seconds += time.perf_counter() - before

    emit({"event": "log", "message": f"運算裝置：{device}；使用本機權重，不自動下載模型"})
    try:
        for attempt in range(2):
            try:
                resource, hit = cache.acquire(key, load)
                emit({"event": "log", "message": "重用已載入模型" if hit else "已載入本次使用的模型"})
                result = (run_detection(request, output, emit, device, predict=resource) if detection else
                          run_action(request, output, emit, device, loaded_model=resource))
                result["runtime"] = {"model_cache_hit": bool(hit and not attempt),
                    "model_load_seconds": round(load_seconds, 6),
                    "total_seconds": round(time.perf_counter() - started, 6),
                    "oom_retry": bool(attempt), "worker_pid": os.getpid()}
                return result
            except Exception as error:
                resource = None
                cache.discard(key, release=False)
                retry = attempt == 0 and _is_out_of_memory(error)
                _clear_exception_tensors(error, detach=retry)
                if not retry:
                    release_unused_memory()
                    if first_oom is not None:
                        raise error from first_oom
                    raise
                # No traceback frames/tensors survive into the next attempt.
                first_oom = error
                cache.clear(release=False)
            release_unused_memory()
            emit({"event": "log", "message": "記憶體不足，已釋放模型快取；重新載入並重試一次"})
    finally:
        resource = None
        if model_cache is None:
            cache.close()
