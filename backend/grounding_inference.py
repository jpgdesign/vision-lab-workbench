"""One cached model, either free text or the complete saved training vocabulary."""
import json
from pathlib import Path


def create_grounding_predictor(model, processor, source_dir, params, device, *, is_checkpoint):
    import torch
    from .detection_expansion import grounding_predictor
    from .grounding_prompts import build_caption_chunks, validate_caption_chunks

    metadata = None
    chunks = None

    def predict(image, inference_params=None):
        nonlocal metadata, chunks
        current = params if inference_params is None else inference_params
        if current.get("use_training_classes", False):
            if not is_checkpoint:
                raise ValueError("使用完整微調類別需要選擇微調 checkpoint。")
            if metadata is None:
                metadata = json.loads((Path(source_dir) / "workbench_training.json").read_text(encoding="utf-8"))
            classes = metadata.get("classes")
            if not isinstance(classes, list) or not classes:
                raise ValueError("微調 checkpoint 缺少完整類別名稱。")
            if chunks is None:
                saved = metadata.get("caption_chunks")
                if saved is None:
                    saved = build_caption_chunks(classes, processor.tokenizer, model.config.max_text_len)
                chunks = validate_caption_chunks(saved, classes, processor.tokenizer, model.config.max_text_len)
            size = metadata.get("params", {}).get("imgsz", 640)
            if type(size) is not int or not 224 <= size <= 1024 or size % 32:
                raise ValueError("微調 checkpoint 的輸入尺寸無效。")
            native = grounding_predictor(model, processor, "", {"imgsz": size, "confidence": current["conf"]},
                                         device, caption_chunks=chunks)
            return [{**row, "label": classes[row["class_id"]]} for row in native(image)]

        prompt = current["prompt"].strip().lower()
        if not prompt.endswith("."):
            prompt += " ."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        maximum = getattr(getattr(model, "config", None), "max_text_len", None)
        if maximum and inputs["input_ids"].shape[1] > maximum:
            raise ValueError(f"偵測目標超過模型的 {maximum} tokens 上限；請縮短文字，或對微調權重啟用完整微調類別。")
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
            threshold=current["conf"], text_threshold=current["text_threshold"],
            target_sizes=[(image.height, image.width)])[0]
        labels = results.get("text_labels", results.get("labels", []))
        return [{"label": str(label), "score": float(score), "box": box.cpu().tolist()}
                for label, score, box in zip(labels, results["scores"], results["boxes"])]
    return predict
