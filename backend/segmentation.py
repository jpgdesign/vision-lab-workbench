"""Exact instance masks for artifacts, compact summaries for UI job state."""

SEGMENTATION_IDS = {"yolov12x_seg", "yolo26x_seg", "rf_detr_seg_2xl"}


def attach_mask(prediction, mask, image_size):
    import cv2
    import numpy as np
    from pycocotools import mask as mask_api
    width, height = image_size
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("分割模型必須回傳每個物件的二維遮罩。")
    values = (values > 0.5).astype(np.uint8)
    if values.shape != (height, width):
        values = cv2.resize(values, (width, height), interpolation=cv2.INTER_NEAREST)
    encoded = mask_api.encode(np.asfortranarray(values))
    return {**prediction, "mask": {"size": [height, width], "counts": encoded["counts"].decode("ascii")},
            "mask_area": int(values.sum()), "mask_encoding": "coco_rle"}


def decode_mask(mask, shape):
    from pycocotools import mask as mask_api
    if not isinstance(mask, dict) or mask.get("size") != list(shape) or not isinstance(mask.get("counts"), str):
        raise ValueError("分割遮罩的尺寸或 COCO RLE 格式不正確。")
    try:
        return mask_api.decode({"size": mask["size"], "counts": mask["counts"].encode("ascii")}).astype(bool)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("無法解碼分割遮罩。") from exc


def overlay_masks(rgb, predictions, colors, outline_width):
    import cv2
    import numpy as np
    from PIL import ImageColor
    result = np.asarray(rgb).copy()
    for index, prediction in enumerate(predictions):
        if "mask" not in prediction:
            continue
        mask = decode_mask(prediction["mask"], result.shape[:2])
        color = ImageColor.getrgb(colors[index % len(colors)])
        result[mask] = np.rint(result[mask] * 0.6 + np.asarray(color) * 0.4).astype(np.uint8)
        # All contours, including holes, retain the model's actual geometry.
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, outline_width)
    return result


def prediction_summaries(predictions):
    """Full masks stay in downloadable JSON/JSONL, not the repeatedly polled jobs."""
    return [{key: value for key, value in prediction.items() if key != "mask"} for prediction in predictions]


def create_rf_seg_predictor(path, params, device):
    from pathlib import Path
    from unittest.mock import patch
    import torch
    from rfdetr import RFDETRSeg2XLarge
    from rfdetr.models.lwdetr import LWDETR

    class LocalSegmentationModel(RFDETRSeg2XLarge):
        def maybe_download_pretrain_weights(self):
            if not Path(self.model_config.pretrain_weights).is_file():
                raise ValueError("本機 RF-DETR 分割權重不存在，請先下載或綁定路徑。")

    original_load = torch.nn.Module.load_state_dict

    def strict_segmentation_load(module, state, strict=True, **kwargs):
        # Upstream defaults to strict=False. Reject missing/random mask heads
        # and incompatible variants instead of claiming a valid segmenter.
        if getattr(module, "segmentation_head", None) is None:
            raise ValueError("RF-DETR 權重沒有實例分割架構。")
        try:
            return original_load(module, state, strict=True, **kwargs)
        except RuntimeError as exc:
            raise ValueError("權重與 RF-DETR-Seg-2XL 分割架構不符；不能使用偵測版權重。") from exc

    with patch("rfdetr.main.download_pretrain_weights", side_effect=ValueError("本機 RF-DETR 分割權重無法讀取。")), \
            patch.object(LWDETR, "load_state_dict", strict_segmentation_load):
        rf_device = "cuda" if str(device).startswith("cuda") else device
        model = LocalSegmentationModel(pretrain_weights=str(path), device=rf_device)
    classes = model.class_names

    def predict(image, inference_params=None):
        current = params if inference_params is None else inference_params
        detections = model.predict(image, threshold=current["conf"])
        masks = detections.mask
        if len(detections) and (masks is None or len(masks) != len(detections)):
            raise ValueError("RF-DETR 分割結果缺少物件遮罩。")
        return [attach_mask({"label": classes.get(int(cls), str(int(cls))) if isinstance(classes, dict) else classes[int(cls)],
                             "score": float(score), "box": box.tolist()}, masks[index], image.size)
                for index, (box, score, cls) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id))]
    return predict
