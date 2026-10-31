"""Real pretrained ST-GCN++ skeleton action inference for one tracked person.

The caller owns tracking: never concatenate different people into one sequence.
This is closed-set NTU120 classification, not a model of arbitrary workflow
steps, and its softmax is not an open-set or safety-event probability.
"""
from pathlib import Path
import math


VENDOR = Path(__file__).parent / "vendor" / "pyskl_stgcnpp"
CLASS_LABELS = tuple((VENDOR / "labels_ntu120.txt").read_text(encoding="utf-8").splitlines())
WEIGHT_URL = "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth"
WEIGHT_SHA256 = "c6e074210fa30c01081460665ce9c1c221ba4a60d7eeb10456cde6e8ac40486e"
WEIGHT_BYTES = 5915801
MIN_SEQUENCE_FRAMES = 30


def prepare_sequence(sequence, image_size):
    """Return float32 [1,2,100,17,3], rejecting insufficient/invalid evidence.

    ``sequence`` must be [T,17,3] pixel x/y/confidence from one stable track.
    The quality gate is an application constraint, separate from the official
    normalization: at least 30 frames and half the window must have six joints
    above .1 confidence, including two shoulder/hip points. No missing poses
    are fabricated; upstream normalization zeroes missing x/y and sampling
    repeats short valid sequences to its fixed 100-frame model input.
    """
    import numpy as np
    from .vendor.pyskl_stgcnpp.preprocessing import coco_joint_input

    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in image_size)):
        raise ValueError("image_size 必須是有效的 (width, height)。")
    try:
        values = np.asarray(sequence, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("骨架資料必須是數值 [T,17,3]。") from exc
    if values.ndim != 3 or values.shape[1:] != (17, 3):
        raise ValueError("骨架動作需要同一追蹤人物的 [T,17,3] COCO 關節；不能混合不同人物。")
    if not MIN_SEQUENCE_FRAMES <= len(values) <= 3000:
        raise ValueError(f"骨架動作需要至少 {MIN_SEQUENCE_FRAMES} 個影格，最多 3000 個；目前資料不足或超出上限。")
    if not np.isfinite(values).all():
        raise ValueError("骨架座標與信心值不可包含 NaN 或 Inf。")
    confidence = values[..., 2]
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("關節信心值必須介於 0 和 1。")
    visible = confidence > 0.1
    usable = (visible.sum(axis=1) >= 6) & (visible[:, [5, 6, 11, 12]].sum(axis=1) >= 2)
    if int(usable.sum()) < max(MIN_SEQUENCE_FRAMES, math.ceil(len(values) / 2)):
        raise ValueError("骨架資料不足：有效人物姿態少於 30 幀或視窗的一半，暫不判定動作。")
    return coco_joint_input(values, image_size)


class SkeletonActionModel:
    """Portable official NTU120 COCO17 joint checkpoint, without MMCV."""
    class_labels = CLASS_LABELS
    labels = CLASS_LABELS
    model_id = "stgcnpp_ntu120"
    keypoint_format = "coco17_xy_confidence_pixels"

    def __init__(self, weight_path, device="cpu"):
        import torch
        from .vendor.pyskl_stgcnpp.model import STGCNPPNTU120

        path = Path(weight_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("找不到 ST-GCN++ NTU120 權重檔；請先下載官方 checkpoint。")
        try:
            selected = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("骨架模型裝置必須是 cpu 或 cuda 裝置。") from exc
        if selected.type not in {"cpu", "cuda"}:
            raise ValueError("骨架模型目前支援 cpu 或 cuda 裝置。")
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("指定 CUDA，但目前無法使用 GPU。")
        model = STGCNPPNTU120()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("ST-GCN++ 權重必須包含完整 tensor state_dict。")
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("ST-GCN++ 權重包含 NaN 或 Inf。")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError("權重與 ST-GCN++ NTU120 COCO17 joint 架構不相符。") from exc
        self.device = selected
        self.weight_path = path
        self.model = model.to(selected).eval()

    def predict(self, sequence, image_size):
        """Return the top five learned NTU120 classes with zero-based IDs."""
        import torch

        inputs = torch.from_numpy(prepare_sequence(sequence, image_size)).to(self.device)
        with torch.inference_mode():
            logits = self.model(inputs)
            if logits.shape != (1, 120) or not torch.isfinite(logits).all():
                raise RuntimeError("ST-GCN++ 模型輸出無效。")
            probabilities = logits.float().softmax(-1)[0]
            scores, indices = probabilities.topk(5)
        return [{"label": self.class_labels[int(index)], "score": float(score), "class_id": int(index)}
                for score, index in zip(scores.cpu(), indices.cpu())]
