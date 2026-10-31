"""Compatibility for the YOLOv12 author's segmentation checkpoints.

The author's Seg branch stores separate qk/v projections and a 5x5 positional
convolution. Ultralytics 8.4's AAttn expects qkv instead. Bind the matching
forward only to the old checkpoint instances, keeping their learned weights.

Adapted from the original non-FlashAttention forward, AGPL-3.0:
https://github.com/sunsmarterjie/yolov12/blob/Seg/ultralytics/nn/modules/block.py
"""

from types import MethodType


def _seg_area_attention_forward(self, features):
    import torch

    batch, channels, height, width = features.shape
    tokens = height * width
    query_key = self.qk(features).flatten(2).transpose(1, 2)
    value_map = self.v(features)
    positional = self.pe(value_map)
    values = value_map.flatten(2).transpose(1, 2)

    if self.area > 1:
        query_key = query_key.reshape(batch * self.area, tokens // self.area, channels * 2)
        values = values.reshape(batch * self.area, tokens // self.area, channels)
        batch, tokens, _ = query_key.shape

    queries, keys = query_key.split([channels, channels], dim=2)
    queries = queries.transpose(1, 2).view(batch, self.num_heads, self.head_dim, tokens)
    keys = keys.transpose(1, 2).view(batch, self.num_heads, self.head_dim, tokens)
    values = values.transpose(1, 2).view(batch, self.num_heads, self.head_dim, tokens)
    logits = (queries.transpose(-2, -1) @ keys) * (self.head_dim ** -0.5)
    exponentials = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    attention = exponentials / exponentials.sum(dim=-1, keepdim=True)
    attended = (values @ attention.transpose(-2, -1)).permute(0, 3, 1, 2)

    if self.area > 1:
        attended = attended.reshape(batch // self.area, tokens * self.area, channels)
        batch, tokens, _ = attended.shape

    attended = attended.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    return self.proj(attended + positional)


def prepare_yolo12_seg_model(model):
    """Bind the author's forward to qk/v AAttn instances; return patch count.

    Accept a YOLO wrapper or a PyTorch model. Standard qkv modules are untouched,
    and repeat calls are harmless. No package classes or source files change.
    """
    network = getattr(model, "model", model)
    patched = 0
    for module in network.modules():
        if (type(module).__name__ == "AAttn"
                and hasattr(module, "qk") and hasattr(module, "v")
                and not hasattr(module, "qkv")
                and not getattr(module, "_workbench_yolo12_seg_compat", False)):
            module.forward = MethodType(_seg_area_attention_forward, module)
            module._workbench_yolo12_seg_compat = True
            patched += 1
    return patched
