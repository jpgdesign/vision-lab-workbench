"""Complete Grounding DINO vocabularies split without changing class identity.

Each fixed caption is a conditional detection task. A target is supervised in
exactly the chunk containing its class; all other phrases in that chunk remain
negative labels. Every image visits every chunk, including empty-target chunks.
"""
from __future__ import annotations

import re


def validated_caption(classes, tokenizer, max_tokens):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    if not classes or any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ValueError("Grounding DINO 類別名稱必須是非空字串。")
    if any(re.search(r"[.?!\[\]]", name) for name in classes):
        raise ValueError("Grounding DINO 類別名稱不可含句點、問號、驚嘆號或特殊 token 括號；請改用完整單一詞組。")
    normalized = [name.strip().lower() for name in classes]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Grounding DINO 類別名稱轉小寫後不可重複。")
    caption = " . ".join(normalized) + " ."
    ids = tokenizer(caption, return_tensors="pt", truncation=False)["input_ids"]
    count = ids.shape[1]
    if count > max_tokens:
        raise ValueError(f"所有類別提示共 {count} tokens，超過本機 Grounding DINO 上限 {max_tokens}；請縮短類別名稱或拆分提示。")
    maps = build_label_maps(torch.empty((1, 1, max_tokens)), ids)
    if maps[0].shape[0] != len(classes) or (maps[0].sum(-1) == 0).any():
        raise ValueError("類別名稱無法與 Grounding DINO 的文字 token 一一對應，請移除特殊標記與分隔符。")
    return caption, int(count)


def build_caption_chunks(classes, tokenizer, max_tokens):
    """Pack stable, consecutive global IDs using the actual tokenizer budget."""
    if type(max_tokens) is not int or max_tokens < 3:
        raise ValueError("Grounding DINO 的提示 token 上限無效。")
    # Validate all identities before grouping; groups cannot hide duplicates.
    normalized = [name.strip().lower() if isinstance(name, str) else name for name in classes]
    if not normalized or any(not isinstance(name, str) or not name for name in normalized):
        raise ValueError("Grounding DINO 類別名稱必須是非空字串。")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Grounding DINO 類別名稱轉小寫後不可重複。")
    chunks, ids = [], []

    def append(current):
        caption, count = validated_caption([classes[i] for i in current], tokenizer, max_tokens)
        chunks.append({"caption": caption, "global_class_ids": list(current), "text_tokens": count})

    for index, name in enumerate(classes):
        # A single over-budget/ambiguous phrase is an error, never discarded.
        validated_caption([name], tokenizer, max_tokens)
        candidate = ids + [index]
        text = " . ".join(normalized[i] for i in candidate) + " ."
        # Measuring an over-budget candidate is expected; it never enters the
        # model. Suppress only the tokenizer's misleading model-length warning.
        count = tokenizer(text, return_tensors="pt", truncation=False, verbose=False)["input_ids"].shape[1]
        if ids and count > max_tokens:
            append(ids)
            ids = [index]
        else:
            ids = candidate
    append(ids)
    return chunks


def prepare_caption_chunks(classes, tokenizer, config):
    """Set only the contrastive-head capacity before model construction."""
    original = getattr(config, "max_text_len", None)
    maximum = getattr(getattr(config, "text_config", None), "max_position_embeddings", None)
    if any(type(value) is not int or value < 1 for value in (original, maximum)):
        raise ValueError("Grounding DINO 設定缺少有效的 max_text_len 或文字位置上限。")
    if original > maximum:
        raise ValueError("Grounding DINO 的提示上限超過文字骨幹位置上限。")
    chunks = build_caption_chunks(classes, tokenizer, maximum)
    effective = max(original, max(chunk["text_tokens"] for chunk in chunks))
    config.max_text_len = effective
    return chunks, {"base_max_text_len": original, "max_text_len": effective,
                    "text_backbone_max_position_embeddings": maximum}


def validate_caption_chunks(chunks, classes, tokenizer, max_tokens):
    """Verify saved mappings before using their global IDs for predictions."""
    normalized = [name.strip().lower() if isinstance(name, str) else name for name in classes]
    if not normalized or any(not isinstance(name, str) or not name for name in normalized):
        raise ValueError("Grounding DINO 類別名稱必須是非空字串。")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Grounding DINO 類別名稱轉小寫後不可重複。")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Grounding DINO checkpoint 缺少完整分組提示。")
    visited = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Grounding DINO 分組提示格式不正確。")
        ids = chunk.get("global_class_ids")
        if (not isinstance(ids, list) or not ids
                or any(type(i) is not int or not 0 <= i < len(classes) for i in ids)):
            raise ValueError("Grounding DINO 分組類別索引不正確。")
        caption, count = validated_caption([classes[i] for i in ids], tokenizer, max_tokens)
        if chunk.get("caption") != caption or chunk.get("text_tokens") != count:
            raise ValueError("Grounding DINO 分組文字與類別對應不一致。")
        visited.extend(ids)
    if visited != list(range(len(classes))):
        raise ValueError("Grounding DINO 分組必須依原順序完整涵蓋每個類別一次。")
    return chunks


def remap_chunk_labels(labels, global_class_ids):
    """Keep every box for this caption and remap only its category indices."""
    import torch
    result = []
    lookup = {global_id: local_id for local_id, global_id in enumerate(global_class_ids)}
    for target in labels:
        source_ids = target["class_labels"]
        keep = torch.tensor([int(i) in lookup for i in source_ids], dtype=torch.bool, device=source_ids.device)
        local_ids = torch.tensor([lookup[int(i)] for i in source_ids[keep]], dtype=torch.long, device=source_ids.device)
        result.append({"class_labels": local_ids, "boxes": target["boxes"][keep]})
    return result


def chunk_loss_weight(batch_samples, accumulation_samples, chunk_count):
    """Average fixed caption tasks, then weight the real accumulation window."""
    if min(batch_samples, accumulation_samples, chunk_count) < 1 or batch_samples > accumulation_samples:
        raise ValueError("分組損失的樣本數或梯度累積數無效。")
    return batch_samples / (accumulation_samples * chunk_count)
