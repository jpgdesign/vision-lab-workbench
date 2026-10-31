"""Inspect complete Grounding training prompts before entering the GPU queue.

The API imports only stdlib here. Tokenizers, config and label-map validation
run in a short-lived, offline CPU process; no model weights are loaded.
"""
from __future__ import annotations

import copy
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

RESULT_PREFIX = "GROUNDING_PREFLIGHT_RESULT="
TIMEOUT_SECONDS = 90
_TOKENIZER_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
)


def _source_signature(request):
    source = request.get("checkpoint_path") or request.get("model_weight_path")
    if not source:
        raise ValueError("Grounding DINO 預檢缺少已驗證的本機來源權重。")
    path = Path(source).resolve(strict=True)
    folder = path if path.is_dir() else path.parent
    if not (folder / "config.json").is_file():
        raise ValueError("Grounding DINO 來源缺少 config.json；請補齊本機模型設定與 tokenizer。")
    signature = []
    for name in _TOKENIZER_FILES:
        item = folder / name
        if item.is_file():
            stat = item.stat()
            signature.append((name, stat.st_size, stat.st_mtime_ns))
    return str(folder), tuple(signature)


def _validate_plan(value, classes):
    if not isinstance(value, dict):
        raise ValueError("Grounding DINO 預檢沒有回傳有效的提示計畫。")
    chunks = value.get("caption_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Grounding DINO 預檢缺少完整提示分組。")
    seen, tokens = [], []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("caption"), str) or not chunk["caption"]:
            raise ValueError("Grounding DINO 預檢的提示分組格式不正確。")
        ids, count = chunk.get("global_class_ids"), chunk.get("text_tokens")
        if (not isinstance(ids, list) or not ids or any(type(index) is not int for index in ids)
                or type(count) is not int or count < 1):
            raise ValueError("Grounding DINO 預檢的類別對照或 token 計數不正確。")
        seen.extend(ids)
        tokens.append(count)
    if seen != list(range(len(classes))):
        raise ValueError("Grounding DINO 預檢未完整保留每個類別一次，不能開始訓練。")
    limits = [value.get(key) for key in ("base_max_text_len", "max_text_len", "text_backbone_max_position_embeddings")]
    if any(type(limit) is not int or limit < 1 for limit in limits):
        raise ValueError("Grounding DINO 預檢缺少有效的文字長度上限。")
    base, effective, maximum = limits
    if not base <= effective <= maximum or max(tokens) > effective:
        raise ValueError("Grounding DINO 預檢分組仍超過文字長度上限。")
    return {**value, "class_count": len(classes), "chunk_count": len(chunks),
            "caption_mode": "chunked" if len(chunks) > 1 else "single",
            "text_tokens": tokens, "classes_per_chunk": [len(chunk["global_class_ids"]) for chunk in chunks]}


@functools.lru_cache(maxsize=32)
def _cached_prompt_plan(folder, signature, classes):
    # signature is part of the key even though the subprocess reads the files.
    # Editing/replacing config or tokenizer metadata must invalidate a preview.
    result = subprocess.run(
        [sys.executable, "-m", "backend.grounding_preflight", "--inspect"],
        input=json.dumps({"folder": folder, "classes": list(classes)}, ensure_ascii=False),
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8",
        timeout=TIMEOUT_SECONDS, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1"},
    )
    lines = [line[len(RESULT_PREFIX):] for line in result.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        raise ValueError("Grounding DINO 預檢程序未回傳唯一結果。" + (f" {result.stderr[-1200:].strip()}" if result.stderr else ""))
    try:
        value = json.loads(lines[0])
    except (ValueError, TypeError) as exc:
        raise ValueError("Grounding DINO 預檢結果無法解析。") from exc
    if not isinstance(value, dict) or value.get("ok") is not True or result.returncode:
        error = value.get("error") if isinstance(value, dict) else None
        raise ValueError("Grounding DINO 文字預檢失敗：" + str(error or "無法讀取本機 tokenizer／config。")[:1600])
    return _validate_plan(value.get("plan"), classes)


def preflight(request, checked):
    """Enrich a valid expansion plan, or return an actionable pre-queue error."""
    if request.get("model_id") != "grounding_dino_base" or not checked.get("valid"):
        return checked
    result = copy.deepcopy(checked)
    result.setdefault("plan", {})
    result.setdefault("errors", [])
    result.setdefault("warnings", [])
    try:
        mode = request.get("training_mode", "standard")
        values = (checked.get("plan", {}).get("classes") if mode in {"expand_frozen", "expand_classes"}
                  else request.get("dataset", {}).get("classes"))
        if (not isinstance(values, list) or not values
                or any(not isinstance(name, str) or not name.strip() for name in values)):
            raise ValueError("Grounding DINO 預檢需要完整且非空的訓練類別清單。")
        folder, signature = _source_signature(request)
        plan = copy.deepcopy(_cached_prompt_plan(folder, signature, tuple(values)))
        result["plan"]["grounding_prompts"] = plan
        if plan["chunk_count"] > 1:
            result["warnings"].append(
                f"Grounding DINO 的 {plan['class_count']} 個完整類別會分成 {plan['chunk_count']} 組文字提示，"
                f"各組為 {', '.join(map(str, plan['text_tokens']))} tokens；全部類別均參與訓練，不會截斷。"
                "每張影像會依序處理全部分組，訓練及完整類別推論的耗時會增加。")
        elif plan["max_text_len"] > plan["base_max_text_len"]:
            result["warnings"].append(
                f"Grounding DINO 保留全部 {plan['class_count']} 個類別，文字提示上限將由 "
                f"{plan['base_max_text_len']} 提高至 {plan['max_text_len']} tokens，仍在文字骨幹上限內。")
    except subprocess.TimeoutExpired:
        result["errors"].append(f"Grounding DINO 文字預檢超過 {TIMEOUT_SECONDS} 秒；請確認本機 tokenizer 與 config 完整後重試。")
    except (ValueError, TypeError, OSError, subprocess.SubprocessError) as exc:
        result["errors"].append(str(exc))
    result["valid"] = not result["errors"]
    return result


def _inspect(payload):
    # This is called only by the CLI subprocess, never by the API preflight.
    from transformers import AutoConfig, AutoTokenizer
    from .grounding_prompts import prepare_caption_chunks
    folder, classes = payload["folder"], payload["classes"]
    config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    chunks, limits = prepare_caption_chunks(classes, tokenizer, config)
    return {**limits, "caption_chunks": chunks}


def _main():
    if sys.argv[1:] != ["--inspect"]:
        raise SystemExit("Usage: python -m backend.grounding_preflight --inspect")
    try:
        payload = json.load(sys.stdin)
        value = {"ok": True, "plan": _inspect(payload)}
    except Exception as exc:
        value = {"ok": False, "error": str(exc)}
    print(RESULT_PREFIX + json.dumps(value, ensure_ascii=False), flush=True)
    raise SystemExit(0 if value["ok"] else 1)


if __name__ == "__main__":
    _main()
