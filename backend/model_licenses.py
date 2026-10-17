"""Read explicit, version-specific license acceptance saved for this runtime."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
import threading
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
LICENSE_FILE = "model_license_acceptance.json"
RF_DETR_LICENSE_ID = "PML-1.0"
RF_DETR_LICENSE_URL = "https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE"
_lock = threading.RLock()


def license_record_path() -> Path:
    from .model_registry import runtime_path
    return runtime_path(APP_ROOT) / LICENSE_FILE


def accept_model_license(model_id: str) -> dict:
    if model_id != "rf_detr_2xl":
        raise ValueError("此模型沒有需要記住的額外授權確認。")
    from .jobs import write_json_atomic
    path = license_record_path()
    with _lock:
        saved = {"schema_version": 1, "models": {}}
        if path.exists():
            if path.is_symlink() or path.stat().st_size > 64 * 1024:
                raise ValueError("授權紀錄檔不正確，未覆寫原始資料。")
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(saved, dict) or saved.get("schema_version") != 1 or not isinstance(saved.get("models"), dict):
                raise ValueError("授權紀錄檔不正確，未覆寫原始資料。")
        saved["models"][model_id] = {"license_id": RF_DETR_LICENSE_ID, "license_url": RF_DETR_LICENSE_URL,
            "accepted": True, "accepted_at": datetime.now(timezone.utc).isoformat(), "source": "user_confirmation"}
        write_json_atomic(path, saved)
        return saved["models"][model_id]


def is_model_license_accepted(model_id: str) -> bool:
    if model_id != "rf_detr_2xl":
        return False
    path = license_record_path()
    try:
        if path.is_symlink():
            return False
        with path.open("rb") as handle:
            content = handle.read(64 * 1024 + 1)
        if len(content) > 64 * 1024:
            return False
        saved = json.loads(content.decode("utf-8-sig"))
        if not isinstance(saved, dict) or type(saved.get("schema_version")) is not int or saved["schema_version"] != 1:
            return False
        models = saved.get("models")
        record = models.get(model_id) if isinstance(models, dict) else None
        return (isinstance(record, dict) and record.get("accepted") is True
                and record.get("license_id") == RF_DETR_LICENSE_ID
                and record.get("license_url") == RF_DETR_LICENSE_URL)
    except (OSError, UnicodeDecodeError, ValueError):
        # Missing, malformed, or inaccessible records never imply consent.
        return False
