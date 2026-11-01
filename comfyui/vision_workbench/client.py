"""Local-only bridge. ComfyUI never downloads weights or changes its ML stack."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import requests

BASE_URL = "http://127.0.0.1:8765"
_cache = {}
_lock = threading.RLock()


def request_json(method, path, payload=None, timeout=20):
    if not isinstance(path, str) or not path.startswith("/api/") or "://" in path:
        raise ValueError("Vision Workbench 路由無效。")
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.request(method, BASE_URL + path, json=payload, timeout=timeout, allow_redirects=False)
        if not 200 <= response.status_code < 300:
            try:
                detail = response.json().get("detail", "")
            except (ValueError, AttributeError):
                detail = ""
            raise RuntimeError(f"Vision Workbench HTTP {response.status_code}：{str(detail)[:1500]}")
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError("無法連接 Vision Workbench；請先開啟 http://127.0.0.1:8765/，再重新執行。") from exc


def resources(refresh=False):
    """Short-lived metadata cache keeps INPUT_TYPES fast across many nodes."""
    with _lock:
        if not refresh and _cache.get("expires", 0) > time.monotonic():
            return _cache["value"]
        value = {"models": [], "checkpoints": [], "media": [], "integration": {}, "available": False}
        try:
            for name in ("models", "checkpoints", "media"):
                value[name] = request_json("GET", "/api/" + name, timeout=3).get(name, [])
            value["integration"] = request_json("GET", "/api/workflows/integrations", timeout=3).get("munolith", {})
            value["available"] = True
        except RuntimeError:
            pass
        _cache.update(value=value, expires=time.monotonic() + (10 if value["available"] else 3))
        return value


def artifact_path(url):
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        if f"{parsed.scheme}://{parsed.netloc}" != BASE_URL:
            raise ValueError("拒絕非本機 Workbench 的輸出連結。")
    if not parsed.path.startswith("/api/artifacts/") or parsed.query or parsed.fragment or ".." in parsed.path.split("/"):
        raise ValueError("Vision Workbench 輸出路徑無效。")
    return parsed.path


def download_artifact(url, target, maximum=512 * 1024 * 1024):
    path = artifact_path(url)
    target = Path(target)
    temporary = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(BASE_URL + path, stream=True, timeout=(10, 60), allow_redirects=False) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"無法讀取 Workbench 輸出（HTTP {response.status_code}）。")
                count = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if count > maximum:
                            raise RuntimeError("輸出超過 512 MB，請直接從 Vision Workbench 結果頁下載。")
                        handle.write(chunk)
        temporary.replace(target)
    finally:
        if temporary.is_file():
            temporary.unlink()
    return target


def wait_for_job(job_id, interrupt=lambda: None, progress=lambda row: None):
    try:
        while True:
            interrupt()
            row = request_json("GET", f"/api/jobs/{job_id}")
            progress(row)
            if row.get("status") in {"completed", "failed", "cancelled"}:
                if row["status"] != "completed":
                    raise RuntimeError(row.get("error") or row.get("message") or f"工作已{row['status']}")
                return row
            for _ in range(4):
                time.sleep(.25)
                interrupt()
    except BaseException:
        try:
            request_json("POST", f"/api/jobs/{job_id}/cancel", timeout=5)
        except Exception:
            pass
        raise


def catalogue():
    # Shipping a schema snapshot allows registration while Workbench is closed.
    snapshot = json.loads(Path(__file__).with_name("catalog.json").read_text(encoding="utf-8"))
    try:
        values = request_json("GET", "/api/workflows/catalog", timeout=3).get("node_types")
        if isinstance(values, list) and len(values) >= len(snapshot):
            return values
    except RuntimeError:
        pass
    return snapshot
