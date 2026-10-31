"""Content-based provenance for inputs of completed training jobs (stdlib only)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading


MANIFEST = "training_inputs.json"


def _stat_key(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_files(request):
    datasets = [("new", request.get("dataset", {}))]
    if request.get("training_mode") == "expand_classes":
        datasets.append(("replay", request.get("replay_dataset", {})))
    seen = set()
    for role, dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        details = dataset.get("details", {})
        if not isinstance(details, dict):
            continue
        values = details.get("train_images")
        if values is None:
            values = [sample.get("path") for sample in details.get("train_samples", []) if isinstance(sample, dict)]
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not Path(value).is_absolute():
                continue
            path = Path(value).resolve()
            if path not in seen:
                seen.add(path)
                yield role, path


def _write_once(path, document):
    # Job outputs are single-writer. Exclusive creation prevents a retry or a
    # legacy API backfill from replacing the original fingerprint snapshot.
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        pass


def record_training_inputs(request, output):
    """Snapshot original train inputs before expansion copies/materialization."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / MANIFEST
    if path.exists():
        return path
    files = []
    for role, source in _training_files(request):
        before = _stat_key(source)
        digest = _sha256(source)
        if _stat_key(source) != before:
            raise OSError("訓練來源檔案在計算內容指紋時改變，請確認資料集後重試。")
        files.append({"source_role": role, "source_path": str(source), "sha256": digest,
                      "bytes": before[2], "mtime_ns": before[3]})
    _write_once(path, {"schema_version": 1, "basis": "snapshot_before_training",
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "training_mode": request.get("training_mode", "standard"), "files": files})
    return path


class TrainingUsageIndex:
    """Annotate local media without exposing source paths or importing ML code.

    Only completed jobs count. A saved pre-training manifest remains immutable
    even after a source file is edited or removed. Legacy completed jobs have no
    historical content hash: backfill only exact request paths whose filesystem
    timestamps do not postdate the job start, and persist that weaker basis.
    """

    def __init__(self, job_root):
        self.job_root = Path(job_root).resolve()
        self._lock = threading.RLock()
        self._manifests = {}
        self._file_hashes = {}

    def _fingerprint(self, path):
        key = _stat_key(path)
        cached = self._file_hashes.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        value = _sha256(path)
        if _stat_key(path) != key:
            raise OSError("Media changed while hashing")
        self._file_hashes[path] = (key, value)
        return value

    def _backfill(self, directory, job):
        request_path = directory / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("Invalid historical training request")
        timestamp = job.get("started_at") or job.get("created_at")
        if isinstance(timestamp, str) and timestamp:
            started = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            cutoff_ns = int(started.timestamp() * 1_000_000_000)
        else:
            cutoff_ns = request_path.stat().st_mtime_ns
        files, skipped = [], 0
        for role, source in _training_files(request):
            try:
                before = _stat_key(source)
                if max(before[3], before[4]) > cutoff_ns:
                    skipped += 1
                    continue
                digest = self._fingerprint(source)
                files.append({"source_role": role, "source_path": str(source), "sha256": digest,
                              "bytes": before[2], "mtime_ns": before[3]})
            except OSError:
                skipped += 1
        _write_once(directory / MANIFEST, {"schema_version": 1,
                    "basis": "legacy_request_unchanged_files",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "training_mode": request.get("training_mode", "standard"),
                    "skipped_unverifiable_files": skipped, "files": files})

    def _job_hashes(self, job):
        job_id = job.get("id", "")
        if (job.get("kind") != "training" or job.get("status") != "completed"
                or not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id)):
            return set()
        directory = self.job_root / job_id
        path = directory / MANIFEST
        try:
            if not path.exists():
                self._backfill(directory, job)
            signature = _stat_key(path)
            cached = self._manifests.get(job_id)
            if cached is not None and cached[0] == signature:
                return cached[1]
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or not isinstance(document.get("files"), list):
                raise ValueError("Invalid training input manifest")
            hashes = {entry["sha256"] for entry in document.get("files", [])
                      if isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
                      and re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])}
            self._manifests[job_id] = (signature, hashes)
            return hashes
        except (OSError, ValueError, TypeError, KeyError):
            return set()

    def annotate(self, entries, jobs):
        """Return copies; callers strip private entry.path through _media_public."""
        with self._lock:
            index = {}
            for job in jobs.values() if isinstance(jobs, dict) else jobs:
                for digest in self._job_hashes(job):
                    index.setdefault(digest, set()).add(job["id"])
            result = []
            for entry in entries:
                job_ids = []
                try:
                    path = Path(entry["path"]).resolve()
                    if index and path.is_file():
                        job_ids = sorted(index.get(self._fingerprint(path), ()))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                result.append({**entry, "training_usage": {"used": bool(job_ids), "job_ids": job_ids}})
            return result
