"""Plan, download or verify optional workflow model assets.

Default prints a plan only. --download fetches pinned official URLs and verifies
byte counts/SHA256 before publishing files; --verify never changes any file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "backend" / "assets" / "workflow_model_sources.json"
CHUNK_SIZE = 1024 * 1024


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            result.update(chunk)
    return result.hexdigest()


def linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def load_manifest(path=MANIFEST):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("version") != 1 or not isinstance(value.get("models"), list):
        raise ValueError("Unsupported workflow asset manifest")
    seen = set()
    for model in value["models"]:
        if not re.fullmatch(r"[a-z0-9_]+", model.get("id", "")) or model["id"] in seen:
            raise ValueError("Invalid or repeated model ID")
        seen.add(model["id"])
        names = set()
        for item in model["files"]:
            name = item["name"]
            url = urlparse(item["url"])
            if (not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."} or name in names
                    or type(item["bytes"]) is not int or item["bytes"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    or url.scheme != "https" or url.hostname not in {"github.com", "download.openmmlab.com", "huggingface.co", "storage.googleapis.com"}
                    or url.username or url.password):
                raise ValueError("Invalid pinned asset declaration")
            names.add(name)
    return value


def target_path(root, model_id, filename):
    root = Path(root).expanduser().resolve()
    path = root / model_id / filename
    if not path.resolve().is_relative_to(root):
        raise ValueError("Asset path escapes the selected output directory")
    for candidate in (path.parent, path):
        if candidate.exists() and linked(candidate):
            raise ValueError("Asset destinations cannot be symbolic links or junctions")
    return path


def file_status(path, entry):
    if not path.is_file():
        return "missing"
    if linked(path) or path.stat().st_size != entry["bytes"]:
        return "invalid"
    return "verified" if sha256(path) == entry["sha256"] else "invalid"


def download_one(entry, target, *, replace_invalid=False, restart_partial=False, opener=urlopen, progress=print):
    target = Path(target)
    if target.exists() and linked(target):
        raise ValueError("Refusing a linked destination")
    status = file_status(target, entry)
    if status == "verified":
        return "already_verified"
    if target.exists() and not replace_invalid:
        raise ValueError(f"Existing {target.name} does not match the pinned asset; use --replace-invalid to replace only after verification")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download.part")
    if partial.exists() and linked(partial):
        raise ValueError("Refusing a linked partial download")
    if restart_partial:
        partial.unlink(missing_ok=True)
    offset = partial.stat().st_size if partial.is_file() else 0
    expected = entry["bytes"]
    if offset > expected:
        raise ValueError("Partial asset is too large; use --restart-partial")
    if offset < expected:
        headers = {"User-Agent": "Vision-Lab-Workbench-workflow-assets/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with opener(Request(entry["url"], headers=headers), timeout=60) as response:
            status_code = response.status
            if status_code not in (200, 206):
                raise ValueError(f"Unexpected download HTTP status {status_code}")
            if status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or int(match[1]) != offset or int(match[3]) != expected or int(match[2]) < offset or int(match[2]) >= expected:
                    raise ValueError("Server returned an invalid resume range")
                mode = "ab"
            else:
                # A server may ignore Range. Restart only our generated partial.
                offset, mode = 0, "wb"
            received, last_report = offset, 0.0
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > expected:
                        raise ValueError("Downloaded data exceeds the pinned byte count")
                    stream.write(chunk)
                    now = time.monotonic()
                    if now - last_report >= 1:
                        progress(f"{target.name}: {received:,}/{expected:,} bytes")
                        last_report = now
    if partial.stat().st_size != expected or sha256(partial) != entry["sha256"]:
        raise ValueError("Size/SHA256 verification failed; existing destination was preserved. Retry with --restart-partial after checking the source")
    os.replace(partial, target)
    return "downloaded_verified"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["pose", "skeleton", "relation"], help="pose, skeleton, relation, hand, mask, or exact model IDs; default: pose skeleton relation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models", help="Asset root; default project models/; model subfolders are created inside")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Download only selected official pinned assets")
    mode.add_argument("--verify", action="store_true", help="Check existing sizes/SHA256 without downloads or modifications")
    parser.add_argument("--replace-invalid", action="store_true", help="Replace an existing mismatched file only after a verified download")
    parser.add_argument("--restart-partial", action="store_true", help="Discard this tool's partial download before retrying")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    choices = {key: model for model in manifest["models"] for key in (model["id"], model["alias"])}
    if any(name not in choices for name in args.models):
        parser.error("Unknown model; choose pose, skeleton, relation, hand, mask or a manifest model ID")
    selected = list({choices[name]["id"]: choices[name] for name in args.models}.values())
    root = args.output_dir.expanduser().resolve()
    if not args.download and not args.verify:
        print(json.dumps({"mode": "plan_only", "output_dir": str(root), "total_bytes": sum(f["bytes"] for m in selected for f in m["files"]),
                          "models": selected, "next": "Add --download to fetch; --verify checks existing assets only."}, indent=2, ensure_ascii=False))
        return 0
    failed = False
    for model in selected:
        for entry in model["files"]:
            try:
                target = target_path(root, model["id"], entry["name"])
                result = (download_one(entry, target, replace_invalid=args.replace_invalid, restart_partial=args.restart_partial)
                          if args.download else file_status(target, entry))
                print(f"{model['id']}/{entry['name']}: {result}")
                failed |= result in {"missing", "invalid"}
            except (OSError, ValueError, HTTPError, URLError) as exc:
                # No API token is accepted or printed. Existing verified files
                # remain untouched when a remote transfer fails.
                print(f"{model['id']}/{entry['name']}: failed ({type(exc).__name__}: {exc})")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
