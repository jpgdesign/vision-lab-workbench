"""Export the running workbench's templates into a ComfyUI workflow folder."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

from backend.comfy_workflows import convert_workflow


# Avoid .json so ComfyUI does not offer bookkeeping as a loadable workflow.
MANIFEST = ".vision-workbench-exports.manifest"
README_PREAMBLE = (
    "視覺應用工作檯 / ComfyUI 中文工作流\n\n"
    "從 ComfyUI 工作流清單依中文名稱開啟 JSON；畫布上方同時顯示專案名稱。載入不會自動執行模型或開攝影機。\n"
    "未修改的舊預設英文檔名會移至 .backups；自行修改或無法確認來源的檔案會保留，不覆寫。\n\n"
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _title_filename(name):
    """Keep readable Chinese, with a bounded Windows-safe filename component."""
    if not isinstance(name, str):
        raise ValueError("工作流名稱必須是文字。")
    value = unicodedata.normalize("NFC", name)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "－", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    result, units = [], 0
    for char in value:
        width = len(char.encode("utf-16-le")) // 2
        if units + width > 48:
            break
        result.append(char)
        units += width
    return "".join(result).strip(" .") or "未命名工作流"


def _file(directory, name):
    if (not isinstance(name, str) or not name or Path(name).name != name
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith((" ", "."))):
        raise ValueError("匯出紀錄含有無效檔名。")
    path = directory / name
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise ValueError("匯出路徑不可連結到工作流資料夾外。")
    return path


def _write(path, content):
    # Unique temporary names avoid collisions between launcher/manual export.
    handle, temporary = tempfile.mkstemp(prefix=".vwb-export-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _archive(path, backup, expected_hash):
    """Move only byte-verified generated files to an in-folder backup."""
    if path.is_symlink() or not path.is_file() or _digest(path) != expected_hash:
        return False
    backup.mkdir(parents=True, exist_ok=True)
    destination = backup / (path.name + ".bak")
    if not destination.resolve().is_relative_to(path.parent.resolve()) or destination.exists():
        raise ValueError("備份位置不安全或已存在。")
    path.replace(destination)
    return True


def _legacy_document(document):
    # Previous exporter differed only in the newly visible canvas name group.
    legacy = dict(document)
    legacy["groups"] = []
    return legacy


def write_templates(templates, resources, destination, source_mode="auto"):
    """Write managed Chinese exports; never replace an edited/unmanaged file."""
    directory = Path(destination).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = _file(directory, MANIFEST)
    previous = {"version": 1, "entries": {}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(previous, dict) or previous.get("version") != 1
                or not isinstance(previous.get("entries"), dict)
                or any(not isinstance(item, dict) for item in previous["entries"].values())):
            raise ValueError("工作流匯出清單格式不正確，未覆寫既有檔案。")
        for item in previous["entries"].values():
            _file(directory, item.get("filename"))
    backup = directory / ".backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not backup.resolve().is_relative_to(directory):
        raise ValueError("備份資料夾不可連結到匯出路徑外。")
    exported, descriptions, preserved, migrated, entries = [], [], [], [], {}
    identifiers = set()
    for index, template in enumerate(templates, 1):
        ident = template["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident) or ident in identifiers:
            raise ValueError("範例 ID 無效或重複，無法作為安全的檔名。")
        identifiers.add(ident)
        sample = template.get("sample", {})
        mode = source_mode
        if mode == "auto":
            mode = "camera" if ident.startswith("live-") or sample.get("origin") == "user_required" else "media"
        document = convert_workflow(template["graph"], name=template["name"], source_mode=mode,
                                    media_id=sample.get("media_id") or "",
                                    options=template.get("recommended_options", {}), resources=resources)
        content = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        label = _title_filename(template["name"])
        preferred = f"{index:02d}-{label}--{ident}.json"
        record = previous["entries"].get(ident, {})
        # Reuse a collision-safe managed name until the template label changes.
        if record.get("title") == template["name"] and record.get("filename"):
            candidate = _file(directory, record["filename"])
            target = candidate if candidate.is_file() and _digest(candidate) == record.get("sha256") else _file(directory, preferred)
        else:
            target = _file(directory, preferred)
        if target.exists() and _digest(target) != content_hash:
            managed = record.get("filename") == target.name and record.get("sha256") == _digest(target)
            if managed:
                _archive(target, backup, record["sha256"])
            else:
                preserved.append(target.name)
                # Keep the edited file and make a distinct generated default.
                target = _file(directory, f"{index:02d}-{label}（預設-{content_hash[:8]}）--{ident}.json")
                if target.exists() and _digest(target) != content_hash:
                    # This content-derived default was also edited: never overwrite it.
                    preserved.append(target.name)
                    number = 2
                    while target.exists() and _digest(target) != content_hash:
                        target = _file(directory, f"{index:02d}-{label}（預設-{content_hash[:8]}-{number}）--{ident}.json")
                        number += 1
        if not target.exists():
            _write(target, content)
        # Retire a prior generated name only when its exact saved hash matches.
        if record.get("filename") and record["filename"] != target.name:
            old = _file(directory, record["filename"])
            if old.exists():
                if _archive(old, backup, record.get("sha256")):
                    migrated.append(old.name)
                else:
                    preserved.append(old.name)
        # First migration: only exact documents produced by the old exporter
        # are recognized. A user-edited English workflow is left untouched.
        for old in directory.glob(f"[0-9]*-{ident}.json"):
            if not re.fullmatch(r"[0-9]+-" + re.escape(ident) + r"\.json", old.name) or old.is_symlink():
                continue
            try:
                known = json.loads(old.read_text(encoding="utf-8")) == _legacy_document(document)
            except (ValueError, OSError):
                known = False
            if known and _archive(old, backup, _digest(old)):
                migrated.append(old.name)
            else:
                preserved.append(old.name)
        entries[ident] = {"filename": target.name, "sha256": content_hash, "title": template["name"]}
        exported.append(str(target))
        descriptions.append(f"{target.name}\n{template['name']}\n範例 ID：{ident}\n來源模式：{mode}\n範例：{sample.get('title', '')}\n{sample.get('expected', '')}\n{sample.get('notes', '')}\n{sample.get('source_url', '')}\n")
    # ComfyUI lists non-hidden user files as workflows, including README.txt.
    # Retire only a README carrying our complete generated preamble. Preserve
    # unrelated/user-written README files in place.
    legacy_readme = _file(directory, "README.txt")
    if legacy_readme.is_file():
        payload = legacy_readme.read_bytes()
        try:
            known_readme = payload.decode("utf-8-sig").replace("\r\n", "\n").startswith(README_PREAMBLE)
        except UnicodeDecodeError:
            known_readme = False
        if known_readme and _archive(legacy_readme, backup, hashlib.sha256(payload).hexdigest()):
            migrated.append(legacy_readme.name)
        else:
            preserved.append(legacy_readme.name)
    readme = _file(directory, ".README.txt")
    if readme.exists():
        try:
            known_readme = readme.read_text(encoding="utf-8-sig").startswith(README_PREAMBLE)
        except (UnicodeDecodeError, OSError):
            known_readme = False
        if not known_readme:
            preserved.append(readme.name)
            readme = None
    if readme is not None:
        _write(readme, README_PREAMBLE + "\n".join(descriptions)
               + ("\n保留的自訂／未確認檔案：\n" + "\n".join(sorted(set(preserved))) if preserved else ""))
    _write(manifest_path, json.dumps({"version": 1, "exporter": "vision-workbench", "entries": entries,
        "preserved_edited_or_unmanaged_files": sorted(set(preserved)), "archived_generated_files": migrated}, ensure_ascii=False, indent=2) + "\n")
    return exported


def export_templates(base_url, destination, source_mode="auto"):
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
        raise ValueError("請使用本機工作檯的 http://127.0.0.1:8765 網址。")
    base_url = base_url.rstrip("/")
    opener = build_opener(ProxyHandler({}))

    def get(path):
        with opener.open(base_url + "/api" + path, timeout=30) as response:
            return json.load(response)

    templates = get("/workflows/templates")["templates"]
    resources = {key: get("/" + key)[key] for key in ("models", "checkpoints", "media")}
    return write_templates(templates, resources, destination, source_mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--destination", required=True, help="ComfyUI/user/default/workflows/VisionWorkbench")
    parser.add_argument("--source-mode", choices=["auto", "media", "camera", "vdo"], default="auto")
    args = parser.parse_args()
    files = export_templates(args.base_url, args.destination, args.source_mode)
    print(f"Exported {len(files)} ComfyUI workflows to {Path(args.destination).resolve()}")


if __name__ == "__main__":
    main()
