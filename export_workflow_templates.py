"""Export the running workbench's templates into a ComfyUI workflow folder."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

from backend.comfy_workflows import convert_workflow


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
    directory = Path(destination).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    backup = directory / ".backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    exported, descriptions = [], []
    for index, template in enumerate(templates, 1):
        ident = template["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident):
            raise ValueError("範例 ID 無法作為安全的檔名。")
        sample = template.get("sample", {})
        mode = source_mode
        if mode == "auto":
            mode = "camera" if ident.startswith("live-") or sample.get("origin") == "user_required" else "media"
        options = template.get("recommended_options", {})
        document = convert_workflow(template["graph"], name=template["name"], source_mode=mode,
                                    media_id=sample.get("media_id") or "", options=options, resources=resources)
        target = directory / f"{index:02d}-{ident}.json"
        content = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            if target.is_file():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup / target.name)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(target)
        exported.append(str(target))
        descriptions.append(f"{target.name}\n{template['name']}\n來源模式：{mode}\n範例：{sample.get('title', '')}\n{sample.get('expected', '')}\n{sample.get('notes', '')}\n{sample.get('source_url', '')}\n")
    (directory / "README.txt").write_text("Vision Workbench / ComfyUI 工作流\n\n先啟動 Vision Workbench 與已安裝外掛的 ComfyUI，再從 ComfyUI 工作流清單開啟 JSON。載入不會自動執行模型或開攝影機。\n已存在且內容不同的檔案先備份至 .backups。\n\n" + "\n".join(descriptions), encoding="utf-8")
    return exported


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
