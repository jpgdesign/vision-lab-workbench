"""Chinese workflow naming must not change execution or overwrite user work."""
import copy
import json
from pathlib import Path
import re
import uuid

import pytest

from backend.comfy_workflows import convert_workflow, _prepare
from export_workflow_templates import MANIFEST, README_PREAMBLE, _title_filename, write_templates


@pytest.fixture
def template():
    return {"id": "sample-detection", "name": "物件偵測／分割：範例", "sample": {"media_id": "image-1"},
            "graph": {"version": 1, "nodes": [
                {"id": "src", "type": "source", "position": {"x": 0, "y": 20}},
                {"id": "det", "type": "detector", "position": {"x": 400, "y": 20}},
                {"id": "out", "type": "output", "position": {"x": 800, "y": 20}}], "edges": [
                {"source": "src", "sourceHandle": "frame", "target": "det", "targetHandle": "frame"},
                {"source": "src", "sourceHandle": "frame", "target": "out", "targetHandle": "frame"},
                {"source": "det", "sourceHandle": "detections", "target": "out", "targetHandle": "observations"}]}}


def document(template):
    return convert_workflow(template["graph"], name=template["name"], media_id="image-1", resources={})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_chinese_group_preserves_uuid_node_and_resource_ids(template):
    resources = {"media": [{"id": "image-1", "name": "便當圖"}],
                 "models": [{"id": "yolo26x", "name": "YOLO26x"}]}
    result = convert_workflow(template["graph"], name=template["name"], media_id="image-1", resources=resources)
    graph, ids, _, _, _, source, name = _prepare(template["graph"], template["name"], "media", "image-1", None, resources)
    stable = json.dumps({"name": name, "graph": graph, "source": source}, ensure_ascii=False, sort_keys=True)
    assert result["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "vision-workbench:" + stable))
    assert result["extra"]["vision_workbench"]["node_ids"] == {str(value): key for key, value in ids.items()}
    assert result["nodes"][0]["widgets_values"][1] == "image-1 | 便當圖"
    assert result["nodes"][1]["widgets_values"][0] == "yolo26x | YOLO26x"
    group = result["groups"][0]
    assert group["title"] == template["name"]
    x, y, width, height = group["bounding"]
    assert all(x <= node["pos"][0] and y <= node["pos"][1]
               and node["pos"][0] + node["size"][0] <= x + width
               and node["pos"][1] + node["size"][1] <= y + height for node in result["nodes"])


def test_windows_safe_chinese_filename():
    assert _title_filename("中文骨架 → 動作") == "中文骨架 → 動作"
    sanitized = _title_filename(' /中文:<>"\\|?*\x00 尾端. ')
    assert not re.search(r'[<>:"/\\|?*\x00-\x1f]', sanitized)
    assert sanitized[-1] not in " ."
    assert len(_title_filename("😀" * 50).encode("utf-16-le")) <= 96
    assert _title_filename("  ... ") == "未命名工作流"


def test_exports_are_chinese_and_idempotent(tmp_path, template):
    first = write_templates([template], {}, tmp_path)
    assert "物件偵測／分割：範例--sample-detection.json" in Path(first[0]).name
    initial = Path(first[0]).read_bytes()
    assert write_templates([template], {}, tmp_path) == first
    assert Path(first[0]).read_bytes() == initial
    assert not (tmp_path / ".backups").exists()
    assert read(tmp_path / MANIFEST)["entries"][template["id"]]["filename"] == Path(first[0]).name


def test_unmodified_legacy_migrates_but_custom_legacy_remains(tmp_path, template):
    old = document(template)
    old["groups"] = []
    legacy = tmp_path / "01-sample-detection.json"
    legacy.write_text(json.dumps(old), encoding="utf-8")
    custom = tmp_path / "02-sample-detection.json"
    old["nodes"][0]["pos"][0] = 900
    custom.write_text(json.dumps(old), encoding="utf-8")
    custom_bytes = custom.read_bytes()
    write_templates([template], {}, tmp_path)
    assert not legacy.exists()
    assert len(list((tmp_path / ".backups").glob("*/01-sample-detection.json.bak"))) == 1
    assert custom.read_bytes() == custom_bytes
    assert custom.name in read(tmp_path / MANIFEST)["preserved_edited_or_unmanaged_files"]


def test_changed_managed_export_archived_before_update(tmp_path, template):
    original = Path(write_templates([template], {}, tmp_path)[0])
    original_bytes = original.read_bytes()
    changed = copy.deepcopy(template)
    changed["graph"]["nodes"][1]["params"] = {"conf": .6}
    updated = Path(write_templates([changed], {}, tmp_path)[0])
    assert updated == original
    assert updated.read_bytes() != original_bytes
    assert next((tmp_path / ".backups").glob("*/*.json.bak")).read_bytes() == original_bytes


def test_edited_managed_export_survives_and_uses_stable_alternative(tmp_path, template):
    original = Path(write_templates([template], {}, tmp_path)[0])
    edited = read(original)
    edited["nodes"][0]["pos"][0] = 12345
    original.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
    edited_bytes = original.read_bytes()
    replacement = Path(write_templates([template], {}, tmp_path)[0])
    assert replacement != original
    assert "預設-" in replacement.name
    assert original.read_bytes() == edited_bytes
    assert Path(write_templates([template], {}, tmp_path)[0]) == replacement
    assert original.read_bytes() == edited_bytes


def test_unmanaged_target_is_preserved(tmp_path, template):
    original = tmp_path / f"01-{_title_filename(template['name'])}--{template['id']}.json"
    original.write_text("unrelated user's project", encoding="utf-8")
    replacement = Path(write_templates([template], {}, tmp_path)[0])
    assert replacement != original
    assert original.read_text(encoding="utf-8") == "unrelated user's project"


def test_new_exports_only_use_hidden_readme(tmp_path, template):
    write_templates([template], {}, tmp_path)
    assert not (tmp_path / "README.txt").exists()
    assert (tmp_path / ".README.txt").read_text(encoding="utf-8").startswith(README_PREAMBLE)


def test_generated_visible_readme_migrates_to_backup(tmp_path, template):
    legacy = tmp_path / "README.txt"
    legacy.write_text(README_PREAMBLE + "old exported template list", encoding="utf-8")
    original_bytes = legacy.read_bytes()
    write_templates([template], {}, tmp_path)
    assert not legacy.exists()
    assert next((tmp_path / ".backups").glob("*/README.txt.bak")).read_bytes() == original_bytes
    assert "README.txt" in read(tmp_path / MANIFEST)["archived_generated_files"]
    assert (tmp_path / ".README.txt").is_file()


@pytest.mark.parametrize("name, content", [
    ("README.txt", "my own project documentation"),
    ("README.txt", "視覺應用工作檯 / ComfyUI 中文工作流\n\nmy own documentation"),
    (".README.txt", "my own hidden documentation")])
def test_user_readme_never_overwritten_or_archived(tmp_path, template, name, content):
    existing = tmp_path / name
    existing.write_text(content, encoding="utf-8")
    original_bytes = existing.read_bytes()
    write_templates([template], {}, tmp_path)
    assert existing.read_bytes() == original_bytes
    assert not (tmp_path / ".backups").exists()
    assert name in read(tmp_path / MANIFEST)["preserved_edited_or_unmanaged_files"]


@pytest.mark.parametrize("bad_id", ["../outside", "bad/id", "bad:stream", "", "a" * 81])
def test_unsafe_template_ids_rejected(tmp_path, template, bad_id):
    template["id"] = bad_id
    with pytest.raises(ValueError):
        write_templates([template], {}, tmp_path)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("record", [[], {"filename": "../outside.json"}, {"filename": "C:\\outside.json"}, {"filename": "bad:stream"}])
def test_bad_manifest_cannot_touch_outside_files(tmp_path, template, record):
    (tmp_path / MANIFEST).write_text(json.dumps({"version": 1, "entries": {template["id"]: record}}), encoding="utf-8")
    with pytest.raises(ValueError):
        write_templates([template], {}, tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / MANIFEST]
