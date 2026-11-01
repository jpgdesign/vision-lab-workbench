from pathlib import Path
import pytest
from install_comfyui import install


def core(tmp_path):
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "main.py").touch()
    (root / "folder_paths.py").touch()
    return root


def test_bridge_install_preserves_other_extensions_and_backs_up_replaced_files(tmp_path):
    root = core(tmp_path)
    other = root / "custom_nodes/other/__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("unchanged")
    target = root / "custom_nodes/vision_workbench/__init__.py"
    target.parent.mkdir()
    target.write_text("older bridge")
    install(root)
    assert other.read_text() == "unchanged"
    backups = list((root / "user/vision_workbench_backups").glob("*/__init__.py"))
    assert len(backups) == 1 and backups[0].read_text() == "older bridge"
    before = sorted((path.relative_to(root), path.read_bytes()) for path in root.rglob("*") if path.is_file())
    install(root)
    after = sorted((path.relative_to(root), path.read_bytes()) for path in root.rglob("*") if path.is_file())
    assert before == after


def test_nested_plugin_link_is_rejected_before_external_file_overwrite(tmp_path):
    root = core(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = root / "custom_nodes/vision_workbench/web"
    target.parent.mkdir(parents=True)
    sentinel = outside / "vision_workbench.js"
    sentinel.write_text("keep this file")
    try:
        target.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="連結"):
        install(root)
    assert sentinel.read_text() == "keep this file"
