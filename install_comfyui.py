"""Install the small ComfyUI bridge; model weights stay in Vision Workbench."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil


def install(root):
    root = Path(root).expanduser().resolve()
    if not (root / "main.py").is_file() or not (root / "folder_paths.py").is_file():
        raise ValueError("請指定含 main.py 與 folder_paths.py 的 ComfyUI 目錄。")
    source = Path(__file__).resolve().parent / "comfyui/vision_workbench"
    destination = root / "custom_nodes/vision_workbench"
    if destination.is_symlink() or destination.resolve().parent != (root / "custom_nodes").resolve():
        raise ValueError("自訂節點目錄不能重新導向其他位置。")
    backup = root / "user/vision_workbench_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists() and target.read_bytes() != path.read_bytes():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    print(f"Installed {copied} bridge files: {destination}")
    print("模型、微調權重、素材沿用 Vision Workbench；無須複製或重新安裝模型環境。")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI_windows_portable/ComfyUI"))
    install(parser.parse_args().comfy_root)
