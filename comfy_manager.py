"""Provision a pinned, isolated ComfyUI bridge host or reuse an existing install.

Planning never downloads or installs. Existing environments are only inspected;
only an explicitly owned managed environment may receive Python packages.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections import deque
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlparse
from urllib.request import build_opener, ProxyHandler, Request
import zipfile

COMFY_VERSION = "0.28.0"
COMFY_REVISION = "700821e1364eaab0e8f21c538a2131719fec57bf"
COMFY_URL = "https://codeload.github.com/Comfy-Org/ComfyUI/zip/" + COMFY_REVISION
COMFY_SHA256 = "493732242c4564b8a119cd29f3381d3347983a7d03933ce607606b63e9f9cd38"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
SOURCE_MARKER = ".vision-workbench-managed.json"
ENV_MARKER = ".vision-workbench-comfy-env.json"
TORCH_PINS = ("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0")


def absolute(path):
    # Resolving an executable symlink would discard Linux venv semantics.
    return Path(os.path.abspath(Path(path).expanduser()))


def _linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _guard(path, root):
    path, root = absolute(path), absolute(root)
    if not path.is_relative_to(root):
        raise ValueError("ComfyUI managed path escapes its project directory.")
    current = path
    while current != root:
        if _linked(current):
            raise ValueError("ComfyUI managed paths must not contain symbolic links or junctions.")
        current = current.parent
    return path


def _json(path):
    if _linked(path) or path.stat().st_size > 64_000:
        raise ValueError("Invalid ComfyUI configuration file.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid ComfyUI configuration object.")
    return value


def _write_json(path, value):
    if _linked(path) or _linked(path.parent):
        raise ValueError("ComfyUI configuration cannot be written through a link.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _core_exists(root):
    return (root / "main.py").is_file() and (root / "folder_paths.py").is_file()


def _env_python(root):
    return absolute(root / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def managed_environment(project_root):
    """Keep Windows Torch header paths below MAX_PATH without OS changes."""
    project = Path(project_root).resolve()
    local = project / ".tools/comfyui-venv"
    if platform.system() == "Windows" and len(str(local)) > 80:
        identity = hashlib.sha256(os.path.normcase(str(project)).encode("utf-8")).hexdigest()[:12]
        return Path.home().resolve() / ".vwb/comfy" / identity / "venv"
    return local


def _guard_environment(path, project_root):
    project = Path(project_root).resolve()
    expected = managed_environment(project)
    if absolute(path) != expected:
        raise ValueError("ComfyUI environment path does not match this project's managed location.")
    boundary = project if expected.is_relative_to(project) else Path.home().resolve()
    return _guard(expected, boundary)


def _owner(project_root):
    return {"owner": str(Path(project_root).resolve()), "revision": COMFY_REVISION}


def _check_owned(path, marker, project_root):
    if marker == ENV_MARKER:
        _guard_environment(path, project_root)
    else:
        _guard(path, project_root)
    try:
        value = _json(path / marker)
    except (OSError, ValueError) as exc:
        raise ValueError("Refusing to modify an unowned ComfyUI directory: " + str(path)) from exc
    if any(value.get(key) != item for key, item in _owner(project_root).items()):
        raise ValueError("ComfyUI managed directory belongs to another project or source revision: " + str(path))
    return value


def load_comfy_config(project_root, *, allow_incomplete=False):
    """Return a verified local registration, None if absent; never run software."""
    project = Path(project_root).resolve()
    path = _guard(project / "runtime/comfyui.json", project)
    if not path.exists():
        return None
    try:
        value = _json(path)
        if set(value) - {"root", "python", "managed", "device", "standalone", "version", "revision", "registered_at"}:
            raise ValueError("Unknown ComfyUI configuration fields.")
        root, python = Path(value["root"]), Path(value["python"])
        if not root.is_absolute() or not python.is_absolute() or not _core_exists(root):
            raise ValueError("Registered ComfyUI is missing; rerun setup with --comfy-root/--comfy-python.")
        if type(value["managed"]) is not bool or type(value["standalone"]) is not bool or value["device"] not in {"auto", "cpu"}:
            raise ValueError("Invalid ComfyUI launch settings.")
        if value["managed"]:
            environment = managed_environment(project)
            if root != project / ".tools/ComfyUI" or python != _env_python(environment) or value["device"] != "cpu":
                raise ValueError("Managed ComfyUI paths do not match this project.")
            _check_owned(root, SOURCE_MARKER, project)
            _guard_environment(environment, project)
            if environment.exists():
                _check_owned(environment, ENV_MARKER, project)
            if not python.is_file() or not (environment / "pyvenv.cfg").is_file():
                if allow_incomplete:
                    return {**value, "_repair_required": True}
                raise ValueError("Managed ComfyUI Python is missing; rerun python setup.py --install to restore its owned environment.")
        elif not python.is_file():
            raise ValueError("Registered ComfyUI Python is missing; select --comfy-python for its existing environment.")
        return value
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid saved ComfyUI configuration: " + str(exc)) from exc


def _find_python(root):
    candidates = [root / ".venv/Scripts/python.exe", root / ".venv/bin/python",
                  root / "venv/Scripts/python.exe", root / "venv/bin/python",
                  root.parent / "python_embeded/python.exe", root.parent / "python_embedded/python.exe"]
    return next((absolute(path) for path in candidates if path.is_file()), None)


def build_comfy_plan(project_root, *, comfy_root=None, comfy_python=None, creator_python=None, device=None):
    """Read-only plan: explicit install, saved config, managed install, known homes."""
    project = Path(project_root).resolve()
    managed_root = project / ".tools/ComfyUI"
    managed_env = managed_environment(project)
    errors, notes, saved = [], [], None
    if device not in {None, "auto", "cpu"}:
        errors.append("ComfyUI device must be auto or cpu.")
    if comfy_root is None:
        try:
            saved = load_comfy_config(project, allow_incomplete=True)
        except ValueError as exc:
            errors.append(str(exc))
    root = Path(comfy_root).expanduser().resolve() if comfy_root is not None else None
    if root is None and saved:
        root = Path(saved["root"])
    if root is None and managed_root.exists():
        root = managed_root
    if root is None:
        root = next((path for path in [Path.home()/"ComfyUI_windows_portable/ComfyUI", Path.home()/"ComfyUI"] if _core_exists(path)), managed_root)
    managed = root == managed_root
    action = "reuse" if _core_exists(root) else "install"
    if comfy_root is not None and not _core_exists(root):
        errors.append("--comfy-root must name an existing ComfyUI directory containing main.py and folder_paths.py. Omit it for an automatic managed download.")
    if managed:
        try:
            _guard(managed_root, project)
            _guard_environment(managed_env, project)
            if managed_root.exists():
                _check_owned(managed_root, SOURCE_MARKER, project)
            if managed_env.exists():
                _check_owned(managed_env, ENV_MARKER, project)
        except ValueError as exc:
            errors.append(str(exc))
        python = _env_python(managed_env)
        if comfy_python and absolute(comfy_python) != python:
            errors.append("Managed ComfyUI uses its isolated managed environment; --comfy-python is for an existing installation.")
        action = "reuse" if saved and saved.get("managed") and not saved.get("_repair_required") else "install"
        chosen_device = "cpu"
        notes.append("Managed ComfyUI runs its bridge on CPU in a separate environment. Vision Workbench model inference keeps its own detected GPU/CUDA environment.")
        if not managed_env.is_relative_to(project):
            notes.append("Windows project path is long; use a short, separately owned per-project environment under ~/.vwb/comfy to avoid MAX_PATH. No Windows settings are changed.")
        if saved and saved.get("_repair_required"):
            notes.append("Restore the missing interpreter in this project's previously registered, owned ComfyUI environment.")
    else:
        python = absolute(comfy_python) if comfy_python else absolute(saved["python"]) if saved else _find_python(root)
        if python is None or not python.is_file():
            errors.append("Cannot identify this ComfyUI Python environment. Supply --comfy-python PATH to its working Python executable; its packages will not be changed.")
        chosen_device = device or (saved or {}).get("device", "auto")
        notes.append("Reuse existing ComfyUI and preserve its Python, PyTorch and custom nodes. Install only the Vision Workbench bridge with backups.")
    if managed and action == "install":
        machine, system = platform.machine().lower(), platform.system()
        if not ((system in {"Windows", "Linux"} and machine in {"amd64", "x86_64"}) or (system in {"Linux", "Darwin"} and machine in {"arm64", "aarch64"})):
            errors.append("No validated managed CPU PyTorch wheel path for this OS/architecture; provide an existing ComfyUI installation.")
    creator = absolute(creator_python or sys.executable)
    if managed and action == "install" and not creator.is_file():
        errors.append("The Python executable for creating ComfyUI's environment is missing.")
    return {"root": str(root), "python": str(python) if python else "", "managed": managed, "action": action,
            "device": chosen_device, "standalone": bool(python and python.parent.name in {"python_embeded", "python_embedded"}),
            "errors": errors, "notes": notes, "creator_python": str(creator), "venv": str(managed_env) if managed else None,
            "version": COMFY_VERSION if managed else None, "revision": COMFY_REVISION if managed else None,
            "archive_url": COMFY_URL if managed else None, "archive_sha256": COMFY_SHA256 if managed else None}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(target):
    """Cache only the exact pinned HTTPS artifact, verifying the completed bytes."""
    target = absolute(target)
    if _linked(target) or _linked(target.parent):
        raise ValueError("ComfyUI download cache cannot be a link.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size <= MAX_ARCHIVE_BYTES and _sha256(target) == COMFY_SHA256:
        return target
    partial = target.with_suffix(".zip.part")
    if _linked(partial):
        raise ValueError("ComfyUI partial download cannot be a link.")
    opener = build_opener(ProxyHandler({}))
    print("Downloading official ComfyUI " + COMFY_VERSION + " source (about 12 MB)…", flush=True)
    with opener.open(Request(COMFY_URL, headers={"User-Agent":"Vision-Workbench-Installer"}), timeout=90) as response:
        final_url = urlparse(response.geturl())
        if final_url.scheme != "https" or final_url.hostname != "codeload.github.com":
            raise ValueError("Unexpected ComfyUI archive download destination.")
        count = 0
        with partial.open("wb") as output:
            while chunk := response.read(1024*1024):
                count += len(chunk)
                if count > MAX_ARCHIVE_BYTES:
                    raise ValueError("ComfyUI archive exceeds the download limit.")
                output.write(chunk)
    if _sha256(partial) != COMFY_SHA256:
        raise ValueError("ComfyUI archive SHA256 does not match the pinned release. Existing installation was not modified.")
    partial.replace(target)
    return target


def extract_archive(archive, destination):
    """Validate all names/types/sizes before writing into a fresh directory."""
    destination = absolute(destination)
    if destination.exists():
        raise ValueError("ComfyUI extraction destination must be new.")
    prefix = "ComfyUI-" + COMFY_REVISION
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if len(entries) > 30_000 or sum(value.file_size for value in entries) > MAX_EXPANDED_BYTES:
            raise ValueError("ComfyUI archive exceeds extraction limits.")
        seen, checked = set(), []
        for info in entries:
            # ZipInfo normalizes backslashes on Windows; validate the raw name.
            name = info.orig_filename
            parts = PurePosixPath(name).parts
            mode = stat.S_IFMT(info.external_attr >> 16)
            if (not parts or parts[0] != prefix or name.startswith("/") or "\\" in name or ":" in name or "\x00" in name or
                any(part in {"..", "."} or part.rstrip(" .") != part or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part) for part in parts) or
                mode not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise ValueError("Unsafe path or link in ComfyUI archive.")
            relative = Path(*parts[1:])
            if not parts[1:]:
                if not info.is_dir():
                    raise ValueError("ComfyUI archive root must be a directory.")
                continue
            key = "/".join(parts[1:]).casefold()
            if key in seen:
                raise ValueError("Duplicate path in ComfyUI archive.")
            seen.add(key)
            checked.append((info, relative))
        destination.mkdir()
        for info, relative in checked:
            target = _guard(destination / relative, destination)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024*1024)
    if not _core_exists(destination) or not (destination / "requirements.txt").is_file():
        raise ValueError("ComfyUI archive lacks its core entry points or requirements.")
    return destination


def _run(command, *, cwd, timeout=None, capture=False):
    print("Running: " + subprocess.list2cmdline([str(value) for value in command]), flush=True)
    command = [str(value) for value in command]
    options = dict(cwd=cwd, text=True, encoding="utf-8", errors="replace",
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                   env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"))
    if capture:
        return subprocess.run(command, check=True, timeout=timeout, capture_output=True, **options)
    # Hidden Windows children do not reliably inherit redirected console handles.
    # Explicitly relay a pipe so pip's progress and failures reach the caller.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options)
    tail = deque(maxlen=100)
    def relay():
        for line in process.stdout:
            tail.append(line)
            print(line, end="", flush=True)
    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    try:
        result = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join(timeout=5)
        raise
    reader.join(timeout=5)
    process.stdout.close()
    if result:
        raise subprocess.CalledProcessError(result, command, output="".join(tail)[-5000:])
    return subprocess.CompletedProcess(command, result)


def verify_environment(root, python, *, managed=False):
    """Check this interpreter without opening a port, camera or model checkpoint."""
    _run([python, "-s", "-c", "import torch, torchvision, torchaudio, requests, numpy, PIL, aiohttp, av, comfyui_frontend_package; print('ComfyUI runtime imports OK')"], cwd=root, timeout=90, capture=True)
    _run([python, "-s", "main.py", "--help"], cwd=root, timeout=60, capture=True)
    if managed:
        _run([python, "-s", "-c", "import torch; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda is None, 'Managed bridge must keep CPU torch'"], cwd=root, timeout=30, capture=True)


def install_comfy_plan(plan, *, project_root):
    """Execute a reviewed plan, registering success only after verification."""
    project = Path(project_root).resolve()
    if plan.get("errors"):
        raise ValueError("\n".join(plan["errors"]))
    root, python = absolute(plan["root"]), absolute(plan["python"])
    managed = plan["managed"]
    if managed:
        tools = _guard(project / ".tools", project)
        expected_root, environment = tools / "ComfyUI", _guard_environment(managed_environment(project), project)
        if root != expected_root or python != _env_python(environment):
            raise ValueError("Managed ComfyUI plan paths do not belong to this project.")
        if plan["action"] != "reuse":
            creator = absolute(plan["creator_python"])
            try:
                _run([creator, "-c", "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), 'Python 3.10-3.12 is required to create managed ComfyUI'"], cwd=project, timeout=15, capture=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError("Managed ComfyUI requires a working Python 3.10-3.12 creator; no ComfyUI download or environment changes were made.") from exc
        tools.mkdir(exist_ok=True)
        if root.exists():
            _check_owned(root, SOURCE_MARKER, project)
        else:
            cache = _guard(tools / "downloads", project)
            cache.mkdir(exist_ok=True)
            archive = download_archive(cache / ("comfyui-" + COMFY_REVISION + ".zip"))
            # TemporaryDirectory removes only its own freshly created folder.
            with tempfile.TemporaryDirectory(prefix="comfy-extract-", dir=tools) as temp:
                staged = extract_archive(archive, Path(temp) / "ComfyUI")
                _write_json(staged / SOURCE_MARKER, _owner(project))
                if root.exists():
                    raise ValueError("ComfyUI destination appeared during extraction; refusing to replace it.")
                staged.rename(root)
        if plan["action"] != "reuse":
            if environment.exists():
                _check_owned(environment, ENV_MARKER, project)
            else:
                environment.mkdir(parents=True)
                _write_json(environment / ENV_MARKER, _owner(project))
            if not python.is_file() or not (environment / "pyvenv.cfg").is_file():
                _run([creator, "-m", "venv", environment], cwd=project)
            pip = [python, "-m", "pip", "--isolated"]
            _run(pip + ["install", "--upgrade", "pip", "--index-url", "https://pypi.org/simple"], cwd=root)
            index = "https://pypi.org/simple" if platform.system() == "Darwin" else "https://download.pytorch.org/whl/cpu"
            _run(pip + ["install", *TORCH_PINS, "--index-url", index], cwd=root)
            with tempfile.TemporaryDirectory(prefix="comfy-constraints-", dir=tools) as temp:
                constraints = Path(temp) / "constraints.txt"
                constraints.write_text("\n".join(TORCH_PINS) + "\n", encoding="utf-8")
                _run(pip + ["install", "--index-url", "https://pypi.org/simple", "--constraint", constraints, "--requirement", root / "requirements.txt"], cwd=root)
            _run(pip + ["check"], cwd=root)
    if not _core_exists(root) or not python.is_file():
        raise ValueError("ComfyUI installation or selected Python is missing.")
    try:
        verify_environment(root, python, managed=managed)
    except (OSError, subprocess.SubprocessError) as exc:
        details = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc))
        if isinstance(details, bytes):
            details = details.decode("utf-8", errors="replace")
        raise ValueError("ComfyUI environment validation failed. Existing environments are never repaired automatically. Select its working --comfy-python or install its dependencies separately.\n" + str(details)[-2500:]) from exc
    from install_comfyui import install
    install(root)
    config = {key: plan[key] for key in ("root", "python", "managed", "device", "standalone", "version", "revision")}
    config["registered_at"] = datetime.now(timezone.utc).isoformat()
    path = _guard(project / "runtime/comfyui.json", project)
    _write_json(path, config)
    return config
