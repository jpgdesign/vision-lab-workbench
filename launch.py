"""Start the local workbench and its configured ComfyUI, reusing running services."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("VISION_WORKBENCH_RUNTIME", str(ROOT / "runtime"))).expanduser().resolve()
URL = "http://127.0.0.1:8765"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_START_URL = COMFY_URL + "/?vision_workflow=live-multi-model&source=camera"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, "Local service redirects are not accepted.", headers, fp)


def local_json(url, timeout=2, maximum=32 * 1024 * 1024):
    """Local service discovery must never follow configured HTTP proxies."""
    with build_opener(ProxyHandler({}), _NoRedirect()).open(url, timeout=timeout) as response:
        content = response.read(maximum + 1)
        if len(content) > maximum:
            raise ValueError("Local service response is too large.")
        return json.loads(content)


def environment_python(root=ROOT, environ=None, current=None):
    """Explicit override, registered venv, local venv, active venv, legacy fallback."""
    # POSIX venv executables commonly link to the system Python. Dereferencing
    # that link discards the venv directory Python uses to find pyvenv.cfg.
    def executable_path(value):
        return Path(os.path.abspath(value))

    environ = os.environ if environ is None else environ
    current = Path(current or sys.executable)
    suffix = Path("Scripts/python.exe" if os.name == "nt" else "bin/python")
    override = environ.get("VISION_WORKBENCH_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise ValueError("VISION_WORKBENCH_PYTHON does not point to a Python executable.")
        return executable_path(candidate)
    override = environ.get("VISION_WORKBENCH_VENV")
    if override:
        candidate = Path(override).expanduser() / suffix
        if not candidate.is_file():
            raise ValueError("VISION_WORKBENCH_VENV does not contain a Python executable.")
        return executable_path(candidate)
    registration = Path(root) / "runtime/workbench_environment.json"
    if registration.exists() or registration.is_symlink():
        try:
            if not registration.is_file() or registration.stat().st_size > 16_384:
                raise ValueError("registration is not a small JSON file")
            saved = json.loads(registration.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or any(not isinstance(saved.get(key), str) or not saved[key] for key in ("project", "venv", "python")):
                raise ValueError("registration fields are missing")
            project, venv, candidate = (Path(saved[key]).expanduser() for key in ("project", "venv", "python"))
            if not all(path.is_absolute() for path in (project, venv, candidate)):
                raise ValueError("registered paths must be absolute")
            if project.resolve() != Path(root).resolve():
                raise ValueError("registration belongs to a different project")
            if executable_path(candidate) != executable_path(venv / suffix):
                raise ValueError("Python does not belong to the registered virtual environment")
            if not (venv / "pyvenv.cfg").is_file() or not candidate.is_file():
                raise ValueError("registered virtual environment or Python is missing")
            return executable_path(candidate)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError("Registered Workbench environment is invalid: %s. Run python setup.py --install again to repair the registration, or set VISION_WORKBENCH_PYTHON explicitly." % exc) from None
    local = root / ".venv" / suffix
    if local.is_file():
        return executable_path(local)
    if (current.parent.parent / "pyvenv.cfg").is_file():
        return executable_path(current)
    legacy = root.parent.parent / "work/vision_workbench/.venv" / suffix
    if legacy.is_file():
        return executable_path(legacy)
    return executable_path(current)


def alive():
    try:
        return local_json(URL + "/api/health").get("status") == "ok"
    except Exception:
        return False


def start_workbench():
    if not alive():
        if port_in_use(8765):
            raise ValueError("Port 8765 is occupied, but it does not identify itself as a ready Vision Workbench. The existing service was left running.")
        try:
            python = environment_python(ROOT)
        except ValueError as exc:
            raise SystemExit(str(exc))
        check = subprocess.run([str(python), "-c", "import fastapi, uvicorn"], capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if check.returncode:
            raise SystemExit("API dependencies are missing in %s. Run python setup.py --install, or select an existing venv with VISION_WORKBENCH_VENV." % python)
        RUNTIME.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        with (RUNTIME / "server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([str(python), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8765"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (RUNTIME / "server.pid").write_text(str(process.pid), encoding="ascii")
        for _ in range(90):
            if alive():
                break
            if process.poll() is not None:
                raise SystemExit("Server did not start. See runtime/server.log.")
            time.sleep(1)
        else:
            raise SystemExit("Startup timed out. See runtime/server.log.")
    print("Vision Workbench: " + URL)


def comfy_selection(args):
    """Resolve an installed environment; starting never downloads or installs."""
    if args.without_comfyui:
        return None
    from comfy_manager import build_comfy_plan, load_comfy_config
    explicit = bool(args.comfy_root or args.comfy_python)
    config = None if explicit else load_comfy_config(ROOT)
    if config is not None:
        return config
    if not (args.with_comfyui or explicit):
        return None
    plan = build_comfy_plan(ROOT, comfy_root=args.comfy_root, comfy_python=args.comfy_python)
    if plan.get("errors"):
        raise ValueError("\n".join(plan["errors"]))
    if plan.get("action") != "reuse":
        raise ValueError("ComfyUI is not installed. Run python setup.py --install --with-comfyui first.")
    return plan


def comfy_probe(project_root=ROOT):
    """Identify ComfyUI plus every shipped bridge node, rather than a live port."""
    try:
        stats = local_json(COMFY_URL + "/system_stats")
        if not isinstance(stats, dict) or not isinstance(stats.get("system"), dict) or not stats["system"].get("comfyui_version"):
            return False, "Port 8188 does not identify itself as ComfyUI."
        definitions = local_json(COMFY_URL + "/object_info", timeout=5)
        catalogue = json.loads((Path(project_root) / "comfyui/vision_workbench/catalog.json").read_text(encoding="utf-8"))
        expected = {"VisionWB_" + node["type"] for node in catalogue}
        if not isinstance(definitions, dict):
            return False, "ComfyUI returned an invalid node catalogue."
        missing = expected - definitions.keys()
        if missing:
            return False, ("ComfyUI is running, but %s Vision Workbench nodes are missing. "
                           "Run python setup.py --install --with-comfyui, then restart ComfyUI when its jobs finish." % len(missing))
        return True, "ComfyUI and %s Vision Workbench nodes are ready." % len(expected)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, "ComfyUI is not ready: " + str(exc)


def port_in_use(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.5):
            return True
    except OSError:
        return False


def comfy_command(config):
    command = [str(config["python"]), "-s", str(Path(config["root"]) / "main.py"),
               "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"]
    if config.get("standalone"):
        command.append("--windows-standalone-build")
    if config.get("device") == "cpu":
        command.append("--cpu")
    return command


def start_comfyui(config, *, timeout=120):
    core = Path(config["root"])
    if not (core / "custom_nodes/vision_workbench/__init__.py").is_file():
        raise ValueError("The Vision Workbench bridge is not installed in the selected ComfyUI. Run python setup.py --install --with-comfyui first.")
    ready, reason = comfy_probe(ROOT)
    if not ready and port_in_use(8188):
        raise ValueError(reason + " The existing service was left running.")
    from export_workflow_templates import export_templates
    files = export_templates(URL, core / "user/default/workflows/VisionWorkbench")
    print("Prepared %s ComfyUI workflows with this computer's registered resources." % len(files), flush=True)
    if ready:
        print("Reusing ComfyUI: " + COMFY_URL, flush=True)
        return {"reused": True, "url": COMFY_START_URL, "workflows": len(files)}
    RUNTIME.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with (RUNTIME / "comfyui.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(comfy_command(config), cwd=core, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (RUNTIME / "comfyui.pid").write_text(str(process.pid), encoding="ascii")
    deadline = time.monotonic() + timeout
    next_progress = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError("ComfyUI exited during startup. See %s; no other service was stopped." % (RUNTIME / "comfyui.log"))
        ready, reason = comfy_probe(ROOT)
        if ready:
            print("ComfyUI: " + COMFY_URL, flush=True)
            return {"reused": False, "pid": process.pid, "url": COMFY_START_URL, "workflows": len(files)}
        now = time.monotonic()
        if now >= next_progress:
            print("Waiting for ComfyUI and its Vision Workbench nodes…", flush=True)
            next_progress = now + 15
        time.sleep(.5)
    raise ValueError("ComfyUI startup timed out. %s See %s. The launched process was left running; rerun after checking the log." % (reason, RUNTIME / "comfyui.log"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="Start/reuse services without opening another browser tab")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--with-comfyui", action="store_true", help="Start installed ComfyUI with the workbench (automatic when configured)")
    mode.add_argument("--without-comfyui", action="store_true", help="Start only the workbench, even when ComfyUI is configured")
    parser.add_argument("--comfy-root", type=Path, help="Use this existing ComfyUI directory; does not install or change its dependencies")
    parser.add_argument("--comfy-python", type=Path, help="Python executable belonging to the selected existing ComfyUI")
    args = parser.parse_args(argv)
    if args.without_comfyui and (args.comfy_root or args.comfy_python):
        parser.error("--without-comfyui cannot be combined with --comfy-root or --comfy-python")
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise SystemExit("Frontend is not built. Run python setup.py --install (requires Node.js/npm).")
    try:
        config = comfy_selection(args)
        start_workbench()
        destination = start_comfyui(config)["url"] if config is not None else URL
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if not args.no_browser:
        webbrowser.open(destination)
    return 0


if __name__ == "__main__":
    main()
