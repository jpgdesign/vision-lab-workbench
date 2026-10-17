"""Double-click START.cmd to open the local workbench, reusing a running server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("VISION_WORKBENCH_RUNTIME", str(ROOT / "runtime"))).expanduser().resolve()
URL = "http://127.0.0.1:8765"


def environment_python(root=ROOT, environ=None, current=None):
    """Explicit override, clone-local venv, active venv, original-install fallback."""
    environ = os.environ if environ is None else environ
    current = Path(current or sys.executable)
    suffix = Path("Scripts/python.exe" if os.name == "nt" else "bin/python")
    override = environ.get("VISION_WORKBENCH_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise ValueError("VISION_WORKBENCH_PYTHON does not point to a Python executable.")
        return candidate.resolve()
    override = environ.get("VISION_WORKBENCH_VENV")
    if override:
        candidate = Path(override).expanduser() / suffix
        if not candidate.is_file():
            raise ValueError("VISION_WORKBENCH_VENV does not contain a Python executable.")
        return candidate.resolve()
    local = root / ".venv" / suffix
    if local.is_file():
        return local.resolve()
    if (current.parent.parent / "pyvenv.cfg").is_file():
        return current.resolve()
    legacy = root.parent.parent / "work/vision_workbench/.venv" / suffix
    if legacy.is_file():
        return legacy.resolve()
    return current.resolve()


def alive():
    try:
        with urlopen(URL + "/api/health", timeout=2) as response:
            data = json.load(response)
            return data.get("status") == "ok"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true", help="Start/reuse the server without opening another browser tab")
    args = parser.parse_args()
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise SystemExit("Frontend is not built. Run python setup.py --install (requires Node.js/npm).")
    if not alive():
        try:
            python = environment_python()
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
    if not args.no_browser:
        webbrowser.open(URL)
    print("Vision Workbench: " + URL)


if __name__ == "__main__":
    main()
