"""Portable environment planner. Nothing is installed without --install.

Examples: python setup.py; python setup.py --install; python setup.py --install --profile base
This is an application bootstrap script, not a setuptools package entry point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

from backend.system_info import collect_system_info, probe_torch

ROOT = Path(__file__).resolve().parent
TORCH_VERSION = "2.8.0"
VISION_VERSION = "0.23.0"
SOURCES = [
    "https://pytorch.org/get-started/previous-versions/",
    "https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128",
    "https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html",
]
# Conservative corresponding-toolkit driver floors, not CUDA minor-compatibility minima.
DRIVERS = {"Windows": {"cu126": (561, 17), "cu128": (572, 61)},
           "Linux": {"cu126": (560, 35, 5), "cu128": (570, 124, 6)}}
# Known desktop/server capabilities supported by this pinned wheel family.
# Unknown newer architectures are not assumed to work merely by numeric range.
CAPABILITIES = {"cu126": {5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0},
                "cu128": {7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0}}


def venv_python(directory, system=None):
    return Path(directory) / ("Scripts/python.exe" if (system or platform.system()) == "Windows" else "bin/python")


def version_tuple(value):
    pieces = re.findall(r"\d+", str(value or ""))
    return tuple(int(piece) for piece in pieces[:3]) + (0,) * max(0, 3 - len(pieces))


def choose_torch(system_info, requested="auto"):
    """Pure conservative wheel selection; never recommends installing a driver."""
    system = system_info["platform"]["system"]
    machine = system_info["platform"]["machine"].lower()
    x86 = machine in {"amd64", "x86_64"}
    arm = machine in {"arm64", "aarch64"}
    reasons = []
    error = None
    supported = (system in {"Windows", "Linux"} and x86) or (system in {"Linux", "Darwin"} and arm)
    if not supported:
        error = "This platform has no validated PyTorch 2.8 wheel path in this installer: %s/%s." % (system, machine)
    nvidia = system_info.get("nvidia", {})
    gpu_rows = nvidia.get("gpus", [])
    capabilities = []
    for gpu in gpu_rows:
        try:
            capabilities.append(float(gpu.get("compute_capability")))
        except (TypeError, ValueError):
            capabilities.append(None)
    driver = version_tuple(nvidia.get("driver_version"))
    candidates = []
    # ARM CUDA environments (e.g. Jetson) require vendor-specific consideration.
    if system in DRIVERS and x86 and nvidia.get("query_ok") and capabilities and all(cap is not None for cap in capabilities):
        if all(cap in CAPABILITIES["cu128"] for cap in capabilities) and driver >= version_tuple(".".join(map(str, DRIVERS[system]["cu128"]))):
            candidates.append("cu128")
        if all(cap in CAPABILITIES["cu126"] for cap in capabilities) and driver >= version_tuple(".".join(map(str, DRIVERS[system]["cu126"]))):
            candidates.append("cu126")
    variant = candidates[0] if candidates else "cpu"
    if requested != "auto":
        variant = requested
        if variant != "cpu" and variant not in candidates:
            error = "Requested %s cannot be verified against the detected OS, GPU architecture and conservative driver floor. Use CPU or update the driver yourself and diagnose again." % variant
    if variant == "cpu":
        if system == "Darwin":
            reasons.append("macOS uses the official default PyPI wheel, with CPU selected by the workbench; MPS is not enabled.")
        elif nvidia.get("detected"):
            reasons.append("CUDA wheel compatibility was not fully verified for all detected GPUs; CPU is the conservative default.")
        else:
            reasons.append("No compatible NVIDIA driver/GPU combination was verified; use the CPU wheel.")
    else:
        reasons.append("Selected %s using GPU compute capability and a conservative driver floor; no system CUDA toolkit installation is required." % variant)
    index = "https://pypi.org/simple" if system == "Darwin" else "https://download.pytorch.org/whl/" + variant
    return {"variant": variant, "index_url": index, "torch_version": TORCH_VERSION,
            "torchvision_version": VISION_VERSION, "reasons": reasons, "error": error}


def build_plan(system_info, target_venv, *, profile="full", requested="auto", repair=False,
               with_rfdetr_plus=False, with_dev=False, skip_frontend=False, target_exists=False):
    target = Path(target_venv).resolve()
    wheel = choose_torch(system_info, requested)
    existing = system_info.get("torch", {}) if target_exists else {}
    preserve = bool(existing.get("working") and not repair)
    errors = []
    if not system_info["python"]["supported"]:
        errors.append("Python 3.10–3.12 is required; Python 3.11 is recommended.")
    if profile == "full" and not preserve and wheel["error"]:
        errors.append(wheel["error"])
    if profile == "full" and existing.get("installed") and not existing.get("working") and not repair:
        errors.append("Existing PyTorch/torchvision failed its check. Use --repair-torch to explicitly replace it.")
    if with_rfdetr_plus and profile != "full":
        errors.append("--with-rfdetr-plus requires --profile full.")
    requirements = ["requirements.txt" if profile == "full" else "requirements-base.txt"]
    if with_rfdetr_plus:
        requirements.append("requirements-rfdetr-plus.txt")
    if with_dev:
        requirements.append("requirements-dev.txt")
    notes = list(wheel["reasons"])
    if preserve:
        notes.append("Preserve the existing working torch/torchvision pair, including its CUDA build. --repair-torch is required to replace it.")
    notes.append("Model weights and datasets are separate downloads; this setup does not fetch them.")
    notes.append("NVIDIA driver detection does not prove torch.cuda availability; the installed interpreter is checked separately.")
    if with_rfdetr_plus:
        notes.append("Optional rfdetr-plus has separate PML terms. Installation does not grant rights or record license acceptance.")
    return {"install_requested": False, "profile": profile, "venv": str(target),
            "python": str(venv_python(target, system_info["platform"]["system"])),
            "create_venv": not target_exists, "torch_action": "skip" if profile == "base" else "preserve" if preserve else "install",
            "torch": wheel, "preserved_versions": {key: existing.get(key) for key in ("version", "torchvision_version")} if preserve else None,
            "requirements": requirements, "build_frontend": not skip_frontend,
            "notes": notes, "errors": errors, "sources": SOURCES}


def _run(command, *, cwd=ROOT):
    print("Running: " + subprocess.list2cmdline([str(value) for value in command]), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)


def execute_plan(plan, *, creator_python=None, repair=False):
    if plan["errors"]:
        raise ValueError("\n".join(plan["errors"]))
    target = Path(plan["venv"])
    python = Path(plan["python"])
    if target.exists() and not (target / "pyvenv.cfg").is_file():
        raise ValueError("Refusing to modify an existing directory without pyvenv.cfg. Choose a project virtual environment with --venv.")
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if plan["build_frontend"] and not npm:
        raise ValueError("Node.js/npm is required to build the frontend. Install Node.js LTS, or use --skip-frontend with an existing build.")
    if not target.exists():
        _run([creator_python or sys.executable, "-m", "venv", str(target)])
    if not python.is_file():
        raise ValueError("The selected virtual environment has no Python executable: " + str(python))
    pip = [str(python), "-m", "pip", "--isolated"]
    _run(pip + ["install", "--upgrade", "pip", "--index-url", "https://pypi.org/simple"])
    if plan["torch_action"] == "install":
        command = pip + ["install", "torch==" + TORCH_VERSION, "torchvision==" + VISION_VERSION,
                         "--index-url", plan["torch"]["index_url"]]
        if repair:
            command.append("--force-reinstall")
        _run(command)
    installed = probe_torch(python)
    if plan["profile"] == "full" and not installed.get("working"):
        raise ValueError("PyTorch/torchvision verification failed: " + str(installed.get("error")))
    # Protect the selected/preserved wheel from model dependencies selecting another CUDA build.
    with tempfile.TemporaryDirectory(prefix="vision-setup-") as temporary:
        constraints = Path(temporary) / "torch-constraints.txt"
        pins = []
        for key, package in (("version", "torch"), ("torchvision_version", "torchvision")):
            if installed.get(key):
                pins.append(package + "==" + installed[key])
        constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
        command = pip + ["install", "--index-url", "https://pypi.org/simple"]
        if pins:
            command += ["--constraint", str(constraints)]
        for requirement in plan["requirements"]:
            command += ["--requirement", str(ROOT / requirement)]
        _run(command)
    _run(pip + ["check"])
    if plan["build_frontend"]:
        _run([npm, "ci"], cwd=ROOT / "frontend")
        _run([npm, "run", "build"], cwd=ROOT / "frontend")
    if plan["profile"] == "full":
        checked = probe_torch(python)
        if not checked.get("working"):
            raise ValueError("Final PyTorch check failed: " + str(checked.get("error")))
        if plan["torch_action"] == "install" and plan["torch"]["variant"] != "cpu" and not checked.get("cuda_available"):
            raise ValueError("CUDA wheels installed, but torch.cuda is unavailable. No driver was changed. Diagnose with this environment before running models.")
    print("Setup complete. Launch with: %s %s" % (subprocess.list2cmdline([str(python)]), subprocess.list2cmdline([str(ROOT / "launch.py")])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Create/use only the selected venv and install the displayed plan")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv", help="Virtual environment path (default: project .venv)")
    parser.add_argument("--profile", choices=("full", "base"), default="full", help="base: API/UI only; full: model runtimes, no weights")
    parser.add_argument("--torch", choices=("auto", "cpu", "cu126", "cu128"), default="auto")
    parser.add_argument("--repair-torch", action="store_true", help="Explicitly replace an existing torch/torchvision pair")
    parser.add_argument("--with-rfdetr-plus", action="store_true", help="Opt in to the separate PML-licensed runtime; does not grant license rights")
    parser.add_argument("--with-dev", action="store_true", help="Install test dependencies")
    parser.add_argument("--skip-frontend", action="store_true", help="Do not run npm ci/build")
    parser.add_argument("--json", action="store_true", help="Print diagnostics and plan as JSON")
    args = parser.parse_args(argv)
    target = args.venv.expanduser().resolve()
    target_python = venv_python(target)
    exists = target_python.is_file() and (target / "pyvenv.cfg").is_file()
    info = collect_system_info(ROOT, refresh=True, python_executable=target_python if exists else sys.executable)
    plan = build_plan(info, target, profile=args.profile, requested=args.torch,
                      repair=args.repair_torch, with_rfdetr_plus=args.with_rfdetr_plus,
                      with_dev=args.with_dev, skip_frontend=args.skip_frontend, target_exists=exists)
    plan["install_requested"] = args.install
    if args.json:
        print(json.dumps({"system": info, "plan": plan}, ensure_ascii=False, indent=2))
    else:
        print("Vision Workbench setup — " + ("INSTALL" if args.install else "DIAGNOSE ONLY; no changes"))
        print("OS: %(system)s %(machine)s" % info["platform"])
        print("Python: %s | supported: %s | recommended: 3.11" % (info["python"]["version"], info["python"]["supported"]))
        print("RAM: %s bytes | disk free: %s bytes" % (info["memory"]["total_bytes"], info["disk"]["free_bytes"]))
        print("NVIDIA detected: %s | driver: %s | Torch CUDA available: %s" % (info["nvidia"]["detected"], info["nvidia"]["driver_version"], info["torch"]["cuda_available"]))
        print("Venv: %s | profile: %s | torch: %s (%s)" % (plan["venv"], plan["profile"], plan["torch_action"], plan["torch"]["variant"]))
        for note in plan["notes"]:
            print("- " + note)
        for error in plan["errors"]:
            print("ERROR: " + error)
        if not args.install:
            print("Add --install to execute this plan. Use --venv PATH to inspect an existing environment.")
    if args.install:
        try:
            execute_plan(plan, repair=args.repair_torch)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print("Setup stopped: " + str(exc), file=sys.stderr)
            return 1
    return 1 if plan["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
