"""Read-only host diagnostics. NVIDIA detection and PyTorch readiness are separate.

Only the isolated probe imports torch; API startup stays independent of ML packages.
No model, driver, or CUDA toolkit is installed or downloaded by this module.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import ctypes
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}
_LOCK = threading.Lock()
_CACHE_SECONDS = 60
_MARKER = "VISION_SYSTEM_JSON:"

# Keep probe output machine-readable even if an installed extension prints warnings.
_TORCH_PROBE = r'''
import importlib.util, json, platform, sys
result = {"installed": False, "version": None, "torchvision_version": None,
          "cuda_build": None, "cuda_available": False, "devices": [],
          "working": False, "error": None,
          "python_version": platform.python_version()}
try:
    result["installed"] = importlib.util.find_spec("torch") is not None
    if result["installed"]:
        import torch
        result["version"] = str(torch.__version__)
        result["cuda_build"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(index)
                result["devices"].append({"id": "cuda:" + str(index), "name": prop.name,
                    "memory_total_bytes": prop.total_memory,
                    "capability": str(prop.major) + "." + str(prop.minor)})
        try:
            import torchvision
            result["torchvision_version"] = str(torchvision.__version__)
            # A tiny CPU operation checks the paired compiled torchvision extension.
            torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
            result["working"] = True
        except Exception as exc:
            result["error"] = "torchvision check failed: " + str(exc)[:300]
    else:
        result["error"] = "PyTorch is not installed in this interpreter."
except Exception as exc:
    result["cuda_available"] = False
    result["devices"] = []
    result["error"] = type(exc).__name__ + ": " + str(exc)[:300]
print("VISION_SYSTEM_JSON:" + json.dumps(result))
'''


def _run(command, timeout=10):
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def probe_torch(python_executable=None):
    """Inspect an interpreter without changing its environment or importing ML here."""
    result = {"installed": False, "version": None, "torchvision_version": None,
              "cuda_build": None, "cuda_available": False, "devices": [],
              "working": False, "error": None, "python_version": None}
    try:
        process = _run([str(python_executable or sys.executable), "-c", _TORCH_PROBE], timeout=30)
        for line in reversed(process.stdout.splitlines()):
            if line.startswith(_MARKER):
                result.update(json.loads(line[len(_MARKER):]))
                return result
        result["error"] = "PyTorch probe did not return diagnostics (exit %s)." % process.returncode
    except subprocess.TimeoutExpired:
        result["error"] = "PyTorch probe timed out after 30 seconds."
    except (OSError, ValueError) as exc:
        result["error"] = "PyTorch probe failed: " + str(exc)[:300]
    return result


def _number(value, cast=float):
    try:
        return cast(value.strip())
    except (ValueError, TypeError, AttributeError):
        return None


def detect_nvidia():
    result = {"detected": False, "query_ok": False, "driver_version": None,
              "gpus": [], "error": None}
    binary = shutil.which("nvidia-smi")
    if not binary and platform.system() == "Windows":
        candidate = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe"
        if candidate.is_file():
            binary = str(candidate)
    if not binary:
        result["error"] = "nvidia-smi was not found; NVIDIA hardware/driver is unverified."
        return result
    try:
        fields = "index,name,driver_version,memory.total,compute_cap"
        process = _run([binary, "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
        with_capability = process.returncode == 0
        if not with_capability:
            process = _run([binary, "--query-gpu=index,name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"])
        if process.returncode:
            result["error"] = "nvidia-smi failed; the driver may be absent or unavailable."
            return result
        result["query_ok"] = True
        for row in csv.reader(io.StringIO(process.stdout)):
            if len(row) < 4:
                continue
            memory_mib = _number(row[3])
            result["gpus"].append({"index": _number(row[0], int), "name": row[1].strip(),
                "memory_total_bytes": int(memory_mib * 1024**2) if memory_mib is not None else None,
                "compute_capability": row[4].strip() if with_capability and len(row) > 4 and _number(row[4]) is not None else None})
            result["driver_version"] = row[2].strip()
        result["detected"] = bool(result["gpus"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = "NVIDIA query failed: " + type(exc).__name__
    return result


def memory_info():
    result = {"total_bytes": None, "available_bytes": None}
    try:
        if platform.system() == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                                        "virtual_total", "virtual_available", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                result.update(total_bytes=status.total, available_bytes=status.available)
        elif platform.system() == "Linux":
            entries = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                entries[key] = int(value.strip().split()[0]) * 1024
            result.update(total_bytes=entries.get("MemTotal"), available_bytes=entries.get("MemAvailable"))
        elif platform.system() == "Darwin":
            process = _run(["sysctl", "-n", "hw.memsize"])
            result["total_bytes"] = _number(process.stdout, int)
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        pass
    return result


def devices_from_torch(torch_info, memory_total_bytes=None):
    """Never expose a CUDA option solely because nvidia-smi found a GPU."""
    devices = [{"id": "cpu", "name": "CPU", "memory_total_bytes": memory_total_bytes}]
    if torch_info.get("cuda_available"):
        devices.extend(deepcopy(torch_info.get("devices") or []))
    return devices


def collect_system_info(project_root=None, *, refresh=False, python_executable=None):
    root = Path(project_root or ROOT).resolve()
    executable = str(python_executable or sys.executable)
    key = (str(root), executable)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and not refresh and time.monotonic() - cached[0] < _CACHE_SECONDS:
            return deepcopy(cached[1])
        memory = memory_info()
        try:
            usage = shutil.disk_usage(root)
            disk = {"free_bytes": usage.free, "total_bytes": usage.total}
        except OSError:
            disk = {"free_bytes": None, "total_bytes": None}
        torch_info = probe_torch(executable)
        version = torch_info.get("python_version") or platform.python_version()
        version_parts = tuple(int(part) for part in version.split(".")[:2])
        devices = devices_from_torch(torch_info, memory["total_bytes"])
        nvidia = detect_nvidia()
        notes = ["Batch size 1 is a conservative starting point, not a model memory-fit guarantee.",
                 "Large models may exceed available RAM/VRAM even at batch size 1."]
        if nvidia["detected"] and not torch_info["cuda_available"]:
            notes.append("NVIDIA hardware was detected, but CUDA is unavailable in this Python environment.")
        if platform.system() == "Darwin":
            notes.append("This workbench exposes CPU on macOS; MPS adapters are not validated.")
        result = {"schema_version": 1,
            "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "python": {"version": version, "executable": executable,
                       "supported": (3, 10) <= version_parts <= (3, 12), "recommended": "3.11"},
            "cpu": {"logical_count": os.cpu_count()}, "memory": memory, "disk": disk,
            "nvidia": nvidia, "torch": torch_info, "devices": devices,
            "recommendations": {"default_device": devices[1]["id"] if len(devices) > 1 else "cpu",
                                "batch_size": 1, "notes": notes}}
        _CACHE[key] = (time.monotonic(), result)
        return deepcopy(result)


def available_device_options():
    return ["auto"] + [device["id"] for device in collect_system_info()["devices"]]
