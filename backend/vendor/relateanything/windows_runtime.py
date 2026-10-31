"""Preload the installed Windows MSVC runtime before NumPy/PyTorch imports.

Some older Anaconda distributions bundle MSVC 14.27 beside python.exe. ORT
1.21+ requires 14.40 or newer; a modern system runtime can otherwise be shadowed.
No DLL is installed, replaced, copied, or unloaded by this helper.
"""
import os

_handles = []
_ready = False
_load_error = None


def prepare_windows_runtime(*, strict=True):
    """Call at process startup, before importing native numerical libraries.

    Uses only the Windows system directory, never a DLL from the working
    directory or PATH. A process which already loaded an old MSVC library must
    be restarted; loading a second version cannot reliably repair it.

    General app/worker startup may use strict=False: unavailable system DLLs
    then return False without blocking unrelated models. RelationModel keeps
    the strict default. Failed attempts are cached until process restart.
    """
    global _ready, _load_error
    if os.name != "nt" or _ready:
        return True
    if _load_error is not None:
        if strict:
            raise _load_error
        return False
    import ctypes
    from pathlib import Path

    loaded = []
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
        if not 0 < size < len(buffer):
            raise OSError("Cannot locate the Windows system directory for the MSVC runtime.")
        directory = Path(buffer.value)
        for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
                     "msvcp140_1.dll", "msvcp140_2.dll"):
            loaded.append(ctypes.WinDLL(str(directory / name)))
    except OSError as exc:
        # Keep any successfully loaded handles alive, but only _ready denotes
        # a complete preload. Partial success must never bypass strict checks.
        _handles.extend(loaded)
        _load_error = RuntimeError(
            "RelateAnything requires the Microsoft Visual C++ 2015–2022 "
            "x64 Redistributable (MSVC runtime >= 14.40). Install it and restart."
        )
        _load_error.__cause__ = exc
        if strict:
            raise _load_error from exc
        return False
    _handles.extend(loaded)
    _ready = True
    return True
