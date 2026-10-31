"""Optional startup must not turn workflow DLLs into a global requirement."""
import ctypes
from types import SimpleNamespace

import pytest

from backend.vendor.relateanything import windows_runtime as runtime


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runtime, "_handles", [])
    monkeypatch.setattr(runtime, "_ready", False)
    monkeypatch.setattr(runtime, "_load_error", None)
    def system_directory(buffer, size):
        buffer.value = "C:/Windows/System32"
        return len(buffer.value)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(GetSystemDirectoryW=system_directory)), raising=False)


def test_optional_failure_does_not_abort_app_and_strict_still_fails(windows, monkeypatch):
    calls = []
    def loader(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("test missing runtime")
        return object()
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    assert runtime.prepare_windows_runtime(strict=False) is False
    assert len(runtime._handles) == 1 and runtime._ready is False
    assert runtime.prepare_windows_runtime(strict=False) is False
    assert len(calls) == 2  # Cached failure does not keep loading DLLs.
    with pytest.raises(RuntimeError, match="Visual C"):
        runtime.prepare_windows_runtime()
    assert len(calls) == 2


def test_strict_first_failure_retains_cause(windows, monkeypatch):
    def missing(path):
        raise OSError("native DLL failed")
    monkeypatch.setattr(ctypes, "WinDLL", missing, raising=False)
    with pytest.raises(RuntimeError) as caught:
        runtime.prepare_windows_runtime(strict=True)
    assert isinstance(caught.value.__cause__, OSError)
    assert runtime._ready is False


def test_success_is_cached_only_after_all_five_system_dlls(windows, monkeypatch):
    calls = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda path: calls.append(path) or object(), raising=False)
    assert runtime.prepare_windows_runtime(strict=False) is True
    assert len(calls) == 5 and len(runtime._handles) == 5 and runtime._ready
    assert all("Windows/System32/" in path.replace("\\", "/") for path in calls)
    assert runtime.prepare_windows_runtime() is True
    assert len(calls) == 5


def test_unavailable_system_directory_is_optional(windows, monkeypatch):
    monkeypatch.setattr(ctypes.windll.kernel32, "GetSystemDirectoryW", lambda buffer, size: 0)
    assert runtime.prepare_windows_runtime(strict=False) is False
    with pytest.raises(RuntimeError):
        runtime.prepare_windows_runtime()


def test_non_windows_is_successful_noop(monkeypatch):
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="posix"))
    assert runtime.prepare_windows_runtime() is True
