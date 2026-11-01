"""Combined startup never installs dependencies or terminates other processes."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comfy_launcher_tests", ROOT / "launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def make_config(tmp_path, device="cpu", standalone=False):
    core = tmp_path / "ComfyUI"
    bridge = core / "custom_nodes/vision_workbench/__init__.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("# bridge", encoding="utf-8")
    return {"root": str(core), "python": str(tmp_path / "python"),
            "device": device, "managed": not standalone, "standalone": standalone}


def no_process(*args, **kwargs):
    pytest.fail("A ready or conflicting service must not start another process")


def registered_environment(project, venv, **overrides):
    import os
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("python")
    (venv / "pyvenv.cfg").write_text("home = base")
    registration = project / "runtime/workbench_environment.json"
    registration.parent.mkdir(parents=True, exist_ok=True)
    value = {"project": str(project.resolve()), "venv": str(venv.absolute()), "python": str(candidate.absolute()), **overrides}
    registration.write_text(json.dumps(value), encoding="utf-8")
    return candidate, registration


def test_registered_short_environment_precedes_local_venv(tmp_path):
    import os
    project = tmp_path / "project"
    selected, _ = registered_environment(project, tmp_path / "short/venv")
    local = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    local.parent.mkdir(parents=True)
    local.write_text("another Python")
    assert launcher.environment_python(project, environ={}) == selected.absolute()


def test_explicit_environment_overrides_take_priority_over_stale_registration(tmp_path):
    project = tmp_path / "project"
    selected, _ = registered_environment(project, tmp_path / "short/venv")
    selected.unlink()
    override = tmp_path / "explicit-python"
    override.write_text("explicit")
    assert launcher.environment_python(project, environ={"VISION_WORKBENCH_PYTHON": str(override)}) == override.absolute()


@pytest.mark.parametrize("problem", ["missing_python", "missing_venv_config", "different_project", "outside_python", "relative_path", "malformed_json"])
def test_invalid_registered_environment_requires_setup_instead_of_falling_back(tmp_path, problem):
    project = tmp_path / "project"
    venv = tmp_path / "short/venv"
    selected, registration = registered_environment(project, venv)
    saved = json.loads(registration.read_text())
    if problem == "missing_python":
        selected.unlink()
    elif problem == "missing_venv_config":
        (venv / "pyvenv.cfg").unlink()
    elif problem == "different_project":
        saved["project"] = str(tmp_path / "another-project")
    elif problem == "outside_python":
        saved["python"] = str(tmp_path / "unrelated-python")
        Path(saved["python"]).write_text("unrelated")
    elif problem == "relative_path":
        saved["venv"] = "relative-venv"
    registration.write_text("not JSON" if problem == "malformed_json" else json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="Run python setup.py --install"):
        launcher.environment_python(project, environ={}, current=Path(sys.executable))


@pytest.mark.parametrize("selection", ["explicit_python", "explicit_venv", "registered", "local", "active", "legacy", "fallback"])
def test_environment_python_preserves_venv_executable_symlink(tmp_path, selection):
    import os
    project = tmp_path / "outputs/project"
    project.mkdir(parents=True)
    suffix = Path("Scripts/python.exe" if os.name == "nt" else "bin/python")
    system_python = tmp_path / "system-python"
    system_python.write_text("system Python")
    directory = project / ".venv" if selection == "local" else tmp_path / "work/vision_workbench/.venv" if selection == "legacy" else tmp_path / "selected-venv"
    linked_python = directory / suffix
    linked_python.parent.mkdir(parents=True)
    try:
        linked_python.symlink_to(system_python)
    except (OSError, NotImplementedError):
        pytest.skip("Executable symlinks are unavailable on this host")
    if selection in {"active", "registered"}:
        (directory / "pyvenv.cfg").write_text("home = system")
    if selection == "registered":
        registration = project / "runtime/workbench_environment.json"
        registration.parent.mkdir()
        registration.write_text(json.dumps({"project": str(project.absolute()), "venv": str(directory.absolute()), "python": str(linked_python.absolute())}), encoding="utf-8")
    environ = {"VISION_WORKBENCH_PYTHON": str(linked_python)} if selection == "explicit_python" else {"VISION_WORKBENCH_VENV": str(directory)} if selection == "explicit_venv" else {}
    current = linked_python if selection in {"active", "fallback"} else system_python
    selected = launcher.environment_python(project, environ=environ, current=current)
    assert selected == Path(os.path.abspath(linked_python))
    assert selected != linked_python.resolve()


def test_workbench_port_conflict_does_not_spawn_or_check_environment(monkeypatch):
    checked = []
    monkeypatch.setattr(launcher, "alive", lambda: False)
    monkeypatch.setattr(launcher, "port_in_use", lambda port: checked.append(port) or True)
    monkeypatch.setattr(launcher, "environment_python", no_process)
    monkeypatch.setattr(launcher.subprocess, "Popen", no_process)
    with pytest.raises(ValueError, match="Port 8765 is occupied"):
        launcher.start_workbench()
    assert checked == [8765]


def test_local_probe_rejects_redirects():
    from urllib.request import Request
    from urllib.error import HTTPError
    with pytest.raises(HTTPError, match="redirects are not accepted"):
        launcher._NoRedirect().redirect_request(Request(launcher.COMFY_URL), None, 302, "Found", {}, "https://example.com/")


def test_probe_requires_all_shipped_nodes(monkeypatch):
    nodes = json.loads((ROOT / "comfyui/vision_workbench/catalog.json").read_text(encoding="utf-8"))
    definitions = {"VisionWB_" + row["type"]: {} for row in nodes}
    def response(url, **kwargs):
        return {"system": {"comfyui_version": "0.28.0"}} if url.endswith("system_stats") else definitions
    monkeypatch.setattr(launcher, "local_json", response)
    assert launcher.comfy_probe(ROOT)[0]
    definitions.pop("VisionWB_pose")
    ready, reason = launcher.comfy_probe(ROOT)
    assert not ready and "1 Vision Workbench nodes are missing" in reason


def test_probe_does_not_accept_an_unrelated_http_server(monkeypatch):
    monkeypatch.setattr(launcher, "local_json", lambda *a, **k: {"status": "ok"})
    assert launcher.comfy_probe(ROOT)[0] is False


def test_reuses_ready_comfy_and_exports_machine_specific_templates(tmp_path, monkeypatch):
    import export_workflow_templates
    config = make_config(tmp_path)
    calls = []
    monkeypatch.setattr(launcher, "comfy_probe", lambda root: (True, "ready"))
    monkeypatch.setattr(launcher.subprocess, "Popen", no_process)
    monkeypatch.setattr(export_workflow_templates, "export_templates", lambda url, folder: calls.append((url, folder)) or list(range(15)))
    result = launcher.start_comfyui(config)
    assert result["reused"] and result["workflows"] == 15
    assert calls == [(launcher.URL, Path(config["root"]) / "user/default/workflows/VisionWorkbench")]


def test_conflicting_service_is_left_alone_without_export(tmp_path, monkeypatch):
    import export_workflow_templates
    config = make_config(tmp_path)
    monkeypatch.setattr(launcher, "comfy_probe", lambda root: (False, "Wrong server"))
    monkeypatch.setattr(launcher, "port_in_use", lambda port: True)
    monkeypatch.setattr(launcher.subprocess, "Popen", no_process)
    monkeypatch.setattr(export_workflow_templates, "export_templates", no_process)
    with pytest.raises(ValueError, match="existing service was left running"):
        launcher.start_comfyui(config)


def test_missing_bridge_does_not_start_or_install(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "Popen", no_process)
    with pytest.raises(ValueError, match="bridge is not installed"):
        launcher.start_comfyui({"root": str(tmp_path)})


def test_new_cpu_process_has_isolated_executable_localhost_and_hidden_flags(tmp_path, monkeypatch):
    import export_workflow_templates
    config = make_config(tmp_path)
    calls = []
    probes = iter([(False, "offline"), (True, "ready")])
    monkeypatch.setattr(launcher, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(launcher, "comfy_probe", lambda root: next(probes))
    monkeypatch.setattr(launcher, "port_in_use", lambda port: False)
    monkeypatch.setattr(export_workflow_templates, "export_templates", lambda *args: list(range(15)))
    monkeypatch.setattr(launcher.subprocess, "CREATE_NO_WINDOW", 123, raising=False)
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4321, poll=lambda: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    result = launcher.start_comfyui(config)
    command, kwargs = calls[0]
    assert command[0] == config["python"]
    assert command[2] == str(Path(config["root"]) / "main.py")
    assert "--cpu" in command and "--windows-standalone-build" not in command
    assert command[command.index("--listen") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8188"
    assert kwargs["cwd"] == Path(config["root"])
    assert kwargs["creationflags"] == 123
    assert result["pid"] == 4321
    assert (tmp_path / "runtime/comfyui.pid").read_text() == "4321"


def test_existing_portable_keeps_its_gpu_mode(tmp_path):
    command = launcher.comfy_command(make_config(tmp_path, device="auto", standalone=True))
    assert "--cpu" not in command
    assert "--windows-standalone-build" in command


def test_failed_child_is_not_reported_ready_and_no_process_termination(tmp_path, monkeypatch):
    import export_workflow_templates
    config = make_config(tmp_path)
    monkeypatch.setattr(launcher, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(launcher, "comfy_probe", lambda root: (False, "offline"))
    monkeypatch.setattr(launcher, "port_in_use", lambda port: False)
    monkeypatch.setattr(export_workflow_templates, "export_templates", lambda *args: [])
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=7, poll=lambda: 1))
    with pytest.raises(ValueError, match="exited during startup"):
        launcher.start_comfyui(config)


def test_timeout_does_not_kill_child_or_existing_service(tmp_path, monkeypatch):
    import export_workflow_templates
    config = make_config(tmp_path)
    monkeypatch.setattr(launcher, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(launcher, "comfy_probe", lambda root: (False, "nodes not ready"))
    monkeypatch.setattr(launcher, "port_in_use", lambda port: False)
    monkeypatch.setattr(export_workflow_templates, "export_templates", lambda *args: [])
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=7, poll=lambda: None))
    with pytest.raises(ValueError, match="process was left running"):
        launcher.start_comfyui(config, timeout=0)


def setup_cli(tmp_path, monkeypatch, config):
    frontend = tmp_path / "frontend/dist/index.html"
    frontend.parent.mkdir(parents=True)
    frontend.write_text("ready")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "comfy_manager", SimpleNamespace(
        load_comfy_config=lambda root: config,
        build_comfy_plan=lambda *a, **k: {"action": "install", "errors": []}))
    calls = []
    monkeypatch.setattr(launcher, "start_workbench", lambda: calls.append("workbench"))
    monkeypatch.setattr(launcher, "start_comfyui", lambda selected: calls.append("comfyui") or {"url": launcher.COMFY_START_URL})
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: calls.append(url))
    return calls


def test_configured_default_launch_starts_both_before_opening_camera_template(tmp_path, monkeypatch):
    calls = setup_cli(tmp_path, monkeypatch, {"root": "existing"})
    assert launcher.main([]) == 0
    assert calls == ["workbench", "comfyui", launcher.COMFY_START_URL]


def test_no_browser_suppresses_all_tabs_and_without_comfy_preserves_original_entry(tmp_path, monkeypatch):
    calls = setup_cli(tmp_path, monkeypatch, {"root": "existing"})
    launcher.main(["--no-browser"])
    assert calls == ["workbench", "comfyui"]
    calls.clear()
    launcher.main(["--without-comfyui"])
    assert calls == ["workbench", launcher.URL]


def test_unconfigured_default_preserves_original_entry_and_explicit_request_does_not_install(tmp_path, monkeypatch):
    calls = setup_cli(tmp_path, monkeypatch, None)
    launcher.main([])
    assert calls == ["workbench", launcher.URL]
    calls.clear()
    with pytest.raises(SystemExit, match="setup.py --install --with-comfyui"):
        launcher.main(["--with-comfyui"])
    assert calls == []


def test_invalid_config_is_not_silently_ignored(tmp_path, monkeypatch):
    calls = setup_cli(tmp_path, monkeypatch, None)
    def invalid(root):
        raise ValueError("configured Python is missing")
    sys.modules["comfy_manager"].load_comfy_config = invalid
    with pytest.raises(SystemExit, match="configured Python is missing"):
        launcher.main([])
    assert calls == []


def test_mutually_exclusive_and_conflicting_options_rejected(tmp_path, monkeypatch):
    calls = setup_cli(tmp_path, monkeypatch, None)
    with pytest.raises(SystemExit):
        launcher.main(["--with-comfyui", "--without-comfyui"])
    with pytest.raises(SystemExit):
        launcher.main(["--without-comfyui", "--comfy-root", "somewhere"])
    assert calls == []
