"""Installer plans are tested without downloading or changing any environment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import system_info

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module("workbench_setup", ROOT / "setup.py")
launcher = load_module("workbench_launcher", ROOT / "launch.py")


def hardware(system="Windows", driver="590.0", capabilities=(8.6,), machine="AMD64", working=False):
    return {"platform": {"system": system, "machine": machine},
        "python": {"supported": True, "version": "3.11.4"},
        "nvidia": {"query_ok": bool(capabilities), "detected": bool(capabilities), "driver_version": driver,
                   "gpus": [{"compute_capability": str(cap) if cap is not None else None} for cap in capabilities]},
        "torch": {"installed": working, "working": working, "version": "2.8.0+cu128" if working else None,
                  "torchvision_version": "0.23.0+cu128" if working else None, "cuda_available": working},
        "memory": {"total_bytes": 32 * 1024**3}, "disk": {"free_bytes": 60 * 1024**3}}


@pytest.mark.parametrize("system,driver,capabilities,variant", [
    ("Windows", "572.61", (8.6,), "cu128"),
    ("Windows", "572.60", (8.6,), "cu126"),
    ("Windows", "561.17", (6.1,), "cu126"),
    ("Windows", "561.16", (8.6,), "cpu"),
    ("Linux", "570.124.06", (8.6,), "cu128"),
    ("Linux", "570.124.05", (8.6,), "cu126"),
    ("Linux", "560.35.05", (5.2,), "cu126"),
    ("Linux", "550.90", (8.6,), "cpu"),
    ("Windows", "590.0", (12.0,), "cu128"),
    ("Windows", "561.17", (12.0,), "cpu"),
    ("Windows", "590.0", (6.1, 12.0), "cpu"),
    ("Windows", "590.0", (None,), "cpu"),
    ("Windows", "590.0", (3.7,), "cpu"),
    ("Windows", "590.0", (11.0,), "cpu"),
    ("Windows", "590.0", (8.9,), "cu128"),
    ("Windows", "590.0", (), "cpu"),
])
def test_conservative_driver_and_architecture_selection(system, driver, capabilities, variant):
    result = setup.choose_torch(hardware(system, driver, capabilities))
    assert result["variant"] == variant
    assert result["index_url"] == "https://download.pytorch.org/whl/" + variant
    assert result["error"] is None


def test_mac_arm_is_cpu_not_mps_and_intel_mac_is_explicitly_unsupported():
    result = setup.choose_torch(hardware("Darwin", capabilities=(), machine="arm64"))
    assert result["variant"] == "cpu"
    assert result["index_url"] == "https://pypi.org/simple"
    assert result["error"] is None
    assert setup.choose_torch(hardware("Darwin", capabilities=(), machine="x86_64"))["error"]


def test_linux_arm_cuda_is_not_automatically_treated_as_desktop_cuda():
    result = setup.choose_torch(hardware("Linux", capabilities=(8.7,), machine="aarch64"))
    assert result["variant"] == "cpu"


def test_explicit_cuda_cannot_bypass_unverified_driver():
    assert setup.choose_torch(hardware(driver="500.0"), "cu128")["error"]
    assert setup.choose_torch(hardware(), "cpu")["variant"] == "cpu"


def test_working_torch_preserved_even_if_user_requests_different_wheel(tmp_path):
    plan = setup.build_plan(hardware(working=True), tmp_path, target_exists=True, requested="cpu")
    assert plan["torch_action"] == "preserve"
    assert plan["preserved_versions"]["version"] == "2.8.0+cu128"
    assert setup.build_plan(hardware(working=True), tmp_path, target_exists=True, repair=True)["torch_action"] == "install"


def test_global_torch_not_mistaken_for_existing_target_environment(tmp_path):
    assert setup.build_plan(hardware(working=True), tmp_path, target_exists=False)["torch_action"] == "install"


def test_broken_existing_torch_requires_explicit_repair(tmp_path):
    info = hardware()
    info["torch"]["installed"] = True
    assert setup.build_plan(info, tmp_path, target_exists=True)["errors"]
    assert not setup.build_plan(info, tmp_path, target_exists=True, repair=True)["errors"]


def test_base_profile_omits_torch_and_plus_is_explicit_opt_in(tmp_path):
    base = setup.build_plan(hardware(), tmp_path, profile="base")
    assert base["torch_action"] == "skip"
    assert base["requirements"] == ["requirements-base.txt"]
    assert not any(line.startswith("rfdetr-plus") for line in (ROOT / "requirements.txt").read_text().splitlines())
    normal = setup.build_plan(hardware(), tmp_path)
    assert "requirements-rfdetr-plus.txt" not in normal["requirements"]
    optional = setup.build_plan(hardware(), tmp_path, with_rfdetr_plus=True)
    assert "requirements-rfdetr-plus.txt" in optional["requirements"]
    assert setup.build_plan(hardware(), tmp_path, profile="base", with_rfdetr_plus=True)["errors"]


def test_unsupported_python_reported_before_install(tmp_path):
    info = hardware()
    info["python"]["supported"] = False
    assert setup.build_plan(info, tmp_path)["errors"]


def test_default_main_is_diagnose_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup, "collect_system_info", lambda *args, **kwargs: hardware())
    monkeypatch.setattr(setup, "execute_plan", lambda *args, **kwargs: pytest.fail("default must not install"))
    assert setup.main(["--venv", str(tmp_path / "new-env"), "--json"]) == 0
    assert '"install_requested": false' in capsys.readouterr().out
    assert not (tmp_path / "new-env").exists()


def test_execution_refuses_non_virtual_environment(tmp_path):
    plan = setup.build_plan(hardware(), tmp_path, skip_frontend=True)
    with pytest.raises(ValueError, match="pyvenv.cfg"):
        setup.execute_plan(plan)


def test_preserved_versions_constrain_dependency_install(tmp_path, monkeypatch):
    python = setup.venv_python(tmp_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    (tmp_path / "pyvenv.cfg").write_text("home = test")
    info = hardware(working=True)
    plan = setup.build_plan(info, tmp_path, target_exists=True, skip_frontend=True)
    commands = []
    def record(command, **kwargs):
        commands.append(command)
        if "--constraint" in command:
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            assert "torch==2.8.0+cu128" in constraints
            assert "torchvision==0.23.0+cu128" in constraints
    monkeypatch.setattr(setup, "_run", record)
    monkeypatch.setattr(setup, "probe_torch", lambda *args: info["torch"])
    setup.execute_plan(plan)
    assert all("torch==2.8.0" not in command for command in commands)
    assert all(command[0] == str(python) for command in commands)
    assert any("--constraint" in command for command in commands)


def test_python_paths_for_windows_and_linux(tmp_path):
    assert setup.venv_python(tmp_path, "Windows") == tmp_path / "Scripts/python.exe"
    assert setup.venv_python(tmp_path, "Linux") == tmp_path / "bin/python"


def test_launcher_local_venv_and_explicit_override(tmp_path):
    local = setup.venv_python(tmp_path / ".venv")
    local.parent.mkdir(parents=True)
    local.touch()
    other = tmp_path / "custom-python"
    other.touch()
    assert launcher.environment_python(tmp_path, environ={}, current=other) == local.resolve()
    assert launcher.environment_python(tmp_path, environ={"VISION_WORKBENCH_PYTHON": str(other)}, current=other) == other.resolve()
    with pytest.raises(ValueError, match="VISION_WORKBENCH_PYTHON"):
        launcher.environment_python(tmp_path, environ={"VISION_WORKBENCH_PYTHON": str(tmp_path / "missing")})
