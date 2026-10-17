import json
from types import SimpleNamespace

from backend import system_info as info


def test_nvidia_presence_is_not_torch_readiness():
    assert info.devices_from_torch({"cuda_available": False, "devices": [{"id": "cuda:0"}]}) == [
        {"id": "cpu", "name": "CPU", "memory_total_bytes": None}]


def test_all_visible_cuda_devices_and_no_mps():
    gpu = [{"id": "cuda:0", "name": "A"}, {"id": "cuda:1", "name": "B"}]
    result = info.devices_from_torch({"cuda_available": True, "devices": gpu}, 1024)
    assert [device["id"] for device in result] == ["cpu", "cuda:0", "cuda:1"]
    result[1]["name"] = "mutated"
    assert gpu[0]["name"] == "A"


def test_nvidia_csv_and_compute_capability(monkeypatch):
    monkeypatch.setattr(info.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(info, "_run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="0, NVIDIA RTX 3060, 590.1, 12288, 8.6\n1, NVIDIA T4, 590.1, 16384, 7.5\n"))
    result = info.detect_nvidia()
    assert result["query_ok"] and result["detected"]
    assert result["gpus"][0]["memory_total_bytes"] == 12 * 1024**3
    assert result["gpus"][1]["compute_capability"] == "7.5"
    assert "cuda_available" not in result


def test_older_nvidia_smi_can_report_gpu_with_unknown_capability(monkeypatch):
    monkeypatch.setattr(info.shutil, "which", lambda name: "nvidia-smi")
    def query(command, **kwargs):
        if "compute_cap" in command[1]:
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout="0, GPU, 550.0, 4096\n")
    monkeypatch.setattr(info, "_run", query)
    result = info.detect_nvidia()
    assert result["detected"]
    assert result["gpus"][0]["compute_capability"] is None


def test_torch_probe_parses_only_marker_and_handles_failure(monkeypatch):
    payload = {"installed": True, "cuda_available": False, "version": "2.8.0+cpu"}
    monkeypatch.setattr(info, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=0,
        stdout="extension warning\n" + info._MARKER + json.dumps(payload)))
    result = info.probe_torch("not-executed")
    assert result["installed"] and not result["cuda_available"]
    monkeypatch.setattr(info, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    assert info.probe_torch("not-executed")["error"]


def test_diagnostics_cached_copied_and_refreshable(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(info, "probe_torch", lambda *args: calls.append(1) or {
        "installed": True, "cuda_available": False, "devices": [], "python_version": "3.11.9"})
    monkeypatch.setattr(info, "detect_nvidia", lambda: {"detected": True, "gpus": []})
    monkeypatch.setattr(info, "memory_info", lambda: {"total_bytes": 1024, "available_bytes": 512})
    first = info.collect_system_info(tmp_path)
    first["recommendations"]["default_device"] = "bad"
    second = info.collect_system_info(tmp_path)
    assert len(calls) == 1
    assert second["recommendations"]["default_device"] == "cpu"
    assert second["python"]["supported"]
    assert any("NVIDIA hardware" in note for note in second["recommendations"]["notes"])
    info.collect_system_info(tmp_path, refresh=True)
    assert len(calls) == 2


def test_training_device_choices_follow_current_torch_devices(monkeypatch):
    from backend import training, training_grounding, training_rfdetr
    options = ["auto", "cpu", "cuda:0", "cuda:1"]
    for module in (training, training_grounding, training_rfdetr):
        monkeypatch.setattr(module, "available_device_options", lambda: list(options))
    for capabilities in (training.training_capabilities("x3d_l"),
                         training_grounding.training_capabilities(), training_rfdetr.training_capabilities()):
        device = next(param for param in capabilities["parameters"] if param["key"] == "device")
        assert device["options"] == options
    options[:] = ["auto", "cpu"]
    device = next(param for param in training.training_capabilities("x3d_l")["parameters"] if param["key"] == "device")
    assert device["options"] == ["auto", "cpu"]
