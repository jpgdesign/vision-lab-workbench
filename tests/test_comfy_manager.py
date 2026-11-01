"""Provisioning contracts use small archives and fake interpreters, never pip."""
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest

import comfy_manager as manager


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)


def core(path):
    path.mkdir(parents=True)
    for name in ("main.py", "folder_paths.py", "requirements.txt"):
        (path/name).write_text("", encoding="utf-8")
    return path


def interpreter(root):
    python = manager._env_python(root)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    (root/"pyvenv.cfg").write_text("home=test")
    return python


def archive(path, extra=()):
    prefix = "ComfyUI-" + manager.COMFY_REVISION + "/"
    with zipfile.ZipFile(path, "w") as value:
        for filename in ("main.py", "folder_paths.py", "requirements.txt"):
            value.writestr(prefix+filename, "# core")
        for filename, content in extra:
            info = zipfile.ZipInfo(filename)
            info.filename = filename  # Windows ZipInfo otherwise normalizes backslashes.
            value.writestr(info, content)
    return path


def test_plan_is_read_only_and_new_environment_is_cpu_isolated(tmp_path):
    project = tmp_path/"project"
    project.mkdir()
    plan = manager.build_comfy_plan(project)
    assert plan["action"] == "install" and plan["managed"]
    assert plan["device"] == "cpu"
    assert plan["root"] == str(project/".tools/ComfyUI")
    assert not (project/".tools").exists()
    assert not (project/"runtime").exists()
    assert plan["archive_sha256"] == manager.COMFY_SHA256


def test_existing_portable_discovered_without_using_system_python(tmp_path):
    project = tmp_path/"project"
    project.mkdir()
    existing = core(Path.home()/"ComfyUI_windows_portable/ComfyUI")
    python = existing.parent/"python_embeded/python.exe"
    python.parent.mkdir()
    python.touch()
    plan = manager.build_comfy_plan(project)
    assert not plan["errors"]
    assert plan["root"] == str(existing)
    assert plan["python"] == str(python)
    assert plan["standalone"] and not plan["managed"] and plan["action"] == "reuse"
    assert plan["device"] == "auto"


def test_existing_core_without_known_interpreter_is_actionable(tmp_path):
    existing = core(tmp_path/"existing")
    plan = manager.build_comfy_plan(tmp_path, comfy_root=existing)
    assert any("--comfy-python" in message for message in plan["errors"])
    assert plan["action"] == "reuse"


def test_explicit_missing_core_does_not_download_into_user_path(tmp_path):
    plan = manager.build_comfy_plan(tmp_path, comfy_root=tmp_path/"missing")
    assert any("--comfy-root" in message for message in plan["errors"])
    with pytest.raises(ValueError):
        manager.install_comfy_plan(plan, project_root=tmp_path)
    assert not (tmp_path/"missing").exists()


def test_existing_install_registration_preserves_environment_and_runs_no_pip(tmp_path, monkeypatch):
    existing = core(tmp_path/"existing")
    python = interpreter(existing/".venv")
    plan = manager.build_comfy_plan(tmp_path, comfy_root=existing)
    checked, installed = [], []
    monkeypatch.setattr(manager,"verify_environment",lambda *args,**kwargs:checked.append((args,kwargs)))
    monkeypatch.setattr(manager,"_run",lambda *args,**kwargs:pytest.fail("Existing installs must not run pip or venv creation"))
    import install_comfyui
    monkeypatch.setattr(install_comfyui,"install",lambda path:installed.append(path))
    config = manager.install_comfy_plan(plan,project_root=tmp_path)
    assert checked[0][0] == (existing,python)
    assert installed == [existing]
    assert config == manager.load_comfy_config(tmp_path)
    again = manager.build_comfy_plan(tmp_path)
    assert again["root"] == str(existing) and again["action"] == "reuse"


def test_failed_environment_check_does_not_register_success(tmp_path, monkeypatch):
    existing=core(tmp_path/"existing")
    interpreter(existing/".venv")
    plan=manager.build_comfy_plan(tmp_path,comfy_root=existing)
    def fail(*args,**kwargs): raise subprocess.CalledProcessError(1,["python","-c","imports"],stderr="ModuleNotFoundError: No module named 'av'")
    monkeypatch.setattr(manager,"verify_environment",fail)
    with pytest.raises(ValueError,match="No module named 'av'"):
        manager.install_comfy_plan(plan,project_root=tmp_path)
    assert manager.load_comfy_config(tmp_path) is None


def test_missing_registration_and_corrupt_registration(tmp_path):
    assert manager.load_comfy_config(tmp_path) is None
    (tmp_path/"runtime").mkdir()
    (tmp_path/"runtime/comfyui.json").write_text("{}")
    with pytest.raises(ValueError,match="Invalid saved"):
        manager.load_comfy_config(tmp_path)


def test_extract_pinned_archive_in_new_directory(tmp_path):
    source=archive(tmp_path/"source.zip")
    dest=manager.extract_archive(source,tmp_path/"core")
    assert (dest/"main.py").read_text() == "# core"
    with pytest.raises(ValueError,match="must be new"):
        manager.extract_archive(source,dest)


@pytest.mark.parametrize("bad", ["../escape.py", "/absolute.py", "ComfyUI-"+manager.COMFY_REVISION+"/../escape.py",
                                  "ComfyUI-"+manager.COMFY_REVISION+"/C:/escape.py", "ComfyUI-"+manager.COMFY_REVISION+"/a\\escape.py",
                                  "ComfyUI-"+manager.COMFY_REVISION+"/NUL.txt", "ComfyUI-"+manager.COMFY_REVISION+"/main.py"])
def test_reject_zip_traversal_windows_devices_and_duplicate_before_writing(tmp_path,bad):
    source=archive(tmp_path/"bad.zip",[(bad,"malicious")])
    with pytest.raises(ValueError):
        manager.extract_archive(source,tmp_path/"extract")
    assert not (tmp_path/"extract").exists()
    assert not (tmp_path/"escape.py").exists()


def test_reject_archive_symlink_and_oversize(tmp_path,monkeypatch):
    source=archive(tmp_path/"source.zip")
    with zipfile.ZipFile(source,"a") as value:
        link=zipfile.ZipInfo("ComfyUI-"+manager.COMFY_REVISION+"/link")
        link.create_system=3
        link.external_attr=(stat.S_IFLNK|0o777)<<16
        value.writestr(link,"../../elsewhere")
    with pytest.raises(ValueError,match="Unsafe"):
        manager.extract_archive(source,tmp_path/"extract")
    source=archive(tmp_path/"large.zip")
    monkeypatch.setattr(manager,"MAX_EXPANDED_BYTES",1)
    with pytest.raises(ValueError,match="limits"):
        manager.extract_archive(source,tmp_path/"extract")


def test_cached_verified_download_needs_no_network(tmp_path,monkeypatch):
    source=tmp_path/"source.zip"
    source.write_bytes(b"verified")
    monkeypatch.setattr(manager,"COMFY_SHA256",manager._sha256(source))
    monkeypatch.setattr(manager,"build_opener",lambda *args:pytest.fail("No request for verified cached archive"))
    assert manager.download_archive(source)==source


def test_bad_download_checksum_does_not_replace_existing_cache(tmp_path,monkeypatch):
    target=tmp_path/"cached.zip"
    target.write_bytes(b"old cache")
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def geturl(self):return manager.COMFY_URL
        def read(self,size):
            if getattr(self,"done",False):return b""
            self.done=True
            return b"tampered download"
    class Opener:
        def open(self,*args,**kwargs):return Response()
    monkeypatch.setattr(manager,"build_opener",lambda *args:Opener())
    with pytest.raises(ValueError,match="SHA256"):
        manager.download_archive(target)
    assert target.read_bytes()==b"old cache"


def test_managed_install_uses_cpu_pins_constraints_and_resumes_without_pip(tmp_path,monkeypatch):
    source=archive(tmp_path/"core.zip")
    plan=manager.build_comfy_plan(tmp_path)
    commands=[]
    monkeypatch.setattr(manager,"download_archive",lambda target:source)
    def run(command,**kwargs):
        command=list(map(str,command)); commands.append(command)
        if "venv" in command:
            interpreter(Path(plan["venv"]))
        if "--constraint" in command:
            text=Path(command[command.index("--constraint")+1]).read_text()
            assert all(pin in text for pin in manager.TORCH_PINS)
    monkeypatch.setattr(manager,"_run",run)
    monkeypatch.setattr(manager,"verify_environment",lambda *args,**kwargs:None)
    import install_comfyui
    monkeypatch.setattr(install_comfyui,"install",lambda path:None)
    result=manager.install_comfy_plan(plan,project_root=tmp_path)
    pip=[cmd for cmd in commands if "pip" in cmd]
    assert pip and all(cmd[0]==plan["python"] for cmd in pip)
    assert any(all(pin in cmd for pin in manager.TORCH_PINS) for cmd in pip)
    assert all("cu128" not in " ".join(cmd) for cmd in commands)
    assert result["managed"] and result["device"]=="cpu"
    assert manager.load_comfy_config(tmp_path)==result
    again=manager.build_comfy_plan(tmp_path)
    assert again["action"]=="reuse"
    commands.clear()
    manager.install_comfy_plan(again,project_root=tmp_path)
    assert not commands


def test_unsupported_creator_fails_before_download_or_managed_directory(tmp_path,monkeypatch):
    plan=manager.build_comfy_plan(tmp_path)
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,["python","-c","version"])
    monkeypatch.setattr(manager,"_run",fail)
    monkeypatch.setattr(manager,"download_archive",lambda path:pytest.fail("Download must not start"))
    with pytest.raises(ValueError,match="Python 3.10-3.12"):
        manager.install_comfy_plan(plan,project_root=tmp_path)
    assert not (tmp_path/".tools").exists()


def test_owned_partial_install_is_resumable_but_unknown_directory_is_not(tmp_path):
    (tmp_path/".tools/ComfyUI").mkdir(parents=True)
    plan=manager.build_comfy_plan(tmp_path)
    assert any("unowned" in value for value in plan["errors"])
    manager._write_json(tmp_path/".tools/ComfyUI"/manager.SOURCE_MARKER,manager._owner(tmp_path))
    plan=manager.build_comfy_plan(tmp_path)
    assert not plan["errors"] and plan["action"]=="install"


def test_windows_long_project_path_uses_short_deterministic_private_environment(tmp_path,monkeypatch):
    monkeypatch.setattr(manager.platform,"system",lambda:"Windows")
    project=tmp_path/("long-project-name-"*8)
    plan=manager.build_comfy_plan(project)
    environment=Path(plan["venv"])
    assert environment.is_relative_to(Path.home()/".vwb/comfy")
    assert environment.name=="venv" and len(environment.parent.name)==12
    assert environment==manager.managed_environment(project)
    assert environment!=manager.managed_environment(project/"another")
    assert Path(plan["root"])==project/".tools/ComfyUI"
    assert not environment.exists()


def test_missing_owned_managed_python_is_repaired_without_trusting_unowned_directory(tmp_path,monkeypatch):
    project=tmp_path/"project"
    project.mkdir()
    root=core(project/".tools/ComfyUI")
    environment=manager.managed_environment(project)
    python=interpreter(environment)
    manager._write_json(root/manager.SOURCE_MARKER,manager._owner(project))
    manager._write_json(environment/manager.ENV_MARKER,manager._owner(project))
    config={"root":str(root),"python":str(python),"managed":True,"device":"cpu","standalone":False}
    manager._write_json(project/"runtime/comfyui.json",config)
    python.unlink()
    with pytest.raises(ValueError,match="Python is missing"):
        manager.load_comfy_config(project)
    plan=manager.build_comfy_plan(project)
    assert plan["action"]=="install" and not plan["errors"]
    (environment/manager.ENV_MARKER).unlink()
    assert manager.build_comfy_plan(project)["errors"]


def test_streamed_subprocess_output_and_failure_are_visible(tmp_path,capsys):
    manager._run([sys.executable,"-c","print('install progress', flush=True)"],cwd=tmp_path)
    assert "install progress" in capsys.readouterr().out
    with pytest.raises(subprocess.CalledProcessError) as result:
        manager._run([sys.executable,"-c","import sys; print('dependency failed', flush=True); sys.exit(3)"],cwd=tmp_path)
    assert "dependency failed" in result.value.output
    assert "dependency failed" in capsys.readouterr().out


def test_captured_probe_retains_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        manager._run([sys.executable,"-c","import time; time.sleep(30)"],cwd=tmp_path,capture=True,timeout=.1)


def test_linux_python_symlink_path_is_preserved(tmp_path):
    executable=tmp_path/"base-python"
    executable.touch()
    python=tmp_path/"env/bin/python"
    python.parent.mkdir(parents=True)
    try:python.symlink_to(executable)
    except OSError:pytest.skip("Symlinks unavailable")
    assert manager.absolute(python)==python
    assert manager.absolute(python)!=python.resolve()
