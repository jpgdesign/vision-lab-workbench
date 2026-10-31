import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("workflow_downloads",ROOT/"download_workflow_models.py")
downloads=importlib.util.module_from_spec(spec); spec.loader.exec_module(downloads)


def entry(content=b"official model bytes"):
    return {"name":"model.pt","url":"https://github.com/example/releases/download/v1/model.pt","bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}


class Response(io.BytesIO):
    def __init__(self,content,status=200,headers=None):
        super().__init__(content); self.status=status; self.headers=headers or {}


def test_default_plan_neither_downloads_nor_creates_folders(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(downloads,"download_one",lambda *a,**k:pytest.fail("plan must not download"))
    output=tmp_path/"not-created"
    assert downloads.main(["--output-dir",str(output)])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan["mode"]=="plan_only" and len(plan["models"])==3
    assert plan["total_bytes"]==991941416 and not output.exists()


def test_pinned_manifest_has_official_fixed_urls_and_verified_hashes():
    manifest=downloads.load_manifest()
    assert {x["id"] for x in manifest["models"]}=={"yolo26n_pose","stgcnpp_ntu120","relateanything_convnext_large"}
    relation=next(x for x in manifest["models"] if x["alias"]=="relation")
    assert len(relation["files"])==7
    assert all(relation["revision"] in f["url"] for f in relation["files"])
    assert relation["weight_license"]=="DINOv3 License"


def test_download_verifies_then_reuses_without_network(tmp_path):
    content=b"official model bytes"; target=tmp_path/"model.pt"
    assert downloads.download_one(entry(content),target,opener=lambda *a,**k:Response(content),progress=lambda x:None)=="downloaded_verified"
    assert target.read_bytes()==content and not target.with_name("model.pt.download.part").exists()
    assert downloads.download_one(entry(content),target,opener=lambda *a,**k:pytest.fail("already verified"))=="already_verified"


def test_bad_download_preserves_existing_file_until_verified(tmp_path):
    target=tmp_path/"model.pt"; target.write_bytes(b"user's old file")
    with pytest.raises(ValueError,match="replace-invalid"):
        downloads.download_one(entry(),target,opener=lambda *a,**k:pytest.fail("must not download"))
    with pytest.raises(ValueError,match="SHA256"):
        downloads.download_one(entry(),target,replace_invalid=True,opener=lambda *a,**k:Response(b"wrong"),progress=lambda x:None)
    assert target.read_bytes()==b"user's old file"


def test_resume_range_and_server_range_ignored_are_both_correct(tmp_path):
    content=b"official model bytes"; target=tmp_path/"model.pt"; partial=tmp_path/"model.pt.download.part"
    partial.write_bytes(content[:4]); requests=[]
    def resume(request,**kwargs):
        requests.append(request); return Response(content[4:],206,{"Content-Range":f"bytes 4-{len(content)-1}/{len(content)}"})
    downloads.download_one(entry(content),target,opener=resume,progress=lambda x:None)
    assert requests[0].get_header("Range")=="bytes=4-" and target.read_bytes()==content
    other=tmp_path/"other.pt"; (tmp_path/"other.pt.download.part").write_bytes(b"bad prefix")
    downloads.download_one(entry(content),other,opener=lambda *a,**k:Response(content),progress=lambda x:None)
    assert other.read_bytes()==content


def test_bad_resume_does_not_publish_or_overwrite_partial(tmp_path):
    content=b"official model bytes"; target=tmp_path/"model.pt"; partial=tmp_path/"model.pt.download.part"
    partial.write_bytes(content[:4])
    with pytest.raises(ValueError,match="resume range"):
        downloads.download_one(entry(content),target,opener=lambda *a,**k:Response(content[4:],206,{"Content-Range":"bytes 5-19/20"}))
    assert not target.exists() and partial.read_bytes()==content[:4]


def test_restart_partial_download_and_verify_readonly(tmp_path,monkeypatch):
    content=b"official model bytes"; target=tmp_path/"model.pt"; partial=tmp_path/"model.pt.download.part"
    partial.write_bytes(b"corrupt partial that is far too large")
    downloads.download_one(entry(content),target,restart_partial=True,opener=lambda *a,**k:Response(content),progress=lambda x:None)
    assert target.read_bytes()==content
    untouched=tmp_path/"verify-only"
    monkeypatch.setattr(downloads,"download_one",lambda *a,**k:pytest.fail("verify must not download"))
    assert downloads.main(["--verify","--models","pose","--output-dir",str(untouched)])==1
    assert not untouched.exists()


def test_manifest_traversal_and_injected_url_rejected(tmp_path):
    manifest=downloads.load_manifest()
    manifest["models"][0]["files"][0]["name"]="../secret"
    file=tmp_path/"manifest.json"; file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): downloads.load_manifest(file)
    manifest=downloads.load_manifest(); manifest["models"][0]["files"][0]["url"]="http://attacker.invalid/weight"
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): downloads.load_manifest(file)


def test_path_escape_and_linked_destination_rejected(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="escapes"):
        downloads.target_path(tmp_path,"..","outside.pt")
    target=tmp_path/"model.pt"; target.write_bytes(b"old")
    monkeypatch.setattr(downloads,"linked",lambda p:Path(p)==target)
    with pytest.raises(ValueError,match="linked"):
        downloads.download_one(entry(),target,replace_invalid=True)
