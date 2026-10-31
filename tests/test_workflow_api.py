import copy
import importlib
import io
import json
from pathlib import Path
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import workflow_api, workflow_graph, workflow_integrations
from backend.workflow_templates import templates


def graph(name="zone-guard"):
    return next(x["graph"] for x in templates()["templates"] if x["id"] == name)


def chosen(value, kind):
    return next(x for x in value["nodes"] if x["type"] == kind)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "module-runtime"))
    app_module = importlib.import_module("backend.app")
    definitions = {
        "yolo26x": ("detection", ["device", "conf", "imgsz", "max_det"]),
        "grounding_dino_base": ("detection", ["device", "conf", "prompt", "use_training_classes", "text_threshold"]),
        "rf_detr_2xl": ("detection", ["device", "conf", "accept_platform_license"]),
        "x3d_l": ("action", ["device", "top_k"]),
    }
    models = [{"id":mid,"task":task,"available":True,"environment_available":True,"inference_supported":True,
               "capabilities":{"input_types":["image","video"],"inference_params":[{"key":k} for k in keys]}}
              for mid,(task,keys) in definitions.items()]
    weights={}
    for mid in definitions:
        path=tmp_path/(mid+".pt"); path.write_bytes(b"fake-only-test-no-model-load"); weights[mid]=path
    checkpoints={}
    normalizations=[]
    monkeypatch.setattr(app_module,"get_models",lambda:copy.deepcopy(models))
    def normalize(mid,kind,params,*,allow_missing_weights=False):
        assert kind=="inference"
        assert not set(params)-set(definitions[mid][1])
        normalizations.append((mid,dict(params),allow_missing_weights))
        values={"device":"auto",**params}
        if mid=="grounding_dino_base": values.setdefault("text_threshold",.25)
        if mid=="rf_detr_2xl": values["accept_platform_license"]=True
        return values
    monkeypatch.setattr(app_module,"validate_job_request",normalize)
    monkeypatch.setattr(app_module.ModelRegistry,"resolve",lambda self,mid,**kw:weights[mid])
    def checkpoint(self,cid,mid,manager=None):
        if cid not in checkpoints: raise ValueError("private path /secret/weights was not found")
        return checkpoints[cid]
    monkeypatch.setattr(app_module.ModelRegistry,"checkpoint_for",checkpoint)
    monkeypatch.setattr(workflow_api,"_dependency_available",lambda name:True)
    assets=tmp_path/"assets"
    monkeypatch.setattr(workflow_graph,"asset_path",lambda mid,app_root=None:assets/mid/workflow_graph.ASSETS[mid][1])
    application=app_module.create_app(tmp_path/"runtime",tmp_path/"samples",start_worker=False)
    media=tmp_path/"movie.mp4"; media.write_bytes(b"not decoded by API")
    application.state.media.add({"id":"movie","name":"movie.mp4","kind":"video","path":str(media)})
    with TestClient(application,base_url="http://127.0.0.1") as client:
        yield {"client":client,"app":application,"models":models,"weights":weights,"checkpoints":checkpoints,
               "normalizations":normalizations,"assets":assets,"tmp":tmp_path}


def request_file(api,response):
    return json.loads((api["app"].state.jobs.job_root/response.json()["id"]/"request.json").read_text(encoding="utf-8"))


def create_assets(api,kind):
    mid={"pose":"yolo26n_pose","skeleton_action":"stgcnpp_ntu120","relation":"relateanything_convnext_large"}[kind]
    directory=api["assets"]/mid; directory.mkdir(parents=True,exist_ok=True)
    names=[workflow_graph.ASSETS[mid][1]]
    if kind=="relation": names=["relateanything.onnx","relateanything.json","predicate_bank.npz","thresholds.json"]
    for name in names: (directory/name).write_bytes(b"nonempty-test-asset-not-loaded")
    if kind=="relation":
        (directory/"thresholds.json").write_text(json.dumps({"predicates":[{"name":x} for x in ["holding","wearing","riding","next to","on"]]}),encoding="utf-8")
    return directory


def test_catalog_and_all_original_templates_are_structurally_valid(api):
    client=api["client"]
    result=client.get("/api/workflows/catalog")
    assert result.status_code==200 and len(result.json()["node_types"])>=10
    assert "path" not in result.text
    values=client.get("/api/workflows/templates").json()["templates"]
    assert len(values)>=8
    for item in values:
        assert workflow_graph.validate_graph(item["graph"])["valid"],item["id"]


def test_save_incomplete_draft_list_and_delete_exact_registered_file(api):
    client=api["client"]
    draft={"version":1,"nodes":[{"id":"input","type":"source"}],"edges":[]}
    saved=client.post("/api/workflows",json={"name":" 未接完的草稿 ","graph":draft})
    assert saved.status_code==201 and saved.json()["name"]=="未接完的草稿"
    saved_id=saved.json()["id"]
    folder=api["app"].state.runtime/"workflows"
    bad_id=uuid.uuid4().hex; (folder/(bad_id+".json")).write_text("[]",encoding="utf-8")
    (folder/"unregistered.json").write_text("{}",encoding="utf-8")
    assert [x["id"] for x in client.get("/api/workflows").json()["workflows"]]==[saved_id]
    assert client.delete("/api/workflows/"+bad_id).status_code==404
    assert client.delete("/api/workflows/unregistered").status_code==404
    assert (folder/"unregistered.json").exists()
    assert client.delete("/api/workflows/"+saved_id).json()["deleted"] is True
    assert not (folder/(saved_id+".json")).exists()


def test_validate_does_not_enqueue_or_publish_resolved_paths(api):
    result=api["client"].post("/api/workflows/validate",json={"graph":graph(),"media_id":"movie"})
    assert result.status_code==200 and result.json()["valid"],result.text
    assert result.json()["plan"]["validation_scope"]=="graph_models_media"
    assert api["app"].state.jobs.list()==[]
    assert "resolved_nodes" not in result.text and str(api["tmp"]) not in result.text
    without_media=api["client"].post("/api/workflows/validate",json={"graph":graph()}).json()
    assert without_media["valid"] and without_media["plan"]["validation_scope"]=="graph_models"
    assert any("media_id" in x for x in without_media["warnings"])


def test_run_stores_server_resolution_and_supports_existing_cancel(api):
    result=api["client"].post("/api/workflows/run",json={"graph":graph(),"media_id":"movie","options":{"device":"cpu","frame_stride":4}})
    assert result.status_code==201,result.text
    request=request_file(api,result)
    assert request["kind"]==request["model_id"]=="workflow"
    assert request["runtime_path"]==str(api["app"].state.runtime)
    assert "runtime_path" not in result.text
    assert request["params"]=={"device":"cpu","frame_stride":4,"max_frames":300,"max_side":1280}
    resolved=request["resolved_nodes"]["detect"]
    assert Path(resolved["model_weight_path"])==api["weights"]["yolo26x"]
    assert resolved["params"]=={"device":"cpu","conf":.25,"imgsz":640,"max_det":40}
    assert "checkpoint_path" not in result.text and "model_weight_path" not in result.text
    cancelled=api["client"].post("/api/jobs/"+result.json()["id"]+"/cancel")
    assert cancelled.status_code==200 and cancelled.json()["status"]=="cancelled"


def test_action_only_uses_existing_normalizer_device_and_top_k(api):
    result=api["client"].post("/api/workflows/run",json={"graph":graph("rgb-action-rules"),"media_id":"movie"})
    assert result.status_code==201,result.text
    request=request_file(api,result)
    assert request["resolved_nodes"]["action"]["params"]=={"device":"auto","top_k":5}
    assert chosen(request["workflow_graph"],"video_action")["params"]["window_seconds"]==2


def test_rf_license_is_normalizer_state_not_new_checkbox(api):
    value=graph(); chosen(value,"detector")["params"]["model_id"]="rf_detr_2xl"
    result=api["client"].post("/api/workflows/run",json={"graph":value,"media_id":"movie"})
    assert result.status_code==201,result.text
    assert request_file(api,result)["resolved_nodes"]["detect"]["params"]["accept_platform_license"] is True


def test_checkpoint_works_without_base_weight_and_keeps_grounding_training_classes(api):
    value=graph(); params=chosen(value,"detector")["params"]
    params.update(model_id="grounding_dino_base",checkpoint_id="trained",use_training_classes=True)
    directory=api["tmp"]/"gd-checkpoint"; directory.mkdir()
    (directory/"workbench_training.json").write_text(json.dumps({"classes":["helmet","vest"]}),encoding="utf-8")
    api["checkpoints"]["trained"]=directory
    next(x for x in api["models"] if x["id"]=="grounding_dino_base")["available"]=False
    result=api["client"].post("/api/workflows/run",json={"graph":value,"media_id":"movie"})
    assert result.status_code==201,result.text
    request=request_file(api,result); resolved=request["resolved_nodes"]["detect"]
    assert resolved["checkpoint_path"]==str(directory) and "model_weight_path" not in resolved
    assert resolved["params"]["use_training_classes"] is True and resolved["params"]["text_threshold"]==.25
    assert api["normalizations"][-1][2] is True


def test_dual_bundle_rejected_without_exposing_path(api):
    directory=api["tmp"]/"private-dual"; directory.mkdir()
    (directory/"branch_bundle.json").write_text("{}",encoding="utf-8")
    api["checkpoints"]["dual"]=directory
    value=graph(); chosen(value,"detector")["params"]["checkpoint_id"]="dual"
    result=api["client"].post("/api/workflows/run",json={"graph":value,"media_id":"movie"})
    assert result.status_code==422 and "單模型" in result.text
    assert "private-dual" not in result.text and not api["app"].state.jobs.list()


def test_invalid_rules_wrong_task_and_missing_media_do_not_queue(api):
    value=graph(); chosen(value,"zone")["params"]["zones"]=[{"name":"bad","points":[[0,0],[0,0],[0,0]]}]
    invalid=api["client"].post("/api/workflows/validate",json={"graph":value,"media_id":"movie"})
    assert invalid.status_code==200 and not invalid.json()["valid"]
    assert "nonzero" in invalid.text
    value=graph(); chosen(value,"detector")["params"]["model_id"]="x3d_l"
    assert api["client"].post("/api/workflows/run",json={"graph":value,"media_id":"movie"}).status_code==422
    assert api["client"].post("/api/workflows/run",json={"graph":graph(),"media_id":"missing"}).status_code==422
    api["weights"]["yolo26x"].unlink()
    missing=api["client"].post("/api/workflows/validate",json={"graph":graph(),"media_id":"movie"})
    assert not missing.json()["valid"] and not api["app"].state.jobs.list()
    assert str(api["tmp"]) not in missing.text


def test_asset_files_and_dependencies_are_preflighted(api,monkeypatch):
    value=graph("posture-fall")
    missing=api["client"].post("/api/workflows/validate",json={"graph":value,"media_id":"movie"})
    assert not missing.json()["valid"] and "yolo26n-pose.pt" in missing.text
    create_assets(api,"pose")
    queued=api["client"].post("/api/workflows/run",json={"graph":value,"media_id":"movie"})
    assert queued.status_code==201,queued.text
    assert request_file(api,queued)["resolved_nodes"]["pose"]["weight_path"].endswith("yolo26n-pose.pt")
    relation=graph("object-relations"); directory=create_assets(api,"relation")
    (directory/"predicate_bank.npz").unlink()
    missing=api["client"].post("/api/workflows/validate",json={"graph":relation,"media_id":"movie"})
    assert not missing.json()["valid"] and "predicate_bank.npz" in missing.text
    create_assets(api,"relation")
    monkeypatch.setattr(workflow_api,"_dependency_available",lambda name:name!="onnxruntime")
    missing=api["client"].post("/api/workflows/validate",json={"graph":relation,"media_id":"movie"})
    assert not missing.json()["valid"] and "onnxruntime" in missing.text
    assert api["client"].post("/api/workflows/validate",json={"graph":graph(),"media_id":"movie"}).json()["valid"]


def test_vlm_unavailable_blocks_only_vlm_graph_and_uses_saved_model_id(api,monkeypatch):
    blocked=api["client"].post("/api/workflows/validate",json={"graph":graph("cloud-review"),"media_id":"movie"})
    assert not blocked.json()["valid"] and "雲端" in blocked.text
    assert api["client"].post("/api/workflows/validate",json={"graph":graph(),"media_id":"movie"}).json()["valid"]
    monkeypatch.setattr(workflow_integrations,"public_config",lambda runtime:{"munolith":{"available":True,"model_id":"vision-model","models":["vision-model"],"key_file_path":"secret-file","api_key":"sk-secret"}})
    result=api["client"].post("/api/workflows/run",json={"graph":graph("cloud-review"),"media_id":"movie"})
    assert result.status_code==201,result.text
    saved=request_file(api,result)
    assert saved["resolved_nodes"]["review"]=={"model_id":"vision-model"}
    assert "sk-secret" not in json.dumps(saved) and "secret-file" not in json.dumps(saved)
    public=api["client"].get("/api/workflows/integrations")
    assert "sk-secret" not in public.text and "key_file_path" not in public.text


def test_relation_unknown_predicates_fail_before_model_load(api):
    create_assets(api,"relation")
    value=graph("object-relations"); chosen(value,"relation")["params"]["predicates"]=["imaginary predicate"]
    result=api["client"].post("/api/workflows/run",json={"graph":value,"media_id":"movie"})
    assert result.status_code==422 and "預載詞庫" in result.text and not api["app"].state.jobs.list()


def test_integration_uses_fixed_service_and_never_returns_key_or_file(api,monkeypatch):
    secret="sk-test-only-sensitive-123456789"
    keyfile=api["tmp"]/"private-key.txt"; keyfile.write_text("API_KEY="+secret,encoding="utf-8")
    calls=[]
    class Response:
        status_code=200
        def json(self): return {"data":[{"id":"vision-model"}]}
    class Session:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def get(self,url,**kwargs):
            calls.append((url,kwargs)); return Response()
    monkeypatch.setattr(workflow_integrations,"_session",Session)
    result=api["client"].post("/api/workflows/integrations/munolith",json={"key_file_path":str(keyfile),"model_id":"vision-model"})
    assert result.status_code==200 and result.json()["munolith"]["available"]
    assert calls[0][0]==workflow_integrations.BASE_URL+"/models"
    assert calls[0][1]["headers"]["Authorization"]=="Bearer "+secret
    assert secret not in result.text and str(keyfile) not in result.text and "key_file_path" not in result.text
    stored=(api["app"].state.runtime/"workflow_integrations.json").read_text(encoding="utf-8")
    assert secret not in stored
    invalid=api["client"].post("/api/workflows/integrations/munolith",json={"key_file_path":str(keyfile),"base_url":"https://evil.invalid","api_key":secret})
    assert invalid.status_code==422 and secret not in invalid.text and len(calls)==1


def test_unreadable_integration_file_errors_do_not_echo_paths(api):
    result=api["client"].post("/api/workflows/integrations/munolith",json={"key_file_path":str(api["tmp"]/"sensitive-private-name")})
    assert result.status_code==422 and "sensitive-private-name" not in result.text


@pytest.mark.parametrize("extra", [{"resolved_nodes":{"secret":"private"}},{"api_key":"sk-secret"},{"options":{"max_frames":True}},
    {"options":{"max_frames":0}},{"options":{"device":"cuda:999"}},{"options":{"arbitrary":"bad"}}])
def test_run_forbids_extra_fields_and_bad_options_without_echoing_secrets(api,extra):
    result=api["client"].post("/api/workflows/run",json={"graph":graph(),"media_id":"movie",**extra})
    assert result.status_code==422 and "sk-secret" not in result.text and not api["app"].state.jobs.list()


def test_invalid_graph_secret_fields_not_echoed_in_validate(api):
    value=graph(); chosen(value,"detector")["params"]["weight_path"]="/sensitive/private/model.pt"
    result=api["client"].post("/api/workflows/validate",json={"graph":value})
    assert result.status_code==200 and not result.json()["valid"] and result.json()["graph"] is None
    assert "/sensitive/private" not in result.text


def test_live_reuses_existing_ingress_status_images_and_stop(api):
    client=api["client"]
    result=client.post("/api/workflows/live",json={"graph":graph(),"capture_fps":10,"max_duration_seconds":None})
    assert result.status_code==201,result.text
    saved=request_file(api,result); jid=result.json()["id"]
    assert saved["kind"]=="live" and saved["model_id"]=="workflow" and saved["max_duration_seconds"] is None
    assert saved["runtime_path"]==str(api["app"].state.runtime) and "runtime_path" not in result.text
    assert saved["models"]==[{"key":"0","model_id":"workflow","params":{}}]
    image=io.BytesIO(); Image.new("RGB",(24,20),"red").save(image,format="JPEG")
    uploaded=client.post(f"/api/live/sessions/{jid}/frames",files={"frame":("camera.jpg",image.getvalue(),"image/jpeg")})
    assert uploaded.status_code==202,uploaded.text
    status=client.get(f"/api/live/sessions/{jid}")
    assert status.status_code==200 and status.json()["model_id"]=="workflow"
    assert client.get(f"/api/live/sessions/{jid}/images/0").status_code==404  # No fake output before the worker.
    stopped=client.post(f"/api/live/sessions/{jid}/stop")
    assert stopped.status_code==200 and stopped.json()["status"]=="cancelled"


def test_live_busy_queue_is_conflict_without_new_job(api):
    client=api["client"]
    assert client.post("/api/workflows/run",json={"graph":graph(),"media_id":"movie"}).status_code==201
    result=client.post("/api/workflows/live",json={"graph":graph()})
    assert result.status_code==409 and len(api["app"].state.jobs.list())==1


def test_preflight_does_not_import_ml_frameworks(api,monkeypatch):
    import builtins
    actual=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split(".")[0] in {"torch","transformers","ultralytics","onnxruntime"}:
            raise AssertionError("API must not import model runtime: "+name)
        return actual(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    create_assets(api,"pose"); create_assets(api,"skeleton_action"); create_assets(api,"relation")
    for name in ("zone-guard","posture-fall","skeleton-actions","object-relations"):
        result=api["client"].post("/api/workflows/validate",json={"graph":graph(name),"media_id":"movie"})
        assert result.status_code==200 and result.json()["valid"],result.text


def test_registered_file_link_is_never_deleted(api,monkeypatch):
    result=api["client"].post("/api/workflows",json={"name":"Saved","graph":graph()})
    ident=result.json()["id"]; path=api["app"].state.runtime/"workflows"/(ident+".json")
    actual=workflow_api._linked
    monkeypatch.setattr(workflow_api,"_linked",lambda value:Path(value)==path or actual(value))
    assert api["client"].delete("/api/workflows/"+ident).status_code==404
    assert path.is_file()
