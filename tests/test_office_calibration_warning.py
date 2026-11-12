"""Missing seat calibration is actionable feedback, not a block on partial poses."""
import copy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import workflow_api, workflow_graph
from backend.workflow_templates import templates


def office_graph(mode="occlusion_assisted", zones=None):
    graph = copy.deepcopy(next(item["graph"] for item in templates()["templates"]
                               if item["id"] == "live-office-crossing"))
    office = next(item for item in graph["nodes"] if item["type"] == "office_monitor")
    office["params"].update(posture_mode=mode, zones=[] if zones is None else zones)
    return graph


@pytest.fixture
def api(tmp_path, monkeypatch):
    asset = tmp_path / "pose.pt"
    asset.write_bytes(b"preflight fixture only; never load a model")
    monkeypatch.setattr(workflow_graph, "asset_path", lambda *args: asset)
    monkeypatch.setattr(workflow_api, "_dependency_available", lambda _: True)
    media_file = tmp_path / "camera-demo.mp4"
    media_file.write_bytes(b"preflight does not decode media")
    media = {"id": "demo", "kind": "video", "path": str(media_file)}
    submissions = []

    def submit(request):
        submissions.append(request)
        return {"id": "office-calibration-test", "status": "queued"}

    app = FastAPI()
    manager = SimpleNamespace(app_root=tmp_path, submit=submit)
    workflow_api.register_workflow_routes(app, manager,
                                         SimpleNamespace(get=lambda key: media if key == "demo" else None),
                                         lambda: [], lambda *a, **k: {}, lambda *a, **k: {}, tmp_path)
    with TestClient(app) as client:
        yield client, submissions


def calibration_warnings(result):
    return [warning for warning in result["warnings"] if "尚未校正坐姿肩部區域" in warning]


def test_missing_calibration_warns_without_rejecting_valid_partial_pose_workflow(api):
    client, _ = api
    response = client.post("/api/workflows/validate", json={"graph": office_graph(), "media_id": "demo"})
    assert response.status_code == 200
    result = response.json()
    assert result["valid"], result["errors"]
    warnings = calibration_warnings(result)
    assert len(warnings) == 1
    assert "局部骨架" in warnings[0]
    assert "雙肩中點" in warnings[0]


@pytest.mark.parametrize("mode,zones", [
    ("strict_full_body", []),
    ("occlusion_assisted", [{"name": "座位肩部", "points": [[.2, .5], [.5, .5], [.5, .7], [.2, .7]]}]),
])
def test_strict_or_calibrated_workflows_do_not_request_calibration(api, mode, zones):
    client, _ = api
    response = client.post("/api/workflows/validate", json={"graph": office_graph(mode, zones), "media_id": "demo"})
    result = response.json()
    assert result["valid"], result["errors"]
    assert not calibration_warnings(result)


@pytest.mark.parametrize("endpoint", ["run", "live"])
def test_missing_calibration_remains_nonblocking_for_both_execution_modes(api, endpoint):
    client, submissions = api
    payload = {"graph": office_graph()}
    if endpoint == "run":
        payload["media_id"] = "demo"
    response = client.post(f"/api/workflows/{endpoint}", json=payload)
    assert response.status_code == 201, response.text
    assert len(submissions) == 1
    submitted_office = next(item for item in submissions[0]["workflow_graph"]["nodes"]
                            if item["type"] == "office_monitor")
    assert submitted_office["params"]["posture_mode"] == "occlusion_assisted"
    assert submitted_office["params"]["zones"] == []
