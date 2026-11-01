import copy

from fastapi.testclient import TestClient
import pytest

from backend.workflow_templates import templates


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKBENCH_RUNTIME", str(tmp_path / "bootstrap"))
    from backend.app import create_app
    with TestClient(create_app(tmp_path / "runtime", tmp_path / "samples", start_worker=False), base_url="http://127.0.0.1:8765") as value:
        yield value


def graph():
    return copy.deepcopy(next(t["graph"] for t in templates()["templates"] if t["id"] == "live-multi-model"))


def test_launch_keeps_multi_model_graph_and_never_starts_camera_or_job(client):
    response = client.post("/api/comfy/workflows/launch", json={"graph": graph(), "source_mode": "vdo", "options": {"device": "cpu"}})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["viewer_url"] == f"http://127.0.0.1:8765/?embedded_workflow={result['workflow_id']}&source=vdo"
    saved = client.get("/api/comfy/workflows/" + result["workflow_id"]).json()
    assert saved["source_mode"] == "vdo" and saved["options"]["device"] == "cpu"
    assert sum(node["type"] == "detector" for node in saved["graph"]["nodes"]) == 2
    assert sum(node["type"] == "pose" for node in saved["graph"]["nodes"]) == 1
    assert client.get("/api/jobs").json()["jobs"] == []


def test_embed_is_limited_to_comfy_origin_and_normal_pages_keep_deny(client):
    ident = client.post("/api/comfy/workflows/launch", json={"graph": graph(), "source_mode": "camera"}).json()["workflow_id"]
    embedded = client.get("/?embedded_workflow=" + ident, headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "iframe", "Sec-Fetch-Mode": "navigate"})
    assert embedded.status_code == 200
    assert "X-Frame-Options" not in embedded.headers
    assert embedded.headers["Content-Security-Policy"] == "frame-ancestors 'self' http://127.0.0.1:8188 http://localhost:8188"
    assert client.get("/").headers["X-Frame-Options"] == "DENY"
    assert client.get("/?embedded_workflow=bad", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "iframe"}).status_code == 403
    assert client.get("/api/comfy/templates", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_malformed_graph_or_source_is_rejected_without_saving(client):
    invalid = graph()
    invalid["edges"] = []
    assert client.post("/api/comfy/workflows/launch", json={"graph": invalid, "source_mode": "camera"}).status_code == 422
    assert client.post("/api/comfy/workflows/launch", json={"graph": graph(), "source_mode": "remote", "url": "https://example.com"}).status_code == 422
    assert client.get("/api/comfy/workflows/" + "0" * 32).status_code == 404


def test_all_templates_export_native_comfy_nodes_instead_of_one_opaque_node(client):
    response = client.get("/api/comfy/templates?source_mode=camera")
    assert response.status_code == 200, response.text
    values = response.json()["templates"]
    assert len(values) >= 15
    for template in values:
        native = template["workflow"]
        assert len(native["nodes"]) == len(template["graph"]["nodes"])
        assert all(node["type"].startswith("VisionWB_") for node in native["nodes"])
        source = next(node for node in native["nodes"] if node["type"] == "VisionWB_source")
        assert source["widgets_values"][0] == "camera"
