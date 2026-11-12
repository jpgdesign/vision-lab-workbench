"""Cross-component contracts for office and handwashing examples (CPU fakes)."""
import copy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageChops

from backend import workflow_api
from backend.comfy_workflows import WIRE_TYPES, convert_workflow, to_api_prompt
from backend.workflow_graph import ASSETS, CATALOG, asset_path, validate_graph
from backend.workflow_hands import HAND_MODEL_KINDS, HandModelNode
from backend.workflow_runtime import WorkflowEngine
from backend.workflow_templates import build, templates


TEMPLATE_IDS = ("live-office-crossing", "handwashing-hybrid", "handwashing-detection-mask")


def example(identifier):
    return copy.deepcopy(next(t for t in templates()["templates"] if t["id"] == identifier))


@pytest.mark.parametrize("identifier", TEMPLATE_IDS)
def test_new_examples_validate_with_defaults_and_preserve_chinese_titles(identifier):
    template = example(identifier)
    checked = validate_graph(template["graph"])
    assert checked["valid"], checked["errors"]
    assert checked["plan"]["order"][0] == "input"
    assert checked["plan"]["order"][-1] == "out"
    assert any("\u4e00" <= ch <= "\u9fff" for ch in template["name"])
    for item in checked["graph"]["nodes"]:
        assert set(item["params"]) == {p["key"] for p in CATALOG[item["type"]]["params"]}


@pytest.mark.parametrize("identifier", TEMPLATE_IDS)
def test_comfy_exports_have_real_typed_links_and_complete_widgets(identifier):
    template = example(identifier)
    document = convert_workflow(template["graph"], name=template["name"], source_mode="camera")
    api_prompt = to_api_prompt(template["graph"], name=template["name"], source_mode="camera")
    assert document["groups"][0]["title"] == template["name"]
    assert document["extra"]["vision_workbench"]["source_mode"] == "camera"
    nodes = {n["id"]: n for n in document["nodes"]}
    for link_id, source_id, source_slot, target_id, target_slot, wire_type in document["links"]:
        output = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert output["type"] == target["type"] == wire_type
        assert link_id in output["links"]
        assert target["link"] == link_id
        assert api_prompt[str(target_id)]["inputs"][target["name"]] == [str(source_id), source_slot]
    for item in nodes.values():
        kind = item["type"].removeprefix("VisionWB_")
        assert item["title"] == CATALOG[kind]["label"]
        if kind == "source":
            assert api_prompt[str(item["id"])]["inputs"]["source_mode"] == "camera"
            continue
        assert len(item["widgets_values"]) == len(CATALOG[kind]["params"])
        for definition in CATALOG[kind]["params"]:
            assert definition["key"] in api_prompt[str(item["id"])]["inputs"]
        if kind in HAND_MODEL_KINDS:
            assert all(output["type"] == "VWB_OBSERVATIONS" for output in item["outputs"])
            assert all(output["type"] != "IMAGE" for output in item["outputs"])


def test_hybrid_hand_mask_and_body_skeleton_branches_remain_separate():
    template = example("handwashing-hybrid")
    graph = template["graph"]
    edges = {(e["source"], e["sourceHandle"], e["target"], e["targetHandle"]) for e in graph["edges"]}
    assert ("hands", "poses", "mask", "objects") in edges
    assert ("hands", "poses", "wash", "hands") in edges
    assert ("pose", "poses", "track", "observations") in edges
    assert ("track", "tracks", "skeleton", "tracks") in edges
    assert not any(a == "hands" and c in {"track", "skeleton"} for a, _, c, _ in edges)
    doc = convert_workflow(graph, name=template["name"])
    wash = next(n for n in doc["nodes"] if n["properties"]["vision_workbench_id"] == "wash")
    actions = [p for p in wash["inputs"] if p["name"].startswith("actions") and p["link"] is not None]
    assert [p["name"] for p in actions] == ["actions", "actions_2"]
    assert all(p["type"] == WIRE_TYPES["actions"] for p in actions)


def test_cross_field_invalid_office_configuration_fails_graph_preflight():
    graph = example("live-office-crossing")["graph"]
    office = next(n for n in graph["nodes"] if n["type"] == "office_monitor")
    office["params"]["line_x"] = .01  # individually in range; inside hysteresis margin
    checked = validate_graph(graph)
    assert not checked["valid"]
    assert any("office" in error for error in checked["errors"])


def test_real_engine_keeps_tracker_identity_counts_directions_and_renders_overlay():
    graph = example("live-office-crossing")["graph"]
    next(n for n in graph["nodes"] if n["type"] == "office_monitor")["params"]["cooldown_seconds"] = .5
    centers = [.25, .25, .35, .45, .55, .65, .65, .55, .45, .35, .35]
    loaded = []

    class FakePose:
        def __init__(self, kind, params, resolved, device):
            assert kind == "pose"
            loaded.append(kind)
            self.actual_device = "cpu"

        def process(self, inputs, frame):
            x = centers[frame["frame_index"]] * frame["width"]
            item = {"box": [x - 120, 150., x + 120, 650.], "label": "person", "score": .95,
                    "keypoints": [[0., 0., 0.] for _ in range(17)], "observed": True}
            return {"poses": {"items": [item], "frame": frame, "timestamp": frame["timestamp"], "fresh": True}}

        def reset(self):
            pass

    engine = WorkflowEngine({"id": "office-integration", "workflow_graph": graph, "params": {"device": "cpu"}}, model_factory=FakePose)
    engine.device = "cpu"  # Exercise the engine without importing ML device discovery.
    source = Image.new("RGB", (1000, 800), "black")
    seen_events, identities = [], set()
    for index in range(len(centers)):
        rendered, events, observations = engine.process(source, index * .2, index)
        seen_events.extend(events)
        identities.add(observations["track"]["tracks"]["items"][0]["track_key"])
    assert loaded == ["pose"]
    assert len(identities) == 1
    assert [e["label"] for e in seen_events if e["kind"] == "line_crossing"] == ["出去", "進來"]
    assert all(e["node_id"] == "office" for e in seen_events)
    assert observations["office"]["events"]["counts"] == {"in": 1, "out": 1}
    assert observations["office"]["actions"]["items"][0]["kind"] == "unknown"
    assert observations["office"]["actions"]["items"][0]["infers_attention"] is False
    assert rendered.size == source.size and ImageChops.difference(source, rendered).getbbox() is not None
    assert "frame" not in observations["office"]["actions"]
    assert engine.event_count == 2
    # Rewinding a video clears tracking/crossing evidence; the old count cannot
    # contaminate a fresh replay or count its first appearance as an entry.
    _, events, replay = engine.process(source, 0., 0)
    assert not events
    assert replay["office"]["events"]["counts"] == {"in": 0, "out": 0}
    assert replay["track"]["tracks"]["items"][0]["track_key"] not in identities


def hand_graph(mask=True):
    nodes = [("src", "source", 0, 0, {}), ("hand", "hand_landmarks", 350, 0, {}), ("out", "output", 1100, 0, {})]
    edges = [("src", "frame", "hand", "frame"), ("src", "frame", "out", "frame"), ("hand", "poses", "out", "observations")]
    if mask:
        nodes.append(("mask", "prompt_mask", 700, 0, {}))
        edges.extend([("src", "frame", "mask", "frame"), ("hand", "poses", "mask", "objects"), ("mask", "detections", "out", "observations")])
    return build("hands-test", "手部測試", "", nodes, edges, [])["graph"]


@pytest.fixture
def preflight_client(tmp_path):
    app = FastAPI()
    workflow_api.register_workflow_routes(app, SimpleNamespace(app_root=tmp_path),
                                         SimpleNamespace(get=lambda _: None, list=lambda: []),
                                         lambda: [], lambda *a, **k: {}, lambda *a, **k: {}, tmp_path)
    with TestClient(app) as client:
        yield client, tmp_path


def test_preflight_reports_both_missing_hand_assets_without_loading_models(preflight_client, monkeypatch):
    client, _ = preflight_client
    monkeypatch.setattr(workflow_api, "_dependency_available", lambda _: True)
    response = client.post("/api/workflows/validate", json={"graph": hand_graph()})
    assert response.status_code == 200
    checked = response.json()
    assert not checked["valid"]
    errors = " ".join(checked["errors"])
    assert ASSETS["mediapipe_hand_landmarker"][1] in errors
    assert ASSETS["sam2_1_tiny"][1] in errors
    assert "缺少模型資產" in errors


def test_preflight_reports_missing_dependencies_and_accepts_prepared_assets(preflight_client, monkeypatch):
    client, root = preflight_client
    for model in ("mediapipe_hand_landmarker", "sam2_1_tiny"):
        path = asset_path(model, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"isolated preflight fixture; must never be loaded")
    monkeypatch.setattr(workflow_api, "_dependency_available", lambda name: name not in {"mediapipe", "pycocotools"})
    unavailable = client.post("/api/workflows/validate", json={"graph": hand_graph()}).json()
    assert not unavailable["valid"]
    assert "mediapipe" in " ".join(unavailable["errors"])
    assert "pycocotools" in " ".join(unavailable["errors"])
    monkeypatch.setattr(workflow_api, "_dependency_available", lambda _: True)
    prepared = client.post("/api/workflows/validate", json={"graph": hand_graph()}).json()
    assert prepared["valid"], prepared["errors"]
    planned = {n["id"]: n for n in prepared["plan"]["nodes"]}
    assert planned["hand"]["model_id"] == "mediapipe_hand_landmarker"
    assert planned["hand"]["device"] == "cpu"
    assert planned["mask"]["model_id"] == "sam2_1_tiny"


@pytest.mark.parametrize("kind", sorted(HAND_MODEL_KINDS))
def test_hand_model_missing_assets_are_explicit_errors_without_download(tmp_path, kind):
    model = HandModelNode(kind, {}, {"weight_path": str(tmp_path / "not-downloaded")}, device="cpu")
    with pytest.raises(ValueError, match="模型檔案不存在"):
        model._load()
    assert model.model is None
    assert not list(tmp_path.iterdir())
