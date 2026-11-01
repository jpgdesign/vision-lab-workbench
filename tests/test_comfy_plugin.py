"""Exercise native graph packets and bridge execution without loading ComfyUI/ML."""
import copy
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
from PIL import Image

from backend.comfy_workflows import to_api_prompt
from backend.workflow_graph import CATALOG, validate_graph
from backend.workflow_templates import templates


PLUGIN = Path(__file__).resolve().parents[1] / "comfyui" / "vision_workbench"
PACKAGE = "_vision_workbench_plugin_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN)]
sys.modules[PACKAGE] = package
client = importlib.import_module(PACKAGE + ".client")
schema = importlib.import_module(PACKAGE + ".schema")
graph = importlib.import_module(PACKAGE + ".graph")
# Registration normally uses the local API; unit tests use its shipped schema.
original_catalogue = client.catalogue
client.catalogue = lambda: json.loads((PLUGIN / "catalog.json").read_text(encoding="utf-8"))
nodes = importlib.import_module(PACKAGE + ".nodes")
client.catalogue = original_catalogue


@pytest.mark.parametrize("template", templates()["templates"], ids=lambda value: value["id"])
def test_every_template_executes_native_graph_packets(template, monkeypatch):
    captured = []
    monkeypatch.setattr(nodes, "execute_output", lambda value, unique_id: captured.append(value) or (value,))
    prompt = to_api_prompt(template, media_id="sample")
    resolved = {}
    def run(ident):
        if ident in resolved:
            return resolved[ident]
        item = prompt[ident]
        values = {key: run(value[0])[value[1]] if isinstance(value, list) else value for key, value in item["inputs"].items()}
        resolved[ident] = nodes.NODE_CLASS_MAPPINGS[item["class_type"]]().build(unique_id=ident, **values)
        return resolved[ident]
    for ident, item in prompt.items():
        if item["class_type"] == "VisionWB_output": run(ident)
    assert len(captured) == 1
    built = graph.as_graph(captured[0])
    checked = validate_graph(built)
    assert checked["valid"], checked["errors"]
    assert len(built["nodes"]) == len(template["graph"]["nodes"])
    assert len(built["edges"]) == len(template["graph"]["edges"])
    assert captured[0].source["media_id"] == "sample"


def test_registration_matches_all_catalogue_types():
    assert set(nodes.NODE_CLASS_MAPPINGS) == {"VisionWB_" + kind for kind in CATALOG}
    assert nodes.NODE_CLASS_MAPPINGS["VisionWB_embedding"].RETURN_TYPES == ("VWB_SIGNAL", "VWB_SIGNAL")
    assert nodes.NODE_CLASS_MAPPINGS["VisionWB_output"].RETURN_TYPES == ("IMAGE", "STRING")


def test_model_image_size_and_counts_remain_integers_in_native_widgets():
    resources = {"models": [], "checkpoints": [], "media": [], "integration": {}}
    detector = schema.input_types(CATALOG["detector"], resources)["required"]
    assert detector["imgsz"] == ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32})
    assert detector["max_det"][0] == "INT"
    assert detector["conf"][0] == "FLOAT"
    action = schema.input_types(CATALOG["video_action"], resources)["required"]
    assert action["window_seconds"][0] == "FLOAT"
    assert action["max_people"][0] == "INT"


def test_source_fragments_never_store_camera_urls_or_credentials():
    value = graph.source_value("1", "vdo", "unused | sample", max_frames=40)
    assert set(value.source) == {"node_id", "source_mode", "media_id", "options"}
    assert set(value.source["options"]) == {"device", "frame_stride", "max_frames", "max_side"}
    assert "vdo" not in json.dumps(graph.as_graph(value))
    with pytest.raises(TypeError):
        graph.source_value("1", "vdo", view_url="https://vdo.ninja/?view=private")


def test_graph_rejects_different_sources_and_semantically_incompatible_signal():
    first = graph.source_value("1")
    second = graph.source_value("2")
    pose = graph.build_values(CATALOG["pose"], "3", {"frame": second})[0]
    with pytest.raises(ValueError, match="同一個影像來源"):
        graph.build_values(CATALOG["output"], "4", {"frame": first, "observations": pose})
    embedding = graph.build_values(CATALOG["embedding"], "5", {"frame": first})[0]
    with pytest.raises(ValueError, match="資料類型"):
        graph.build_values(CATALOG["condition"], "6", {"actions": embedding})


def test_merge_deduplicates_shared_ancestry_and_preserves_input_fragments():
    source = graph.source_value("1")
    before = copy.deepcopy(source)
    a = graph.build_values(CATALOG["detector"], "2", {"frame": source, "model_id": "yolo26x | YOLO26x"})[0]
    b = graph.build_values(CATALOG["detector"], "3", {"frame": source, "model_id": "yolov12x | YOLOv12x"})[0]
    final = graph.build_values(CATALOG["output"], "4", {"frame": source, "observations": a, "observations_2": b})[0]
    assert source == before
    assert len(final.nodes) == 4 and len(final.edges) == 5
    assert len({edge["id"] for edge in graph.as_graph(final)["edges"]}) == 5
    with pytest.raises(ValueError, match="重複"):
        graph.build_values(CATALOG["output"], "4", {"frame": source, "observations": a, "observations_2": a})


def test_invalid_json_is_actionable():
    source = graph.source_value("1")
    detection = graph.build_values(CATALOG["detector"], "2", {"frame": source})[0]
    with pytest.raises(ValueError, match="有效 JSON"):
        graph.build_values(CATALOG["tracker"], "3", {"observations": detection, "labels": "not json"})


@pytest.mark.parametrize("value", ["https://example.com/api/artifacts/x", "http://localhost:8765/api/artifacts/x", "/api/models", "/api/artifacts/../secrets", "/api/artifacts/x?download=y"])
def test_artifacts_are_limited_to_local_output_route(value):
    with pytest.raises(ValueError): client.artifact_path(value)


def _output_value(mode="media"):
    source = graph.source_value("1", mode, "sample")
    return graph.build_values(CATALOG["output"], "2", {"frame": source})[0]


def test_live_output_prepares_viewer_without_starting_capture(monkeypatch):
    calls = []
    def request(method, path, payload=None, **kwargs):
        calls.append((method, path, payload))
        return {"viewer_url": client.BASE_URL + "/?embedded_workflow=abc123&source=camera"}
    monkeypatch.setattr(client, "request_json", request)
    monkeypatch.setattr(nodes, "_image_tensor", lambda image: image.size)
    result = nodes.execute_output(_output_value("camera"), "2")
    assert [item[1] for item in calls] == ["/api/comfy/workflows/launch"]
    assert calls[0][2]["source_mode"] == "camera"
    assert result["ui"]["vision_workbench"][0]["mode"] == "camera"
    assert "尚未取得影格" in result["result"][1]


def test_file_output_returns_actual_downloaded_image(monkeypatch, tmp_path):
    calls = []
    def request(method, path, payload=None, **kwargs):
        calls.append(path)
        return {"valid": True} if path.endswith("validate") else {"id": "job123"}
    monkeypatch.setattr(client, "request_json", request)
    monkeypatch.setattr(client, "wait_for_job", lambda *args: {"status": "completed", "result": {"media_url": "/api/artifacts/job123/image.jpg", "media_kind": "image", "frames": 1}})
    def download(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (27, 19), (20, 40, 60)).save(target)
    monkeypatch.setattr(client, "download_artifact", download)
    monkeypatch.setattr(nodes, "_output_paths", lambda: (tmp_path / "temp", tmp_path / "output"))
    monkeypatch.setattr(nodes, "_image_tensor", lambda image: (image.size, image.getpixel((0, 0))))
    result = nodes.execute_output(_output_value(), "2")
    assert calls == ["/api/workflows/validate", "/api/workflows/run"]
    assert result["result"][0][0] == (27, 19)
    assert json.loads(result["result"][1])["frames"] == 1
    assert (tmp_path / "temp" / "vision-workbench" / result["ui"]["images"][0]["filename"]).is_file()
    assert not list((tmp_path / "temp").rglob("*.jpg"))


def test_interrupt_cancels_remote_work(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "request_json", lambda method, path, **kwargs: calls.append((method, path)))
    def interrupt(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): client.wait_for_job("job1", interrupt)
    assert calls == [("POST", "/api/jobs/job1/cancel")]
