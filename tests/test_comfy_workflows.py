import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

from backend.comfy_workflows import PREFIX, SOURCE_WIDGETS, convert_workflow, input_slots, output_slots, to_api_prompt
from backend.workflow_graph import CATALOG, validate_graph
from backend.workflow_templates import templates


def _graph(ident="zone-guard"):
    return next(item["graph"] for item in templates()["templates"] if item["id"] == ident)


def _roundtrip(original, document, prompt):
    """Decode actual Comfy prompt links/widgets into the original typed graph."""
    ids = document["extra"]["vision_workbench"]["node_ids"]
    nodes, edges = [], []
    for ident, item in prompt.items():
        kind = item["class_type"].removeprefix(PREFIX)
        params = {}
        for definition in CATALOG[kind]["params"]:
            value = item["inputs"][definition["key"]]
            if definition["type"] == "json": value = json.loads(value)
            elif definition["key"] == "checkpoint_id" and value == "原始預訓練權重": value = ""
            elif definition["key"] == "model_id" and value == "使用已設定的 API 模型": value = ""
            params[definition["key"]] = value
        nodes.append({"id": ids[ident], "type": kind, "params": params})
        for slot in input_slots(kind):
            if slot["name"] not in item["inputs"]: continue
            source, out_slot = item["inputs"][slot["name"]]
            source_kind = prompt[source]["class_type"].removeprefix(PREFIX)
            edges.append({"source": ids[source], "sourceHandle": output_slots(source_kind)[out_slot]["name"],
                          "target": ids[ident], "targetHandle": slot["port"]})
    result = validate_graph({"version": 1, "nodes": nodes, "edges": edges})
    assert result["valid"], result["errors"]
    assert {n["id"]: (n["type"], n["params"]) for n in result["graph"]["nodes"]} == {n["id"]: (n["type"], n["params"]) for n in original["nodes"]}
    endpoints = lambda values: {(e["source"], e["sourceHandle"], e["target"], e["targetHandle"]) for e in values}
    assert endpoints(result["graph"]["edges"]) == endpoints(original["edges"])


@pytest.mark.parametrize("template", templates()["templates"], ids=lambda template: template["id"])
def test_templates_preserve_all_nodes_parameters_and_typed_edges(template):
    original = copy.deepcopy(template["graph"])
    doc = convert_workflow(template, media_id="sample-abc")
    prompt = to_api_prompt(template, media_id="sample-abc")
    assert template["graph"] == original
    assert doc["version"] == .4 and len(doc["nodes"]) == len(original["nodes"])
    assert len(doc["links"]) == len(original["edges"])
    assert doc == convert_workflow(template, media_id="sample-abc")
    _roundtrip(original, doc, prompt)
    for edge_id, source, source_slot, target, target_slot, kind in doc["links"]:
        a = next(node for node in doc["nodes"] if node["id"] == source)
        b = next(node for node in doc["nodes"] if node["id"] == target)
        assert edge_id in a["outputs"][source_slot]["links"]
        assert b["inputs"][target_slot]["link"] == edge_id
        assert kind == a["outputs"][source_slot]["type"] == b["inputs"][target_slot]["type"]


def test_all_30_catalog_nodes_can_be_exported_not_only_template_subset():
    covered = {node["type"] for template in templates()["templates"] for node in template["graph"]["nodes"]}
    for kind in set(CATALOG) - covered:
        if kind == "merge":
            graph = _graph()
            graph["nodes"].append({"id": "merge", "type": "merge"})
            edge = next(edge for edge in graph["edges"] if edge["source"] == "zone" and edge["sourceHandle"] == "events")
            edge["target"], edge["targetHandle"] = "merge", "events"
            graph["edges"].append({"source": "merge", "sourceHandle": "events", "target": "out", "targetHandle": "events"})
        else:
            assert [p["name"] for p in CATALOG[kind]["inputs"]] == ["frame"]
            graph = {"version": 1, "nodes": [{"id": "s", "type": "source"}, {"id": "cv", "type": kind}, {"id": "o", "type": "output"}],
                     "edges": [{"source": "s", "sourceHandle": "frame", "target": "cv", "targetHandle": "frame"},
                               {"source": "cv", "sourceHandle": CATALOG[kind]["outputs"][0]["name"], "target": "o", "targetHandle": CATALOG[kind]["outputs"][0]["name"]}]}
            if CATALOG[kind]["outputs"][0]["name"] != "frame":
                graph["edges"].append({"source": "s", "sourceHandle": "frame", "target": "o", "targetHandle": "frame"})
        normalized = validate_graph(graph)
        assert normalized["valid"], normalized["errors"]
        doc, prompt = convert_workflow(graph), to_api_prompt(graph)
        _roundtrip(normalized["graph"], doc, prompt)
        covered.add(kind)
    assert covered == set(CATALOG) and len(covered) >= 30


def multi_graph(count=3):
    nodes = [{"id": "source", "type": "source"}, {"id": "out", "type": "output"}]
    edges = [{"source": "source", "sourceHandle": "frame", "target": "out", "targetHandle": "frame"}]
    for index in range(count):
        kind = "pose" if index == 2 else "detector"
        node_id = "model_" + str(index)
        nodes.append({"id": node_id, "type": kind, "params": {"model_id": "rf_detr_2xl" if index == 1 else "yolo26x"} if kind == "detector" else {}})
        edges += [{"source": "source", "sourceHandle": "frame", "target": node_id, "targetHandle": "frame"},
                  {"source": node_id, "sourceHandle": "poses" if kind == "pose" else "detections", "target": "out", "targetHandle": "observations"}]
    return {"version": 1, "nodes": nodes, "edges": edges}


def test_parallel_detectors_and_pose_connect_to_different_output_sockets():
    graph = multi_graph()
    doc, prompt = convert_workflow(graph, source_mode="camera"), to_api_prompt(graph, source_mode="camera")
    out = next(item for item in prompt.values() if item["class_type"] == "VisionWB_output")
    assert all(key in out["inputs"] for key in ("frame", "observations", "observations_2", "observations_3"))
    source = next(node for node in doc["nodes"] if node["type"] == "VisionWB_source")
    assert source["widgets_values"][0] == "camera"
    assert output_slots("output") == [{"name": "preview", "type": "IMAGE"}, {"name": "report", "type": "STRING"}]
    _roundtrip(validate_graph(graph)["graph"], doc, prompt)


def test_ninth_branch_rejected_instead_of_silently_lost():
    with pytest.raises(ValueError, match="最多可轉換 8"):
        convert_workflow(multi_graph(9))


@pytest.mark.parametrize("mode", ["media", "camera", "vdo", "live"])
def test_source_widget_order_and_values_match_plugin(mode):
    options = {"device": "cuda:0", "frame_stride": 3, "max_frames": 600, "max_side": 960}
    doc = convert_workflow(_graph(), source_mode=mode, media_id="sample-a", options=options)
    source = next(node for node in doc["nodes"] if node["type"] == "VisionWB_source")
    assert dict(zip(SOURCE_WIDGETS, source["widgets_values"])) == {"source_mode": "camera" if mode == "live" else mode, "media_id": "sample-a", **options}


def test_json_arrays_are_widget_strings_never_misread_as_comfy_links():
    doc, prompt = convert_workflow(_graph("zone-sequence")), to_api_prompt(_graph("zone-sequence"))
    zone = next(node for node in doc["nodes"] if node["type"] == "VisionWB_zone")
    assert isinstance(zone["widgets_values"][0], str)
    action = next(node for node in prompt.values() if node["class_type"] == "VisionWB_sequence")
    assert json.loads(action["inputs"]["steps"]) == ["zone_enter:check", "zone_enter:work", "zone_enter:exit"]


def test_resource_labels_and_checkpoint_values_preserved():
    graph = _graph()
    next(node for node in graph["nodes"] if node["type"] == "detector")["params"]["checkpoint_id"] = "trained-bento"
    resources = {"models": [{"id": "yolo26x", "name": "YOLO26x"}],
                 "checkpoints": [{"id": "trained-bento", "model_id": "yolo26x", "name": "便當微調"}],
                 "media": [{"id": "demo", "name": "測試影像.jpg"}]}
    prompt = to_api_prompt(graph, media_id="demo", resources=resources)
    source = next(item for item in prompt.values() if item["class_type"] == "VisionWB_source")
    model = next(item for item in prompt.values() if item["class_type"] == "VisionWB_detector")
    assert source["inputs"]["media_id"] == "demo | 測試影像.jpg"
    assert model["inputs"]["model_id"] == "yolo26x | YOLO26x"
    assert model["inputs"]["checkpoint_id"] == "trained-bento | yolo26x · 便當微調"
    assert "便當微調" in json.dumps(prompt, ensure_ascii=False)


@pytest.mark.parametrize("options", [{"max_frames": True}, {"max_side": 99999}, {"device": "remote"}, {"token": "not-allowed"}])
def test_invalid_source_options_are_rejected(options):
    with pytest.raises(ValueError): convert_workflow(_graph(), options=options)


def test_incompatible_graph_is_rejected_before_export():
    graph = _graph()
    next(edge for edge in graph["edges"] if edge["target"] == "track")["sourceHandle"] = "frame"
    with pytest.raises(ValueError): to_api_prompt(graph)


def test_converter_matches_real_plugin_socket_types_and_widget_order():
    path = Path(__file__).resolve().parents[1] / "comfyui/vision_workbench/schema.py"
    spec = importlib.util.spec_from_file_location("_vision_workbench_schema_contract", path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    resources = {"media": [], "models": [], "checkpoints": [], "integration": {}}
    for kind, definition in CATALOG.items():
        assert input_slots(kind) == schema.expanded_inputs(definition)
        if kind != "output":
            assert [p["type"] for p in output_slots(kind)] == [schema.TYPE_MAP[p["type"]] for p in definition["outputs"]]
        fields = schema.input_types(definition, resources)
        widgets = [key for key, value in fields["required"].items() if not value[1].get("forceInput")]
        assert widgets == (list(SOURCE_WIDGETS) if kind == "source" else [p["key"] for p in definition["params"]])


def test_export_cli_uses_sample_metadata_and_preserves_edited_workflow(tmp_path, monkeypatch):
    import export_workflow_templates as cli
    entry = next(item for item in templates()["templates"] if item["id"] == "zone-guard")
    entry.update(sample={"media_id": "clip", "title": "走道", "origin": "original_project"},
                 recommended_options={"frame_stride": 1, "max_frames": 252, "max_side": 640})
    replies = {"/workflows/templates": {"templates": [entry]}, "\u002fmodels": {"models": [{"id": "yolo26x", "name": "YOLO26x"}]},
               "/checkpoints": {"checkpoints": []}, "/media": {"media": [{"id": "clip", "name": "廠區.mp4"}]}}
    class FakeOpener:
        def open(self, url, timeout):
            assert url.startswith("http://127.0.0.1:8765/api") and timeout == 30
            return io.BytesIO(json.dumps(replies[url.removeprefix("http://127.0.0.1:8765/api")]).encode())
    monkeypatch.setattr(cli, "build_opener", lambda *args: FakeOpener())
    files = cli.export_templates("http://127.0.0.1:8765", tmp_path)
    path = Path(files[0])
    doc = json.loads(path.read_text(encoding="utf-8"))
    source = next(node for node in doc["nodes"] if node["type"] == "VisionWB_source")
    assert source["widgets_values"] == ["media", "clip | 廠區.mp4", "auto", 1, 252, 640]
    doc["nodes"][0]["pos"] = [1250, 750]
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    user_bytes = path.read_bytes()
    replacement = Path(cli.export_templates("http://127.0.0.1:8765", tmp_path)[0])
    assert replacement != path
    assert path.read_bytes() == user_bytes
    generated = json.loads(replacement.read_text(encoding="utf-8"))
    assert generated["version"] == .4
    assert generated["nodes"][0]["pos"] != [1250, 750]
    assert generated["groups"][0]["title"] == entry["name"]
    manifest = json.loads((tmp_path / cli.MANIFEST).read_text(encoding="utf-8"))
    assert path.name in manifest["preserved_edited_or_unmanaged_files"]
    assert manifest["entries"][entry["id"]]["filename"] == replacement.name
    assert cli.export_templates("http://127.0.0.1:8765", tmp_path) == [str(replacement)]
    assert path.read_bytes() == user_bytes


def test_export_cli_rejects_remote_resource_endpoint(tmp_path):
    from export_workflow_templates import export_templates
    with pytest.raises(ValueError, match="本機"):
        export_templates("https://example.com", tmp_path)
