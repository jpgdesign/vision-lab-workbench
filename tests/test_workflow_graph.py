"""Portable workflow contracts, without loading models or calling services."""
import copy
import math

import pytest

from backend.workflow_graph import normalize_graph, validate_graph
from backend.workflow_templates import templates


def template(name="zone-guard"):
    return copy.deepcopy(next(item["graph"] for item in templates()["templates"] if item["id"] == name))


def node(graph, identifier):
    return next(item for item in graph["nodes"] if item["id"] == identifier)


def edge(source, output, target, input_, identifier="extra"):
    return {"id": identifier, "source": source, "sourceHandle": output, "target": target, "targetHandle": input_}


def test_every_published_template_is_typed_acyclic_and_reaches_output():
    values = templates()["templates"]
    assert len(values) >= 8
    assert len({item["id"] for item in values}) == len(values)
    for item in values:
        checked = validate_graph(item["graph"])
        assert checked["valid"], (item["id"], checked["errors"])
        order = checked["plan"]["order"]
        assert len(order) == len(item["graph"]["nodes"])
        assert all(order.index(e["source"]) < order.index(e["target"]) for e in item["graph"]["edges"])
        assert all(source["url"].startswith("https://") for source in item["sources"])


def test_pose_and_detection_are_both_accepted_by_tracker_without_accepting_frames():
    graph = template()
    node(graph, "detect").update(type="pose", params={})
    next(e for e in graph["edges"] if e["source"] == "detect")["sourceHandle"] = "poses"
    assert validate_graph(graph)["valid"]
    graph["edges"] = [e for e in graph["edges"] if e["target"] != "track"]
    graph["edges"].append(edge("input", "frame", "track", "observations"))
    checked = validate_graph(graph)
    assert not checked["valid"] and any("不相容" in message for message in checked["errors"])


def test_multiple_output_inputs_allow_distinct_branches_but_not_duplicate_wires():
    graph = template()
    graph["edges"].append(edge("detect", "detections", "out", "observations"))
    assert validate_graph(graph)["valid"]
    graph["edges"].append(edge("detect", "detections", "out", "observations", "different-id"))
    assert not validate_graph(graph)["valid"]


def test_single_input_rejects_second_producer_even_with_matching_type():
    graph = template()
    graph["nodes"].append({"id": "detect2", "type": "detector"})
    graph["edges"].extend([edge("input", "frame", "detect2", "frame"), edge("detect2", "detections", "track", "observations", "e2")])
    assert any("一個輸入" in message for message in validate_graph(graph)["errors"])


def test_cycle_is_rejected_even_when_every_handle_has_compatible_type():
    graph = template()
    graph["edges"] = [e for e in graph["edges"] if not (e["source"] == "input" and e["target"] == "detect")]
    graph["edges"].append(edge("out", "frame", "detect", "frame"))
    checked = validate_graph(graph)
    assert not checked["valid"] and any("循環" in message for message in checked["errors"])


def test_person_action_requires_tracks_and_unconnected_nodes_are_not_silently_ignored():
    graph = template("rgb-action-rules")
    node(graph, "action")["params"]["scope"] = "person"
    assert any("追蹤框" in message for message in validate_graph(graph)["errors"])
    graph = template()
    graph["nodes"].append({"id": "unused", "type": "detector"})
    graph["edges"].append(edge("input", "frame", "unused", "frame"))
    assert any("沒有通往" in message for message in validate_graph(graph)["errors"])


@pytest.mark.parametrize("key,value", [("conf", math.nan), ("conf", math.inf), ("conf", True), ("conf", 1.2), ("max_det", 1.5), ("use_training_classes", "true"), ("api_key", "do-not-save"), ("checkpoint_path", "C:/private/model.pt")])
def test_invalid_model_parameters_and_credentials_are_rejected(key, value):
    graph = template()
    node(graph, "detect")["params"][key] = value
    assert not validate_graph(graph)["valid"]


def test_defaults_are_detached_and_cloud_warning_is_only_for_remote_nodes():
    graph = {"version": 1, "nodes": [{"id": "zone", "type": "zone"}], "edges": []}
    first = normalize_graph(graph)
    first["nodes"][0]["params"]["zones"][0]["points"][0][0] = .99
    assert normalize_graph(graph)["nodes"][0]["params"]["zones"][0]["points"][0][0] != .99
    assert "params" not in graph["nodes"][0]
    assert not any("雲端" in text for text in validate_graph(template("object-relations"))["warnings"])
    assert any("雲端" in text for text in validate_graph(template("cloud-review"))["warnings"])
