from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from backend.workflow_embedding import EmbeddingNode
from backend.workflow_model_nodes import ModelNode


def frame(stamp=0., source="one", size=(16, 12)):
    return {"image": Image.new("RGB", size, (100, 50, 25)), "timestamp": stamp, "frame_index": int(stamp * 30),
            "source_id": source, "width": size[0], "height": size[1]}


class Backbone(torch.nn.Module):
    def __init__(self, values=(3., 4., 0.)):
        super().__init__()
        self.register_buffer("features", torch.tensor([values]))
        self.calls = 0
        self.inputs = []

    def forward(self, pixel_values):
        assert torch.is_inference_mode_enabled()
        self.calls += 1
        self.inputs.append(pixel_values.clone())
        return SimpleNamespace(pooler_output=self.features)


def processor(images, return_tensors):
    assert return_tensors == "pt"
    return {"pixel_values": torch.from_numpy(np.stack([np.asarray(image).transpose(2, 0, 1).copy() for image in images])).float()}


def adapter(*, classified=False, **params):
    node = EmbeddingNode(params, {"model_id": "dinov3_7b"}, "cpu")
    backbone = Backbone()
    if classified:
        classifier = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        node.model = SimpleNamespace(backbone=backbone, classifier=classifier)
        node.metadata = {"mode": "classification", "labels": {0: "first", 1: "second"}}
    else:
        node.model, node.metadata = backbone, {"mode": "embedding"}
    node.processor = processor
    return node, backbone


def test_real_cls_vector_normalized_without_inventing_a_class():
    node, backbone = adapter()
    packet = node.process({}, frame())
    item = packet["embeddings"]["items"][0]
    assert item["vector"] == pytest.approx([.6, .8, 0.])
    assert item["dimensions"] == 3 and item["pooling"] == "cls" and item["normalized"] is True
    assert "vector" not in packet["embeddings"]["display_items"][0]
    assert packet["actions"]["items"][0]["label"] == "unknown"
    assert packet["actions"]["items"][0]["reason"] == "embedding_only"
    assert backbone.inputs[0].shape == (1, 3, 12, 16)
    assert backbone.inputs[0][0, :, 0, 0].tolist() == [100, 50, 25]


def test_raw_features_and_classifier_logits_use_the_same_backbone_pass():
    node, backbone = adapter(classified=True, normalize=False)
    result = node.process({}, frame())
    assert backbone.calls == 1
    assert result["embeddings"]["items"][0]["vector"] == [3., 4., 0.]
    action = result["actions"]["items"][0]
    assert action["label"] == "second" and action["scope"] == "scene"
    assert action["score"] == pytest.approx(.7310586)
    assert [candidate["label"] for candidate in action["candidates"]] == ["second", "first"]


def test_classifier_below_threshold_is_unknown_with_candidates():
    node, _ = adapter(classified=True, min_score=.99)
    action = node.process({}, frame())["actions"]["items"][0]
    assert action["label"] == "unknown" and action["score"] == 0
    assert action["candidates"][0]["label"] == "second"


def test_cadence_never_reuses_cached_vector_or_class_as_new_evidence():
    node, backbone = adapter(classified=True, every_seconds=2.)
    node.process({}, frame(0.))
    skipped = node.process({}, frame(.5))
    assert backbone.calls == 1
    for port in skipped.values():
        assert port["fresh"] is False and port["items"] == []
        assert port["display_items"][0]["evidence_timestamp"] == 0.
        assert port["display_items"][0]["stale"] is True
    node.process({}, frame(2.))
    assert backbone.calls == 2


@pytest.mark.parametrize("changed", [frame(10.), frame(0.), frame(.5, source="other"), frame(.5, size=(20, 12))])
def test_gap_rewind_source_or_size_resets_cadence(changed):
    node, backbone = adapter(every_seconds=120, max_gap_seconds=5)
    node.process({}, frame(.1))
    assert node.process({}, changed)["embeddings"]["fresh"] is True
    assert backbone.calls == 2


def test_duplicate_frame_is_not_recomputed():
    node, backbone = adapter()
    node.process({}, frame(0.))
    assert node.process({}, frame(0.))["embeddings"]["fresh"] is False
    assert backbone.calls == 1


def test_nonfinite_features_and_incompatible_classifier_metadata_rejected():
    node, backbone = adapter()
    backbone.features[0, 0] = torch.nan
    with pytest.raises(ValueError, match="CLS"):
        node.process({}, frame())
    node, _ = adapter(classified=True)
    node.metadata["labels"] = {0: "only-one"}
    with pytest.raises(ValueError, match="標籤"):
        node.process({}, frame())


def test_persistent_modelnode_uses_existing_offline_loader_once_and_forwards_checkpoint(monkeypatch):
    from backend import inference
    backbone, calls = Backbone(), []
    def create(device, checkpoint_path=None, model_weight_path=None):
        calls.append((device, checkpoint_path, model_weight_path))
        return backbone, processor, {"mode": "embedding"}
    monkeypatch.setattr(inference, "create_dinov3_model", create)
    monkeypatch.setattr(inference, "resolve_device", lambda device: "cpu")
    node = ModelNode("embedding", {}, {"model_id": "dinov3_7b", "checkpoint_path": "/registered/checkpoint"}, "cpu")
    node.process({}, frame(0.)); node.process({}, frame(1.))
    assert calls == [("cpu", "/registered/checkpoint", None)] and backbone.calls == 2
    node.reset()
    node.process({}, frame(0.))
    assert len(calls) == 1 and backbone.calls == 3


def test_embedding_vector_reaches_artifacts_and_render_summary_without_large_model(monkeypatch):
    from backend.workflow_runtime import WorkflowEngine
    graph = {"version": 1, "nodes": [{"id": "input", "type": "source"}, {"id": "features", "type": "embedding"}, {"id": "out", "type": "output"}], "edges": [
        {"source": "input", "sourceHandle": "frame", "target": "features", "targetHandle": "frame"},
        {"source": "input", "sourceHandle": "frame", "target": "out", "targetHandle": "frame"},
        {"source": "features", "sourceHandle": "embeddings", "target": "out", "targetHandle": "embeddings"}]}
    node, _ = adapter()
    engine = WorkflowEngine({"id": "embedding-test", "workflow_graph": graph}, model_factory=lambda *args, **kw: node)
    engine.device = "cpu"
    image, events, records = engine.process(Image.new("RGB", (320, 240)), 0., 0)
    assert image.size == (320, 240) and not events
    vector = records["features"]["embeddings"]["items"][0]["vector"]
    assert vector == pytest.approx([.6, .8, 0.])
    assert records["features"]["embeddings"]["coordinate_space"]["width"] == 320
