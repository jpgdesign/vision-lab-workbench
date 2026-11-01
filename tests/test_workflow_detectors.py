import json

from backend import workflow_detectors as adapters


def test_dual_branch_predictor_keeps_distinct_scores_and_class_mapping(tmp_path, monkeypatch):
    from backend import inference
    manifest = {"classes": ["person", "bento"], "branches": {
        "original": {"classes": ["person"], "label_map": [0]},
        "new": {"classes": ["bento"], "label_map": [1]}}}
    (tmp_path / adapters.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(adapters, "is_bundle", lambda path: True)
    monkeypatch.setattr(adapters, "validate_bundle", lambda path, model: tmp_path)
    monkeypatch.setattr(adapters, "branch_request", lambda request, root, name: {"params": {"branch": name}, "checkpoint_path": name})
    created = []
    def factory(model, params, device, checkpoint, weight):
        created.append(checkpoint)
        return lambda image, selected: [{"label": "person" if checkpoint == "original" else "bento", "score": .4 if checkpoint == "original" else .8, "box": [0, 0, 10, 10]}]
    monkeypatch.setattr(inference, "create_detector", factory)
    predict = adapters.create_workflow_detector("yolo26x", {}, "cpu", "bundle")
    first = predict(None)
    second = predict(None)
    assert created == ["original", "new"]  # Loaded once, reused for every frame.
    assert first == second
    assert [row["score"] for row in first] == [.4, .8]
    assert [row["branch"] for row in first] == ["original", "new"]
    assert [row["union_class_id"] for row in first] == [0, 1]
