"""Relation semantics, source identity and official preprocessing regressions."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from backend.workflow_relations import RelationModel
from backend.vendor.relateanything.runtime import OnnxRelationHead, load_predicate_bank
from backend.vendor.relateanything.scoring import ScoreContract


class FakeHead:
    max_boxes = 32
    contract = ScoreContract(calib_a=.5825, calib_b=-1.8911)
    calls = 0

    def set_predicates(self, names):
        self.names = list(names)

    def __call__(self, frame, boxes):
        self.calls += 1
        self.boxes = boxes.copy()
        if hasattr(self, "outputs"):
            return self.outputs
        return (np.full((1, len(self.names)), 4., np.float32),
                np.array([2.], np.float32), np.array([0]), np.array([1]), np.array([True]))


@pytest.fixture
def model():
    value = RelationModel.__new__(RelationModel)
    value.head = FakeHead()
    value.default_predicates = ("holding",)
    value.predicate_labels = ("holding", "wearing", "riding", "next to", "on")
    value._lock = threading.Lock()
    value.last_diagnostics = {}
    return value


@pytest.fixture
def frame():
    return np.zeros((100, 200, 3), np.uint8)


def detections():
    return [dict(box=[10, 10, 30, 50], label="cup", score=.1, track_key="cup:8"),
            dict(box=[40, 10, 100, 90], label="person", score=.9, track_key="person:4")]


def test_original_indices_and_track_identity_survive_score_order(model, frame):
    result = model.predict(frame, detections(), ["holding"], threshold=.8)
    assert len(result) == 1
    row = result[0]
    assert row["subject_index"] == 1 and row["object_index"] == 0
    assert row["subject_label"] == "person" and row["object_label"] == "cup"
    assert row["subject_box"] == [40, 10, 100, 90]
    assert row["scope"] == "relation" and row["label"] == row["predicate"] == "holding"
    assert row["track_key"] == row["subject_track_key"] == "person:4"
    assert row["object_track_key"] == "cup:8"
    assert json.loads(row["relation_key"]) == ["person:4", "holding", "cup:8"]
    # Threshold/reporting are calibrated relation scores, NOT multiplied by
    # detector confidence .9*.1 (which is used only to rank candidates).
    expected = 1 / (1 + math.exp(-(.5825 * (4 + 2) - 1.8911)))
    assert row["score"] == pytest.approx(expected, abs=1e-6)
    assert model.predict(frame, detections(), ["holding"], threshold=.9) == []


def test_untracked_regions_do_not_fabricate_person_scope_or_identity(model, frame):
    boxes = detections()
    for box in boxes:
        box.pop("track_key")
    row = model.predict(frame, boxes)[0]
    assert row["scope"] == "relation"
    assert not {"track_key", "subject_track_key", "object_track_key", "relation_key"} & row.keys()


def test_max_32_regions_selection_is_visible_and_stably_mapped(model, frame):
    boxes = [dict(box=[i, 1, i+2, 8], label=str(i), score=i/40) for i in range(35)]
    row = model.predict(frame, boxes)[0]
    assert (row["subject_index"], row["object_index"]) == (34, 33)
    assert model.last_diagnostics["dropped_count"] == 3
    assert model.last_diagnostics["selected_indices"] == list(range(34, 2, -1))
    assert len(model.head.boxes) == 32


def test_single_and_empty_regions_do_not_run_head(model, frame):
    assert model.predict(frame, []) == []
    assert model.predict(frame, detections()[:1]) == []
    assert model.head.calls == 0


def test_unknown_predicate_fails_even_without_regions(model, frame):
    with pytest.raises(ValueError, match="不包含"):
        model.predict(frame, [], ["silently-substitute-me"])
    assert model.head.calls == 0


def test_predicate_switch_and_default_reset(model, frame):
    model.predict(frame, detections(), ["on", "holding"])
    assert model.head.names == ["on", "holding"]
    model.predict(frame, detections())
    assert model.head.names == ["holding"]


def test_padding_and_negative_indices_cannot_become_relations(model, frame):
    model.head.outputs = (np.full((4, 1), 4., np.float32), np.full(4, 2., np.float32),
                          np.array([0, -1, 0, 0]), np.array([1, 0, 31, 0]), np.ones(4, bool))
    result = model.predict(frame, detections())
    assert len(result) == 1 and result[0]["subject_index"] == 1


@pytest.mark.parametrize("outputs", [
    (np.zeros((1, 1)), np.zeros(1), np.array([0.]), np.array([1]), np.ones(1, bool)),
    (np.zeros((1, 2)), np.zeros(1), np.array([0]), np.array([1]), np.ones(1, bool)),
    (np.full((1, 1), np.nan), np.zeros(1), np.array([0]), np.array([1]), np.ones(1, bool)),
])
def test_invalid_model_outputs_are_not_displayed_as_success(model, frame, outputs):
    model.head.outputs = outputs
    with pytest.raises(RuntimeError):
        model.predict(frame, detections())


@pytest.mark.parametrize("override", [
    dict(box=[0, 0, 0, 1]), dict(box=[0, 0, float("nan"), 2]), dict(box=[1, 2]),
    dict(score=float("nan")), dict(score=2), dict(score=True),
])
def test_invalid_detection_rejected(model, frame, override):
    boxes = detections()
    boxes[0].update(override)
    with pytest.raises(ValueError):
        model.predict(frame, boxes)


@pytest.mark.parametrize("kwargs", [dict(threshold=-1), dict(threshold=True), dict(top_k=0),
                                    dict(top_k=1.5), dict(predicates=[]), dict(predicates=["on", "on"])])
def test_invalid_controls_rejected(model, frame, kwargs):
    with pytest.raises(ValueError):
        model.predict(frame, detections(), **kwargs)


def test_official_square_resize_rgb_and_box_normalization():
    head = OnnxRelationHead.__new__(OnnxRelationHead)
    head.img_size = 448
    head.max_boxes = 32
    head.vocab_mode = "input"
    head._W = np.zeros((2, 512), np.float32)
    head._alpha = np.ones(2, np.float32)
    frame = np.empty((10, 20, 3), np.uint8)
    frame[:] = [10, 20, 30]
    feed = head.make_feed(frame, np.array([[2, 1, 10, 9]], np.float32))
    assert feed["image"].shape == (1, 3, 448, 448)
    np.testing.assert_allclose(feed["image"][0, :, 0, 0], np.array([30, 20, 10])/255)
    np.testing.assert_allclose(feed["boxes"][0, 0], [.3, .5, .4, .8])
    assert not feed["boxes"][0, 1:].any()
    np.testing.assert_array_equal(feed["box_counts"], [1])
    assert feed["image"].flags.c_contiguous


def test_unrecognized_object_pickle_bank_is_not_unpickled(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez(path, names=np.array(["on"], dtype=object))
    with load_predicate_bank(path) as bank:
        with pytest.raises(ValueError, match="Object arrays"):
            bank["names"]


def test_unicode_bank_does_not_require_pickle(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez(path, names=np.array(["on"], dtype="U2"))
    with load_predicate_bank(path) as bank:
        assert bank["names"].tolist() == ["on"]


def test_invalid_bundle_rejected_before_native_session(tmp_path, monkeypatch):
    for name in ("relateanything.onnx", "predicate_bank.npz", "thresholds.json"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "relateanything.json").write_text(json.dumps(dict(img_size=224)), encoding="utf-8")
    monkeypatch.setattr("backend.workflow_relations.prepare_windows_runtime", lambda: None)
    with pytest.raises(ValueError, match="448px"):
        RelationModel(tmp_path)


def test_official_bundle_cpu_in_fresh_process():
    """Optional real-artifact smoke; no downloads and no CUDA allocation."""
    root = Path(__file__).resolve().parents[3]
    model_dir = root / "outputs/model_weights/relateanything_convnext_large"
    if not (model_dir / "relateanything.onnx").is_file():
        pytest.skip("Official ONNX weights are not present in this checkout")
    script = """
from backend.workflow_relations import prepare_windows_runtime, RelationModel
prepare_windows_runtime()
import numpy as np
model = RelationModel(MODEL_DIR)
assert model.head.sess.get_providers() == ['CPUExecutionProvider']
assert {'holding','wearing','riding','next to','on'}.issubset(model.available_predicates())
image = np.full((80, 120, 3), 128, np.uint8)
regions = [dict(box=[1,1,50,70]), dict(box=[60,1,110,70])]
rows = model.predict(image, regions, ['holding', 'next to'], threshold=0)
assert len(rows) == 2
assert all(0 <= r['score'] <= 1 and r['scope'] == 'relation' for r in rows)
assert 'torch' not in __import__('sys').modules
print('real CPU graph passed')
""".replace("MODEL_DIR", repr(str(model_dir)))
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script],
                               cwd=Path(__file__).resolve().parents[1],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                               capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "real CPU graph passed" in completed.stdout
