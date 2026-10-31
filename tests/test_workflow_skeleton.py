"""Native preprocessing parity and strict real-checkpoint CPU inference."""
from pathlib import Path
import os

import numpy as np
import pytest

from backend.workflow_skeleton import CLASS_LABELS, SkeletonActionModel, prepare_sequence


FIXTURE = Path(__file__).parent / "fixtures" / "stgcnpp_preprocess_golden.npz"


def sequence(frames=100):
    value = np.zeros((frames, 17, 3), dtype=np.float32)
    value[..., :2] = [640, 360]
    value[..., 2] = .9
    return value


@pytest.mark.parametrize("frames", [50, 150, 337])
def test_preprocessing_matches_unmodified_official_pipeline_for_every_sampling_branch(frames):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        source = fixture[f"input_{frames}"]
        original = source.copy()
        expected = fixture[f"expected_{frames}"]
        actual = prepare_sequence(source, (1280, 720))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.shape == (1, 2, 100, 17, 3)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert np.count_nonzero(actual[:, 1]) == 0


def test_normalization_masks_only_low_confidence_xy_and_does_not_change_global_rng():
    value = sequence()
    value[:, 0] = [200, 100, .005]
    state = np.random.get_state()
    result = prepare_sequence(value, (1280, 720))
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    np.testing.assert_array_equal(result[0, 0, :, 0, :2], 0)
    np.testing.assert_array_equal(result[0, 0, :, 0, 2], np.float32(.005))
    np.testing.assert_array_equal(result[0, 0, :, 1, :2], 0)


@pytest.mark.parametrize("value, message", [
    (np.zeros((29, 17, 3)), "至少"),
    (np.zeros((100, 17, 3)), "資料不足"),
    (np.zeros((2, 100, 17, 3)), "不同人物"),
    (np.zeros((100, 25, 3)), "COCO"),
    (np.full((100, 17, 3), np.nan), "NaN"),
    (np.full((100, 17, 3), 2.), "信心值"),
])
def test_invalid_or_unobservable_sequences_are_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        prepare_sequence(value, (1280, 720))


def test_low_coverage_and_missing_torso_are_not_guessed():
    value = sequence()
    value[:75] = 0
    with pytest.raises(ValueError, match="資料不足"):
        prepare_sequence(value, (1280, 720))
    value = sequence()
    value[:, [5, 6, 11, 12], 2] = 0
    with pytest.raises(ValueError, match="資料不足"):
        prepare_sequence(value, (1280, 720))


@pytest.mark.parametrize("size", [(0, 720), (1280, float('inf')), (True, 720), (1280,)])
def test_invalid_frame_dimensions(size):
    with pytest.raises(ValueError, match="image_size"):
        prepare_sequence(sequence(), size)


def test_official_label_order_is_complete():
    assert len(CLASS_LABELS) == 120 and len(set(CLASS_LABELS)) == 120
    assert CLASS_LABELS[0] == "drink water" and CLASS_LABELS[42] == "falling"
    assert CLASS_LABELS[79] == "squat down"
    assert CLASS_LABELS[-1] == "finger-guessing game (playing rock-paper-scissors)"


def test_real_official_checkpoint_strict_cpu_forward_and_wrong_head_rejected(tmp_path):
    import torch
    default = Path(__file__).resolve().parents[2] / "model_weights" / "stgcnpp_ntu120" / "stgcnpp_ntu120_hrnet_j.pth"
    weight = Path(os.environ.get("WORKBENCH_STGCN_TEST_WEIGHT", default))
    if not weight.is_file():
        pytest.skip("Official checkpoint optional: set WORKBENCH_STGCN_TEST_WEIGHT; tests never download weights.")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = SkeletonActionModel(weight, device="cpu")
        assert all(parameter.device.type == "cpu" for parameter in model.model.parameters())
        prediction = model.predict(sequence(), (1280, 720))
        assert len(prediction) == 5
        assert [row["score"] for row in prediction] == sorted([row["score"] for row in prediction], reverse=True)
        assert all(0 <= row["score"] <= 1 and row["label"] == CLASS_LABELS[row["class_id"]] for row in prediction)
        assert sum(row["score"] for row in prediction) <= 1.000001
        state = torch.load(weight, map_location="cpu", weights_only=True)
        state["cls_head.fc_cls.weight"] = state["cls_head.fc_cls.weight"][:60]
        invalid = tmp_path / "wrong-ntu60.pth"
        torch.save(state, invalid)
        with pytest.raises(ValueError, match="架構"):
            SkeletonActionModel(invalid, device="cpu")
    finally:
        torch.set_num_threads(old_threads)
