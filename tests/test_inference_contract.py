"""Regression tests for cross-model score semantics and temporal inputs."""
import numpy as np
import pytest
import torch

from backend.inference import classification_probabilities, kinetics400_labels, prepare_action_tensor


def test_raw_logits_become_probabilities_without_losing_rank():
    scores = classification_probabilities(torch.tensor([[11.1, 9.1, -3.0]]), False)
    assert torch.isclose(scores.sum(), torch.tensor(1.0))
    assert scores.argmax().item() == 0
    assert (scores <= 1).all()


def test_normalized_scores_are_not_softmaxed_twice():
    scores = torch.tensor([[0.8, 0.15, 0.05]])
    assert torch.equal(classification_probabilities(scores, True), scores)


@pytest.mark.parametrize('scores', [[[11.1, 9.1]], [[float('nan'), 1.0]], [[0.2, 0.2]]])
def test_invalid_probability_adapter_fails_loudly(scores):
    with pytest.raises(ValueError):
        classification_probabilities(torch.tensor(scores), True)


def test_kinetics_class_map_has_every_index():
    labels = kinetics400_labels()
    assert set(labels) == set(range(400))
    assert labels[0] == 'abseiling'


def test_slowfast_temporal_pathways_preserve_rgb_order():
    frames = np.zeros((32, 80, 120, 3), dtype=np.uint8)
    frames[..., 0] = 255
    slow, fast = prepare_action_tensor(frames, {
        'side': 64, 'short_side': 64, 'slowfast_alpha': 4,
        'mean': [0, 0, 0], 'std': [1, 1, 1],
    }, 'cpu')
    assert slow.shape == (1, 3, 8, 64, 64)
    assert fast.shape == (1, 3, 32, 64, 64)
    assert torch.allclose(fast[:, 0], torch.ones_like(fast[:, 0]))
    assert torch.count_nonzero(fast[:, 1:]) == 0
