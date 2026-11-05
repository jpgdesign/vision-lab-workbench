"""CPU-only preservation/calibration regressions; no pretrained models or downloads."""
from copy import deepcopy

import pytest
import torch
from PIL import Image
from torch import nn

from bento_research.hierarchical_refiner import FineRefiner, masked_logits, parent_mask
from evaluate_bento_hierarchical_refiner import calibrate, core, enrich, fine_scores, known_fine_annotations


CLASSES = [
    {'id': 0, 'name': 'carrot', 'parent_id': 1},
    {'id': 1, 'name': 'broccoli', 'parent_id': 1},
    {'id': 2, 'name': 'steamed_egg', 'parent_id': 2},
]


def test_parent_mask_prevents_cross_parent_gradient_updates():
    head = nn.Linear(2, 3)
    mask = parent_mask(CLASSES)
    logits = masked_logits(head, torch.tensor([[1., 2.]]), torch.tensor([1]), mask)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    assert torch.count_nonzero(head.weight.grad[:2]) > 0
    assert torch.count_nonzero(head.weight.grad[2]) == 0
    assert head.bias.grad[2] == 0
    assert logits.softmax(1)[0, 2] == 0
    with pytest.raises(ValueError, match='no child outputs'):
        masked_logits(head, torch.zeros((1, 2)), torch.tensor([0]), mask)


class DummyFrozenFeatures(nn.Module):
    def forward(self, inputs):
        return torch.zeros((inputs.shape[0], 2), device=inputs.device)


def dummy_refiner():
    # Bypass disk/model loading; exercise the actual prediction and fallback code.
    refiner = FineRefiner.__new__(FineRefiner)
    refiner.classes = deepcopy(CLASSES)
    refiner.device = torch.device('cpu')
    refiner.config = {'imgsz': 64}
    refiner.extractor = DummyFrozenFeatures()
    refiner.head = nn.Linear(2, len(CLASSES))
    with torch.no_grad():
        refiner.head.weight.zero_()
        refiner.head.bias.copy_(torch.tensor([2., 0., 5.]))
    refiner.mean = torch.zeros(2)
    refiner.std = torch.ones(2)
    refiner.mask = parent_mask(CLASSES)
    refiner.thresholds = {'parents': {'1': {'enabled': True, 'threshold': .8},
                                      '2': {'enabled': True, 'threshold': .8}}}
    return refiner


def test_no_child_single_child_and_disabled_calibration_fall_back():
    refiner = dummy_refiner()
    image = Image.new('RGB', (25, 40), 'green')
    none, single, supported = refiner.predict_crops([image] * 3, [0, 2, 1])
    assert none['accepted'] is False and none['confidence'] is None
    assert none['reason'] == 'no_supported_children'
    assert single['accepted'] is False and single['fine_class_name'] is None
    assert single['reason'] == 'single_child_parent_not_discriminative'
    assert supported['accepted'] is True and supported['fine_class_name'] == 'carrot'
    assert set(supported['probabilities']) == {'carrot', 'broccoli'}
    refiner.thresholds = {'parents': {}}
    result = refiner.predict_crops([image], [1])[0]
    assert result['candidate_class_name'] == 'carrot'
    assert result['accepted'] is False and result['fine_class_name'] is None
    refiner.thresholds = {'parents': {'1': {'enabled': True, 'threshold': .99}}}
    result = refiner.predict_crops([image], [1])[0]
    assert result['reason'] == 'fine_confidence_below_threshold'
    assert result['fine_class_name'] is None


def calibration_fixture():
    # Two correct known fine labels, one known but unsupported label, one broad unknown.
    predictions = [{'image_id': 1, 'category_id': 1, 'bbox': [float(i)*10, 0., 8., 8.], 'score': .9}
                   for i in range(4)]
    predictions.append({'image_id': 1, 'category_id': 1, 'bbox': [45., 0., 8., 8.], 'score': .1})
    data = {'predictions': predictions, 'matches': {0: 1, 1: 2, 2: 3, 3: 4},
            'known': {1: {'class_name': 'carrot', 'supported_by_head': True},
                      2: {'class_name': 'carrot', 'supported_by_head': True},
                      3: {'class_name': 'spinach', 'supported_by_head': False}}}
    candidates = {i: {'parent_id': 1, 'candidate_class_id': 0,
                      'candidate_class_name': 'carrot', 'confidence': .9} for i in range(4)}
    return data, candidates


def test_unsupported_known_targets_prevent_optimistic_calibration():
    data, candidates = calibration_fixture()
    thresholds = calibrate(CLASSES, data, candidates, target_precision=.9, minimum=2)
    parent = thresholds['parents']['1']
    assert parent['known_fine_matched_validation_predictions'] == 3
    assert parent['unsupported_known_fine_validation_predictions'] == 1
    assert parent['enabled'] is False
    assert parent['grid'][0]['accepted'] == 3
    assert parent['grid'][0]['correct'] == 2
    assert thresholds['test_used_for_selection'] is False


def test_enrichment_never_changes_boxes_scores_or_coarse_classes():
    data, candidates = calibration_fixture()
    before = deepcopy(data)
    thresholds = calibrate(CLASSES, data, candidates, target_precision=.6, minimum=2)
    output = enrich(data, candidates, thresholds)
    assert data == before
    assert [core(row) for row in output] == before['predictions']
    assert len(output) == len(before['predictions'])
    assert output[-1]['fine']['accepted'] is False  # Low-confidence coarse row still preserved.
    score = fine_scores(data, output)
    assert score['known_fine_gt_total'] == 3
    assert score['known_fine_gt_supported_by_head'] == 2
    assert score['wrong_accepted_fine'] == 1
    assert score['accepted_fine_with_unknown_gt'] == 1
    assert score['end_to_end_known_fine_recall'] == pytest.approx(2/3)
    assert score['fine_precision_on_matched_known_gt'] == pytest.approx(2/3)


def test_known_target_join_keeps_unsupported_labels_and_rejects_duplicates():
    sample = {'source_id': 10, 'crop_id': 's10_o0', 'class_id': None,
              'class_name': 'spinach', 'parent_id': 1, 'fine_annotation_known': True,
              'bbox_xywh_normalized': [.1, .2, .3, .4]}
    dataset = {'splits': {'val': [sample]}}
    doc = {'images': [{'id': 1, 'file_name': 'example.jpg', 'width': 100, 'height': 100}],
           'annotations': [{'id': 7, 'image_id': 1, 'category_id': 1, 'bbox': [10., 20., 30., 40.]}]}
    manifest = {'example.jpg': {'source_id': 10}}
    known = known_fine_annotations(dataset, 'val', doc, manifest)
    assert known[7]['class_name'] == 'spinach'
    assert known[7]['supported_by_head'] is False
    dataset['splits']['val'].append(deepcopy(sample))
    with pytest.raises(ValueError, match='Duplicate fine annotation'):
        known_fine_annotations(dataset, 'val', doc, manifest)
