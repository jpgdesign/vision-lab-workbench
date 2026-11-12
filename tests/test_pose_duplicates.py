"""Same-frame pose duplicates must not create competing person tracks."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.workflow_model_nodes import ModelNode
from backend.workflow_pose_filter import filter_pose_duplicates


def person(score=.9, dx=0., visible=17):
    return {"label": "person", "score": score, "box": [100., 50., 300., 350.],
            "keypoints": [[130. + i * 5 + dx, 70. + i * 12, .9 if i < visible else .1]
                          for i in range(17)], "observed": True}


def test_near_identical_pose_keeps_highest_scored_whole_record():
    lower, higher = person(.6, dx=2), person(.9)
    higher["custom_metadata"] = "keep me"
    before = deepcopy([lower, higher])
    result, diagnostic = filter_pose_duplicates([lower, higher])
    assert result == [higher] and result[0] is higher
    assert [lower, higher] == before
    assert diagnostic["pose_raw_count"] == 2
    assert diagnostic["pose_output_count"] == 1
    assert diagnostic["pose_suppressed_duplicates"] == 1


def test_same_box_different_people_with_different_joints_are_kept():
    a, b = person(), person(.8, dx=15)
    assert filter_pose_duplicates([a, b])[0] == [a, b]


def test_one_disagreeing_confident_joint_cannot_be_diluted_by_matching_face():
    a, b = person(), person(.8)
    b["keypoints"][9][0] += 30
    assert filter_pose_duplicates([a, b])[0] == [a, b]


@pytest.mark.parametrize("visible", [0, 1, 2, 3])
def test_insufficient_shared_visible_joints_never_suppress(visible):
    a, b = person(visible=visible), person(.8, visible=visible)
    assert filter_pose_duplicates([a, b])[0] == [a, b]


def test_four_matching_confident_joints_are_sufficient():
    a, b = person(visible=4), person(.8, dx=1, visible=4)
    assert filter_pose_duplicates([a, b])[0] == [a]


def test_visibility_must_be_shared_at_same_joint_indices():
    a, b = person(visible=4), person(.8, visible=0)
    for i in range(4, 8):
        b["keypoints"][i][2] = .9
    assert filter_pose_duplicates([a, b])[0] == [a, b]


def test_low_confidence_disagreeing_joint_is_not_evidence():
    a, b = person(visible=4), person(.8, visible=4)
    b["keypoints"][10] = [500, 500, .1]
    assert filter_pose_duplicates([a, b])[0] == [a]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_and_boolean_joint_values_cannot_support_suppression(value):
    a, b = person(visible=4), person(.8, visible=4)
    b["keypoints"][3][0] = value
    assert filter_pose_duplicates([a, b])[0] == [a, b]


@pytest.mark.parametrize("value", [.49, 1.1, float("nan"), True])
def test_joint_confidence_is_validated(value):
    a, b = person(visible=4), person(.8, visible=4)
    b["keypoints"][3][2] = value
    assert filter_pose_duplicates([a, b])[0] == [a, b]


def test_resembling_skeletons_with_nonidentical_boxes_are_kept():
    a, b = person(), person(.8)
    b["box"][0] += 15  # IoU=.925, below conservative duplicate threshold.
    assert filter_pose_duplicates([a, b])[0] == [a, b]


@pytest.mark.parametrize("change", ["label", "producer_node", "producer_model"])
def test_different_class_or_available_producer_is_kept_separate(change):
    a, b = person(), person(.8)
    b[change] = "different"
    assert filter_pose_duplicates([a, b])[0] == [a, b]


def test_equal_scores_deterministically_keep_first_input():
    a, b = person(), person()
    result = filter_pose_duplicates([a, b])[0]
    assert len(result) == 1 and result[0] is a


def test_filter_has_no_cross_frame_memory():
    a = person()
    assert filter_pose_duplicates([a])[0] == [a]
    assert filter_pose_duplicates([a])[0] == [a]


class Tensor:
    def __init__(self, values):
        self.values = np.array(values)

    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values
    def __iter__(self): return iter(self.values)


class Boxes:
    def __init__(self, items):
        self.xyxy = Tensor([x["box"] for x in items])
        self.conf = Tensor([x["score"] for x in items])
        self.length = len(items)

    def __len__(self): return self.length


def test_model_node_packet_contains_filtered_poses_and_diagnostic_counts():
    records = [person(), person(.8, dx=1)]
    output = SimpleNamespace(boxes=Boxes(records), keypoints=SimpleNamespace(data=Tensor([x["keypoints"] for x in records])))
    model = SimpleNamespace(predict=lambda *args, **kwargs: [output])
    node = ModelNode("pose", {}, {"model_id": "fake-pose"})
    node.model = model
    frame = {"image": Image.new("RGB", (640, 360)), "width": 640, "height": 360,
             "timestamp": 0., "source_id": "test-camera"}
    result = node.process({}, frame)["poses"]
    assert len(result["items"]) == len(result["display_items"]) == 1
    assert result["diagnostics"]["pose_suppressed_duplicates"] == 1
    output.boxes = Boxes([])
    result = node.process({}, {**frame, "timestamp": .1})["poses"]
    assert result["items"] == []
    assert result["diagnostics"]["pose_raw_count"] == 0
    assert result["diagnostics"]["pose_suppressed_duplicates"] == 0
