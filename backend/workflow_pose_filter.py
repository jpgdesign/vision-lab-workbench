"""Conservative same-frame duplicate removal for one COCO17 pose producer.

This is not person ReID. Overlapping boxes alone never justify suppression;
four or more matching, confident joints must also agree very closely.
"""
import math


POSE_DUPLICATE_IOU = .95
POSE_DUPLICATE_JOINT_CONFIDENCE = .5
POSE_DUPLICATE_MIN_JOINTS = 4
POSE_DUPLICATE_MAX_JOINT_DISTANCE = .025


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _geometry(item):
    if not isinstance(item, dict) or item.get("label") != "person":
        return None
    box, points, score = item.get("box"), item.get("keypoints"), item.get("score")
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(x) for x in box)
            or box[2] <= box[0] or box[3] <= box[1]
            or not _finite(score) or not 0 <= score <= 1
            or not isinstance(points, (list, tuple)) or len(points) != 17):
        return None
    valid = {}
    for index, point in enumerate(points):
        if (isinstance(point, (list, tuple)) and len(point) == 3 and all(_finite(x) for x in point)
                and POSE_DUPLICATE_JOINT_CONFIDENCE <= point[2] <= 1):
            valid[index] = point
    return box, valid, float(score)


def _duplicate(a, b):
    box_a, joints_a, _ = a
    box_b, joints_b, _ = b
    intersection = (max(0., min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
                    * max(0., min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])))
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if intersection / (area_a + area_b - intersection) < POSE_DUPLICATE_IOU:
        return False
    common = joints_a.keys() & joints_b.keys()
    if len(common) < POSE_DUPLICATE_MIN_JOINTS:
        return False
    # Using the smaller diagonal and the largest joint discrepancy prevents
    # a large box or many matching facial points from hiding a disagreeing limb.
    scale = min(math.dist(box_a[:2], box_a[2:]), math.dist(box_b[:2], box_b[2:]))
    return all(math.dist(joints_a[i][:2], joints_b[i][:2]) <= scale * POSE_DUPLICATE_MAX_JOINT_DISTANCE
               for i in common)


def filter_pose_duplicates(items):
    """Return original surviving records and counts; never synthesize joints.

Call only on a single model's outputs for one frame. Different available
producer identifiers are additionally kept separate. Invalid/insufficient
records are preserved for downstream validation, not silently discarded.
    """
    geometry = [_geometry(item) for item in items]
    order = sorted(range(len(items)), key=lambda i: (-geometry[i][2], i) if geometry[i] else (1., i))
    kept = []
    for index in order:
        if geometry[index] is None:
            kept.append(index)
            continue
        duplicate = False
        for other in kept:
            if geometry[other] is None:
                continue
            if any(items[index].get(key) != items[other].get(key) for key in ("producer_node", "producer_model")):
                continue
            if _duplicate(geometry[index], geometry[other]):
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    result = [items[i] for i in sorted(kept)]
    return result, {
        "pose_raw_count": len(items), "pose_output_count": len(result),
        "pose_suppressed_duplicates": len(items) - len(result),
        "pose_deduplication_method": "same_frame_iou_0.95_and_4_confident_joints_max_distance_0.025_box_diagonal",
    }
