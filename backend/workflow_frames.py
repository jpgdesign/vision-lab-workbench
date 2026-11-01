"""Route connected images and reproject geometry between workflow branches.

Frames carry a homogeneous ``to_source`` transform in pixel coordinates. Rules
and tracking use source coordinates; image/model nodes use their connected image.
This lets a crop feed a model while its boxes are drawn over the original source.
"""
from __future__ import annotations

import numpy as np


IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def transform(frame):
    matrix = np.asarray(frame.get("to_source", IDENTITY), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("影格座標轉換無效。")
    return matrix


def coordinate_space(frame):
    return {"width": frame["width"], "height": frame["height"], "to_source": transform(frame).tolist()}


def _points(points, matrix):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("節點輸出含無效座標。")
    mapped = np.column_stack((values, np.ones(len(values)))) @ matrix.T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("節點座標無法投影到目標影格。")
    return mapped[:, :2] / mapped[:, 2:]


def _box(box, matrix, width, height):
    x1, y1, x2, y2 = box
    points = _points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], matrix)
    low = np.maximum(points.min(axis=0), [0, 0])
    high = np.minimum(points.max(axis=0), [width, height])
    if np.any(high <= low):
        return None
    return [float(low[0]), float(low[1]), float(high[0]), float(high[1])]


def _item(item, matrix, source, target):
    result = dict(item)
    width, height = target["width"], target["height"]
    for key in ("box", "subject_box", "object_box"):
        if key in item:
            result[key] = _box(item[key], matrix, width, height)
            if result[key] is None:
                return None  # This observation is entirely outside this crop.
    if item.get("keypoints"):
        points = _points([p[:2] for p in item["keypoints"]], matrix)
        result["keypoints"] = [[float(x), float(y), float(p[2]) if 0 <= x < width and 0 <= y < height else 0.]
                               for (x, y), p in zip(points, item["keypoints"])]
    if item.get("polygon"):
        result["polygon"] = _points(item["polygon"], matrix).tolist()
    if "area" in item:
        # Contour area refers to the full polygon (before viewport clipping).
        result["area"] = float(item["area"]) * abs(float(np.linalg.det(matrix[:2, :2])))
        if "area_ratio" in item:
            result["area_ratio"] = result["area"] / (width * height)
    if "mask" in item:
        import cv2
        from .segmentation import attach_mask, decode_mask
        mask = decode_mask(item["mask"], (source["height"], source["width"]))
        # Geometric coordinates describe pixel edges, whereas warpPerspective
        # samples integer pixel centers. The half-pixel change also matters for
        # resize and right-angle rotation (otherwise an outer row disappears).
        center = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
        raster_matrix = np.linalg.solve(center, matrix @ center)
        mapped = cv2.warpPerspective(mask.astype(np.uint8), raster_matrix, (width, height),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result = attach_mask(result, mapped, (width, height))
    return result


def reproject_packet(packet, target):
    """Copy packet geometry; never mutate a sibling branch or its image."""
    source = packet.get("frame", target)
    if source.get("source_id") != target.get("source_id"):
        raise ValueError("不能合併不同影像來源的座標。")
    matrix = np.linalg.solve(transform(target), transform(source))
    same = (source["width"], source["height"]) == (target["width"], target["height"]) and np.allclose(matrix, np.eye(3), atol=1e-10)
    result = {**packet, "frame": target}
    if same:
        return result
    for key in ("items", "display_items"):
        if key in packet:
            result[key] = [mapped for item in packet[key] if (mapped := _item(item, matrix, source, target)) is not None]
    if "overlays" in packet:
        result["overlays"] = []
        for overlay in packet["overlays"]:
            copied = dict(overlay)
            if "points" in overlay:
                points = np.asarray(overlay["points"]) * [source["width"], source["height"]]
                copied["points"] = (_points(points, matrix) / [target["width"], target["height"]]).tolist()
            result["overlays"].append(copied)
    return result


def route_inputs(inputs, source_frame):
    # Image consumers obey the actual frame wire. Coordinate-only nodes use the
    # source so the visual zone editor and motion measurements have one basis.
    selected = inputs.get("frame", {}).get("frame", source_frame)
    routed = {}
    for name, value in inputs.items():
        if name == "frame":
            routed[name] = value
        elif isinstance(value, list):
            routed[name] = [reproject_packet(part, selected) for part in value]
        else:
            routed[name] = reproject_packet(value, selected)
    return routed, selected
