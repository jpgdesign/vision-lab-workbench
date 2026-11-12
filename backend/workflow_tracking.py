"""Bounded observation-only IoU tracking; no model imports or installations."""
from __future__ import annotations

import math
from urllib.parse import quote


MAX_ACTIVE_TRACKS = 512


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parameter(value, name, minimum, maximum):
    if not _finite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}")
    return float(value)


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _observation(item):
    if not isinstance(item, dict) or item.get("observed", True) is False:
        return None
    box, score, label = item.get("box"), item.get("score"), item.get("label")
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(v) for v in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1] or not _finite(score) or not 0 <= score <= 1:
        return None
    if not isinstance(label, str) or not label or len(label) > 256:
        return None
    result = dict(item)
    result["box"], result["score"] = list(map(float, box)), float(score)
    if "keypoints" in item:
        points = item["keypoints"]
        if not isinstance(points, (list, tuple)) or not 1 <= len(points) <= 1024:
            return None
        if any(not isinstance(p, (list, tuple)) or len(p) != 3 or not all(_finite(x) for x in p) or not 0 <= p[2] <= 1 for p in points):
            return None
        result["keypoints"] = [list(map(float, p)) for p in points]
    return result


def _class(item):
    # Exact class labels are the portable class identity across detector adapters.
    # A model/label-map switch should reset the containing workflow session.
    return item["label"]


def _solver():
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    return linear_sum_assignment


class TrackerNode:
    """Assign local IDs by same-class IoU; crossing people can switch IDs.

    Lost states may be matched again within max_age_seconds, but no predicted or
    stale boxes are emitted. age_seconds is elapsed source time since first
    observation, not a claim of uninterrupted visibility.
    """

    def __init__(self, params, node_id="track"):
        if not isinstance(params, dict):
            raise ValueError("Tracker params must be an object")
        labels = params.get("labels", [])
        if not isinstance(labels, list) or len(labels) > 512 or any(not isinstance(x, str) or not x.strip() or len(x) > 256 for x in labels):
            raise ValueError("labels must be a list of nonempty class names")
        self.labels = set(labels)
        self.min_iou = _parameter(params.get("min_iou", .15), "min_iou", 0, 1)
        self.max_age_seconds = _parameter(params.get("max_age_seconds", 1), "max_age_seconds", 0, 3600)
        self.association_mode = params.get("association_mode", "iou")
        if self.association_mode not in ("iou", "recent_iou"):
            raise ValueError("association_mode must be iou or recent_iou")
        self.node_id = str(node_id)
        self._assignment = _solver()
        self.method = "hungarian_iou_same_class" if self._assignment else "greedy_iou_same_class"
        if self.association_mode == "recent_iou":
            self.method += "_recent_first"
        self.reset()

    def reset(self):
        self._generation = getattr(self, "_generation", 0) + 1
        self._tracks, self._next_id = {}, 1
        self._source, self._last_time = None, None
        self._last_observed_ids, self._last_fresh_time = set(), None

    def _matches(self, observations):
        ids = sorted(self._tracks)
        if self.association_mode == "recent_iou":
            # First keep continuity with tracks observed in the previous fresh
            # frame. Only remaining detections may revive a temporarily lost
            # track; a stale, slightly better box cannot steal an active match.
            active = [track_id for track_id in ids if track_id in self._last_observed_ids]
            matches = self._assign(active, observations)
            remaining = [index for index in range(len(observations)) if index not in matches]
            lost = [track_id for track_id in ids if track_id not in self._last_observed_ids]
            recovered = self._assign(lost, [observations[index] for index in remaining])
            matches.update({remaining[index]: track_id for index, track_id in recovered.items()})
            return matches
        return self._assign(ids, observations)

    def _assign(self, ids, observations):
        if not ids or not observations:
            return {}
        candidates = []
        costs = [[1e6 for _ in observations] for _ in ids]
        for row, track_id in enumerate(ids):
            state = self._tracks[track_id]
            for column, item in enumerate(observations):
                if _class(item) != state["label"]:
                    continue
                overlap = _iou(state["box"], item["box"])
                if overlap > 0 and overlap >= self.min_iou:
                    costs[row][column] = 1 - overlap
                    candidates.append((-overlap, track_id, column))
        if self._assignment:
            rows, columns = self._assignment(costs)
            return {int(column): ids[int(row)] for row, column in zip(rows, columns) if costs[int(row)][int(column)] < 1e6}
        # Explicit deterministic fallback, never represented as Hungarian/ReID.
        matches, used = {}, set()
        for _, track_id, column in sorted(candidates):
            if track_id not in used and column not in matches:
                matches[column] = track_id
                used.add(track_id)
        return matches

    def process(self, inputs, frame):
        if not isinstance(inputs, dict) or not isinstance(frame, dict):
            raise ValueError("inputs and frame must be objects")
        t = _parameter(frame.get("timestamp"), "frame.timestamp", 0, 1e15)
        _parameter(frame.get("width"), "frame.width", 1, 100000)
        _parameter(frame.get("height"), "frame.height", 1, 100000)
        source = frame.get("source_id")
        if not isinstance(source, str) or not source:
            raise ValueError("frame.source_id must be a nonempty string")
        scope = (source, str(frame.get("session_id", "")))
        reset_reason = None
        if self._source != scope:
            reset_reason = "new_source_or_session"
            self.reset()
        elif self._last_time is not None and t < self._last_time:
            reset_reason = "source_time_reversed"
            self.reset()
        self._source, self._last_time = scope, t
        expired = [i for i, state in self._tracks.items() if t - state["last_seen"] > self.max_age_seconds]
        for track_id in expired:
            self._tracks.pop(track_id)
        packet = inputs.get("observations", {})
        if not isinstance(packet, dict) or not isinstance(packet.get("items", []), list):
            raise ValueError("observations must be a packet with an items list")
        if self.association_mode == "recent_iou" and (
                packet.get("fresh", True) is False or packet.get("timestamp", t) != t
                or self._last_fresh_time == t):
            # A repeated/cached inference packet is not a new observation and
            # must not keep either identity TTL or posture dwell alive.
            return {"tracks": {"items": [], "timestamp": t, "frame": dict(frame), "fresh": False,
                               "tracking": self._diagnostics(reset_reason, len(expired), stale=True)}}
        valid, invalid_count, label_filtered = [], 0, 0
        for index, raw in enumerate(packet.get("items", [])):
            item = _observation(raw)
            if item is None:
                invalid_count += 1
            elif self.labels and item["label"] not in self.labels:
                label_filtered += 1
            else:
                valid.append((index, item))
        limited = max(0, len(valid) - MAX_ACTIVE_TRACKS)
        if limited:
            valid = sorted(sorted(valid, key=lambda pair: (-pair[1]["score"], pair[0]))[:MAX_ACTIVE_TRACKS])
        observations = [item for _, item in valid]
        matches = self._matches(observations)
        needed = len(observations) - len(matches)
        eviction_count = max(0, len(self._tracks) + needed - MAX_ACTIVE_TRACKS)
        assigned_ids = set(matches.values())
        removable = sorted((state["last_seen"], track_id) for track_id, state in self._tracks.items() if track_id not in assigned_ids)
        for _, track_id in removable[:eviction_count]:
            self._tracks.pop(track_id)
        output = []
        for index, item in enumerate(observations):
            track_id = matches.get(index)
            if track_id is None:
                track_id, self._next_id = self._next_id, self._next_id + 1
                state = {"first_seen": t}
                self._tracks[track_id] = state
            else:
                state = self._tracks[track_id]
            state.update(box=item["box"], label=item["label"], last_seen=t)
            track_key = "/".join(quote(str(x), safe="") for x in (scope[0], scope[1], self.node_id, self._generation, track_id))
            output.append({**item, "track_id": track_id, "track_key": track_key, "observed": True,
                           "age_seconds": t - state["first_seen"]})
        if self.association_mode == "recent_iou":
            self._last_observed_ids = {item["track_id"] for item in output}
            self._last_fresh_time = t
        diagnostics = self._diagnostics(reset_reason, len(expired), invalid=invalid_count, filtered=label_filtered,
                                        limited=limited, evictions=eviction_count)
        return {"tracks": {"items": output, "timestamp": t, "frame": dict(frame), "tracking": diagnostics,
                           **({"fresh": True} if self.association_mode == "recent_iou" else {})}}

    def _diagnostics(self, reset_reason, expired, *, invalid=0, filtered=0, limited=0, evictions=0, stale=False):
        return {"method": self.method, "association_mode": self.association_mode,
                       "active_tracks": len(self._tracks), "max_active_tracks": MAX_ACTIVE_TRACKS,
                       "invalid_observations": invalid, "label_filtered": filtered,
                       "capacity_filtered_observations": limited, "capacity_evictions": evictions,
                       "expired_tracks": expired, "reset_reason": reset_reason, "stale_packet_ignored": stale,
                       "identity_scope": "source/session/node/generation/local_id", "ghost_predictions": False,
                       "limitation": "IoU association has no motion prediction or appearance ReID; crossings/occlusion may switch IDs."}
