"""Per-session office crossing and conservative seated-posture rules.

This module does not infer attention. ``認真`` is a user-assigned name for a
visible seated-pose rule, not an engagement, productivity or mental-state score.
All state belongs to one node instance and is reset when its source changes.
"""
import math
import uuid


OFFICE_KINDS = {"office_monitor"}
MAX_TRACKS = 512
DEFAULTS = {
    "line_x": .5, "line_y_start": 0., "line_y_end": 1.,
    "hysteresis": .025, "min_travel": .08, "side_dwell_seconds": .15,
    "max_gap_seconds": 1., "cooldown_seconds": 1.5, "track_ttl_seconds": 5.,
    "min_score": .25, "keypoint_confidence": .5, "sitting_dwell_seconds": 1.,
    "seated_label": "認真（坐姿規則）", "max_tracks": MAX_TRACKS,
}


def office_catalog(node, param, port):
    return [node("office_monitor", "辦公室 · 進出計數與坐姿規則", "規則",
                 "中央線左→右為出去、右→左為進來；坐姿顯示使用者定義的『認真（坐姿規則）』。此為可見姿態規則，不推斷專注或工作表現；腿部遮擋時回報不明。",
                 [port("tracks", "tracks")], [port("actions", "actions"), port("events", "events")], [
                     param("line_x", "中線水平位置（0 左／1 右）", .5, min=.002, max=.998, step=.005),
                     param("line_y_start", "線段上端（相對高度）", 0., min=0, max=1, step=.01),
                     param("line_y_end", "線段下端（相對高度）", 1., min=0, max=1, step=.01),
                     param("hysteresis", "中線兩側忽略帶（畫面寬比例）", .025, min=.001, max=.25, step=.005),
                     param("min_travel", "跨線最小位移（畫面寬比例）", .08, min=0, max=1, step=.01),
                     param("side_dwell_seconds", "跨側穩定確認（秒）", .15, min=0, max=60, step=.05),
                     param("cooldown_seconds", "同人跨線事件間隔（秒）", 1.5, min=0, max=60, step=.1),
                     param("max_gap_seconds", "有效觀察最大間隔（秒）", 1., min=.01, max=60, step=.1),
                     param("track_ttl_seconds", "追蹤狀態保留（秒）", 5., min=.01, max=300, step=.1),
                     param("min_score", "人物信心下限", .25, min=0, max=1, step=.01),
                     param("keypoint_confidence", "關節可見信心下限", .5, min=0, max=1, step=.05),
                     param("sitting_dwell_seconds", "坐姿持續確認（秒）", 1., min=0, max=60, step=.1),
                     param("seated_label", "坐姿規則顯示名稱", "認真（坐姿規則）"),
                     param("max_tracks", "最多保留追蹤數", MAX_TRACKS, min=1, max=MAX_TRACKS, step=1),
                 ])]


def _number(value, name, low=0., high=86400.):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def validate_office_params(params):
    """Return validated defaults, usable by the graph's preflight validator."""
    if not isinstance(params, dict):
        raise ValueError("Office parameters must be an object")
    extra = set(params) - set(DEFAULTS)
    if extra:
        raise ValueError(f"Unknown office parameter: {', '.join(sorted(extra))}")
    result = {**DEFAULTS, **params}
    for name in ("line_x", "line_y_start", "line_y_end", "min_travel", "min_score", "keypoint_confidence"):
        result[name] = _number(result[name], name, 0., 1.)
    result["hysteresis"] = _number(result["hysteresis"], "hysteresis", .001, .25)
    if not result["hysteresis"] < result["line_x"] < 1 - result["hysteresis"]:
        raise ValueError("line_x must leave room for hysteresis on both sides")
    if result["line_y_start"] >= result["line_y_end"]:
        raise ValueError("line_y_start must be below line_y_end in normalized coordinates")
    for name in ("side_dwell_seconds", "cooldown_seconds", "sitting_dwell_seconds"):
        result[name] = _number(result[name], name, 0., 60.)
    result["max_gap_seconds"] = _number(result["max_gap_seconds"], "max_gap_seconds", .01, 60.)
    result["track_ttl_seconds"] = _number(result["track_ttl_seconds"], "track_ttl_seconds", result["max_gap_seconds"], 300.)
    max_tracks = _number(result["max_tracks"], "max_tracks", 1, MAX_TRACKS)
    if max_tracks != int(max_tracks):
        raise ValueError("max_tracks must be an integer")
    result["max_tracks"] = int(max_tracks)
    if not isinstance(result["seated_label"], str) or not result["seated_label"].strip() or len(result["seated_label"]) > 100:
        raise ValueError("seated_label must be 1–100 characters")
    # A custom display label must retain its observable rule qualifier.
    if "坐姿規則" not in result["seated_label"]:
        result["seated_label"] = result["seated_label"].strip() + "（坐姿規則）"
    return result


def _track_key(item):
    key = item.get("track_key")
    if isinstance(key, str) and key:
        return key
    key = item.get("track_id")
    if isinstance(key, (str, int)) and not isinstance(key, bool) and key != "":
        return f"track:{key}"
    return None


def _actor(item):
    return {"scope": "person", **{k: item[k] for k in ("track_id", "track_key", "box") if k in item}}


def _box(item):
    box = item.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in box):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def _point(item, index, threshold, width, height):
    points = item.get("keypoints")
    if not isinstance(points, (list, tuple)) or len(points) != 17:
        return None
    point = points[index]
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
        return None
    x, y, score = point
    if not threshold <= score <= 1 or not 0 <= x <= width or not 0 <= y <= height:
        return None
    return point


def _angle(a, b, c):
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*u) * math.hypot(*v)
    if norm < 1e-6:
        return None
    return math.degrees(math.acos(max(-1., min(1., (u[0] * v[0] + u[1] * v[1]) / norm))))


def seated_pose_evidence(item, width, height, threshold=.5):
    """COCO17 geometry, not a learned action or attention classifier.

    Both hips, knees, ankles and shoulders must be visible. Requiring ankles
    intentionally returns unknown for desk/chair occlusion. A 2-D side-view
    seated candidate is still ambiguous with some crouches; this rule cannot
    certify that someone is sitting on a chair or concentrating on work.
    """
    points = {i: _point(item, i, threshold, width, height) for i in (5, 6, 11, 12, 13, 14, 15, 16)}
    if not all(points.values()):
        return "unknown", 0., {"reason": "both_shoulders_hips_knees_ankles_must_be_visible"}
    shoulder = tuple((points[5][i] + points[6][i]) / 2 for i in (0, 1))
    hip = tuple((points[11][i] + points[12][i]) / 2 for i in (0, 1))
    torso = math.dist(shoulder, hip)
    if torso < 5 or hip[1] <= shoulder[1]:
        return "unknown", 0., {"reason": "degenerate_or_inverted_torso"}
    tilt = math.degrees(math.atan2(abs(hip[0] - shoulder[0]), hip[1] - shoulder[1]))
    knees = [_angle(points[h], points[k], points[a]) for h, k, a in ((11, 13, 15), (12, 14, 16))]
    hips = [_angle(points[s], points[h], points[k]) for s, h, k in ((5, 11, 13), (6, 12, 14))]
    details = {"torso_angle_degrees": tilt, "knee_angles_degrees": knees, "hip_angles_degrees": hips,
               "skeleton_schema": "coco17", "limitations": "2D seated-pose candidate; not proof of attention or chair contact"}
    if any(x is None for x in knees + hips):
        return "unknown", 0., {**details, "reason": "degenerate_leg"}
    score = min(p[2] for p in points.values())
    upright = tilt <= 35
    if (upright and all(155 <= x <= 180 for x in knees)
            and all(points[h][1] < points[k][1] < points[a][1] for h, k, a in ((11, 13, 15), (12, 14, 16)))):
        return "standing", score, details
    seated = (upright and all(50 <= x <= 135 for x in knees)
              and all(45 <= x <= 135 for x in hips)
              and all(-.25 * torso <= points[k][1] - points[h][1] <= .75 * torso
                      and points[a][1] - points[k][1] >= .2 * torso
                      for h, k, a in ((11, 13, 15), (12, 14, 16))))
    return ("seated_candidate", score, details) if seated else ("unknown", 0., {**details, "reason": "no_reliable_standing_or_seated_rule"})


class OfficeNode:
    """Count confirmed crossings and report a temporally supported pose rule."""

    def __init__(self, kind, params, node_id="office"):
        if kind not in OFFICE_KINDS:
            raise ValueError(f"Unsupported office node: {kind}")
        self.kind, self.node_id = kind, str(node_id)
        self.params = validate_office_params(params)
        self._generation = 0
        self._run_token = uuid.uuid4().hex[:12]
        self.reset()

    def reset(self):
        self._states, self.counts = {}, {"in": 0, "out": 0}
        self._source = self._time = None
        self._generation += 1
        self._event_sequence = 0

    def close(self):
        self.reset()

    def _new_state(self, t):
        return {"last_seen": t, "side": None, "anchor": None, "previous": None,
                "candidate": None, "candidate_since": t, "candidate_samples": 0,
                "intersected_line": False, "last_cross": -math.inf,
                "seated_since": None, "seated_samples": 0, "seated_confirmed": False}

    def _event(self, kind, label, item, t, **details):
        self._event_sequence += 1
        return {**_actor(item), "kind": kind, "label": label, "signal": kind,
                "timestamp": t, "severity": "info",
                "event_id": f"{self.node_id}:{self._run_token}:{self._generation}:{self._event_sequence}",
                "details": details}

    def _crossing(self, item, state, point, t):
        x, y = point
        p, events = self.params, []
        previous = state["previous"]
        if previous and (previous[0] - p["line_x"]) * (x - p["line_x"]) <= 0 and x != previous[0]:
            ratio = (p["line_x"] - previous[0]) / (x - previous[0])
            cross_y = previous[1] + ratio * (y - previous[1])
            if p["line_y_start"] <= cross_y <= p["line_y_end"]:
                state["intersected_line"] = True
        state["previous"] = point
        side = -1 if x < p["line_x"] - p["hysteresis"] else 1 if x > p["line_x"] + p["hysteresis"] else 0
        if not side:
            state.update(candidate=None, candidate_samples=0)
            return events
        if side == state["side"]:
            # Use a recently observed origin on the confirmed side, not an
            # ancient point from before a long excursion around a finite line.
            state.update(anchor=point, candidate=None, candidate_samples=0, intersected_line=False)
            return events
        if state["candidate"] != side:
            state.update(candidate=side, candidate_since=t, candidate_samples=1)
            return events
        state["candidate_samples"] += 1
        if state["candidate_samples"] < 2 or t - state["candidate_since"] + 1e-9 < p["side_dwell_seconds"]:
            return events
        old_side, anchor = state["side"], state["anchor"]
        if old_side is not None and anchor is not None and abs(x - anchor[0]) < p["min_travel"]:
            return events
        if old_side is not None and state["intersected_line"] and t - state["last_cross"] >= p["cooldown_seconds"]:
            direction = "out" if old_side == -1 else "in"
            self.counts[direction] += 1
            state["last_cross"] = t
            events.append(self._event("line_crossing", "出去" if direction == "out" else "進來", item, t,
                                      direction=direction, image_direction="left_to_right" if direction == "out" else "right_to_left",
                                      anchor="box_center", from_point=list(anchor), to_point=list(point),
                                      counts=dict(self.counts), line_x=p["line_x"],
                                      rule="stable tracked center crossed the configured line"))
        # Initial appearance and cooldown-suppressed crossings establish the
        # new side but do not create a delayed phantom count on a later frame.
        state.update(side=side, anchor=point, candidate=None, candidate_samples=0, intersected_line=False)
        return events

    def _posture(self, item, state, frame, t):
        label, score, details = seated_pose_evidence(item, frame["width"], frame["height"], self.params["keypoint_confidence"])
        event = None
        if label == "seated_candidate":
            if state["seated_since"] is None:
                state.update(seated_since=t, seated_samples=0)
            state["seated_samples"] += 1
            if state["seated_samples"] >= 2 and t - state["seated_since"] + 1e-9 >= self.params["sitting_dwell_seconds"]:
                label = "seated_rule"
                if not state["seated_confirmed"]:
                    event = self._event("seated_rule", self.params["seated_label"], item, t,
                                        method="coco17_geometry_rule", observed_seconds=t - state["seated_since"],
                                        user_assigned_label=True, infers_attention=False)
                state["seated_confirmed"] = True
        else:
            state.update(seated_since=None, seated_samples=0, seated_confirmed=False)
        display = {"seated_rule": self.params["seated_label"], "seated_candidate": "坐姿候選（等待持續確認）",
                   "standing": "站姿（姿態規則）", "unknown": "姿態不明（遮擋或證據不足）"}[label]
        action = {**_actor(item), "label": display, "signal": label,
                  "kind": "unknown" if label == "unknown" else "posture_rule",
                  "score": score, "timestamp": t, "observed": True,
                  "method": "coco17_geometry_rule", "infers_attention": False,
                  "score_semantics": "minimum_keypoint_confidence_not_action_probability",
                  "details": {**details, "user_assigned_label": self.params["seated_label"],
                              "observed_seconds": 0. if state["seated_since"] is None else t - state["seated_since"]}}
        return action, event

    def _result(self, actions, events, frame, fresh=True):
        overlay = {"kind": "line", "points": [[self.params["line_x"], self.params["line_y_start"]],
                                                 [self.params["line_x"], self.params["line_y_end"]]],
                   "label": f"出去 L→R: {self.counts['out']} / 進來 R→L: {self.counts['in']}",
                   "count": self.counts["in"] + self.counts["out"]}
        common = {"frame": frame, "timestamp": frame["timestamp"], "fresh": fresh, "counts": dict(self.counts),
                  "diagnostics": {"active_tracks": len(self._states), "infers_attention": False,
                                  "direction_coordinate_space": "processed_image_before_any_display_mirroring",
                                  "limitation": "Track ID changes or occluded legs may prevent a count/posture decision"}}
        return {"actions": {**common, "items": actions}, "events": {**common, "items": events, "overlays": [overlay]}}

    def process(self, inputs, frame):
        t = _number(frame.get("timestamp"), "timestamp", 0., 1e15)
        width = _number(frame.get("width"), "width", 1, 100000)
        height = _number(frame.get("height"), "height", 1, 100000)
        source = (frame.get("source_id"), frame.get("session_id"), width, height)
        if not isinstance(source[0], str) or not source[0]:
            raise ValueError("Office frame requires source_id")
        if self._source != source or (self._time is not None and t < self._time):
            self.reset()
        duplicate = self._time is not None and t == self._time
        self._source, self._time = source, t
        for key, state in list(self._states.items()):
            if t - state["last_seen"] > self.params["track_ttl_seconds"]:
                del self._states[key]
        packet = inputs.get("tracks", {})
        if not isinstance(packet, dict) or not isinstance(packet.get("items", []), list):
            raise ValueError("tracks must be a packet with an items list")
        if duplicate or packet.get("fresh", True) is False or packet.get("timestamp", t) != t:
            return self._result([], [], frame, fresh=False)
        actions, events, seen = [], [], set()
        for item in packet.get("items", []):
            if not isinstance(item, dict) or item.get("observed", True) is False:
                continue
            key, box = _track_key(item), _box(item)
            score = item.get("score", 0.)
            if (key is None or key in seen or box is None or item.get("label") not in {"person", "人", "人物"}
                    or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                    or not self.params["min_score"] <= score <= 1
                    or item.get("timestamp", t) != t):
                continue
            cx, cy = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
            if not 0 <= cx <= 1 or not 0 <= cy <= 1:
                continue
            seen.add(key)
            state = self._states.get(key)
            if state is None or t - state["last_seen"] > self.params["max_gap_seconds"]:
                state = self._new_state(t)
                self._states[key] = state
            events.extend(self._crossing(item, state, (cx, cy), t))
            action, event = self._posture(item, state, frame, t)
            actions.append(action)
            if event:
                events.append(event)
            state["last_seen"] = t
        # An explicit fresh frame without this track breaks its dwell evidence.
        # Keep its established side briefly; max_gap prevents teleport counts.
        for key, state in self._states.items():
            if key not in seen:
                state.update(candidate=None, candidate_samples=0, seated_since=None, seated_samples=0, seated_confirmed=False)
        if len(self._states) > self.params["max_tracks"]:
            oldest = sorted(self._states, key=lambda k: (self._states[k]["last_seen"], k))
            for key in oldest[:len(self._states) - self.params["max_tracks"]]:
                del self._states[key]
        return self._result(actions, events, frame)
