"""CPU-only temporal/geometry rules for workflow packets.

These are transparent rules, not learned action recognition. Time is source
time in seconds; missing observations never accumulate positive evidence.
"""
from __future__ import annotations

import math


KINDS = {"zone", "posture", "condition", "sequence", "ppe", "merge"}


def _number(value, name, minimum=0, maximum=86400):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _strings(value, name, *, nonempty=False):
    if not isinstance(value, list) or len(value) > 256 or any(not isinstance(x, str) or not x.strip() or len(x) > 256 for x in value):
        raise ValueError(f"{name} must be a list of nonempty strings")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty")
    return list(value)


def _items(packet):
    if isinstance(packet, list):
        return [item for part in packet for item in _items(part)]
    if not isinstance(packet, dict):
        return []
    return [x for x in packet.get("items", []) if isinstance(x, dict)]


def _identity(item):
    if item.get("scope") == "relation":
        return ("relation", _relation_key(item) or (
            "untracked", str(item.get("subject_index")), str(item.get("object_index")), str(item.get("label"))))
    if item.get("scope") == "scene":
        return ("scene", "scene")
    if item.get("track_key") is not None:
        return ("person", str(item["track_key"]))
    if item.get("track_id") is not None:
        return ("person", str(item["track_id"]))
    # Unattributed action scores remain scene-level, never attached to a person.
    return ("scene", "scene")


def _relation_key(item):
    if item.get("relation_key") is not None:
        return str(item["relation_key"])
    if item.get("subject_track_key") is not None and item.get("object_track_key") is not None:
        return (str(item["subject_track_key"]), str(item.get("predicate", item.get("label"))), str(item["object_track_key"]))
    return None


def _actor(item):
    scope, _ = _identity(item)
    result = {"scope": scope}
    if scope == "person":
        for name in ("track_id", "track_key"):
            if name in item:
                result[name] = item[name]
    elif scope == "relation":
        for name in ("relation_key", "track_id", "track_key", "subject_index", "object_index", "subject_track_key",
                     "object_track_key", "subject_label", "object_label", "subject_box", "object_box", "predicate"):
            if name in item:
                result[name] = item[name]
    return result


def _known(item):
    return (item.get("observed", True) is not False and item.get("label") not in {"unknown", "ppe_unknown"}
            and item.get("kind") != "unknown" and item.get("value") != "unknown")


def _box(item):
    value = item.get("box")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        return None
    x1, y1, x2, y2 = map(float, value)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _inside(point, polygon):
    """Ray casting with boundary counted inside, in normalized coordinates."""
    x, y = point
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-10 and min(ax, bx) - 1e-10 <= x <= max(ax, bx) + 1e-10 and min(ay, by) - 1e-10 <= y <= max(ay, by) + 1e-10:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def _point(item, index, threshold):
    points = item.get("keypoints")
    if not isinstance(points, (list, tuple)) or len(points) != 17:
        return None
    p = points[index]
    if not isinstance(p, (list, tuple)) or len(p) < 3:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in p[:3]):
        return None
    return tuple(map(float, p[:3])) if threshold <= p[2] <= 1 else None


def _angle(a, b, c):
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    length = math.hypot(*u) * math.hypot(*v)
    if length < 1e-9:
        return None
    return math.degrees(math.acos(max(-1, min(1, (u[0] * v[0] + u[1] * v[1]) / length))))


class RuleNode:
    """Process graph packets without model, tensor, network or filesystem imports."""

    # Fits the tracker maximum (512) in all 64 permitted zones. Unbounded live
    # sessions must not retain identities or event IDs after their useful life.
    _MAX_ENTRIES = 32768

    def __init__(self, kind, params, node_id=""):
        if kind not in KINDS:
            raise ValueError(f"Unsupported workflow rule: {kind}")
        if not isinstance(params, dict):
            raise ValueError("Rule params must be an object")
        self.kind, self.node_id, self.params = kind, str(node_id), dict(params)
        p = self.params
        p["max_gap_seconds"] = _number(p.get("max_gap_seconds", 10 if kind == "sequence" else 1), "max_gap_seconds", 0.001, 3600)
        for name, default, maximum in (("min_score", .5, 1), ("keypoint_confidence", .5, 1),
                                       ("cooldown", 10, 86400), ("min_duration", 0, 3600),
                                       ("missing_hold_seconds", 2, 3600), ("dwell_seconds", 3, 3600),
                                       ("fall_transition_seconds", 1.2, 60), ("fall_hold_seconds", 1.5, 60)):
            p[name] = _number(p.get(name, default), name, 0, maximum)
        for name, default in (("require_helmet", True), ("require_vest", False), ("reset_on_complete", True)):
            if not isinstance(p.get(name, default), bool):
                raise ValueError(f"{name} must be boolean")
            p[name] = p.get(name, default)
        p["labels"] = _strings(p.get("labels", []), "labels")
        p["helmet_labels"] = _strings(p.get("helmet_labels", ["helmet", "hardhat", "hard_hat"]), "helmet_labels", nonempty=True)
        p["vest_labels"] = _strings(p.get("vest_labels", ["vest", "safety_vest"]), "vest_labels", nonempty=True)
        if kind == "sequence":
            p["steps"] = _strings(p.get("steps", []), "steps", nonempty=True)
            if len(p["steps"]) < 2 or any(a == b for a, b in zip(p["steps"], p["steps"][1:])):
                raise ValueError("steps need at least two entries; adjacent identical labels cannot identify distinct steps")
        if kind == "zone":
            if p.get("anchor", "foot") not in {"foot", "center"}:
                raise ValueError("anchor must be foot or center")
            p["anchor"] = p.get("anchor", "foot")
            limit = _number(p.get("occupancy_limit", 0), "occupancy_limit", 0, 100000)
            if not limit.is_integer():
                raise ValueError("occupancy_limit must be an integer")
            p["occupancy_limit"] = int(limit)
            zones = p.get("zones", [])
            if not isinstance(zones, list) or len(zones) > 64:
                raise ValueError("zones must be a list with at most 64 polygons")
            names, checked = set(), []
            for zone in zones:
                if not isinstance(zone, dict) or not isinstance(zone.get("name"), str) or not zone["name"].strip() or zone["name"] in names:
                    raise ValueError("Each zone needs a distinct nonempty name")
                points = zone.get("points")
                if not isinstance(points, list) or not 3 <= len(points) <= 128:
                    raise ValueError("A zone needs 3 to 128 points")
                polygon = []
                for point in points:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError("Zone points must be [x,y]")
                    polygon.append([_number(point[0], "zone x", 0, 1), _number(point[1], "zone y", 0, 1)])
                area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1])))
                if area < 1e-8:
                    raise ValueError("Zone polygon must have nonzero area")
                names.add(zone["name"])
                checked.append({"name": zone["name"], "points": polygon})
            p["zones"] = checked
        self.reset()

    def reset(self):
        self._states, self._cooldowns, self._occupancy = {}, {}, {}
        self._sample_times = {}
        self._seen_events = {}
        self._generation = getattr(self, "_generation", 0) + 1
        self._source, self._last_time, self._step, self._event_counter = None, None, 0, 0

    def process(self, inputs, frame):
        if not isinstance(inputs, dict) or not isinstance(frame, dict):
            raise ValueError("inputs and frame must be objects")
        t = _number(frame.get("timestamp"), "frame.timestamp", 0, 1e15)
        _number(frame.get("width"), "frame.width", 1, 100000)
        _number(frame.get("height"), "frame.height", 1, 100000)
        source = (str(frame.get("source_id", "")), str(frame.get("session_id", "")))
        if source != self._source or (self._last_time is not None and t < self._last_time):
            self.reset()
        self._source, self._last_time = source, t
        self._step += 1
        self._frame, self._time = dict(frame), t
        self._prune_history()
        result = getattr(self, "_" + self.kind)(inputs)
        self._prune_history()
        return result

    def _prune_history(self):
        gap, t = self.params["max_gap_seconds"], self._time
        # Incomplete sequences first emit their normal timeout in _sequence;
        # completed non-reset sequences can expire silently after inactivity.
        for key, state in list(self._states.items()):
            if key[0] == "sequence" and not state.get("complete"):
                continue
            if t - state.get("time", state.get("advanced", t)) > gap:
                self._states.pop(key, None)
        for mapping, ttl in ((self._sample_times, gap),
                             (self._cooldowns, max(gap, self.params["cooldown"])),
                             (self._seen_events, gap)):
            for key, timestamp in list(mapping.items()):
                if t - timestamp > ttl:
                    mapping.pop(key, None)
        # Preserve current/most recently observed identities before inactive
        # ones. Sorting is needed only under capacity pressure, not every frame.
        for mapping in (self._states, self._sample_times, self._cooldowns, self._seen_events):
            limit = min(10000, self._MAX_ENTRIES) if mapping is self._seen_events else self._MAX_ENTRIES
            if len(mapping) > limit:
                def last_seen(key):
                    value = mapping[key]
                    if isinstance(value, dict):
                        return value.get("time", value.get("advanced", -math.inf))
                    if mapping is self._cooldowns and key in self._states:
                        return max(value, self._states[key].get("time", value))
                    return value
                for key in sorted(mapping, key=last_seen)[:len(mapping) - limit]:
                    mapping.pop(key, None)

    def _packet(self, items, **extra):
        return {"items": items, "timestamp": self._time, "frame": self._frame, **extra}

    def _event(self, kind, label=None, item=None, severity="warning", **details):
        self._event_counter += 1
        return {"kind": kind, "label": label or kind, "severity": severity, "timestamp": self._time,
                "event_id": f"{self.node_id}:{self._source}:{self._generation}:{self._step}:{self._event_counter}",
                **_actor(item or {"scope": "scene"}), "details": details}

    def _continuous(self, key, *, positive, valid=True):
        """Return observed support seconds; unobserved gaps never contribute."""
        state = self._states.get(key)
        if not valid or not positive:
            self._states.pop(key, None)
            return None
        contiguous = state and state.get("step") == self._step - 1 and 0 <= self._time - state["time"] <= self.params["max_gap_seconds"]
        if not contiguous:
            state = {"support": 0.0, "start": self._time, "emitted": False}
        else:
            state["support"] += self._time - state["time"]
        state.update(time=self._time, step=self._step)
        self._states[key] = state
        return state

    def _expire(self, prefix=None):
        for key, state in list(self._states.items()):
            if (prefix is None or key[0] == prefix) and state.get("step") != self._step:
                self._states.pop(key, None)

    def _zone(self, inputs):
        tracks, events, observed = _items(inputs.get("tracks")), [], set()
        valid_tracks = [x for x in tracks if x.get("observed", True) and _box(x) and _identity(x)[0] == "person"]
        counts = {z["name"]: 0 for z in self.params["zones"]}
        for item in valid_tracks:
            identity = _identity(item)
            if identity in observed:
                continue
            observed.add(identity)
            x1, y1, x2, y2 = _box(item)
            point = ((x1 + x2) / (2 * self._frame["width"]), (y2 if self.params["anchor"] == "foot" else (y1 + y2) / 2) / self._frame["height"])
            for zone in self.params["zones"]:
                key = ("zone", identity, zone["name"])
                state = self._states.get(key)
                continuous = state and state["step"] == self._step - 1 and self._time - state["time"] <= self.params["max_gap_seconds"]
                inside = _inside(point, zone["points"])
                if inside:
                    counts[zone["name"]] += 1
                    if not continuous or not state["inside"]:
                        state = {"inside": True, "support": 0.0, "start": self._time, "dwell": False}
                        event = self._event("zone_enter", item=item, severity="info", zone=zone["name"], anchor=self.params["anchor"])
                        event["signal"] = f"zone_enter:{zone['name']}"
                        events.append(event)
                    else:
                        state["support"] += self._time - state["time"]
                    if not state["dwell"] and state["support"] >= self.params["dwell_seconds"]:
                        event = self._event("zone_dwell", item=item, zone=zone["name"], observed_seconds=state["support"], start_timestamp=state["start"])
                        event["signal"] = f"zone_dwell:{zone['name']}"
                        events.append(event)
                        state["dwell"] = True
                else:
                    if continuous and state["inside"]:
                        event = self._event("zone_exit", item=item, severity="info", zone=zone["name"], observed_seconds=state["support"])
                        event["signal"] = f"zone_exit:{zone['name']}"
                        events.append(event)
                    state = {"inside": False, "support": 0.0, "dwell": False}
                state.update(time=self._time, step=self._step)
                self._states[key] = state
        self._expire("zone")  # A lost track is not an observed exit.
        for zone, count in counts.items():
            over = self.params["occupancy_limit"] > 0 and count > self.params["occupancy_limit"]
            if over and not self._occupancy.get(zone, False):
                event = self._event("zone_occupancy", zone=zone, count=count, limit=self.params["occupancy_limit"])
                event["signal"] = f"zone_occupancy:{zone}"
                events.append(event)
            self._occupancy[zone] = over
        overlays = [{"kind": "polygon", "points": z["points"], "label": z["name"], "count": counts[z["name"]]} for z in self.params["zones"]]
        return {"events": self._packet(events, overlays=overlays), "tracks": self._packet(tracks, overlays=overlays)}

    def _posture(self, inputs):
        actions = []
        threshold = self.params["keypoint_confidence"]
        for item in _items(inputs.get("tracks")):
            if _identity(item)[0] != "person":
                continue
            points = {i: _point(item, i, threshold) for i in range(17)}
            torso = [points[i] for i in (5, 6, 11, 12)]
            label, score, angle = "unknown", 0.0, None
            if item.get("observed", True) and all(torso):
                shoulder = ((torso[0][0] + torso[1][0]) / 2, (torso[0][1] + torso[1][1]) / 2)
                hip = ((torso[2][0] + torso[3][0]) / 2, (torso[2][1] + torso[3][1]) / 2)
                dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
                if math.hypot(dx, dy) > 1e-6:
                    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
                    score = min(x[2] for x in torso)
                    if angle >= 65:
                        label = "lying"
                    elif all(points[i] for i in (13, 14, 15, 16)):
                        knees = [_angle(points[h], points[k], points[a]) for h, k, a in ((11, 13, 15), (12, 14, 16))]
                        score = min(score, *(points[i][2] for i in (13, 14, 15, 16)))
                        if all(x is not None and x <= 135 for x in knees) and angle < 60:
                            label = "squatting"
                        elif all(x is not None and x >= 150 for x in knees) and angle < 35:
                            label = "standing"
            base = {**_actor(item), "timestamp": self._time, "method": "geometry_rule", "score_semantics": "minimum_keypoint_confidence_not_action_probability"}
            actions.append({**base, "label": label, "score": score if label != "unknown" else 0.0, "details": {"torso_angle_from_image_vertical": angle, "skeleton_schema": "coco17"}})
            key = ("posture", _identity(item))
            state = self._states.get(key, {})
            contiguous = state.get("step") == self._step - 1 and self._time - state.get("time", 0) <= self.params["max_gap_seconds"]
            if not contiguous or label == "unknown":
                state = {}
            if label == "standing":
                state = {"upright_time": self._time}
            elif label == "lying":
                if "lying_start" not in state and "upright_time" in state and self._time - state["upright_time"] <= self.params["fall_transition_seconds"]:
                    state["lying_start"] = self._time
                if "lying_start" in state and self._time - state["lying_start"] >= self.params["fall_hold_seconds"]:
                    actions.append({**base, "label": "fall_candidate", "score": score, "details": {"method": "upright_to_lying_rule", "lying_start": state["lying_start"], "not_a_learned_fall_classifier": True}})
            else:
                state.pop("lying_start", None)
            state.update(time=self._time, step=self._step)
            self._states[key] = state
            wrists = [points[i] for i in (5, 6, 9, 10)]
            if item.get("observed", True) and all(wrists) and wrists[2][1] < wrists[0][1] and wrists[3][1] < wrists[1][1]:
                actions.append({**base, "label": "raised_hands", "score": min(p[2] for p in wrists), "details": {"method": "both_wrists_above_shoulders"}})
        self._expire("posture")
        return {"actions": self._packet(actions)}

    def _condition(self, inputs):
        events, seen = [], set()
        for item in _items(inputs.get("actions")) + _items(inputs.get("events")):
            identity = _identity(item)
            label = item.get("label", "")
            if not isinstance(label, str) or not label:
                continue
            signal = item.get("signal", label)
            if identity[0] == "relation" and _relation_key(item) is None and self.params["min_duration"] > 0:
                # Detection list indices are not identities across frames.
                # A persistent relation condition requires tracked endpoints.
                continue
            if not _known(item):
                for key in list(self._states):
                    if key[:2] == ("condition", identity):
                        self._states.pop(key)
                continue
            if self.params["labels"] and label not in self.params["labels"] and signal not in self.params["labels"]:
                continue
            key = ("condition", identity, signal)
            if key in seen:
                continue
            seen.add(key)
            sample_time = item.get("timestamp", self._time)
            if (not isinstance(sample_time, (int, float)) or isinstance(sample_time, bool)
                    or not math.isfinite(sample_time) or sample_time > self._time
                    or self._time - sample_time > self.params["max_gap_seconds"]
                    or sample_time <= self._sample_times.get(key, -math.inf)):
                continue
            self._sample_times[key] = sample_time
            score = item.get("score", 1.0)
            valid_score = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1
            state = self._continuous(key, positive=valid_score and score >= self.params["min_score"])
            if state and not state["emitted"] and state["support"] >= self.params["min_duration"] and self._time - self._cooldowns.get(key, -math.inf) >= self.params["cooldown"]:
                event = self._event("condition", label=label, item=item, signal=signal, score=score, observed_seconds=state["support"], start_timestamp=state["start"])
                event["signal"] = signal
                events.append(event)
                state["emitted"] = True
                self._cooldowns[key] = self._time
        self._expire("condition")
        return {"events": self._packet(events)}

    def _sequence(self, inputs):
        events, steps = [], self.params["steps"]
        incoming = _items(inputs.get("signals")) + _items(inputs.get("events")) + _items(inputs.get("actions"))
        # An explicit unknown must clear evidence before checking a timeout.
        unknown = {_identity(item) for item in incoming if not _known(item)}
        for identity in unknown:
            self._states.pop(("sequence", identity), None)
        for key, state in list(self._states.items()):
            if key[0] == "sequence" and state.get("position", 0) and not state.get("complete") and self._time - state["advanced"] > self.params["max_gap_seconds"]:
                events.append(self._event("sequence_timeout", item=state["actor"], expected=steps[state["position"]], elapsed=self._time - state["advanced"], reason="next_step_not_observed_not_proof_of_person_absence"))
                self._states.pop(key)
        for item in incoming:
            key = ("sequence", _identity(item))
            if _identity(item) in unknown:
                continue
            signal = item.get("signal", item.get("label"))
            if signal not in steps:
                continue
            event_id = item.get("event_id")
            if event_id is not None:
                if str(event_id) in self._seen_events:
                    continue
                self._seen_events[str(event_id)] = self._time
            sample_key = ("sequence", _identity(item), signal)
            if "timestamp" in item:
                sample_time = item["timestamp"]
                if (not isinstance(sample_time, (int, float)) or isinstance(sample_time, bool)
                        or not math.isfinite(sample_time) or sample_time > self._time
                        or self._time - sample_time > self.params["max_gap_seconds"]
                        or sample_time <= self._sample_times.get(sample_key, -math.inf)):
                    continue
                self._sample_times[sample_key] = sample_time
            state = self._states.get(key)
            if state:
                state["time"] = self._time
            if state and (state.get("complete") or state.get("last_signal") == signal):
                continue
            if state is None:
                if signal != steps[0]:
                    continue  # No observed beginning; do not invent a missing past step.
                state = {"position": 1, "last_signal": signal, "advanced": self._time, "time": self._time, "actor": _actor(item)}
                self._states[key] = state
                continue
            expected = steps[state["position"]]
            if self._time <= state["advanced"]:
                continue  # Simultaneous observations do not establish an order.
            if signal != expected:
                events.append(self._event("sequence_out_of_order", item=item, expected=expected, received=signal, completed_steps=steps[:state["position"]]))
                self._states.pop(key, None)
                if signal == steps[0]:
                    self._states[key] = {"position": 1, "last_signal": signal, "advanced": self._time, "time": self._time, "actor": _actor(item)}
                continue
            state.update(position=state["position"] + 1, last_signal=signal, advanced=self._time)
            if state["position"] == len(steps):
                events.append(self._event("sequence_complete", item=item, severity="info", steps=steps))
                if self.params["reset_on_complete"]:
                    self._states.pop(key, None)
                else:
                    state["complete"] = True
        return {"events": self._packet(events)}

    def _ppe(self, inputs):
        tracks = [x for x in _items(inputs.get("tracks")) if _identity(x)[0] == "person"]
        detections, events = _items(inputs.get("detections")), []
        threshold = self.params["keypoint_confidence"]
        for equipment, enabled in (("helmet", self.params["require_helmet"]), ("vest", self.params["require_vest"])):
            if not enabled:
                continue
            regions, visible = {}, {}
            for index, item in enumerate(tracks):
                box = _box(item)
                if not box or not item.get("observed", True):
                    visible[index] = False
                    continue
                x1, y1, x2, y2 = box
                if equipment == "helmet":
                    head = [_point(item, i, threshold) for i in range(5)]
                    visible[index] = item.get("head_visible") is True or (item.get("head_visible") is not False and sum(p is not None for p in head) >= 2)
                    regions[index] = (x1, y1, x2, y1 + .35 * (y2 - y1))
                else:
                    visible[index] = item.get("torso_visible") is True or (item.get("torso_visible") is not False and all(_point(item, i, threshold) for i in (5, 6, 11, 12)))
                    regions[index] = (x1, y1 + .2 * (y2 - y1), x2, y1 + .75 * (y2 - y1))
            matched, ambiguous = set(), set()
            for detection in detections:
                box, score = _box(detection), detection.get("score", 0)
                if not box or detection.get("label") not in self.params[f"{equipment}_labels"] or not isinstance(score, (int, float)) or not math.isfinite(score) or not self.params["min_score"] <= score <= 1:
                    continue
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                candidates = []
                for index, region in regions.items():
                    if not visible[index] or index in matched:
                        continue
                    x1, y1, x2, y2 = region
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        distance = math.hypot((cx - (x1 + x2) / 2) / (x2 - x1), (cy - (y1 + y2) / 2) / (y2 - y1))
                        candidates.append((distance, index))
                candidates.sort()
                if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .15:
                    ambiguous.update(i for _, i in candidates)
                elif candidates:
                    matched.add(candidates[0][1])
            for index, item in enumerate(tracks):
                key = ("ppe", _identity(item), equipment)
                valid = visible.get(index, False) and index not in ambiguous
                state = self._continuous(key, positive=index not in matched, valid=valid)
                if not valid:
                    events.append(self._event("unknown", "ppe_unknown", item, severity="info", equipment=equipment, reason="visibility_or_assignment_unknown"))
                elif state and not state["emitted"] and state["support"] >= self.params["missing_hold_seconds"] and self._time - self._cooldowns.get(key, -math.inf) >= self.params["cooldown"]:
                    events.append(self._event("ppe_missing", f"missing_{equipment}", item, equipment=equipment, observed_seconds=state["support"], method="visible_roi_without_associated_equipment"))
                    state["emitted"] = True
                    self._cooldowns[key] = self._time
        self._expire("ppe")
        return {"events": self._packet(events)}

    def _merge(self, inputs):
        events, seen = [], set()
        for item in _items(inputs.get("events")):
            event_id = item.get("event_id")
            if event_id is not None and str(event_id) in seen:
                continue
            if event_id is not None:
                seen.add(str(event_id))
            events.append(dict(item))
        return {"events": self._packet(events)}
