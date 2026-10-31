# Based on RelateAnything by Maëlic Neau
# https://github.com/Maelic/RelateAnything
# Copyright (C) 2026 Maëlic Neau. AGPL-3.0-only; see LICENSE and NOTICE.
# Modified 2026-11-01: workbench relative imports, relation-only ONNX runtime,
# NumPy-only scoring, and hash-gated loading of the official legacy NPZ bank.

from __future__ import annotations
import hashlib
import json
import os
import warnings
from typing import List, Optional, Sequence
import numpy as np

OFFICIAL_BANK_SHA256 = 'd7e863f1609f1784767f6db69318ac8c23febcc000a005394885988d2fc514fd'


def load_predicate_bank(path):
    # The released bank stores string arrays using NumPy object pickles.
    # Only its verified official bytes may enable pickle. User banks with
    # Unicode arrays are safe; unrecognized object/pickle arrays are rejected.
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return np.load(path, allow_pickle=digest == OFFICIAL_BANK_SHA256)


class OnnxRelationHead:
    """Re-parameterized relation head. Predicate vocabulary is a runtime input
    when the graph was exported with --vocab-mode input."""

    def __init__(self, onnx_path: str, bank_path: str = "", threads: int = 0,
                 providers: Optional[Sequence[str]] = None):
        meta = json.load(open(os.path.splitext(onnx_path)[0] + ".json"))
        self.img_size: int = meta["img_size"]
        self.max_boxes: int = meta["max_boxes"]
        self.vocab_mode: str = meta.get("vocab_mode", "baked")
        # v2 graphs emit RAW LOGITS ("pred_logits"/"pair_logits"); v1 emitted
        # already-sigmoided scores. Key off the graph's own output names rather
        # than the sidecar, so a mismatched pair of files cannot be read wrong.
        out_names = self._build_engine(onnx_path, threads, providers)
        self.emits_logits = "pred_logits" in out_names
        if not self.emits_logits:
            warnings.warn(
                f"{os.path.basename(onnx_path)} is a v1 export emitting "
                f"sigmoided scores ({out_names[:2]}). They will be converted "
                "back to logits, but the raw head saturates above 0.9999 where "
                "fp32 has ~2 significant digits left, so the recovered ranking "
                "is APPROXIMATE. Re-export with deploy/export_onnx.py.",
                RuntimeWarning, stacklevel=2)
        # The score contract ships with the artifact; the sidecar carries the
        # calibration the numbers were produced under.
        from .scoring import ScoreContract
        cal = meta.get("calibration") or {}
        self.contract = ScoreContract(calib_a=float(cal.get("a", 1.0)),
                                      calib_b=float(cal.get("b", 0.0)))

        self.bank = None
        if self.vocab_mode == "input":
            if not bank_path:
                raise ValueError("vocab-mode=input needs a predicate bank")
            z = load_predicate_bank(bank_path)
            self.bank = {
                "names": [str(x) for x in z["names"]],
                "W": z["W"].astype(np.float32),
                "alpha": z["alpha"].astype(np.float32),
                # new-schema arrays (build_predicate_bank >= release): the
                # two-graph type vector and calibrated thresholds. Old banks
                # lack them -> features unavailable, never silently wrong.
                "is_spatial": (z["is_spatial"].astype(bool)
                               if "is_spatial" in z.files else None),
                "thr": (z["thr"].astype(np.float32)
                        if "thr" in z.files else None),
            }
            self.bank_index = {n: i for i, n in enumerate(self.bank["names"])}
            self.set_predicates([str(x) for x in z["default"]])
            z.close()
        else:
            self.predicates = meta["predicates"]

    # -- engine hooks (see OnnxDetector): deploy/ov_runtime.py overrides ----
    def _build_engine(self, path: str, threads: int,
                      providers: Optional[Sequence[str]]) -> List[str]:
        """Create the inference engine; returns the graph's output names."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime could not load. Install requirements-workflows.txt. "
                "On Windows, call prepare_windows_runtime() before importing "
                "NumPy/PyTorch and restart the application; old Anaconda MSVC "
                "DLLs can shadow the installed Visual C++ runtime."
            ) from exc
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            path, so, providers=list(providers or ["CPUExecutionProvider"]))
        self.input_names = [i.name for i in self.sess.get_inputs()]
        return [o.name for o in self.sess.get_outputs()]

    def _run_engine(self, feed: dict):
        return self.sess.run(None, feed)

    # -- the dynamic vocabulary ------------------------------------------
    def set_predicates(self, names: Sequence[str]) -> "OnnxRelationHead":
        """Swap the predicate vocabulary. No re-export, no text encoder — rows
        are sliced out of the precomputed bank."""
        if self.bank is None:
            raise RuntimeError("graph was exported with a baked vocabulary")
        missing = [n for n in names if n not in self.bank_index]
        if missing:
            raise KeyError(
                f"not in the predicate bank: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} — rebuild it with "
                f"deploy/build_predicate_bank.py on the training box")
        idx = np.array([self.bank_index[n] for n in names], np.int64)
        self.predicates = list(names)
        self._W = np.ascontiguousarray(self.bank["W"][idx])
        self._alpha = np.ascontiguousarray(self.bank["alpha"][idx])
        self.is_spatial = (self.bank["is_spatial"][idx]
                           if self.bank.get("is_spatial") is not None else None)
        self.thr = (self.bank["thr"][idx]
                    if self.bank.get("thr") is not None else None)
        return self

    def available_predicates(self) -> List[str]:
        return list(self.bank["names"]) if self.bank else list(self.predicates)

    # -- inference ---------------------------------------------------------
    def make_feed(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> dict:
        """Preprocess one frame + boxes into the graph's feed dict. Split out
        of __call__ so quantization calibration (deploy/export_openvino.py)
        feeds the model EXACTLY what inference will."""
        import cv2
        H, W = frame_bgr.shape[:2]
        N = len(boxes_xyxy)
        # NOTE: a plain square resize, NOT letterbox — the head's geometry
        # features are normalized cxcywh, so subject/object/union boxes and the
        # image are distorted identically and the mapping stays consistent.
        # This matches relsgg.api.RelateAnything._to_chw (and training).
        img = cv2.resize(frame_bgr, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        x = img[:,:,::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        b = boxes_xyxy.astype(np.float32).copy()
        b[:, [0, 2]] /= max(W, 1); b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
        bw = b[:, 2] - b[:, 0];       bh = b[:, 3] - b[:, 1]
        boxes = np.stack([cx, cy, bw, bh], -1).astype(np.float32)

        # the graph was built for a fixed N — zero-pad and declare the true
        # count, exactly as the training collate does
        Nmax = self.max_boxes
        padded = np.zeros((1, Nmax, 4), np.float32)
        n = min(N, Nmax)
        padded[0,:n] = boxes[:n]

        feed = {"image": np.ascontiguousarray(x), "boxes": padded,
                "box_counts": np.array([n], np.int64)}
        if self.vocab_mode == "input":
            feed["W"] = self._W
            feed["alpha"] = self._alpha
        return feed

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray):
        """Returns raw (pred_logits [K,V], pair_logits [K], sub, obj, valid)."""
        feed = self.make_feed(frame_bgr, boxes_xyxy)
        pred, pair, sub, obj, valid = self._run_engine(feed)
        if not self.emits_logits:
            # v1 artifact: invert the sigmoids so the host sees logits either
            # way. Lossy at the top of the range by construction — warned once
            # at load. eps keeps logit(0) and logit(1) finite.
            eps = np.float32(1e-7)
            lg = lambda p: np.log(np.clip(p, eps, 1 - eps)
                                  / (1 - np.clip(p, eps, 1 - eps)))
            pred, pair = lg(pred.astype(np.float32)), lg(pair.astype(np.float32))
        return pred[0], pair[0], sub[0], obj[0], valid[0]
