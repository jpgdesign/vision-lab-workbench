"""Verify actual YOLO26x old/new output equivalence and save a shared bundle.

Uses the existing ten COCO demonstration images as deterministic canaries.
This is output-equivalence evidence on those inputs, not a full COCO benchmark.
Both native branches are unfused FP32 and use identical square letterboxing.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from bento_research.common import ROOT, EXPERIMENT, save, digest, now, coco_score
from bento_research.frozen_yolo_branches import FrozenYOLO26Branches


def tree_hash(value):
    if isinstance(value, torch.Tensor):
        item = value.detach().cpu().contiguous()
        return {"shape": list(item.shape), "dtype": str(item.dtype),
                "sha256": hashlib.sha256(item.numpy().tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {str(key): tree_hash(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree_hash(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value))


def preprocess(path, device):
    original = cv2.imread(str(path))
    if original is None:
        raise ValueError(f"Cannot decode {path}")
    image = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True)(image=original)
    array = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0) / 255., original


def decode(raw, original):
    # YOLO26 already selects predictions in its native end-to-end head.
    # This filters confidence/caps each branch independently; it performs no NMS.
    decoded = raw[0] if isinstance(raw, tuple) else raw
    boxes = decoded[0].detach().clone()
    boxes = boxes[boxes[:, 4] > .001][:100]
    if len(boxes):
        boxes[:, :4] = ops.scale_boxes((640, 640), boxes[:, :4], original.shape)
    return boxes.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-checkpoint", type=Path, default=EXPERIMENT / "yolo26x_frozen_branch/train/weights/best.pt")
    parser.add_argument("--old-checkpoint", type=Path, default=ROOT / "outputs/model_weights/yolo26x/yolo26x.pt")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "yolo26x_frozen_branch/preservation")
    parser.add_argument("--canary-images", type=Path, default=ROOT / "outputs/vision_training_starter/images")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--expected-new-classes", type=int, default=6,
                        help="Explicit expected class count for the new branch; defaults to the earlier six-class pilot")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "bundle/branches.json").exists():
        raise FileExistsError("Use a new output directory to preserve previous audit artifacts")
    images = sorted(args.canary_images.glob("*.jpg"))
    if not images:
        raise ValueError("The fixed canary directory must contain JPG images")
    old_sha, new_sha = digest(args.old_checkpoint), digest(args.new_checkpoint)
    old = YOLO(str(args.old_checkpoint)).model.float().to(args.device).eval()
    new = YOLO(str(args.new_checkpoint)).model.float().to(args.device).eval()
    names = {"old": dict(old.names), "new": dict(new.names)}
    if args.expected_new_classes < 1 or len(names["old"]) != 80 or len(names["new"]) != args.expected_new_classes:
        raise ValueError(f"Expected original 80 classes and {args.expected_new_classes} explicitly configured new classes")
    reference = []
    started = time.perf_counter()
    for index, path in enumerate(images, 1):
        tensor, original = preprocess(path, args.device)
        with torch.inference_mode():
            a, b = old(tensor), new(tensor)
        reference.append({"source": str(path), "sha256": digest(path),
                          "old_raw": tree_hash(a), "new_raw": tree_hash(b),
                          "old_boxes": decode(a, original).tolist(),
                          "new_boxes": decode(b, original).tolist()})
        print(json.dumps({"phase": "native_reference", "image": index}), flush=True)
    shared = FrozenYOLO26Branches(old, new).eval()
    # The wrapper owns the old backbone and two branches. Drop native models so
    # the unused second backbone is actually released.
    del old, new
    gc.collect()
    checks = []
    for index, (path, expected) in enumerate(zip(images, reference), 1):
        tensor, original = preprocess(path, args.device)
        with torch.inference_mode():
            actual = shared(tensor)
        row = {"image": path.name,
               "old_raw_bitwise_equal": tree_hash(actual["old"]) == expected["old_raw"],
               "new_raw_bitwise_equal": tree_hash(actual["new"]) == expected["new_raw"],
               "old_boxes_equal": decode(actual["old"], original).tolist() == expected["old_boxes"],
               "new_boxes_equal": decode(actual["new"], original).tolist() == expected["new_boxes"]}
        checks.append(row)
        if not all(value for key, value in row.items() if key != "image"):
            save(out / "failed_equivalence.json", checks)
            raise RuntimeError(f"Native/shared output mismatch for {path.name}")
        print(json.dumps({"phase": "shared_equivalence", **row}), flush=True)
    shared.save_bundle(out / "bundle")
    parameter_counts = {name: sum(p.numel() for p in getattr(shared, name).parameters())
                        for name in ("shared_backbone", "old_branch", "new_branch")}
    del shared
    gc.collect()
    restored = FrozenYOLO26Branches.load_bundle(out / "bundle", device=args.device)
    reload_checks = []
    for path, expected in zip(images, reference):
        tensor, original = preprocess(path, args.device)
        with torch.inference_mode():
            actual = restored(tensor)
        row = {"image": path.name, "old_equal": tree_hash(actual["old"]) == expected["old_raw"],
               "new_equal": tree_hash(actual["new"]) == expected["new_raw"]}
        reload_checks.append(row)
        if not row["old_equal"] or not row["new_equal"]:
            raise RuntimeError("Portable bundle reload changed raw outputs")
    # Keep class namespaces separate. Application fusion must not suppress old
    # results or silently alias broad food classes to COCO subclasses.
    save(out / "native_canary_predictions.json", reference)
    summary = {"created_at": now(), "status": "passed", "images": len(images),
               "old_checkpoint": str(args.old_checkpoint.resolve()), "old_sha256": old_sha,
               "new_checkpoint": str(args.new_checkpoint.resolve()), "new_sha256": new_sha,
               "source_files_unchanged": digest(args.old_checkpoint) == old_sha and digest(args.new_checkpoint) == new_sha,
               "device": args.device, "precision": "float32", "deterministic_algorithms": True,
               "preprocessing": "640 square letterbox, RGB /255, no Conv/BN fusion",
               "postprocessing": "native head, per-branch confidence .001 and max_det100, no extra NMS",
               "classes": names, "checks": checks, "portable_reload_checks": reload_checks,
               "parameters": parameter_counts, "saved_backbone_copies": 1,
               "bundle_files_sha256": {name: digest(out / "bundle" / name)
                                       for name in ("branches.json", "branches.safetensors")},
               "bundle": str(out / "bundle/branches.json"), "seconds": time.perf_counter() - started,
               "limitations": ["Fixed-image deterministic canary, not full COCO80 AP validation",
                               f"New food branch contains {len(names['new'])} configured source-label classes; label granularity and coverage require separate evaluation",
                               "FP16, ONNX/TensorRT, dynamic shapes and camera pipelines are not verified here",
                               "Original model errors remain; branches return independent namespaced results"]}
    if not summary["source_files_unchanged"]:
        raise RuntimeError("A source checkpoint changed during verification")
    save(out / "preservation_report.json", summary)
    print(json.dumps({"status": "passed", "report": str(out / "preservation_report.json")}), flush=True)


if __name__ == "__main__":
    main()
