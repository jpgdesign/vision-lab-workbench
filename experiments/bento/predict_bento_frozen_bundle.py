"""Run an independent shared-backbone bundle on an image, without the workbench."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import torch
from bento_research.common import save
from bento_research.frozen_yolo_branches import FrozenYOLO26Branches
from verify_bento_frozen_bundle import preprocess, decode


def annotate(image, boxes, names, branch):
    color = (90, 210, 100) if branch == "old" else (40, 170, 255)
    records = []
    for row in boxes.tolist():
        x1, y1, x2, y2, score, cls = row
        name = names[int(cls)]
        records.append({"branch": branch, "class_id": int(cls), "name": name,
                        "confidence": score, "xyxy": [x1, y1, x2, y2]})
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(image, p1, p2, color, 2)
        cv2.putText(image, f"{branch}:{name} {score:.2f}", (p1[0], max(15, p1[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not .001 <= args.confidence <= 1:
        parser.error("confidence must be between .001 and 1")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("old.jpg", "new.jpg", "combined.jpg", "predictions.json")):
        raise FileExistsError("Choose a new output directory; previous results are preserved")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = FrozenYOLO26Branches.load_bundle(args.bundle, device=args.device)
    tensor, original = preprocess(args.source, args.device)
    if tensor.is_cuda: torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(tensor)
    if tensor.is_cuda: torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    combined, records = original.copy(), []
    for branch, names in (("old", model.old_names), ("new", model.new_names)):
        boxes = decode(outputs[branch], original)
        boxes = boxes[boxes[:, 4] > args.confidence]
        standalone = original.copy()
        records.extend(annotate(standalone, boxes, names, branch))
        annotate(combined, boxes, names, branch)
        if not cv2.imwrite(str(args.output / f"{branch}.jpg"), standalone):
            raise OSError("Failed to save annotated result")
    if not cv2.imwrite(str(args.output / "combined.jpg"), combined):
        raise OSError("Failed to save combined result")
    save(args.output / "predictions.json", {
        "source": str(args.source.resolve()), "bundle": str(args.bundle.resolve()),
        "confidence": args.confidence, "precision": "float32", "device": args.device,
        "first_forward_seconds": elapsed, "latency_note": "Single first forward, not a warmed latency benchmark",
        "cross_branch_suppression": False, "class_namespaces": ["old", "new"], "predictions": records,
        "note": "Old and new outputs are kept independently; broad food names are not silently merged with COCO subclasses."
    })
    print(args.output.resolve())


if __name__ == "__main__":
    main()
