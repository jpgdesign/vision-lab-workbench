"""Evaluate the actual shared-backbone YOLO26 bundle, without training.

Both branches run on the same 640-square letterbox input, in unfused FP32.
Each branch independently uses native end-to-end decoding: confidence > .001,
at most 100 predictions, and no additional NMS or cross-branch suppression.
This six-class food evaluation is not a comparison with the old 91-class task.
The ten-image original-branch COCO canary is not a full COCO80 benchmark.
Imports and --help do not load a model or allocate GPU tensors.
"""
from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
from pathlib import Path
import time

from bento_research.common import (
    ROOT, EXPERIMENT, audit_dataset, coco_score, digest, ground_truth, now, save,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def category_mapping(names, categories):
    """Use exact class-name identity, never COCO's sparse IDs as model indices."""
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if len(names) != len(set(names)) or not names:
        raise ValueError("Model class names must be unique and nonempty")
    by_name = {category["name"]: category["id"] for category in categories}
    if len(by_name) != len(categories) or len({c["id"] for c in categories}) != len(categories):
        raise ValueError("Ground-truth category IDs and names must be unique")
    if set(names) != set(by_name):
        raise ValueError(f"Model/ground-truth vocabularies differ: model-only={set(names)-set(by_name)}, "
                         f"ground-truth-only={set(by_name)-set(names)}")
    return {index: by_name[name] for index, name in enumerate(names)}


def detection_rows(boxes, image_id, width, height, mapping):
    """Convert already-decoded native xyxy rows to COCO, without another NMS."""
    rows = boxes.tolist() if hasattr(boxes, "tolist") else boxes
    if len(rows) > 100:
        raise ValueError("Native decode exceeded the per-branch 100-box cap")
    result = []
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError("Each prediction must have six finite xyxy/confidence/class values")
        x1, y1, x2, y2, confidence, cls = map(float, row)
        if cls != int(cls) or int(cls) not in mapping or not .001 < confidence <= 1:
            raise ValueError("Prediction class or confidence violates native decode protocol")
        if x1 < -1e-3 or y1 < -1e-3 or x2 > width + 1e-3 or y2 > height + 1e-3:
            raise ValueError("Decoded prediction is outside the original image")
        if x2 <= x1 or y2 <= y1:
            # Clipping can collapse boxes wholly outside the visible image.
            continue
        result.append({"image_id": image_id, "category_id": mapping[int(cls)],
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "score": confidence})
    return result


def score_native(document, predictions):
    # COCOeval sorts category IDs; common's per-class summary follows document
    # order, so normalize that order before handing it the predictions.
    document = {**document, "categories": sorted(document["categories"], key=lambda c: c["id"])}
    result = coco_score(document, predictions)
    # common.coco_score does not perform NMS. Its generic metadata defaults to
    # .7, which is incorrect for this explicitly NMS-free evaluation pipeline.
    result["nms_iou"] = None
    result["postprocessing"] = "Native independent end-to-end branches; no additional or cross-branch NMS"
    result["ap_confidence_comparison"] = "> 0.001"
    return result


def load_canary(annotations, images_directory):
    from PIL import Image
    document = read(annotations)
    if not all(key in document for key in ("images", "annotations", "categories")):
        raise ValueError("COCO canary JSON is missing images/annotations/categories")
    if len(document["images"]) != 10:
        raise ValueError("This canary protocol requires exactly ten COCO demonstration images")
    document = {**document, "info": document.get("info", {}),
                "images": sorted(document["images"], key=lambda im: im["id"]),
                "categories": sorted(document["categories"], key=lambda category: category["id"])}
    ids = {im["id"] for im in document["images"]}
    cats = {category["id"] for category in document["categories"]}
    if len(ids) != 10 or len(cats) != 80:
        raise ValueError("Canary must have ten unique image IDs and the complete 80-class COCO vocabulary")
    paths = []
    for im in document["images"]:
        path = (images_directory / im["file_name"]).resolve()
        if not path.is_relative_to(images_directory.resolve()) or not path.is_file():
            raise ValueError(f"Canary image missing or outside supplied image directory: {im['file_name']}")
        with Image.open(path) as image:
            if image.size != (im["width"], im["height"]):
                raise ValueError(f"Canary dimensions disagree with annotation: {path}")
        paths.append(path)
    annotation_ids = set()
    for ann in document["annotations"]:
        bbox = ann["bbox"]
        if (ann["id"] in annotation_ids or ann["image_id"] not in ids or ann["category_id"] not in cats
                or len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox)
                or bbox[2] <= 0 or bbox[3] <= 0):
            raise ValueError("Invalid or duplicate COCO canary annotation")
        annotation_ids.add(ann["id"])
    return document, paths


def annotate(path, image_record, document, predictions, destination, branch):
    """Side-by-side GT and >=.25 predictions; saved as an evaluation figure."""
    from PIL import Image, ImageDraw, ImageFont
    original = Image.open(path).convert("RGB")
    font = ImageFont.load_default(size=15)
    colors = ["#009e73", "#0072b2", "#e69f00", "#cc79a7", "#d55e00", "#56b4e9"]
    names = {category["id"]: category["name"] for category in document["categories"]}
    panels = []
    gt = [a for a in document["annotations"] if a["image_id"] == image_record["id"]]
    for heading, rows, predicted in (("Ground truth", gt, False),
                                     (f"{branch} branch | confidence >= 0.25", predictions, True)):
        panel = original.copy()
        draw = ImageDraw.Draw(panel)
        for row in rows:
            if predicted and row["score"] < .25:
                continue
            x, y, width, height = row["bbox"]
            color = colors[row["category_id"] % len(colors)]
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
            text = names[row["category_id"]] + (f" {row['score']:.2f}" if predicted else "")
            text_x, text_y = max(0, x), max(0, y - 18)
            draw.rectangle(draw.textbbox((text_x, text_y), text, font=font), fill=color)
            draw.text((text_x, text_y), text, fill="white", font=font)
        panel.thumbnail((720, 720))
        framed = Image.new("RGB", (panel.width, panel.height + 34), "#18261f")
        framed.paste(panel, (0, 34))
        ImageDraw.Draw(framed).text((8, 8), heading, fill="white", font=font)
        panels.append(framed)
    figure = Image.new("RGB", (sum(p.width for p in panels) + 10, max(p.height for p in panels)), "#edf2ef")
    figure.paste(panels[0], (0, 0)); figure.paste(panels[1], (panels[0].width + 10, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.save(destination)


def evaluate_split(model, document, paths, branch, output, device, preprocess, decode):
    import torch
    names = model.new_names if branch == "new" else model.old_names
    mapping = category_mapping(names, document["categories"])
    if len(paths) != len(document["images"]):
        raise ValueError("Evaluation paths and image records must have matching order and length")
    predictions, records, examples = [], [], []
    started = time.perf_counter()
    for index, (path, im) in enumerate(zip(paths, document["images"]), 1):
        tensor, original = preprocess(path, device)
        if original.shape[:2] != (im["height"], im["width"]):
            raise ValueError(f"Decoded dimensions disagree with ground truth: {path}")
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (1, 3, 640, 640):
            raise ValueError("Preprocessing must produce FP32 NCHW 1x3x640x640")
        # Execute the actual complete shared pipeline, not an independently
        # fused/rectangular YOLO predictor or a reconstructed single branch.
        with torch.inference_mode():
            raw = model(tensor, branch="both")
        boxes = decode(raw[branch], original)
        rows = detection_rows(boxes, im["id"], im["width"], im["height"], mapping)
        predictions.extend(rows)
        record = {"id": im["id"], "file_name": im["file_name"], "path": str(path.resolve()),
                  "sha256": digest(path), "width": im["width"], "height": im["height"],
                  "native_decoded_box_count": len(boxes), "scored_box_count": len(rows), "predictions": rows}
        records.append(record)
        if index <= 8:
            example = f"examples/{index:02d}_{im['id']}.jpg"
            annotate(path, im, document, rows, output / example, branch)
            examples.append({"image_id": im["id"], "source_sha256": record["sha256"], "path": example})
        print(json.dumps({"phase": output.name, "branch": branch, "image": index,
                          "total": len(paths), "predictions": len(rows)}), flush=True)
        del raw, boxes, tensor
    metrics = score_native(document, predictions)
    save(output / "ground_truth.json", document)
    save(output / "predictions.json", predictions)
    save(output / "image_predictions.json", records)
    save(output / "metrics.json", metrics)
    save(output / "per_class.json", metrics["fixed_025_iou_050"]["per_class"])
    result = {"branch": branch, "images": len(paths), "ground_truth_objects": len(document["annotations"]),
              "supported_gt_classes": len({a["category_id"] for a in document["annotations"] if not a.get("iscrowd", 0)}),
              "model_class_to_category_id": mapping, "metrics": metrics, "seconds": time.perf_counter() - started,
              "examples": examples, "artifacts": {"ground_truth": "ground_truth.json", "predictions": "predictions.json",
                "image_predictions": "image_predictions.json", "per_class": "per_class.json", "metrics": "metrics.json"}}
    save(output / "summary.json", result)
    return result


def preservation_reference(bundle, canary_paths):
    """Reference an adjacent prior audit; do not imply a new native comparison."""
    path = bundle.parent / "preservation_report.json"
    result = {"baseline_rerun": False, "equivalence_claim": "Not measured by this evaluator"}
    if not path.is_file():
        return result
    report = read(path)
    result.update(path=str(path), sha256=digest(path), reported_status=report.get("status"))
    references = bundle.parent / "native_canary_predictions.json"
    same_images = False
    if references.is_file():
        prior = read(references)
        same_images = sorted(row["sha256"] for row in prior) == sorted(digest(p) for p in canary_paths)
    checks = report.get("checks", [])
    reload_checks = report.get("portable_reload_checks", [])
    report_bundle = Path(report.get("bundle", "")).resolve()
    expected_bundle_hashes = report.get("bundle_files_sha256")
    bundle_names = ("branches.json", "branches.safetensors")
    bundle_hashes_match = (isinstance(expected_bundle_hashes, dict)
                          and set(expected_bundle_hashes) == set(bundle_names)
                          and all((bundle / name).is_file()
                                  and expected_bundle_hashes[name] == digest(bundle / name)
                                  for name in bundle_names))
    passed = (report.get("status") == "passed" and report.get("source_files_unchanged") is True
              and report.get("images") == 10 and len(checks) == len(reload_checks) == 10 and same_images
              and bundle_hashes_match
              and report_bundle == (bundle / "branches.json").resolve()
              and all(row.get("old_raw_bitwise_equal") is True and row.get("new_raw_bitwise_equal") is True
                      and row.get("old_boxes_equal") is True and row.get("new_boxes_equal") is True for row in checks)
              and all(row.get("old_equal") is True and row.get("new_equal") is True for row in reload_checks))
    result.update(prior_report_checks_and_inputs_match=passed, prior_bundle_hashes_match=bundle_hashes_match,
                  prior_device=report.get("device"),
                  equivalence_claim=("Prior native/shared and portable-reload audit passed on these ten inputs; "
                                     "not rerun here and not a full-COCO preservation claim" if passed
                                     else "Adjacent report does not establish matching ten-input equivalence here"))
    return result


def write_report(output, result):
    labels = {"val": "六類 val", "test": "六類 test", "coco10_canary": "原分支 COCO 10 圖 canary"}
    rows, panels = [], []
    for key, split in result["splits"].items():
        metrics = split["metrics"]; fixed = metrics["fixed_025_iou_050"]["micro"]
        rows.append(f'<tr><td>{labels[key]}</td><td>{split["images"]}</td><td>{fixed["precision"]:.1%}</td>'
                    f'<td>{fixed["recall"]:.1%}</td><td>{metrics["map50"]:.1%}</td><td>{metrics["map50_95"]:.1%}</td></tr>')
        images = ''.join(f'<figure><img loading="lazy" src="{key}/{entry["path"]}"><figcaption>image_id {entry["image_id"]}</figcaption></figure>' for entry in split["examples"])
        panels.append(f'<h2>{labels[key]}</h2><p><a href="{key}/metrics.json">指標</a> · <a href="{key}/per_class.json">逐類結果</a> · <a href="{key}/predictions.json">全部預測</a></p><div class="grid">{images}</div>')
    limitations = ''.join(f'<li>{html.escape(text)}</li>' for text in result["limitations"])
    page = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shared YOLO26 bundle 評估</title><style>body{font:16px/1.6 system-ui;max-width:1280px;margin:30px auto;padding:0 20px;background:#f4f7f5;color:#173324}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #bdcec3}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%;height:auto}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>Shared YOLO26 bundle：實際推論管線評估</h1><p>640 方形 letterbox、FP32、未融合 Conv/BN；同一次共享骨幹執行新舊分支，各自原生解碼且不額外 NMS。P/R 固定 confidence ≥ 0.25、IoU ≥ 0.5。</p><table><tr><th>切分</th><th>影像</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50–95</th></tr>'+''.join(rows)+'</table><ul>'+limitations+'</ul><p><a href="evaluation.json">完整評估紀錄</a></p>'+''.join(panels)+'</html>'
    (output / "report.html").write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Directory containing branches.json and branches.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="New or empty evaluation output directory")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT / "dataset/dataset.yaml", help="Curated local dataset YAML or its directory")
    parser.add_argument("--canary-images", type=Path, default=ROOT / "outputs/vision_training_starter/images")
    parser.add_argument("--canary-annotations", type=Path, default=ROOT / "outputs/vision_training_starter/annotations/instances_val2017_10.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    bundle, output = args.bundle.resolve(), args.output.resolve()
    dataset = args.dataset.resolve()
    if dataset.is_dir():
        dataset = dataset / "dataset.yaml"
    if not dataset.is_file() or not (bundle / "branches.json").is_file():
        raise FileNotFoundError("Local dataset YAML and bundle branches.json are required")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty output directory; previous evaluations are preserved")
    config = read(bundle / "branches.json")
    if config.get("dtype") != "float32":
        raise ValueError("This verified pipeline requires a saved FP32 bundle; no silent dtype conversion")
    names, splits, audit = audit_dataset(dataset)
    if config.get("new_names") != names:
        raise ValueError("Bundle new-class names/order differ from the audited six-class dataset")
    canary_document, canary_paths = load_canary(args.canary_annotations.resolve(), args.canary_images.resolve())
    category_mapping(config["old_names"], canary_document["categories"])
    # Defer model/runtime imports until after --help and all local input checks.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from bento_research.frozen_yolo_branches import FrozenYOLO26Branches
    from verify_bento_frozen_bundle import preprocess, decode
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; CPU fallback is not implicit")
    hashes = {name: digest(bundle / name) for name in ("branches.json", "branches.safetensors")}
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"created_at": now(), "status": "running", "bundle": str(bundle), "bundle_sha256": hashes,
        "dataset_audit": audit, "protocol": {"device": args.device, "dtype": "float32", "imgsz": 640,
            "preprocessing": "verify_bento_frozen_bundle.preprocess: square LetterBox auto=False, RGB /255",
            "conv_bn_fused": False, "forward": "FrozenYOLO26Branches.forward(branch='both')",
            "decode": "verify_bento_frozen_bundle.decode: confidence > .001, max100 separately per branch, no NMS",
            "fixed_confidence": .25, "fixed_matching_iou": .5, "ap_iou_range": [.5, .95],
            "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "additional_nms": False, "cross_branch_suppression": False, "test_threshold_tuning": False,
            "selection": "Bundle supplied after checkpoint selection; this evaluator performs no selection or training"},
        "canary_source": {"annotations": str(args.canary_annotations.resolve()), "annotations_sha256": digest(args.canary_annotations),
                           "images_directory": str(args.canary_images.resolve())},
        "preservation_reference": preservation_reference(bundle, canary_paths), "splits": {},
        "limitations": ["Six coarse food classes are a separate task; these scores are not a same-taxonomy improvement over the old 91-class task.",
            "These scores measure the actual shared 640-square unfused FP32 pipeline; standalone rectangular/fused predictor scores can differ.",
            "The original-branch score covers ten COCO canary images, not full COCO80 evaluation or comprehensive capability retention.",
            "Only val selects training checkpoints; this script does not tune thresholds or select checkpoints on test.",
            "Small, source-annotation-based food splits retain label-quality and rare-class uncertainty; no nutrition accuracy is measured."]}
    save(output / "evaluation.json", result)
    model = None
    try:
        model = FrozenYOLO26Branches.load_bundle(bundle, device=args.device)
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError("Loaded bundle did not retain FP32")
        for split in ("val", "test"):
            document = ground_truth(splits[split], names)
            result["splits"][split] = evaluate_split(model, document, splits[split], "new", output / split,
                                                      args.device, preprocess, decode)
            save(output / "evaluation.json", result)
        result["splits"]["coco10_canary"] = evaluate_split(model, canary_document, canary_paths, "old",
            output / "coco10_canary", args.device, preprocess, decode)
        if any(digest(bundle / name) != value for name, value in hashes.items()):
            raise RuntimeError("Bundle files changed during evaluation")
        result.update(status="complete", completed_at=now(), seconds=time.perf_counter() - started,
                      bundle_files_unchanged=True)
        save(output / "evaluation.json", result)
        write_report(output, result)
        print(json.dumps({"status": "complete", "report": str(output / "report.html"),
                          "evaluation": str(output / "evaluation.json")}), flush=True)
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        save(output / "evaluation.json", result)
        raise
    finally:
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
