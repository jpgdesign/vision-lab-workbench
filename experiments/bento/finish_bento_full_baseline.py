"""Verify and score a completed full-data YOLO26 run; never starts training.

All asset paths are explicit so this script also works from a frozen code
snapshot. --help performs no model loading or GPU work. The caller owns GPU
scheduling; a completed training progress.json is mandatory before verification.
An explicit manifest legacy_samples array opts into additional historical
holdout evaluation after auditing its files, mapped GT, and split separation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Completed trainer output directory")
    parser.add_argument("--dataset", type=Path, required=True, help="The exact local dataset YAML used in training")
    parser.add_argument("--weights", type=Path, required=True, help="Unchanged original YOLO26x pretrained checkpoint")
    parser.add_argument("--canary-images", type=Path, required=True, help="Explicit directory containing the ten original-task JPG canaries")
    parser.add_argument("--taxonomy-review", type=Path, help="Optional exact candidate taxonomy JSON for additional fine/broad/review subset metrics")
    parser.add_argument("--allow-untrained-classes", nargs="+", action="extend", default=[], metavar="CLASS",
                        help="Must exactly match the completed trainer's explicit allowance")
    parser.add_argument("--device", default="cuda:0", help="Single torch device, normally cuda:0 or cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for verification/evaluation")
    args = parser.parse_args()
    if args.threads < 1 or "," in args.device:
        parser.error("threads must be positive and only one device is supported")
    for key in ("run", "dataset", "weights", "canary_images", "taxonomy_review"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    return args


def require_empty(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Preserve existing verification/evaluation artifacts; output is not empty: {path}")


def audit_legacy_samples(dataset, names, splits):
    """Evaluate only an explicitly declared, immutable historical holdout.

    Source SHA verification means the copied evaluation image is byte-identical
    to the original source SHA already bound into this dataset manifest. No
    guessed source root or undeclared image folder is searched.
    """
    from PIL import Image, ImageOps
    from bento_research.full_data_common import digest, label_path, read_boxes

    root = Path(dataset).resolve().parent
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    if "legacy_samples" not in manifest:
        return [], {"status": "not_declared", "enabled": False}
    rows = manifest["legacy_samples"]
    if not isinstance(rows, list):
        raise ValueError("Declared legacy_samples must be an array of explicit image records")
    if not rows:
        return [], {"status": "declared_empty", "enabled": False, "images": 0}

    def local_path(value):
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Declared legacy/active file is absent or outside the dataset: {value}")
        return path

    def image_identity(path):
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            width, height = rgb.size
            rgb_sha = hashlib.sha256(f"RGB:{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
        return digest(path), rgb_sha, width, height

    actual_active_paths = {path.resolve() for paths in splits.values() for path in paths}
    active_records = manifest["images"]
    if {local_path(row["image"]) for row in active_records} != actual_active_paths:
        raise ValueError("Active manifest records disagree with the already-audited train/val/test paths")
    active_groups = {str(row["group_id"]) for row in active_records}
    active_old_groups = {str(row["old_group_id"]) for row in active_records if row.get("old_group_id")}
    active_source_ids = {value for row in active_records for value in row.get("source_ids", [row["source_id"]])}
    active_rgbs = set()
    for row in active_records:
        path = local_path(row["image"])
        sha, rgb, width, height = image_identity(path)
        if sha != row["sha256"] or rgb != row["rgb_sha256"]:
            raise ValueError(f"Active image content no longer matches the manifest: {path}")
        if (width, height) != (row["width"], row["height"]):
            raise ValueError(f"Active image dimensions changed: {path}")
        active_rgbs.add(rgb)

    seen_paths, seen_groups, seen_rgbs, seen_source_ids = set(), set(), set(), set()
    class_images, class_boxes = Counter(), Counter()
    paths, files = [], []
    for row in rows:
        if row.get("split") != "legacy":
            raise ValueError("Every declared legacy sample must explicitly have split=legacy")
        path, label = local_path(row["image"]), local_path(row["label"])
        if label != label_path(path).resolve():
            raise ValueError(f"Declared legacy label does not match native YOLO path resolution: {path}")
        group = str(row["group_id"])
        old_group = str(row["old_group_id"]) if row.get("old_group_id") else None
        ids = set(row.get("source_ids", [row["source_id"]]))
        if (path in actual_active_paths or path in seen_paths or group in active_groups
                or (old_group is not None and old_group in active_old_groups)
                or ids & (active_source_ids | seen_source_ids)):
            raise ValueError(f"Legacy sample overlaps train/val/test or another declared sample: {path}")
        sha, rgb, width, height = image_identity(path)
        if sha != row["sha256"] or sha != row["original_sha256"] or rgb != row["rgb_sha256"]:
            raise ValueError(f"Legacy image/source SHA or RGB identity disagrees: {path}")
        if rgb in active_rgbs or rgb in seen_rgbs:
            raise ValueError(f"Legacy RGB content overlaps an active split or another legacy image: {path}")
        if (width, height) != (row["width"], row["height"]):
            raise ValueError(f"Legacy image dimensions disagree: {path}")
        label_sha = digest(label)
        if label_sha != row["label_sha256"]:
            raise ValueError(f"Legacy label SHA no longer matches the declared GT: {label}")
        boxes = read_boxes(label, len(names))
        objects = row["objects"]
        if len(boxes) != row["object_count"] or len(boxes) != len(objects):
            raise ValueError(f"Legacy GT count differs from manifest: {label}")
        present = {box[0] for box in boxes}
        if present != set(row["class_ids"]):
            raise ValueError(f"Legacy GT class IDs differ from manifest: {label}")
        for box, obj in zip(boxes, objects):
            cls, cx, cy, width_box, height_box = box
            mapped_box = (cx - width_box / 2, cy - height_box / 2, width_box, height_box)
            expected_box = obj["bbox_xywh_normalized"]
            if (cls != obj["class_id"] or names[cls] != obj["class_name"] or len(expected_box) != 4
                    or any(not math.isfinite(float(value)) for value in expected_box)
                    or any(abs(actual - expected) > 2e-8 for actual, expected in zip(mapped_box, expected_box))):
                raise ValueError(f"Legacy GT row differs from its declared mapped source annotation: {label}")
        paths.append(path)
        seen_paths.add(path)
        seen_groups.add(group)
        seen_rgbs.add(rgb)
        seen_source_ids.update(ids)
        class_images.update(present)
        class_boxes.update(box[0] for box in boxes)
        files.append({"image": str(path), "label": str(label), "image_sha256": sha,
                      "original_source_sha256": row["original_sha256"], "label_sha256": label_sha,
                      "rgb_sha256": rgb, "source_path_metadata": row["source_path"],
                      "source_id": row["source_id"], "group_id": group, "objects": len(boxes)})
    declared = manifest.get("summary", {}).get("legacy", {})
    actual_summary = {"images": len(paths), "objects": sum(class_boxes.values()), "groups": len(seen_groups)}
    for key, value in actual_summary.items():
        if key in declared and declared[key] != value:
            raise ValueError(f"Legacy summary {key} does not match actual files/GT")
    distribution = {**actual_summary,
                    "per_class": {name: {"images": class_images[i], "objects": class_boxes[i]}
                                  for i, name in enumerate(names)},
                    "absent_classes": [name for i, name in enumerate(names) if not class_boxes[i]]}
    return paths, {"status": "audited", "enabled": True, **actual_summary,
                   "manifest_sha256": digest(manifest_path), "files": files,
                   "class_distribution": distribution,
                   "checks": {"copied_image_matches_original_source_sha": True, "label_sha_and_gt_rows_match": True,
                              "group_disjoint_from_train_val_test": True, "old_group_disjoint_from_train_val_test": True,
                              "source_ids_disjoint_from_train_val_test": True, "rgb_disjoint_from_train_val_test": True},
                   "scope": "Historical holdout excluded from this run's training and validation selection; not a newly captured external set",
                   "source_verification_scope": "Image bytes match both manifest file SHA and original-source SHA. Raw source roots are not guessed or independently reopened.",
                   "limits": ["Earlier Bento experiments may have used these historical photos for validation",
                              "Source labels remain incomplete or semantically uncertain; this is not newly reviewed ground truth"]}


def taxonomy_groups(path, names):
    """Exclusive diagnostic subsets from metadata, not claims of reviewed GT."""
    if path is None:
        return {}, {"status": "not_requested"}
    from bento_research.full_data_common import digest
    document = read(path)
    canonical = {row["name"] for row in document["canonical_classes"]}
    if canonical != set(names):
        raise ValueError("Taxonomy-review canonical names differ from the actual dataset vocabulary")
    mappings = {name: [] for name in names}
    for row in document["mapping"]:
        if row.get("canonical_name") in mappings and row.get("active_candidate_class", True):
            mappings[row["canonical_name"]].append(row)
    groups = {"fine_named": [], "broad_unspecified": [], "semantic_review": []}
    for name in names:
        rows = mappings[name]
        if not rows:
            raise ValueError(f"No active taxonomy mapping for dataset class {name}")
        kinds = {row["kind"] for row in rows}
        flags = {flag for row in rows for flag in row.get("flags", [])}
        if "semantic_review_required" in flags or "ambiguous_concatenated_source_label" in kinds:
            groups["semantic_review"].append(name)
        elif "broad_or_subtype_unspecified" in kinds:
            groups["broad_unspecified"].append(name)
        elif kinds == {"named_ingredient_or_dish"}:
            groups["fine_named"].append(name)
        else:
            raise ValueError(f"Unrecognized taxonomy grouping for {name}: {sorted(kinds)}")
    assert sum(map(len, groups.values())) == len(names)
    return groups, {"status": "metadata_grouped", "path": str(path), "sha256": digest(path),
                    "groups": groups,
                    "rule": "Exclusive precedence: semantic_review flag/ambiguous kind, then broad kind, then named ingredient/dish",
                    "limits": "Named/fine is a source-taxonomy descriptor, not proof of complete fine annotation. Subset scores retain all images and filter only class IDs."}


def subset_score(document, predictions, names, label):
    from bento_research.full_data_common import trained_classes_only_score
    if not names:
        return {"status": "no_categories", "metrics": None}
    result = trained_classes_only_score(document, predictions, names)
    result["subset_protocol"]["scope"] = f"Additional {label} view; original all-taxonomy metrics remain available"
    result["subset_protocol"]["filter"] = "Keep these declared class IDs in both GT and predictions; retain all original images"
    # COCO uses -1 as an undefined metric when the subset has no ground truth.
    if not result["subset_protocol"]["kept_gt_objects"]:
        return {"status": "no_ground_truth", "metrics": None, "subset_protocol": result["subset_protocol"]}
    return {"status": "scored", "metrics": result}


def percentage(value):
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else "—"


def href(target, parent):
    try:
        return Path(os.path.relpath(target, parent)).as_posix()
    except ValueError:
        return Path(target).as_uri()


def write_report(destination, run, native, result, audit, preservation):
    escape = html.escape
    rows = []

    def metric_row(label, metrics):
        fixed = metrics["fixed_025_iou_050"]["micro"]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            escape(label), str(fixed["support"]), percentage(fixed["precision"]), percentage(fixed["recall"]),
            percentage(metrics["map50"]), percentage(metrics["map50_95"]))) + "</tr>")

    ordered_splits = list(dict.fromkeys([*native.get("splits", {}), *result["splits"]]))
    for split in ordered_splits:
        native_split = native.get("splits", {}).get(split, {})
        if "coco" in native_split:
            metric_row(f"單獨新分支／{split}／全部類別", native_split["coco"])
        if "coco_trained_classes_only" in native_split:
            metric_row(f"單獨新分支／{split}／train有正例類別", native_split["coco_trained_classes_only"])
        shared = result["splits"].get(split)
        if shared:
            metric_row(f"共用骨幹／{split}／全部類別", shared["metrics"])
            metric_row(f"共用骨幹／{split}／train有正例類別", shared["trained_classes_only"])
    groups_html, per_class_html, examples_html = [], [], []
    for split, scored in result["splits"].items():
        for name, subset in scored.get("taxonomy_subsets", {}).items():
            if subset["status"] == "scored":
                metrics = subset["metrics"]
                fixed = metrics["fixed_025_iou_050"]["micro"]
                groups_html.append(f"<tr><td>{escape(split + ' / ' + name)}</td><td>{fixed['support']}</td>"
                                   f"<td>{percentage(fixed['precision'])}</td><td>{percentage(fixed['recall'])}</td>"
                                   f"<td>{percentage(metrics['map50'])}</td><td>{percentage(metrics['map50_95'])}</td></tr>")
            else:
                groups_html.append(f"<tr><td>{escape(split + ' / ' + name)}</td><td colspan='5'>無可評估的類別或GT</td></tr>")
        detail_rows = []
        for name, metric in scored["metrics"]["fixed_025_iou_050"]["per_class"].items():
            train_count = audit["class_distribution"]["train"]["per_class"][name]["objects"]
            note = "沒有train正例" if train_count == 0 else ""
            if not metric["support"]:
                note += "；此切分沒有GT，AP/Recall未驗證"
            detail_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
                escape(name), str(train_count), str(metric["support"]), percentage(metric["precision"]),
                percentage(metric["recall"]), percentage(metric["map50"]), percentage(metric["map50_95"]), escape(note))) + "</tr>")
        per_class_html.append(f"<details><summary>{escape(split)}：逐類結果與支持數</summary><div class='scroll'><table>"
                              "<tr><th>類別</th><th>train框</th><th>此切分GT框</th><th>P</th><th>R</th><th>AP50</th><th>AP50–95</th><th>限制</th></tr>"
                              + "".join(detail_rows) + "</table></div></details>")
        paths = scored.get("example_originals", {})
        figures = []
        for example in scored.get("examples", []):
            picture = run / "shared-evaluation" / split / example["path"]
            source = paths.get(str(example["image_id"]))
            source_link = f" · <a href='{escape(href(Path(source), destination.parent), quote=True)}'>開啟原圖</a>" if source else ""
            figures.append(f"<figure><a href='{escape(href(picture, destination.parent), quote=True)}'><img loading='lazy' src='{escape(href(picture, destination.parent), quote=True)}'></a>"
                           f"<figcaption>image_id {example['image_id']}；左來源標註、右預測{source_link}</figcaption></figure>")
        examples_html.append(f"<h2>{escape(split)} 固定順序前8張圖例</h2><div class='grid'>" + "".join(figures) + "</div>")
    absent = []
    distributions = dict(audit["class_distribution"])
    legacy = result.get("legacy_holdout", {})
    if legacy.get("enabled"):
        distributions["legacy"] = legacy["class_distribution"]
    for split, distribution in distributions.items():
        labels = distribution["absent_classes"]
        absent.append(f"<li>{split} 無正例類別：{escape(', '.join(labels)) if labels else '無'}</li>")
    hashes = "".join(f"<li>{escape(name)}：<code>{escape(value)}</code></li>" for name, value in result["bundle_files_sha256"].items())
    links = " · ".join(f"<a href='{escape(href(run / target, destination.parent), quote=True)}'>{label}</a>" for target, label in (
        ("evaluation.json", "單獨新分支結果"), ("shared-evaluation/evaluation.json", "共用骨幹完整結果"),
        ("preservation/preservation_report.json", "原路徑一致性檢查"), ("training_config.json", "訓練設定")))
    group_section = ""
    if groups_html:
        group_section = ("<h2>依來源語義描述分組的額外診斷</h2><p>分組採語義待審、粗類、具名細類的互斥優先順序，"
                         "不表示細類已人工補標完成。每個子集合仍保留全部影像；總分保留在上表。</p><div class='scroll'><table>"
                         "<tr><th>切分／組別</th><th>GT框</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50–95</th></tr>"
                         + "".join(groups_html) + "</table></div>")
    legacy_section = ""
    if legacy.get("enabled"):
        legacy_section = (f"<h2>歷史便當保留集</h2><p>另外實測 {legacy['images']} 張、{legacy['objects']} 個來源標註物件。"
                          "已核對來源／影像／標註雜湊、GT逐框，以及與train／val／test的group、來源ID和RGB不重疊。"
                          "本輪沒有用它訓練或選模；它是歷史保留集，並非新拍的外部測試。先前便當實驗可能曾使用這些照片作驗證，"
                          "不能解讀為整個研究過程從未看過的資料。</p>")
    page = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO26 全來源基線：訓練後實測</title><style>body{{font:16px/1.7 system-ui;max-width:1280px;margin:28px auto;padding:0 20px;background:#f4f7f5;color:#18352b}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:9px;border:1px solid #c7d7cc;text-align:left}}.scroll{{overflow:auto}}.note{{padding:16px;background:#fff1d9}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;height:auto}}code{{word-break:break-all}}details{{margin:18px 0}}a{{color:#126852}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>YOLO26x 全來源基線：原路徑驗證與共用骨幹實測</h1>
<p>狀態：{escape(result['status'])}。明確輸出類別 {len(result['classes'])} 類；train／val／test 圖數為 {audit['split_counts']['train']}／{audit['split_counts']['val']}／{audit['split_counts']['test']}。</p>
<p class="note">這是保留來源粗細標籤的探索基線，沒有宣稱已完成階層式補標或解決漏標／父子類衝突。訓練缺席類沒有正例學習，額外trained-only分數不取代全部類別分數。所有數值均取自實際記錄，未支持類別顯示「—」。</p>
<h2>原能力路徑</h2><p>{preservation['images']} 張固定COCO圖的原生／共享／儲存重載輸出全部逐位元相等；來源checkpoint保持不變。條件為相同640方形前處理、未融合FP32及確定性 {escape(result['protocol']['device'])} 計算。這不是完整COCO80、模型匯出或即時攝影機驗證。</p>
<h2>新分支指標</h2><p>P/R固定confidence≥0.25、IoU≥0.5；AP使用confidence&gt;0.001、IoU0.50–0.95、max_det100；不追加NMS。共享模型同次執行新舊兩路。單獨原生推論的矩形前處理／融合可能不同，兩條管線分開報告。</p>
<div class="scroll"><table><tr><th>管線／切分／類別範圍</th><th>GT框</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50–95</th></tr>{''.join(rows)}</table></div>
{legacy_section}
<h2>缺席類別與支持範圍</h2><ul>{''.join(absent)}</ul><p>COCO mAP只平均此切分有GT的類別；因此不能把mAP解釋成所有{len(result['classes'])}類都通過驗證。不同類別範圍的平均數不能直接當成同一任務的改善。</p>
{group_section}{''.join(per_class_html)}{''.join(examples_html)}
<h2>檔案與權重身分</h2><p>{links}</p><ul>{hashes}</ul><p>測試集只在驗證集選定best.pt之後計分；本程式不改模型或資料、不調整閾值、不依測試分數選權重。</p></html>"""
    destination.write_text(page, encoding="utf-8")


def main():
    args = arguments()
    progress = read(args.run / "progress.json")
    if progress.get("status") != "completed":
        raise RuntimeError(f"Training is not completed; no verification/GPU work started: {progress.get('status')}")
    from bento_research.full_data_common import (
        audit_dataset, coco_score, digest, ground_truth, now, save, trained_classes_only_score,
    )
    best = args.run / "train/weights/best.pt"
    native = read(args.run / "evaluation.json")
    training = read(args.run / "training_config.json")
    if not best.is_file() or not args.weights.is_file() or not args.dataset.is_file():
        raise FileNotFoundError("Best checkpoint, original weights and original dataset YAML must exist")
    if set(native.get("splits", {})) != {"val", "test"}:
        raise ValueError("Completed native validation and test results are required")
    names, splits, audit = audit_dataset(args.dataset, args.allow_untrained_classes)
    if training.get("classes") != names or native.get("classes") != names:
        raise ValueError("Completed training/native evaluation vocabulary does not match the supplied dataset")
    if set(training.get("allowed_untrained_classes", [])) != set(args.allow_untrained_classes):
        raise ValueError("Untrained-class allowance differs from the completed training run")
    for key in ("yaml_sha256", "manifest_sha256", "label_set_sha256", "split_counts"):
        if training["dataset"].get(key) != audit.get(key):
            raise ValueError(f"Dataset changed after training or audit provenance is missing: {key}")
    legacy_paths, legacy_audit = audit_legacy_samples(args.dataset, names, splits)
    evaluation_splits = {split: splits[split] for split in ("val", "test")}
    if legacy_audit.get("enabled"):
        evaluation_splits["legacy"] = legacy_paths
    old_sha, new_sha = digest(args.weights), digest(best)
    if old_sha != training["weights_sha256"] or new_sha != native["checkpoint_sha256"]:
        raise ValueError("Source/best checkpoint hashes disagree with the completed training records")
    canaries = sorted(args.canary_images.glob("*.jpg"))
    if len(canaries) != 10:
        raise ValueError("Exactly ten existing JPG canaries are required")
    groups, group_metadata = taxonomy_groups(args.taxonomy_review, names)
    preservation_dir, output = args.run / "preservation", args.run / "shared-evaluation"
    require_empty(preservation_dir)
    require_empty(output)
    if (args.run / "finish_status.json").exists():
        raise FileExistsError("finish_status.json already exists; preserve the previous attempt and inspect it before rerunning")
    verifier = Path(__file__).resolve().with_name("verify_bento_frozen_bundle.py")
    if not verifier.is_file():
        raise FileNotFoundError("The frozen code snapshot must include verify_bento_frozen_bundle.py")
    started = time.perf_counter()
    state = {"status": "verifying", "started_at": now(), "pid": os.getpid(),
             "old_checkpoint_sha256": old_sha, "new_checkpoint_sha256": new_sha,
             "script_sha256": digest(Path(__file__)), "verifier_sha256": digest(verifier)}
    save(args.run / "finish_status.json", state)
    model = None
    result = None
    try:
        command = [sys.executable, str(verifier), "--old-checkpoint", str(args.weights),
                   "--new-checkpoint", str(best), "--output", str(preservation_dir),
                   "--canary-images", str(args.canary_images), "--device", args.device,
                   "--threads", str(args.threads), "--expected-new-classes", str(len(names))]
        print(json.dumps({"phase": "verify_native_shared_reload", "classes": len(names)}), flush=True)
        with (args.run / "finish-verification.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
        from evaluate_bento_shared_bundle import evaluate_split, preservation_reference
        bundle = preservation_dir / "bundle"
        preservation = read(preservation_dir / "preservation_report.json")
        reference = preservation_reference(bundle, canaries)
        if (not reference.get("prior_report_checks_and_inputs_match")
                or preservation.get("old_sha256") != old_sha or preservation.get("new_sha256") != new_sha):
            raise RuntimeError("The saved native/shared/reload audit does not prove equivalence for these exact weights/inputs")
        bundle_config = read(bundle / "branches.json")
        if bundle_config.get("new_names") != names or bundle_config.get("dtype") != "float32":
            raise ValueError("Saved bundle vocabulary or precision differs from the verified data contract")
        hashes = {name: digest(bundle / name) for name in ("branches.json", "branches.safetensors")}
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
            raise RuntimeError("Requested CUDA is unavailable; no implicit CPU fallback")
        output.mkdir(parents=True, exist_ok=True)
        result = {"created_at": now(), "status": "running", "classes": names, "dataset_audit": audit,
                  "bundle": str(bundle), "bundle_files_sha256": hashes,
                  "old_checkpoint_sha256": old_sha, "new_checkpoint_sha256": new_sha,
                  "native_evaluation_sha256": digest(args.run / "evaluation.json"),
                  "preservation_reference": reference, "taxonomy_subsets": group_metadata,
                  "legacy_holdout": legacy_audit,
                  "allowed_untrained_classes": audit["allowed_untrained_classes"], "splits": {},
                  "protocol": {"device": args.device, "dtype": "float32", "imgsz": 640,
                      "preprocessing": "640 square LetterBox auto=False, RGB /255", "conv_bn_fused": False,
                      "forward": "FrozenYOLO26Branches.forward(branch='both')", "fixed_confidence": .25,
                      "fixed_matching_iou": .5, "ap_confidence_floor": .001, "ap_confidence_comparison": "> 0.001",
                      "ap_iou_range": [.5, .95], "max_detections_per_branch": 100,
                      "additional_nms": False, "cross_branch_suppression": False,
                      "deterministic_algorithms": True, "test_threshold_tuning": False,
                      "selection": "Training selected best.pt on validation; this script performs no selection or training"},
                  "limitations": ["Mixed coarse/fine source-label baseline, not completed hierarchical annotation or a commercial benchmark",
                                  "Train-absent classes have no positive training evidence; all-taxonomy and trained-only scores are both retained",
                                  "Ten old-task canaries do not establish full COCO80, export-runtime or camera preservation",
                                  "Native standalone and shared square/unfused pipelines are measured separately",
                                  "Source labels can be incomplete and have incompatible granularity; unsupported AP/Recall is not a learned capability"]}
        save(output / "evaluation.json", result)
        if legacy_audit.get("enabled"):
            save(output / "legacy_audit.json", legacy_audit)
        state.update(status="evaluating_shared", updated_at=now())
        save(args.run / "finish_status.json", state)
        model = FrozenYOLO26Branches.load_bundle(bundle, device=args.device)
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("Shared bundle did not retain FP32")
        for split, split_paths in evaluation_splits.items():
            document = ground_truth(split_paths, names)
            destination = output / split
            scored = evaluate_split(model, document, split_paths, "new", destination,
                                    args.device, preprocess, decode)
            predictions = read(destination / "predictions.json")
            # Re-score the same actual predictions with arbitrary-taxonomy support metadata.
            scored["metrics"] = coco_score(document, predictions)
            scored["metrics"]["ap_confidence_comparison"] = "> 0.001"
            scored["trained_classes_only"] = trained_classes_only_score(document, predictions, audit["trained_classes"])
            scored["taxonomy_subsets"] = {group: subset_score(document, predictions, classes, group)
                                           for group, classes in groups.items()}
            scored["example_originals"] = {str(image_id): str(path.resolve())
                                            for image_id, path in enumerate(split_paths, 1) if image_id <= 8}
            if split == "legacy":
                scored["holdout_scope"] = legacy_audit["scope"]
            save(destination / "metrics.json", scored["metrics"])
            save(destination / "per_class.json", scored["metrics"]["fixed_025_iou_050"]["per_class"])
            save(destination / "trained_classes_only.json", scored["trained_classes_only"])
            save(destination / "taxonomy_subsets.json", scored["taxonomy_subsets"])
            save(destination / "summary.json", scored)
            result["splits"][split] = scored
            save(output / "evaluation.json", result)
        if any(digest(bundle / name) != expected for name, expected in hashes.items()):
            raise RuntimeError("Bundle content changed during final evaluation")
        if digest(args.weights) != old_sha or digest(best) != new_sha:
            raise RuntimeError("A source checkpoint changed during final evaluation")
        if legacy_audit.get("enabled"):
            if digest(args.dataset.parent / "manifest.json") != legacy_audit["manifest_sha256"]:
                raise RuntimeError("Dataset manifest changed while evaluating the legacy holdout")
            for record in legacy_audit["files"]:
                if (digest(Path(record["image"])) != record["image_sha256"]
                        or digest(Path(record["label"])) != record["label_sha256"]):
                    raise RuntimeError("A legacy image or GT label changed during evaluation")
        if args.taxonomy_review is not None and digest(args.taxonomy_review) != group_metadata["sha256"]:
            raise RuntimeError("Taxonomy review changed while evaluating")
        result.update(status="complete", completed_at=now(), seconds=time.perf_counter() - started,
                      bundle_files_unchanged=True, source_checkpoints_unchanged=True)
        save(output / "evaluation.json", result)
        write_report(args.run / "report.html", args.run, native, result, audit, preservation)
        write_report(output / "report.html", args.run, native, result, audit, preservation)
        state.update(status="completed", completed_at=now(), seconds=time.perf_counter() - started,
                     shared_evaluation=str(output / "evaluation.json"), report=str(args.run / "report.html"))
        save(args.run / "finish_status.json", state)
        print(json.dumps({"status": "completed", "report": state["report"], "classes": len(names)}), flush=True)
    except BaseException as exc:
        state.update(status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}")
        save(args.run / "finish_status.json", state)
        if result is not None:
            result.update(status="failed", error=state["error"])
            save(output / "evaluation.json", result)
        raise
    finally:
        if model is not None:
            del model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
