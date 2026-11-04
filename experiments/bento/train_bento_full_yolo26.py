"""Train a new YOLO26x food branch with an explicit full-data taxonomy.

The verified original 80-class checkpoint remains immutable. Its first eleven
backbone modules are copied and frozen, including BatchNorm buffers. Only a new
neck/head is optimized; original and new output semantics remain separate.
No training, GPU allocation or downloads occur during --help or --audit-only.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path

from bento_research.full_data_common import (
    audit_dataset, bind_native_dataset, coco_score, digest, ground_truth, now, save,
    trained_classes_only_score,
)


OFFICIAL_YOLO26X_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
BACKBONE_MODULES = 11


def backbone_state(model):
    prefixes = tuple(f"model.{i}." for i in range(BACKBONE_MODULES))
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name.startswith(prefixes)}


def verify_backbone(model, reference):
    import torch
    current = model.state_dict()
    changed = [name for name, value in reference.items()
               if name not in current or current[name].dtype != value.dtype
               or not torch.equal(current[name].detach().cpu(), value)]
    if changed:
        raise RuntimeError("Frozen backbone changed: " + ", ".join(changed[:5]))
    return len(reference)


def ordered_names(model):
    names = model.names
    return [names[i] if i in names else names[str(i)] for i in range(len(names))] if isinstance(names, dict) else list(names)


def verify_source_model(model):
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import DetectionModel
    architecture = model.yaml
    if type(model) is not DetectionModel or type(model.model[-1]) is not Detect:
        raise ValueError("Source must be a native YOLO detection model")
    if (architecture.get("scale") != "x" or "yolo26" not in str(architecture.get("yaml_file", "")).lower()
            or architecture.get("end2end") is not True or architecture.get("reg_max") != 1
            or len(architecture.get("backbone", [])) != BACKBONE_MODULES
            or len(ordered_names(model)) != 80):
        raise ValueError("Expected the verified 80-class YOLO26x end-to-end source architecture")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Local YOLO YAML and adjacent manifest.json")
    parser.add_argument("--weights", type=Path, required=True, help="Verified original yolo26x.pt, not a previous fine-tune")
    parser.add_argument("--output", type=Path, required=True, help="A new or completely empty run directory")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs")
    parser.add_argument("--batch", type=int, default=4, help="Fixed batch size; no automatic OOM fallback")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    parser.add_argument("--device", default="0", help="Single CUDA device, or cpu; DDP is not supported")
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2, help="Data-loader workers")
    parser.add_argument("--allow-untrained-classes", nargs="+", action="extend", default=[], metavar="CLASS",
                        help="Exact class names permitted to have zero train examples; allowance must exactly match actual missing classes; may repeat")
    parser.add_argument("--audit-only", action="store_true", help="Print the data/source-hash audit; create no run files and perform no model/GPU work")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.patience < 0 or args.workers < 0:
        parser.error("epochs/batch must be positive and patience/workers non-negative")
    if "," in args.device:
        parser.error("Only one training device is supported by this experiment")
    args.dataset, args.weights, args.output = (path.resolve() for path in (args.dataset, args.weights, args.output))
    return args


def ensure_fresh_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output is not empty; existing runs are never overwritten: {path}")


def main():
    args = arguments()
    if not args.dataset.is_file() or not args.weights.is_file():
        raise FileNotFoundError("A local dataset YAML and existing official weights are required")
    ensure_fresh_output(args.output)
    source_digest = digest(args.weights)
    if source_digest != OFFICIAL_YOLO26X_SHA256:
        raise ValueError("Source SHA256 differs from this study's verified official YOLO26x checkpoint")
    names, splits, audit = audit_dataset(args.dataset, args.allow_untrained_classes)
    if args.audit_only:
        print(json.dumps({"status": "audited_only", "source_weights_sha256": source_digest,
                          "dataset": audit}, ensure_ascii=True, indent=2))
        return

    # Fail before reserving a run or touching CUDA if the dataset/source is invalid.
    ensure_fresh_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": now()}, stream)
    out = args.output
    state = {"status": "initializing", "started_at": now(), "pid": os.getpid()}
    epochs = []

    def event(kind, **values):
        row = {"time": now(), "event": kind, **values}
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=True), flush=True)

    def progress(**values):
        state.update(values)
        state["updated_at"] = now()
        save(out / "progress.json", state)

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer
        from bento_research.frozen_yolo_branches import make_new_branch_model

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        native_yaml = bind_native_dataset(out, names, splits, audit)
        save(out / "dataset_check.json", audit)
        save(out / "class_distribution.json", audit["class_distribution"])
        model = YOLO(str(args.weights), task="detect")
        original = model.model.float()
        verify_source_model(original)
        frozen_reference = backbone_state(original)
        original_state = {name: value.detach().cpu().clone() for name, value in original.state_dict().items()}
        if any(not torch.isfinite(value).all() for value in original_state.values() if value.is_floating_point()):
            raise ValueError("The initial checkpoint contains NaN or Inf")
        old_names = ordered_names(original)
        save(out / "preservation_contract.json", {
            "source_checkpoint": str(args.weights), "source_sha256": source_digest,
            "old_classes": dict(enumerate(old_names)), "new_classes": names,
            "frozen_backbone_modules": list(range(BACKBONE_MODULES)),
            "trainable_new_branch_modules": list(range(BACKBONE_MODULES, len(original.model))),
            "old_path": "Original checkpoint is never optimized; its independent inference path must be retained",
            "combined_inference": "Compatible with a separately assembled shared immutable backbone and independent neck/head paths",
            "new_checkpoint_alone": "The new food taxonomy only; not the original 80-class output",
            "new_branch_selection": "Validation mAP50-95 conditional on exact backbone invariance",
            "allowed_untrained_classes": audit["allowed_untrained_classes"],
            "split_absent_classes": {split: audit["class_distribution"][split]["absent_classes"] for split in splits},
            "label_model": "One explicit flat output taxonomy; hierarchical supervision and cross-source annotation repair are not implemented by this trainer",
            "scope": "Mixed coarse/fine source-label baseline; no universal old-task preservation or commercial accuracy claim without separate evaluation"})

        class Trainer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                expected = self.data["names"]
                expected = [expected[i] for i in range(len(expected))] if isinstance(expected, dict) else list(expected)
                if expected != names:
                    raise ValueError("Native trainer changed the audited class-name ordering")
                target, transfer = make_new_branch_model(original, names)
                verify_backbone(target, frozen_reference)
                if any(not torch.equal(original.state_dict()[key].detach().cpu(), value)
                       for key, value in original_state.items()):
                    raise RuntimeError("New-branch initialization changed the original model")
                save(out / "initialization_audit.json", transfer)
                return target

            def _model_train(self):
                super()._model_train()
                for module in self.model.model[:BACKBONE_MODULES]:
                    module.eval()

            def _restore_frozen_ema(self):
                verify_backbone(self.model, frozen_reference)
                if self.ema is None:
                    raise RuntimeError("Expected an EMA model for deterministic frozen-state restoration")
                with torch.no_grad():
                    current = self.model.state_dict()
                    for name, value in self.ema.ema.state_dict().items():
                        if name in frozen_reference:
                            value.copy_(current[name])
                verify_backbone(self.ema.ema, frozen_reference)

            def validate(self):
                self._restore_frozen_ema()
                return super().validate()

            def save_model(self):
                self._restore_frozen_ema()
                return super().save_model()

            def final_eval(self):
                self._final_eval = True
                try:
                    return super().final_eval()
                finally:
                    self._final_eval = False

        config = dict(data=str(native_yaml), epochs=args.epochs, batch=args.batch, imgsz=640,
                      patience=args.patience, optimizer="AdamW", lr0=.0003, lrf=.01, cos_lr=True,
                      weight_decay=.0005, momentum=.9, warmup_epochs=3., warmup_bias_lr=0.,
                      amp=False, half=False, workers=args.workers, seed=42, freeze=BACKBONE_MODULES,
                      plots=True, degrees=5., hsv_h=.01, hsv_s=.2, hsv_v=.15,
                      mosaic=.5, close_mosaic=min(10, args.epochs), mixup=0., copy_paste=0.,
                      translate=.05, scale=.2, fliplr=.5, flipud=0., shear=0., perspective=0.,
                      device=args.device, project=str(out), name="train", exist_ok=True,
                      save=True, val=True, pretrained=True, resume=False, cache=False,
                      deterministic=True, nbs=16, max_det=100)
        save(out / "training_config.json", {
            "recipe": "Verified pretrained YOLO26x, immutable original path, new neck/head, full-data explicit taxonomy",
            "weights": str(args.weights), "weights_sha256": source_digest, "classes": names,
            "class_count": len(names), "dataset": audit, "kwargs": config,
            "versions": {key: importlib.metadata.version(key) for key in ("torch", "torchvision", "ultralytics", "numpy")},
            "checkpoint_selection": "Validation mAP50-95; test inference is performed only after selection",
            "initialization": "Transfer compatible pretrained tensors; reset semantic class projections whenever names differ, even at equal nc",
            "risk_metadata": audit["limits"], "executed_script_sha256": digest(Path(__file__)),
            "allowed_untrained_classes": audit["allowed_untrained_classes"],
            "actual_train_absent_classes": audit["actual_train_absent_classes"],
            "val_absent_classes": audit["class_distribution"]["val"]["absent_classes"],
            "test_absent_classes": audit["class_distribution"]["test"]["absent_classes"],
            "full_data_common_sha256": digest(Path(__file__).parent / "bento_research/full_data_common.py")})

        def started(trainer):
            if trainer.amp:
                raise RuntimeError("AMP unexpectedly enabled; this study explicitly uses FP32")
            verify_backbone(trainer.model, frozen_reference)
            if ordered_names(trainer.model) != names:
                raise RuntimeError("Actual model labels disagree with the audited taxonomy")
            if any(value.requires_grad for name, value in trainer.model.named_parameters() if name in frozen_reference):
                raise RuntimeError("Backbone parameters unexpectedly trainable")
            if any(value.dtype != torch.float32 for value in trainer.model.parameters()):
                raise RuntimeError("Trainable model unexpectedly changed floating-point precision")
            save(out / "effective_training.json", {
                "amp": bool(trainer.amp), "device": str(trainer.device), "actual_batch_size": int(trainer.batch_size),
                "ultralytics_args": {key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(trainer.args).items()},
                "train_images": len(splits["train"]), "val_images": len(splits["val"]),
                "test_images_excluded_from_training": len(splits["test"])})
            progress(status="training", total_epochs=args.epochs, completed_epochs=0)
            event("training_started", classes=len(names), batch=int(trainer.batch_size), device=str(trainer.device))

        def finite_loss(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise RuntimeError("Training loss is NaN/Inf; no settings are silently changed")

        def epoch_end(trainer):
            if getattr(trainer, "_final_eval", False):
                return
            verified = verify_backbone(trainer.model, frozen_reference)
            metrics = {str(key): float(value) for key, value in (trainer.metrics or {}).items()}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Validation metric contains NaN/Inf")
            if getattr(trainer.validator.args, "half", False):
                raise RuntimeError("Epoch validation unexpectedly used FP16")
            row = {"epoch": trainer.epoch + 1, "metrics": metrics, "fitness": float(trainer.fitness),
                   "learning_rates": [group["lr"] for group in trainer.optimizer.param_groups],
                   "frozen_tensors_equal": verified}
            epochs.append(row)
            save(out / "epoch_metrics.json", epochs)
            progress(completed_epochs=len(epochs), latest=row, best_fitness=float(trainer.best_fitness))
            event("epoch", **row)

        model.add_callback("on_pretrain_routine_end", started)
        model.add_callback("on_train_batch_end", finite_loss)
        model.add_callback("on_fit_epoch_end", epoch_end)
        model.train(trainer=Trainer, **config)
        if digest(args.weights) != source_digest:
            raise RuntimeError("Original pretrained checkpoint file was modified")
        if any(not torch.equal(original.state_dict()[key].detach().cpu(), value)
               for key, value in original_state.items()):
            raise RuntimeError("Original in-memory source state changed")
        best = out / "train/weights/best.pt"
        if not best.is_file():
            raise RuntimeError("Training produced no best.pt")
        progress(status="evaluating", completed_epochs=len(epochs))
        del model, original, original_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {"created_at": now(), "checkpoint": str(best), "checkpoint_sha256": digest(best),
                  "classes": names, "dataset": audit, "splits": {},
                  "allowed_untrained_classes": audit["allowed_untrained_classes"],
                  "trained_classes": audit["trained_classes"],
                  "pipeline": "Standalone new native branch; shared-bundle equivalence and old-task regression require separate verification"}
        for split in ("val", "test"):
            # No test model inference occurs until best.pt has been selected above.
            model = YOLO(str(best), task="detect")
            model.model.float()
            verify_backbone(model.model, frozen_reference)
            if ordered_names(model.model) != names:
                raise RuntimeError("Saved checkpoint class order changed")
            if any(not torch.isfinite(value).all() for value in model.model.state_dict().values() if value.is_floating_point()):
                raise RuntimeError("Saved best checkpoint contains NaN/Inf")
            doc = ground_truth(splits[split], names)
            save(out / f"{split}_ground_truth.json", doc)
            event("evaluating", split=split, images=len(splits[split]),
                  absent_classes=audit["class_distribution"][split]["absent_classes"])
            native = model.val(data=str(native_yaml), split=split, device=args.device, imgsz=640,
                               batch=args.batch, half=False, workers=args.workers, conf=.001, iou=.7,
                               max_det=100, plots=True, project=str(out), name=f"eval_{split}", exist_ok=True)
            native_metrics = {str(key): float(value) for key, value in native.results_dict.items()}
            if not all(math.isfinite(value) for value in native_metrics.values()):
                raise RuntimeError("Final native metrics contain NaN/Inf")
            detections, previews = [], []
            for image_id, path in enumerate(splits[split], 1):
                pred = model.predict(str(path), device=args.device, imgsz=640, half=False,
                                     conf=.001, iou=.7, max_det=100, verbose=False)[0]
                for box, score, cls in zip(pred.boxes.xyxy.cpu().tolist(), pred.boxes.conf.cpu().tolist(), pred.boxes.cls.cpu().tolist()):
                    if not all(map(math.isfinite, [*box, score])):
                        raise RuntimeError("Non-finite final detection")
                    x1, y1, x2, y2 = box
                    detections.append({"image_id": image_id, "category_id": int(cls), "score": score,
                                       "bbox": [x1, y1, x2 - x1, y2 - y1]})
                if image_id <= 8:
                    view = model.predict(str(path), device=args.device, imgsz=640, half=False,
                                         conf=.25, iou=.7, max_det=100, verbose=False)[0]
                    target = out / "examples" / split / f"{image_id:03d}_{path.stem}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    view.save(filename=str(target))
                    previews.append({"source": str(path), "result": str(target), "detections": len(view.boxes)})
                if image_id % 25 == 0:
                    event("evaluation_progress", split=split, processed=image_id)
            save(out / f"{split}_predictions_coco.json", detections)
            result["splits"][split] = {"native_ultralytics": native_metrics,
                                        "coco": coco_score(doc, detections),
                                        "coco_trained_classes_only": trained_classes_only_score(doc, detections, audit["trained_classes"]),
                                        "examples": previews}
            save(out / "evaluation.json", result)
            del model
        if digest(args.weights) != source_digest:
            raise RuntimeError("Original pretrained source changed during evaluation")
        progress(status="completed", completed_at=now(), result=str(out / "evaluation.json"), checkpoint=str(best))
        event("completed", checkpoint=str(best), results=str(out / "evaluation.json"))
    except BaseException as exc:
        progress(status="failed", error=f"{type(exc).__name__}: {exc}")
        event("failed", error=state["error"])
        raise


if __name__ == "__main__":
    main()
