"""Train an added YOLO26x food branch from pretrained weights.

The original 80-class checkpoint is immutable. This file trains only the new
branch (neck/head) with a frozen pretrained backbone. The separate shared-backbone
inference module composes the old and new branches without cross-branch suppression.
This six-class source-annotation pilot does not replace a formal fine-class study.
"""
from bento_research.common import *
import sys


def backbone_state(model, count=11):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(tuple(f"model.{i}." for i in range(count)))}


def verify_backbone(model, reference):
    import torch
    current = model.state_dict()
    changed = [k for k, v in reference.items()
               if k not in current or not torch.equal(current[k].detach().cpu().float(), v.float())]
    if changed:
        raise RuntimeError("Frozen backbone changed: " + ", ".join(changed[:5]))
    return len(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT / "dataset/dataset.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs/model_weights/yolo26x/yolo26x.pt")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "yolo26x_frozen_branch")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--evaluate-only", action="store_true", help="Score the existing best.pt without training")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.patience < 0:
        parser.error("epochs and batch must be positive, patience non-negative")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    state = {"status": "initializing", "started_at": now(), "pid": os.getpid()}
    log_path = out / "events.jsonl"
    def event(kind, **values):
        row = {"time": now(), "event": kind, **values}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=True), flush=True)
    def progress(**values):
        state.update(values); state["updated_at"] = now(); save(out / "progress.json", state)
    try:
        import importlib.metadata
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer
        torch.set_num_threads(4)
        dataset = args.dataset.resolve(); weights = args.weights.resolve()
        if not dataset.is_file() or not weights.is_file():
            raise FileNotFoundError("Dataset YAML and local official weights must exist before launching")
        names, splits, audit = audit_dataset(dataset)
        # Native Ultralytics resolves relative YAML/TXT paths differently from
        # the manifest auditor. Bind training to the exact audited file lists.
        import yaml
        native_data = {"path": str(out), "nc": len(names), "names": dict(enumerate(names))}
        native_lists = {}
        for split, paths in splits.items():
            listing = out / f"audited_{split}_images.txt"
            listing.write_text("\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8")
            native_data[split] = str(listing)
            native_lists[split] = {"path": str(listing), "sha256": digest(listing)}
        native_yaml = out / "audited_dataset.yaml"
        native_yaml.write_text(yaml.safe_dump(native_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        audit["native_dataset_yaml"] = str(native_yaml)
        audit["native_dataset_yaml_sha256"] = digest(native_yaml)
        audit["native_image_lists"] = native_lists
        save(out / "dataset_check.json", audit)
        ground_truths = {split: ground_truth(splits[split], names) for split in ("val", "test")}
        for split, doc in ground_truths.items(): save(out / f"{split}_ground_truth.json", doc)
        train_dir = out / "train"
        best = train_dir / "weights/best.pt"
        config = dict(data=str(native_yaml), epochs=args.epochs, batch=args.batch, imgsz=640,
            patience=args.patience, optimizer="AdamW", lr0=.0003, lrf=.01, cos_lr=True,
            weight_decay=.0005, momentum=.9, warmup_epochs=3., warmup_bias_lr=0.,
            amp=False, half=False, workers=0, seed=42, freeze=11, plots=True,
            degrees=5., hsv_h=.01, hsv_s=.2, hsv_v=.15, mosaic=.5, close_mosaic=10,
            mixup=0., copy_paste=0., translate=.05, scale=.2, fliplr=.5, flipud=0.,
            shear=0., perspective=0., device=args.device, project=str(out), name="train",
            exist_ok=True, save=True, val=True, pretrained=True, resume=False,
            cache=False, deterministic=True, nbs=16)
        if not args.evaluate_only:
            if best.exists() or (train_dir / "weights/last.pt").exists():
                raise RuntimeError("Existing training weights found; use --evaluate-only or a new output directory")
            save(out / "training_config.json", {"recipe": "YOLO26x immutable original path plus new frozen-backbone branch; six-class pilot",
                 "not_exact_reproduction": True, "weights": str(weights), "weights_sha256": digest(weights),
                 "classes": names, "dataset": audit, "kwargs": config,
                 "versions": {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "ultralytics", "numpy")},
                 "checkpoint_selection": "validation mAP50-95; no test data used for training or selection"})
            class Trainer(DetectionTrainer):
                def _model_train(self):
                    super()._model_train()
                    for module in self.model.model[:11]:
                        module.eval()

                def _restore_frozen_ema(self):
                    verify_backbone(self.model, frozen_reference)
                    # EMA arithmetic may round unchanged tensors. Save the exact
                    # frozen source tensors, including BN buffers, into EMA.
                    with torch.no_grad():
                        current = self.model.state_dict()
                        for name, value in self.ema.ema.state_dict().items():
                            if name in frozen_reference:
                                value.copy_(current[name])

                def validate(self):
                    self._restore_frozen_ema()
                    return super().validate()

                def save_model(self):
                    self._restore_frozen_ema()
                    return super().save_model()

                def final_eval(self):
                    self._final_eval = True
                    try: return super().final_eval()
                    finally: self._final_eval = False
            model = YOLO(str(weights), task="detect")
            architecture = model.model.yaml
            if architecture.get("scale") != "x" or "yolo26" not in str(architecture.get("yaml_file", "")).lower():
                raise ValueError(f"Expected official YOLO26x architecture, got {architecture.get('yaml_file')} / {architecture.get('scale')}")
            if len(architecture.get("backbone", [])) != 11:
                raise ValueError("Unexpected YOLO26 backbone boundary")
            model.model.float()
            frozen_reference = backbone_state(model.model)
            source_digest = digest(weights)
            save(out / "preservation_contract.json", {
                "source_checkpoint": str(weights), "source_sha256": source_digest,
                "old_classes": model.names, "new_classes": names,
                "frozen_backbone_modules": list(range(11)),
                "trainable_new_branch_modules": list(range(11, len(model.model.model))),
                "old_path": "Original checkpoint is unchanged and never optimized",
                "combined_inference": "Shared immutable backbone, separate neck/head and independent outputs",
                "new_checkpoint_alone": "Only the six-class food branch; not the original 80-class output",
                "new_branch_selection": "new validation mAP50-95, conditional on backbone invariance",
                "scope": "six-class pilot; old-output equivalence is verified separately from old-task AP"
            })
            if any(not torch.isfinite(t).all() for t in model.model.state_dict().values() if t.is_floating_point()):
                raise ValueError("Initial weights contain NaN or Inf")
            epochs = []
            def started(trainer):
                if trainer.amp:
                    raise RuntimeError("AMP unexpectedly enabled")
                verify_backbone(trainer.model, frozen_reference)
                unfrozen = [name for name, value in trainer.model.named_parameters()
                            if name in frozen_reference and value.requires_grad]
                if unfrozen:
                    raise RuntimeError("Backbone parameters unexpectedly trainable")
                save(out / "effective_training.json", {"amp": bool(trainer.amp), "device": str(trainer.device),
                    "actual_batch_size": int(trainer.batch_size),
                    "ultralytics_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(trainer.args).items()},
                    "train_images": len(splits['train']), "val_images": len(splits['val']),
                    "test_images_excluded_from_training": len(splits['test'])})
                progress(status="training", total_epochs=args.epochs, completed_epochs=0)
                event("training_started", batch=int(trainer.batch_size), device=str(trainer.device))
            def finite_loss(trainer):
                if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                    raise RuntimeError("Training loss is NaN/Inf; run stopped without changing settings")
            def epoch_end(trainer):
                if getattr(trainer, "_final_eval", False): return
                verified_tensors = verify_backbone(trainer.model, frozen_reference)
                raw = {str(k): float(v) for k, v in (trainer.metrics or {}).items()}
                if not all(math.isfinite(v) for v in raw.values()):
                    raise RuntimeError("Validation metric is NaN/Inf")
                if getattr(trainer.validator.args, "half", False):
                    raise RuntimeError("Validation unexpectedly used FP16")
                row = {"epoch": trainer.epoch + 1, "metrics": raw, "fitness": float(trainer.fitness),
                       "learning_rates": [g["lr"] for g in trainer.optimizer.param_groups],
                       "frozen_tensors_equal": verified_tensors}
                epochs.append(row); save(out / "epoch_metrics.json", epochs)
                progress(completed_epochs=len(epochs), latest=row, best_fitness=float(trainer.best_fitness))
                event("epoch", **row)
            model.add_callback("on_pretrain_routine_end", started)
            model.add_callback("on_train_batch_end", finite_loss)
            model.add_callback("on_fit_epoch_end", epoch_end)
            model.train(trainer=Trainer, **config)
            if digest(weights) != source_digest:
                raise RuntimeError("Original pretrained checkpoint was modified")
            progress(status="evaluating", completed_epochs=len(epochs))
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif not best.is_file():
            raise FileNotFoundError(best)
        if not best.is_file(): raise RuntimeError("Training did not produce best.pt")
        source = YOLO(str(weights), task="detect")
        source.model.float()
        frozen_reference = backbone_state(source.model)
        del source
        result = {"created_at": now(), "checkpoint": str(best), "checkpoint_sha256": digest(best),
                  "classes": names, "dataset": audit, "splits": {}}
        for split in ("val", "test"):
            model = YOLO(str(best), task="detect")
            model.model.float()
            verify_backbone(model.model, frozen_reference)
            if any(not torch.isfinite(t).all() for t in model.model.state_dict().values() if t.is_floating_point()):
                raise RuntimeError("Saved best checkpoint contains NaN/Inf")
            event("evaluating", split=split, images=len(splits[split]))
            native = model.val(data=str(native_yaml), split=split, device=args.device, imgsz=640,
                               batch=args.batch, half=False, workers=0, conf=.001, iou=.7,
                               max_det=100, plots=True, project=str(out), name=f"eval_{split}", exist_ok=True)
            native_metrics = {str(k): float(v) for k, v in native.results_dict.items()}
            if not all(math.isfinite(v) for v in native_metrics.values()):
                raise RuntimeError("Final FP32 native metrics contain NaN/Inf")
            detections, previews = [], []
            for image_id, path in enumerate(splits[split], 1):
                pred = model.predict(str(path), device=args.device, imgsz=640, half=False,
                                     conf=.001, iou=.7, max_det=100, verbose=False)[0]
                for box, score, cls in zip(pred.boxes.xyxy.cpu().tolist(), pred.boxes.conf.cpu().tolist(), pred.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    if not all(map(math.isfinite, [*box, score])): raise RuntimeError("Non-finite detection")
                    detections.append({"image_id": image_id, "category_id": int(cls), "score": score,
                                       "bbox": [x1, y1, x2-x1, y2-y1]})
                if image_id <= 8:
                    # Render only fixed-threshold detections, not the AP confidence tail.
                    view = model.predict(str(path), device=args.device, imgsz=640, half=False,
                                         conf=.25, iou=.7, max_det=100, verbose=False)[0]
                    target = out / "examples" / split / f"{image_id:03d}_{path.stem}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True); view.save(filename=str(target))
                    previews.append({"source": str(path), "result": str(target), "detections": len(view.boxes)})
                if image_id % 20 == 0: event("evaluation_progress", split=split, processed=image_id)
            save(out / f"{split}_predictions_coco.json", detections)
            result["splits"][split] = {"native_ultralytics": native_metrics,
                "coco": coco_score(ground_truths[split], detections), "examples": previews}
            result["splits"][split]["coco"]["postprocessing"] = "Native YOLO26 end-to-end; no extra NMS. iou argument is not a matching threshold."
            result["splits"][split]["coco"]["nms_iou"] = None
            save(out / "evaluation.json", result)
            del model
        progress(status="completed", completed_at=now(), result=str(out / "evaluation.json"), checkpoint=str(best))
        event("completed", checkpoint=str(best), results=str(out / "evaluation.json"))
    except BaseException as exc:
        progress(status="failed", error=f"{type(exc).__name__}: {exc}")
        event("failed", error=state["error"])
        raise


if __name__ == "__main__":
    main()
