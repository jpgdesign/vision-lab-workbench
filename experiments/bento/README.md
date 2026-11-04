# Bento incremental-detection research

These standalone experiments are separate from the teaching UI. The research
objective is to learn new food classes from pretrained features while preserving
the original detector's inference path. YOLO26, RF-DETR and Grounding DINO are the
comparison targets; **this implementation currently covers YOLO26 only**.

## Implemented experiment

`bento_research/frozen_yolo_branches.py` contains one immutable YOLO26 backbone,
the immutable original neck/head, and a separate trainable new neck/head. It
preserves the native YOLO26 routes, clones feature storage between branches,
and keeps independent outputs. There is no cross-branch NMS or shared top-k.

New neck/head training starts from compatible pretrained weights. New class
output projections require new initialization. The original checkpoint is never
optimized. Freezing a backbone alone does **not** preserve a mutable old head;
the original head here is retained as its own fixed path.

The portable research artifact is `branches.json` plus
`branches.safetensors`, with one saved backbone. It is not a normal Ultralytics
checkpoint or a workbench checkpoint. A separately saved food-only `best.pt` is
compatible with normal YOLO inference, but does not contain the original 80-class
output. Do not confuse these two deliverables.

## Environment

Verified environment: Python 3.11, PyTorch 2.8.0+cu128, Ultralytics 8.4.0.
Install the PyTorch build appropriate for your hardware first, then the research
dependencies listed in `requirements.txt`. These scripts do not install or upgrade
your environment automatically. Use an isolated environment and retain the
recorded versions when loading an output-equivalence bundle.

## Dataset contract

The initial experiment is a **six-coarse-class source-annotation pilot**, with
405 images: 287 train, 59 validation, 59 test. Classes are staple, vegetable, egg,
soy, meat and seafood. This is not the original 91-class food-recognition task,
and annotations have not all been manually corrected. Seafood has only 21/5/5
instances in train/validation/test. No commercial reliability claim follows
from this pilot.

The trainer requires a local YOLO dataset YAML with six unique names and a
`manifest.json` beside it. The manifest must contain `images` records with
`image` (relative to the YAML directory or absolute), `split` and `group_id`;
optional `grouping.legacy_holdout_group_ids` identifies excluded old holdout
groups. The trainer checks file/manifest agreement, group-disjoint splits and
decoded-image uniqueness. Dataset YAML download directives are rejected.
These checks establish structural consistency, **not annotation completeness**.

Keep duplicate images, adjacent frames and crops of one original in a single
split before augmentation. Preserve original fine labels and use a versioned
semantic mapping for coarse labels. Broad classes such as vegetable are not
synonyms of COCO subclasses such as broccoli.

The [BentoCal slides](https://www.slideshare.net/slideshow/bentocal-yolov8/281855445)
inspire coarse categories and annotation review. Slides 11/12 change bounding-box
granularity to smaller food pieces; they do not publish a train/validation/test
ratio. Arbitrarily slicing boxes or cropping texture patches is not an equivalent
labeling method. Any new ground-truth definition needs a new version and all
models must be evaluated against the same definition. The slides have incomplete
recipe/class mapping information and inconsistent reported figures, so this is
not a claimed reproduction.

Images, private annotations, trained weights and environment folders are not
included here. Each source dataset and original model retains its own license.

## Completed pilot: 2026-11-04

YOLO26x completed 60 epochs (best validation checkpoint: epoch 57, seed 42).
The actual shared-backbone inference bundle was evaluated after selection:

| Split | Images / boxes | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|
| Validation | 59 / 112 | 70.75% | 66.96% | 58.56% | 35.69% |
| Test | 59 / 123 | 81.82% | 73.17% | 82.52% | 45.18% |

P/R use confidence 0.25 and IoU 0.5. AP uses a 0.001 score floor and max 100
detections. This pipeline uses square 640 preprocessing, unfused FP32 and native
end-to-end predictions without extra NMS. All 498 frozen backbone tensors remained
equal during training. On ten fixed COCO images, both branches matched their
native models bitwise, including a portable save/reload. This is a finite canary
check, not full COCO80 or export/live-camera validation.

**Generalization remains weak on the original difficult Bento photos.** On the
same ten fully mappable legacy images (85 boxes), recall was 17.65% for the prior
food fine-tune and 22.35% for this new branch; 66 boxes were still missed. That
diagnostic collapses fine labels and applies the same NMS to both models, so it
is a different evaluation pipeline. The prior model used those images for
validation selection; the new training excluded their groups. Neither the new
test score nor this comparison establishes commercial readiness or 91-class
improvement. Seafood has only five test boxes.

On one RTX 3060, 30 warmed measurements per phase gave median preprocessing-to-
decoded-output latency of 105.49 ms for two serial full models versus 80.96 ms
for the shared backbone (both branches, FP32). Loading, disk I/O, HTTP and camera
transport are excluded; fixed phase order and system load limit interpretation.

[Recorded results, metrics, protocols and artifact hashes](pilot_2026-11-04.json)
include the execution-version caveat: the pilot restored frozen EMA tensors
before saving; the current trainer also restores them before epoch validation.
The final saved checkpoint was rescored. This is one pilot; the RF-DETR and
Grounding DINO comparison and multi-seed confirmation remain uncompleted.

## Commands

Run from this directory. Replace the example paths with your own existing files.

```powershell
python train_bento_yolo26_frozen_branch.py --dataset C:/data/bento/dataset.yaml --weights C:/models/yolo26x.pt --output C:/runs/bento-frozen --epochs 60 --batch 2 --patience 12
```

This trains only the new native food model with an exactly frozen backbone,
checks frozen tensors every epoch, selects its best checkpoint on validation,
and evaluates validation/test after selection. The immutable old path is assembled
afterwards. The script saves source hashes, effective settings, epoch metrics,
COCO predictions and fixed-confidence metrics.

```powershell
python verify_bento_frozen_bundle.py --old-checkpoint C:/models/yolo26x.pt --new-checkpoint C:/runs/bento-frozen/train/weights/best.pt --canary-images C:/data/old-task-canaries --output C:/runs/bento-frozen/preservation --device cpu
python predict_bento_frozen_bundle.py --bundle C:/runs/bento-frozen/preservation/bundle --source C:/data/example.jpg --output C:/runs/example-prediction
```

The verifier checks both raw native outputs and decoded boxes, then verifies a
safetensors save/reload. It rejects mismatched backbone tensors. It uses the same
640-square FP32 unfused preprocessing for each comparison. Those canary checks
are not a complete COCO80 AP benchmark and do not verify FP16, exported runtimes,
or a camera pipeline. The prediction command writes separate old/new images,
a combined drawing and namespaced JSON; it never discards old predictions to
make room for the new branch.

```powershell
python evaluate_bento_shared_bundle.py --bundle C:/runs/bento-frozen/preservation/bundle --dataset C:/data/bento/dataset.yaml --output C:/runs/bento-frozen/shared-evaluation --device cuda:0 --canary-images C:/data/old-task-canaries --canary-annotations C:/data/old-task-instances.json
python -m pytest bento_research/test_frozen_yolo_branches.py -q
```

Compare two complete native models against one shared-backbone bundle after all
training/evaluation GPU jobs finish:

```powershell
python benchmark_bento_branch_latency.py --old-checkpoint C:/models/yolo26x.pt --new-checkpoint C:/runs/bento-frozen/train/weights/best.pt --bundle C:/runs/bento-frozen/preservation/bundle --images C:/data/old-task-canaries --output C:/runs/bento-frozen/latency.json --device cuda:0
```

The benchmark checks that parameters, buffers, classes and head settings match;
excludes loading and warmup; and reports forward and preprocessing-to-decoded-box
p50/p95 latency plus process-local CUDA tensor memory. It does not measure camera,
HTTP or UI latency. Use the exact checkpoint from which the bundle was built.

The evaluator scores the actual shared-bundle pipeline. Its square preprocessing
and unfused execution can differ from native standalone YOLO validation; both
protocols must be labeled rather than mixing their scores. Class names remain
separate between branches. The tests use randomly initialized native YOLO26n on
CPU, take an actual new-branch optimization step, and check old tensor/output
invariance, feature isolation, invalid backbone rejection and portable reload.

## Full-source baseline and parallel annotation review

`train_bento_full_yolo26.py` is a separate runner for an explicit arbitrary-size
taxonomy. It leaves the six-class experiment unchanged. It verifies this study's
original YOLO26x checkpoint hash, freezes all backbone parameters and buffers,
resets new semantic class projections, and selects the checkpoint on validation
mAP50–95. The original 80-class inference path remains separate and immutable.

The 2026-11-05 baseline uses 1,727 training images, 236 validation images and the
same 59-image test cohort; 14 legacy images are held out separately. All six
sources are represented. Exact-image alternatives, empty annotations, known
quality issues and protected related images are recorded separately. Related
images stay in one split; the earlier representative-image sampling cap is
removed. This is not all 2,197 files placed in training.

There are 149 candidate output names, including broad labels and unresolved
source terms; 144 have training examples. This **flat-label baseline does not
implement hierarchical or partial-label supervision**. A parallel annotation
version preserves hierarchy, preparation and dish attributes with unknown values
and review evidence. Neither name mapping nor visual review of a subset establishes
complete annotations. Source labels and this baseline's frozen dataset are not
overwritten by review work.

```powershell
python train_bento_full_yolo26.py --dataset C:/data/full-bento/dataset.yaml --weights C:/models/yolo26x.pt --output C:/runs/full-bento --epochs 60 --batch 2 --patience 15 --device 0 --workers 0 --allow-untrained-classes cod corn fried_chicken_cube preserved_egg_tofu white_radish
```

The allowance is specific to this dataset: it must exactly match the actual
train-empty classes. Omit it when every class has training examples. It never
moves test images into training. Add `--audit-only` to check data and the source
hash without loading a model, allocating GPU tensors, or reserving a run folder.
Existing nonempty run directories are rejected.

After the trainer finishes, verify and evaluate the actual shared model:

```powershell
python finish_bento_full_baseline.py --run C:/runs/full-bento --dataset C:/data/full-bento/dataset.yaml --weights C:/models/yolo26x.pt --canary-images C:/data/old-task-canaries --taxonomy-review C:/data/full-bento/taxonomy_review.json --allow-untrained-classes cod corn fried_chicken_cube preserved_egg_tofu white_radish --device cuda:0
```

The finisher keeps all-taxonomy scores and adds explicitly labeled trained-only
and optional fine/broad/review category views. Ground-truth-absent categories are
not presented as validated. It checks hashes and requires completed training;
it does not start or alter training. `--expected-new-classes` on the lower-level
bundle verifier now accepts this explicit class count; its default remains six.
When `manifest.json` explicitly declares `legacy_samples`, the finisher validates
their image/label hashes, boxes and split isolation before scoring that historical
holdout separately. It does not discover or silently add arbitrary image folders.

The initial batch-4 attempt failed on its first batch with a cuDNN/CUBLAS execution
error; its logs were preserved and batch 2 was started as a fresh run. No full-source
accuracy result is claimed here before final evaluation. This is an exploratory
baseline, not the completed three-model comparison or a commercial benchmark.

## Research limits and next comparisons

- Frozen-path preservation keeps original errors too; it does not make the old
  detector more accurate.
- Replay, distillation and adaptive shared-path methods are planned comparisons,
  not implemented paper reproductions in this folder.
- RF-DETR and Grounding DINO need their own feature routing, loss and validation
  adapters. This YOLO26 implementation must not be advertised as supporting them.
- Use the same held-out data and tuning budget; record each model's appropriate
  optimizer, effective batch, resolution, prompt/token settings and precision.
- Report new-class AP/Recall, old-class regression, latency and memory separately.
  Confirm final candidates over multiple seeds and an external target-domain set.
