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
error; its logs were preserved and batch 2 was started as a fresh run. The batch-2
run completed 60 epochs. Its actual shared-model 149-class test results were
Precision 87.32%, Recall 50.41%, AP50 63.21% and AP50–95 40.88% on the locked
59-image/123-box test set, using confidence 0.25 and IoU 0.5 for P/R. These are
exploratory mixed-taxonomy results; the test has GT for only 44 classes. They are
not directly comparable with six-coarse-class scores and do not complete the
three-model comparison or establish commercial performance.

## Preserve the six-class detector and add optional fine labels

The hierarchical refinement pilot preserves the successful six-class food
detector, including its learned neck and head. It extracts pooled P3/P4/P5 neck
features from cropped food boxes and trains only an additional linear classifier.
The supplied coarse parent masks incompatible fine outputs. An accepted fine
name is extra metadata: coarse boxes, category IDs and confidence scores remain
unchanged. Missing children, insufficient validation support or low fine
confidence produce a coarse-only result. This preserves existing detections;
it cannot recover food missed by the frozen detector.

This differs from the 149-class baseline, which started another food branch from
the original COCO checkpoint. The refinement runner explicitly loads the
successful **six-class food checkpoint**, not the original 80-class checkpoint.
The separate immutable COCO80/coarse6 shared bundle can be checked with live
before/after canaries. These standalone scripts do not add workbench or ComfyUI
integration, and do not implement a general multi-level hierarchical detector.

The prepared 2026-11-05 pilot uses **20 enabled fine names, 250 training crops
from 138 distinct images**. Twenty-two names met the minimum of five distinct
training images; the two whose parent had only one supported child were disabled.
The minimum is an eligibility rule, not evidence of adequate statistical support.
Original validation/test remain 59/59 images with all 112/123 source GT boxes.
Only 31/29 boxes have supported fine targets; unsupported known fine labels and
broad unknown labels remain explicitly recorded. All related validation/test
and legacy groups are excluded from training.

**Fine targets still come from source annotations, not completed manual
annotation-v2 ground truth.** Preparation excludes broad, ambiguous and known
overlapping label definitions. Unknown targets are never guessed or optimized
as negative fine labels. Residual source errors remain possible; crop accuracy
does not establish end-to-end detector accuracy or commercial readiness.

### Inputs and preparation

Run these commands from this directory, with paths to your existing audited
artifacts. Source images, annotations, checkpoints, personal paths and local
dataset manifests are not bundled with these scripts. Small sanitized result
summary JSON files are included. Dataset/model licenses still apply.

Preparation consumes the preceding research manifests, **not an arbitrary YOLO
ZIP**. `--coarse-dataset` must contain `manifest.json`, `class_mapping.json` and
`legacy_holdout.json`; `--full-dataset` must contain `manifest.json`,
`class_mapping.json` and `validation_checks.json`, together with the referenced
images and labels. Records must retain original source IDs, source-class IDs,
canonical names/flags, RGB and file hashes, old/new related-image group IDs,
and normalized top-left `xywh` boxes. The original pilot's 59-image validation,
59-image test and 149-candidate taxonomy checks are deliberate. Adapting this
to another dataset requires a newly versioned audited protocol.

```powershell
python prepare_bento_hierarchical_crops.py --coarse-dataset C:/data/bento-coarse --full-dataset C:/data/bento-full --output C:/data/bento-fine-crops --min-train-images 5 --min-children-per-parent 2
```

This records the protected groups and hashes, writes exact-box crops without
padding, and creates `manifest.json`, `class_counts.json`, split indices and
`validation_checks.json`. Unsupported or unknown targets retain `class_id: null`.
The original datasets and detector are never modified.

### Train the optional classifier

```powershell
python train_bento_hierarchical_classifier.py --dataset C:/data/bento-fine-crops/manifest.json --coarse-weights C:/runs/bento-frozen/train/weights/best.pt --output C:/runs/bento-fine-head --device cuda:0 --imgsz 224 --batch 16 --epochs 80 --patience 12 --views 2
```

Only the linear head receives optimizer updates. Frozen detector state/file
hashes are checked; feature normalization uses training crops only. The head is
selected on supported validation-crop loss. Artifacts include `head.safetensors`,
`scaler.npz`, model/protocol metadata, crop hashes and GT-crop diagnostics.
GT-crop scores are labeled as supported-class diagnostics. Fine-label acceptance
stays disabled until validation on actual detector-predicted crops.

### Calibrate, preserve and evaluate

```powershell
python evaluate_bento_hierarchical_refiner.py --dataset C:/data/bento-fine-crops/manifest.json --refiner C:/runs/bento-fine-head --coarse-dataset C:/data/bento-coarse --coarse-weights C:/runs/bento-frozen/train/weights/best.pt --coarse-evaluation C:/runs/bento-frozen/shared-evaluation --bundle C:/runs/bento-frozen/preservation/bundle --output C:/runs/bento-fine-evaluation --device cuda:0 --live-canary-count 3 --target-precision 0.9 --minimum-calibration-count 5
```

The evaluator reuses the original saved coarse predictions, runs fresh fine
inference on their actual predicted-box crops, and verifies that enrichment
changes none of their boxes, scores or coarse categories. The reported full
coarse scores therefore come from retained predictions, not a claimed new full
detector run. `--coarse-dataset` resolves image paths, permitting relocation of
the audited data. Live canaries separately rerun both original paths before and
after refinement; they are finite checks, not full COCO validation. Use
`--live-canary-count 0` and omit `--bundle` when only saved-output preservation is
being checked; the report then explicitly makes no fresh-model equivalence claim.

Thresholds use only the fixed validation set and include unsupported known fine
targets as errors. Broad/unknown fine targets are excluded from fine accuracy
and counted separately. An empirical 90% target on five or more accepted samples
is not a deployment precision guarantee. Calibration updates the refiner's
`thresholds.json` and archives its prior value in the evaluation folder.
Test known-fine recall includes unsupported fine GT; unknown/unmatched outputs
are reported separately. The report may correctly show no enabled fine parent
if calibration evidence is insufficient.

### Completed refinement pilot: 2026-11-05

Training stopped after 18 epochs; validation selected epoch 6. Supported GT-crop
accuracy was 22/31 (70.97%) on validation and 22/29 (75.86%) on test. These are
classification diagnostics on annotated crops, not deployed fine-detection
precision. Before applying the acceptance policy, fresh predictions on correctly
matched detector crops had the following diagnostic exact-name counts:

| Split | All matched known fine targets | Supported targets only |
|---|---:|---:|
| Validation | 14/38 (36.84%) | 14/16 (87.50%) |
| Test | 15/32 (46.88%) | 15/18 (83.33%) |

Unsupported known fine targets remain in the first denominator. Neither column
is a deployment result: unmatched/unknown objects and low-confidence coarse
misses require separate accounting.

**No parent met the predefined validation acceptance gate** of at least five
accepted known-fine predictions with empirical precision at least 90%. Therefore
zero fine parents were enabled and every displayed result retained its coarse
label. The frozen test detector scores stayed at Recall 73.17% and AP50 82.52%,
as expected from exact preservation of the saved predictions. Three fresh
validation canaries confirmed bitwise original COCO80/coarse6 outputs before and
after refinement and equality with their saved coarse reference. This is not a
full COCO regression test. The experiment demonstrates the preservation/fallback
mechanism; **it does not demonstrate improved deployable fine recognition**.

[Recorded refinement results and artifact hashes](hierarchical_pilot_2026-11-05.json)
contain the conditional metrics, zero-acceptance result and scope limits without
private paths or source images.

CPU-only regression tests exercise parent masking, gradients, unsupported-GT
calibration, coarse fallback and exact coarse-output preservation with dummy
features; they require no pretrained model downloads:

```powershell
python -m pytest bento_research/test_hierarchical_refiner.py -q
```

### Inspect raw fine-label candidates

When validation disables fine-label display, the retained-output gallery correctly
shows only coarse labels. A separate diagnostic gallery exposes the saved raw fine
candidates, including mistakes and labels that did not pass validation:

```powershell
python build_bento_fine_candidate_gallery.py --evaluation C:/runs/bento-fine-evaluation --coarse-evaluation C:/runs/bento-frozen/shared-evaluation --coarse-dataset C:/data/bento-coarse --output C:/runs/bento-fine-evaluation/candidate-preview --add-navigation
```

The page pairs original coarse detections with raw fine candidates, offers split
and filename filters, and lists full names, scores and top-three candidates for
each box. Candidate scores are within-parent classifier scores, not calibrated
accuracy. The generator checks image hashes and exact coarse-output identity;
it does not rerun inference, enable acceptance gates, or change model thresholds.
The first pilot contains 74 raw fine candidates in each of validation and test,
with zero accepted fine labels. All 59 images per split remain available.

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
