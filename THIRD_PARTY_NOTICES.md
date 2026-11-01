# Third-party notices

Vision Lab Workbench's original code is distributed under AGPL-3.0. This does not relicense third-party code, pretrained models, checkpoints, datasets, or installed dependencies.

## Included source and class labels

- `backend/vendor/videomaev2/modeling_finetune.py` is derived from [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), MIT License, copyright (c) 2023 OpenGVLab. The upstream license is included at [backend/vendor/videomaev2/LICENSE.txt](backend/vendor/videomaev2/LICENSE.txt). The file retains its upstream BEiT/timm/DINO/DeiT attribution. Local integration supports loading the Giant K710 architecture; later changes should retain attribution.
- `backend/assets/kinetics710.txt` preserves the upstream VideoMAEv2 class order.
- `backend/assets/kinetics400.json` comes from the official PySlowFast class-name mapping, cross-checked with the Hugging Face Kinetics-400 mapping. PySlowFast is copyright 2019 Facebook, Inc., Apache-2.0; the license is included at [backend/assets/LICENSE-SlowFast.txt](backend/assets/LICENSE-SlowFast.txt). Exact source URLs, revision and label-map hash are recorded in [backend/assets/PROVENANCE.txt](backend/assets/PROVENANCE.txt).
- `backend/vendor/pyskl_stgcnpp/` adapts the ST-GCN++ architecture and NTU120 labels from [PYSKL](https://github.com/kennymckormick/pyskl), revision `f2bf3a6b08e2e8dec744692d64efdb187fd6719a`, under Apache-2.0. See its [license](backend/vendor/pyskl_stgcnpp/LICENSE), [notice](backend/vendor/pyskl_stgcnpp/NOTICE.md) and [source hashes](backend/vendor/pyskl_stgcnpp/sources.json). The local PyTorch adapter avoids requiring MMCV; this does not change the original checkpoint's class order or imply identical accuracy with a different pose estimator.
- `backend/vendor/relateanything/` contains the required source, license and notices from [RelateAnything](https://github.com/Maelic/RelateAnything), revision `fc626d0d58113efcc8b08252c168a7ad529a171f`. Its code is AGPL-3.0-only with the upstream Section 7 attribution notice. Retain **“Based on RelateAnything by Maëlic Neau”** and a link to the upstream project in the interface, together with its [LICENSE](backend/vendor/relateanything/LICENSE), [NOTICE](backend/vendor/relateanything/NOTICE) and [third-party notices](backend/vendor/relateanything/THIRD_PARTY_NOTICES.md).

## Dependencies installed separately

The Python and npm package managers install the dependencies declared in the requirements and lock files. Their own distributions carry their licenses and notices. Key upstream projects include:

| Component | Upstream / license information |
|---|---|
| Ultralytics YOLO | [AGPL-3.0 / commercial alternatives](https://www.ultralytics.com/license) |
| PyTorch | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| PyTorchVideo | [Apache-2.0](https://github.com/facebookresearch/pytorchvideo/blob/main/LICENSE) |
| Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| RF-DETR | [Upstream repository](https://github.com/roboflow/rf-detr) |
| RF-DETR Plus, optional | [PML-1.0](https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE); not redistributed here |
| VDO.Ninja SDK | [MPL-2.0](https://github.com/steveseguin/ninjasdk); the SDK is bundled into locally built browser assets by Vite |
| ONNX Runtime, optional CPU workflow dependency | [MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE); pinned in `requirements-workflows.txt` |
| React | [MIT](https://github.com/facebook/react/blob/main/LICENSE) |
| React Flow / XYFlow | [MIT](https://github.com/xyflow/xyflow/blob/main/LICENSE) |
| Lucide icons | [ISC](https://github.com/lucide-icons/lucide/blob/main/LICENSE) |
| ComfyUI, downloaded by the full installer or supplied by the user | [Official repository](https://github.com/Comfy-Org/ComfyUI) / [GPL-3.0 license](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) |

## ComfyUI installation and integration

The source release includes the original `comfyui/vision_workbench` bridge and installation tools under this project's AGPL-3.0 license. It does not bundle or relicense ComfyUI itself. A full installation first reuses an existing ComfyUI where available; otherwise it downloads the pinned official source archive recorded in `comfy_manager.py`, verifies its SHA256, and preserves the upstream source tree and `LICENSE` in `.tools/ComfyUI/`.

ComfyUI's Python dependencies are installed separately into the managed ComfyUI environment and retain their respective licenses. Existing user-managed ComfyUI installations retain their code, packages, models, custom nodes and notices. The bridge connects to the local Vision Workbench service; neither its installation nor the included workflow JSON grants rights to third-party model weights or media.

## Bundled teaching media and annotations

The reviewed collection under `examples/` is included in the source release. See [per-file attribution and licenses](examples/media/ATTRIBUTION.md), [the source manifest](examples/media/manifest.json), and [the fine-tuning conversion record](examples/finetuning/coco10_yolo/provenance.json).

- Ten unmodified COCO val2017 images, also copied unchanged into the train/validation layout, retain their individual Creative Commons licenses. Some include NonCommercial, NoDerivatives and/or ShareAlike conditions. They are not relicensed under AGPL.
- COCO annotations/captions and converted YOLO labels are attributed to the COCO Consortium under CC BY 4.0. Subsetting and format conversion are documented.
- `freezy_traffic.webm` is **Freezy Traffic** by **LIFEOFVIDS**, distributed unchanged under **CC BY 3.0**, with source and license-review links in `examples/media/video/source.json`.

No pretrained weight files, privately trained checkpoints, personal consent records, private datasets, or other locally added media are part of the source release. Obtain model weights separately under the respective upstream conditions.

## Optional workflow weights and conceptual references

[The workflow model manifest](backend/assets/workflow_model_sources.json) records official URLs, exact sizes and SHA256 hashes. `download_workflow_models.py` downloads weights only with `--download`; it does not redistribute them in this source tree.

- YOLO26n Pose uses the Ultralytics model terms (AGPL-3.0 or an applicable upstream commercial license).
- ST-GCN++ NTU120 is obtained from the official PYSKL/OpenMMLab checkpoint host. Apache-2.0 covers the source adaptation; users must separately observe the upstream checkpoint and NTU dataset conditions.
- RelateAnything ConvNeXt-Large ONNX and its sidecars are pinned to Hugging Face revision `3f320abdb0c579f685815089a08c472ae6928d4f`. The upstream model card identifies the **DINOv3 License** for these weights, separately from the code license: [model card](https://huggingface.co/maelic/relsgg-convnext-large/blob/3f320abdb0c579f685815089a08c472ae6928d4f/README.md), [license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).

The workflow templates independently implement concepts such as zone entry, dwell time, posture transitions, PPE visibility and event order. [ZoneGuard](https://github.com/K-saif/ZoneGuard) has no explicit LICENSE in the reviewed repository, and [AI Safety Monitoring](https://github.com/nusRying/ai-safety-monitoring) declares Proprietary / All Rights Reserved. They are conceptual references only; no source code from either project is included or relicensed. Links to MMAction2, Supervision and ASFormer in templates are method references, not claims that those applications or complete temporal-segmentation models are integrated.
