# Third-party notices

Vision Lab Workbench's original code is distributed under AGPL-3.0. This does not relicense third-party code, pretrained models, checkpoints, datasets, or installed dependencies.

## Included source and class labels

- `backend/vendor/videomaev2/modeling_finetune.py` is derived from [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), MIT License, copyright (c) 2023 OpenGVLab. The upstream license is included at [backend/vendor/videomaev2/LICENSE.txt](backend/vendor/videomaev2/LICENSE.txt). The file retains its upstream BEiT/timm/DINO/DeiT attribution. Local integration supports loading the Giant K710 architecture; later changes should retain attribution.
- `backend/assets/kinetics710.txt` preserves the upstream VideoMAEv2 class order.
- `backend/assets/kinetics400.json` comes from the official PySlowFast class-name mapping, cross-checked with the Hugging Face Kinetics-400 mapping. PySlowFast is copyright 2019 Facebook, Inc., Apache-2.0; the license is included at [backend/assets/LICENSE-SlowFast.txt](backend/assets/LICENSE-SlowFast.txt). Exact source URLs, revision and label-map hash are recorded in [backend/assets/PROVENANCE.txt](backend/assets/PROVENANCE.txt).

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
| React | [MIT](https://github.com/facebook/react/blob/main/LICENSE) |
| React Flow / XYFlow | [MIT](https://github.com/xyflow/xyflow/blob/main/LICENSE) |
| Lucide icons | [ISC](https://github.com/lucide-icons/lucide/blob/main/LICENSE) |

No pretrained weight files, privately trained checkpoints, personal consent records, or dataset images/videos are part of the source release. Obtain those separately under the respective upstream conditions.
