# PYSKL ST-GCN++ inference subset

Copyright 2018–2019 Open-MMLab. This directory contains adapted source from
[kennymckormick/pyskl](https://github.com/kennymckormick/pyskl), commit
`f2bf3a6b08e2e8dec744692d64efdb187fd6719a`, under Apache-2.0. The complete
license is in `LICENSE`; upstream source paths and SHA256 digests are recorded
in `sources.json`. Upstream states that its repository is no longer maintained.

Changes made for this workbench:

- Extract the official COCO17 STGCN blocks, adaptive graph convolutions,
  multi-scale temporal convolutions, and graph construction.
- Replace MMCV's BN/ReLU builders with equivalent `torch.nn` layers; remove
  registry dependencies and implicit checkpoint loading.
- Implement the official GCNHead average pooling with its zero-dropout setting.
- Keep one test clip instead of the published ten-clip accuracy protocol.
  Preprocessing follows the saved `upstream_config.py` validation pipeline:
  PreNormalize2D, joint features, UniformSample(100), PoseDecode, and
  FormatGCNInput(num_person=2, mode='zero'). There is no PoseCompact in this
  STGCN configuration. The confidence channel is not a depth coordinate.
- Use a local NumPy RandomState with seed 255 instead of mutating global RNG.
- Add application input-quality checks outside the native preprocessing.

The model was trained on HRNet COCO17 poses from NTU RGB+D 120. YOLO supplies
the same joint order but has a different pose-error distribution. A tracked
person fills the first person slot; the second slot is zero. Person-pair
interactions cannot be represented fully by a single-person input. These are
120 predefined action classes, not custom operating procedures or an open-set
event detector. Scores alone are not calibrated safety-event probabilities.

## Reproduce the weight download

Official checkpoint:
[STGCN++ NTU120 XSub HRNet Joint](https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth)

Expected size: **5,915,801 bytes**. SHA256:
`c6e074210fa30c01081460665ce9c1c221ba4a60d7eeb10456cde6e8ac40486e`.

```powershell
Invoke-WebRequest -Uri 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth' -OutFile 'stgcnpp_ntu120_hrnet_j.pth'
Get-FileHash -Algorithm SHA256 -LiteralPath 'stgcnpp_ntu120_hrnet_j.pth'
```

The official label order is preserved in `labels_ntu120.txt`. The checkpoint
contains a plain tensor state dictionary and is loaded with `weights_only=True`,
`map_location='cpu'`, and `strict=True`. No MMCV, MMDetection, MMPose, or custom
compiled operators are installed or imported.

Official sample video:
[ntu_sample.avi](https://raw.githubusercontent.com/kennymckormick/pyskl/f2bf3a6b08e2e8dec744692d64efdb187fd6719a/demo/ntu_sample.avi)
(72 frames, 30 FPS, 1920×1080). Dataset/video rights remain with their authors;
the source-code license is not a separate license grant for every dataset.
