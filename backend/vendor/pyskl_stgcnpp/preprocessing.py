# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Adapted from PYSKL (Apache-2.0), commit f2bf3a6b08e2e8dec744692d64efdb187fd6719a.
# Changes: fixed one-clip COCO17 joint inference path and local RandomState
# instead of changing NumPy's global RNG. See NOTICE.md and sources.json.
import numpy as np


def uniform_indices(total_frames, clip_len=100):
    """Official UniformSampleFrames test path, num_clips=1 and seed=255."""
    rng = np.random.RandomState(255)
    # Preserve the upstream RNG draws for p_interval=1, including its no-op crop.
    ratio = rng.rand() * 0 + 1
    frames = int(ratio * total_frames)
    offset = rng.randint(total_frames - frames + 1)
    if frames < clip_len:
        indices = np.arange(clip_len)
    elif frames < clip_len * 2:
        basic = np.arange(clip_len)
        selected = rng.choice(clip_len + 1, frames - clip_len, replace=False)
        increments = np.zeros(clip_len + 1, dtype=np.int64)
        increments[selected] = 1
        indices = basic + np.cumsum(increments)[:-1]
    else:
        boundaries = np.array([i * frames // clip_len for i in range(clip_len + 1)])
        indices = boundaries[:clip_len] + rng.randint(np.diff(boundaries))
    return ((indices + offset) % total_frames).astype(np.int64)


def coco_joint_input(sequence, image_size):
    """PreNormalize2D -> joint features -> uniform100 -> zero-padded M=2.

    Input is one person's pixel [x,y,confidence] sequence. Output is
    (num_clips=1, M=2, T=100, V=17, C=3), the backbone's N,M,T,V,C format.
    The second person slot is zeros, as FormatGCNInput(mode='zero') specifies.
    """
    width, height = image_size
    keypoint = np.array(sequence, dtype=np.float32, copy=True)
    maskout = keypoint[..., 2] <= 0.01
    keypoint[..., 0] = (keypoint[..., 0] - width / 2) / (width / 2)
    keypoint[..., 1] = (keypoint[..., 1] - height / 2) / (height / 2)
    keypoint[..., 0][maskout] = 0
    keypoint[..., 1][maskout] = 0
    # GenSkeFeat(feats=['j']) retains these three channels without bone/motion.
    sampled = keypoint[uniform_indices(len(keypoint))]
    result = np.zeros((1, 2, 100, 17, 3), dtype=np.float32)
    result[0, 0] = sampled
    return np.ascontiguousarray(result)
