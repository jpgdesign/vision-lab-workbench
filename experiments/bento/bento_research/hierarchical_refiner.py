"""Optional fine labels from frozen food-detector features.

This module never changes the coarse detector's boxes, scores or labels. The
classifier is conditioned on a supplied coarse parent; absence of a calibrated
fine prediction always means retain the original coarse result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F


COARSE_NAMES = ['staple', 'vegetable', 'egg', 'soy', 'meat', 'seafood']


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''): h.update(part)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def crop_preprocess(image, imgsz=224, *, augmentation_seed=None):
    image = image.convert('RGB')
    if augmentation_seed is not None:
        rng = np.random.default_rng(augmentation_seed)
        if rng.random() < .5: image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(.9, 1.1)))
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(.9, 1.1)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(.95, 1.05)))
        image = image.rotate(float(rng.uniform(-4, 4)), resample=Image.Resampling.BILINEAR, fillcolor=(114,114,114))
    scale = min(imgsz/image.width, imgsz/image.height)
    image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.BILINEAR)
    square = Image.new('RGB', (imgsz, imgsz), (114,114,114))
    square.paste(image, ((imgsz-image.width)//2, (imgsz-image.height)//2))
    array = np.array(square, dtype=np.float32).transpose(2,0,1).copy()/255.
    return torch.from_numpy(array)


class FrozenCoarseFeatures(nn.Module):
    """Pool the successful detector's learned neck outputs, before Detect.

    The original six-class training froze its backbone. Pooling only backbone
    features would therefore omit the food-specific learned neck representation.
    """
    def __init__(self, checkpoint, device='cpu'):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import Detect
        path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = file_digest(path)
        model = YOLO(str(path), task='detect').model.float()
        if type(model) is not DetectionModel or type(model.model[-1]) is not Detect:
            raise TypeError('Expected the native, unfused six-class YOLO detector')
        names = model.names
        names = [names.get(i, names.get(str(i))) for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names != COARSE_NAMES:
            raise ValueError(f'Coarse classes differ: {names!r}')
        self.native = model.to(device).eval().requires_grad_(False)
        self.feature_indices = list(model.model[-1].f)
        self.save_indices = set(model.save) | set(self.feature_indices)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        if hasattr(self, 'native'):
            self.native.eval().requires_grad_(False)
            for parameter in self.native.parameters(): parameter.grad = None
        return self

    @torch.inference_mode()
    def forward(self, images):
        self.native.eval()
        saved=[]; current=images
        for module in self.native.model[:-1]:
            if module.f != -1:
                current = saved[module.f] if isinstance(module.f,int) else [current if i == -1 else saved[i] for i in module.f]
            current=module(current)
            saved.append(current if module.i in self.save_indices else None)
        features=[saved[i] if i != -1 else current for i in self.feature_indices]
        if any(value is None or value.ndim != 4 for value in features):
            raise ValueError('Expected spatial neck features for all Detect inputs')
        return torch.cat([F.adaptive_avg_pool2d(value, 1).flatten(1) for value in features],dim=1)


def parent_mask(classes, device='cpu'):
    if [row['id'] for row in classes] != list(range(len(classes))):
        raise ValueError('Fine class IDs must be contiguous')
    mask=torch.zeros((6,len(classes)),dtype=torch.bool,device=device)
    for row in classes: mask[int(row['parent_id']), int(row['id'])]=True
    return mask


def masked_logits(head, features, parents, mask):
    logits=head(features)
    allowed=mask[parents]
    if not bool(allowed.any(dim=1).all()):
        raise ValueError('Cannot compute classifier loss for a parent with no child outputs')
    return logits.masked_fill(~allowed, -1e9)


class FineRefiner:
    def __init__(self, directory, coarse_weights, device='cpu'):
        from safetensors.torch import load_file
        directory=Path(directory)
        self.config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        if file_digest(coarse_weights) != self.config['coarse_checkpoint_sha256']:
            raise ValueError('Coarse checkpoint differs from the trained frozen feature extractor')
        self.classes=self.config['classes']; self.device=torch.device(device)
        self.extractor=FrozenCoarseFeatures(coarse_weights, device)
        self.head=nn.Linear(self.config['feature_dimension'],len(self.classes)).to(device)
        self.head.load_state_dict(load_file(str(directory/'head.safetensors'),device=str(device)),strict=True)
        self.head.eval().requires_grad_(False)
        with np.load(directory/'scaler.npz',allow_pickle=False) as values:
            self.mean=torch.from_numpy(values['mean'].copy()).to(device)
            self.std=torch.from_numpy(values['std'].copy()).to(device)
        self.mask=parent_mask(self.classes,device)
        threshold_path=directory/'thresholds.json'
        self.thresholds=json.loads(threshold_path.read_text(encoding='utf-8')) if threshold_path.is_file() else {'parents':{}}
        if self.thresholds.get('scope') == 'detector_val_crops':
            expected={'dataset_sha256':self.config['dataset_sha256'],
                      'coarse_checkpoint_sha256':self.config['coarse_checkpoint_sha256'],
                      'classifier_config_sha256':file_digest(directory/'config.json'),
                      'classifier_head_sha256':file_digest(directory/'head.safetensors')}
            if any(self.thresholds.get(key) != value for key,value in expected.items()):
                raise ValueError('Deployment thresholds were calibrated for a different dataset or model')

    @torch.inference_mode()
    def predict_crops(self, images, parent_ids):
        if len(images) != len(parent_ids): raise ValueError('Crop and parent counts differ')
        if not images: return []
        output=[]
        for image,parent_id in zip(images,parent_ids):
            parent_id=int(parent_id)
            if not 0 <= parent_id < 6: raise ValueError('Parent is outside the six-class vocabulary')
            allowed=torch.nonzero(self.mask[parent_id],as_tuple=False).flatten()
            result={'parent_id':parent_id,'parent_name':COARSE_NAMES[parent_id], 'candidate_class_id':None,
                    'candidate_class_name':None,'fine_class_id':None,'fine_class_name':None,'confidence':None,
                    'accepted':False,'reason':'no_supported_children','probabilities':{}}
            if len(allowed):
                inputs=crop_preprocess(image,self.config['imgsz']).unsqueeze(0).to(self.device)
                features=(self.extractor(inputs)-self.mean)/self.std
                logits=self.head(features)[0,allowed]
                probabilities=logits.softmax(0)
                score,position=probabilities.max(0); candidate=int(allowed[position])
                result.update(candidate_class_id=candidate,candidate_class_name=self.classes[candidate]['name'],confidence=float(score),
                              probabilities={self.classes[int(index)]['name']:float(probability) for index,probability in zip(allowed,probabilities)})
                threshold=self.thresholds.get('parents',{}).get(str(parent_id),{})
                if len(allowed)<2:
                    result['reason']='single_child_parent_not_discriminative'
                elif not threshold.get('enabled',False):
                    result['reason']=threshold.get('reason','no_detector_validation_calibration')
                elif not 0 <= float(threshold.get('threshold',2.)) <= 1:
                    raise ValueError('Enabled threshold must be finite and inside [0,1]')
                elif float(score)>=float(threshold['threshold']):
                    result.update(accepted=True,reason='fine_threshold_passed',fine_class_id=candidate,fine_class_name=self.classes[candidate]['name'])
                else:
                    result['reason']='fine_confidence_below_threshold'
            output.append(result)
        return output


def load_refiner(run_dir, coarse_weights, device='cpu'):
    return FineRefiner(run_dir,coarse_weights,device)
