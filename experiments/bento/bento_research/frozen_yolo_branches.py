"""Research-only YOLO26: one frozen backbone and two isolated neck/head paths.

Inputs are already-preprocessed NCHW tensors, identical to the native model's
input. Each result retains its native branch output; this module intentionally
does not combine class IDs, suppress boxes across branches, or share a top-k.
The old path always runs in evaluation mode. Only ``new_branch`` is trainable.
No model download, GPU allocation, or training occurs when importing this file.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _names(model) -> list[str]:
    names = model.names
    result = [names[i] if i in names else names[str(i)] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if len(result) != model.model[-1].nc or not result or any(not isinstance(x, str) or not x for x in result):
        raise ValueError("A complete class-name table matching the native Detect head is required.")
    if len(set(result)) != len(result):
        raise ValueError("Class names must be unique within each branch.")
    return result


def _native_model(model):
    from ultralytics.nn.modules import Conv, Detect
    from ultralytics.nn.tasks import DetectionModel
    if type(model) is not DetectionModel or type(model.model[-1]) is not Detect:
        raise TypeError("Pass native, unfused Ultralytics DetectionModel instances, not YOLO wrappers.")
    cfg = model.yaml
    if cfg.get("end2end") is not True or cfg.get("reg_max") != 1:
        raise ValueError("This research adapter supports the YOLO26 end-to-end detection architecture only.")
    if not cfg.get("scale") or not cfg.get("backbone") or not cfg.get("head"):
        raise ValueError("The model must retain its explicit architecture and scale.")
    head = model.model[-1]
    if head.export or head.format is not None or head.cv2 is None or head.cv3 is None:
        raise ValueError("Exported or inference-fused heads cannot be branched or saved portably.")
    if any(isinstance(module, Conv) and not hasattr(module, "bn") for module in model.modules()):
        raise ValueError("Use unfused Conv/BN weights; fused source models cannot be rebuilt from this bundle.")
    _names(model)
    return model


def _backbone_state(model, boundary):
    return {name: tensor for name, tensor in model.state_dict().items()
            if any(name.startswith(f"model.{index}.") for index in range(boundary))}


def make_new_branch_model(source, classes: list[str]):
    """Build a native new-task model with compatible pretrained feature weights.

    The original model is never cast, loaded into, or modified. Classification
    output projections are kept at native initialization whenever class meanings
    differ, even when the new class count happens to equal the old class count.
    The caller may train this native model with its complete backbone frozen,
    or attach it here and optimize only ``new_parameters()`` with native loss.
    """
    from ultralytics.nn.tasks import DetectionModel
    _native_model(source)
    if not classes or len(set(classes)) != len(classes) or any(not isinstance(x, str) or not x for x in classes):
        raise ValueError("New class names must be a nonempty, unique string list.")
    first = next(source.parameters())
    target = DetectionModel(deepcopy(source.yaml), ch=source.yaml.get("channels", 3), nc=len(classes), verbose=False)
    target.to(device=first.device, dtype=first.dtype)
    target.names = dict(enumerate(classes))
    reset = []
    if classes != _names(source):
        head_index = len(target.model) - 1
        for branch in ("cv3", "one2one_cv3"):
            for scale, block in enumerate(getattr(target.model[-1], branch)):
                reset.append(f"model.{head_index}.{branch}.{scale}.{len(block) - 1}.")
    source_state, target_state = source.state_dict(), target.state_dict()
    copied = {name: tensor.detach() for name, tensor in source_state.items()
              if name in target_state and tensor.shape == target_state[name].shape
              and not any(name.startswith(prefix) for prefix in reset)}
    target.load_state_dict(copied, strict=False)
    audit = {"copied_tensor_count": len(copied), "total_target_tensor_count": len(target_state),
             "new_class_projection_prefixes": reset,
             "initialized_tensor_names": sorted(set(target_state) - set(copied)),
             "old_classes": _names(source), "new_classes": list(classes)}
    return target, audit


def _clone_features(value: Any, memo: dict[int, torch.Tensor]):
    """Detach and clone per branch, preserving aliases only inside that branch."""
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.detach().clone()
        return memo[id(value)]
    if isinstance(value, list):
        return [_clone_features(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_features(item, memo) for item in value)
    if isinstance(value, dict):
        return {key: _clone_features(item, memo) for key, item in value.items()}
    return value


class FrozenYOLO26Branches(nn.Module):
    """One shared immutable backbone, immutable old neck/head, trainable new path.

    Module ownership is borrowed from the two native models (no deep copy). After
    construction, release caller references to the native models to release the
    unused new-model backbone. They must start with identical backbone tensors;
    a separately fine-tuned backbone is rejected, not silently substituted.
    """
    SCHEMA_VERSION = 1

    def __init__(self, old_model, new_model):
        super().__init__()
        _native_model(old_model)
        _native_model(new_model)
        self.boundary = len(old_model.yaml["backbone"])
        for key in ("backbone", "head", "scale", "scales", "channels", "end2end", "reg_max"):
            if old_model.yaml.get(key) != new_model.yaml.get(key):
                raise ValueError(f"Branch architecture differs at {key}; shared routing would be unsafe.")
        if len(old_model.model) != len(new_model.model):
            raise ValueError("Branch layer counts differ.")
        for old, new in zip(old_model.model, new_model.model):
            if old.i != new.i or old.f != new.f or type(old) is not type(new):
                raise ValueError("Native f/i routing differs between branches.")
        old_parameter, new_parameter = next(old_model.parameters()), next(new_model.parameters())
        if old_parameter.dtype != new_parameter.dtype or old_parameter.device != new_parameter.device:
            raise ValueError("Bring both models to the same device and floating-point dtype first.")
        old_state, new_state = _backbone_state(old_model, self.boundary), _backbone_state(new_model, self.boundary)
        if old_state.keys() != new_state.keys() or any(not torch.equal(value, new_state[name]) for name, value in old_state.items()):
            raise ValueError("New backbone differs from the frozen original; shared-feature equivalence is not established.")
        old_ids = {id(parameter) for parameter in old_model.parameters()}
        if any(id(parameter) in old_ids for module in new_model.model[self.boundary:] for parameter in module.parameters()):
            raise ValueError("Old and new trainable branches must not share parameter objects.")
        self.old_config, self.new_config = deepcopy(old_model.yaml), deepcopy(new_model.yaml)
        self.old_names, self.new_names = _names(old_model), _names(new_model)
        self.old_save, self.new_save = set(old_model.save), set(new_model.save)
        self.shared_save = self.old_save | self.new_save
        self.shared_backbone = nn.ModuleList(list(old_model.model[:self.boundary]))
        self.old_branch = nn.ModuleList(list(old_model.model[self.boundary:]))
        self.new_branch = nn.ModuleList(list(new_model.model[self.boundary:]))
        self.shared_backbone.requires_grad_(False)
        self.old_branch.requires_grad_(False)
        self.new_branch.requires_grad_(True)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.shared_backbone.eval()
        self.old_branch.eval()
        # Protect against callers having changed requires_grad between epochs.
        self.shared_backbone.requires_grad_(False)
        self.old_branch.requires_grad_(False)
        # A borrowed model might still carry gradients from an earlier trainer.
        # Clear them so an optimizer built over all bundle parameters cannot
        # apply stale gradients or decoupled decay to the immutable path.
        for parameters in (self.shared_backbone.parameters(), self.old_branch.parameters()):
            for parameter in parameters:
                parameter.grad = None
        return self

    def _apply(self, fn):
        # Detect caches are plain tensor attributes in Ultralytics, not buffers.
        super()._apply(fn)
        for branch in (self.old_branch, self.new_branch):
            head = branch[-1]
            for name in ("stride", "anchors", "strides"):
                setattr(head, name, fn(getattr(head, name)))
        return self

    def new_parameters(self):
        return self.new_branch.parameters()

    @staticmethod
    def _run(modules, current, saved, save_indices):
        for module in modules:
            if module.f != -1:
                current = saved[module.f] if isinstance(module.f, int) else [current if index == -1 else saved[index] for index in module.f]
            current = module(current)
            saved.append(current if module.i in save_indices else None)
        return current

    def _features(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Supply the same preprocessed NCHW tensor used by native YOLO inference.")
        # Enforce eval even if someone called train() directly on a child.
        self.shared_backbone.eval()
        self.old_branch.eval()
        with torch.no_grad():
            saved = []
            current = self._run(self.shared_backbone, images, saved, self.shared_save)
        return current, saved

    def forward(self, images, branch: str = "both"):
        if branch not in {"old", "new", "both"}:
            raise ValueError("branch must be old, new, or both")
        current, saved = self._features(images)
        result = {}
        # Independent feature storage prevents an in-place operation in either
        # neck from changing the shared cache or the other branch's input.
        for name, modules, save_indices in (("old", self.old_branch, self.old_save), ("new", self.new_branch, self.new_save)):
            if branch not in {name, "both"}:
                continue
            branch_current, branch_saved = _clone_features((current, saved), {})
            if name == "old":
                with torch.no_grad():
                    result[name] = self._run(modules, branch_current, branch_saved, save_indices)
            else:
                result[name] = self._run(modules, branch_current, branch_saved, save_indices)
        return result if branch == "both" else result[branch]

    def forward_new(self, images):
        return self(images, branch="new")

    def forward_old(self, images):
        return self(images, branch="old")

    def save_bundle(self, directory):
        """Save one backbone plus both necks/heads as safetensors + JSON.

        This is a research bundle, not an Ultralytics .pt checkpoint or a
        workbench registry entry. Call ``load_bundle`` to reconstruct it.
        """
        from safetensors.torch import save_file
        import ultralytics
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config_path, weights_path = directory / "branches.json", directory / "branches.safetensors"
        if config_path.exists() or weights_path.exists():
            raise FileExistsError("Choose a new bundle directory; existing research artifacts are not overwritten.")
        config = {"schema_version": self.SCHEMA_VERSION, "architecture": "frozen_yolo26_shared_backbone",
                  "ultralytics_version": ultralytics.__version__, "boundary": self.boundary,
                  "old_config": self.old_config, "new_config": self.new_config,
                  "old_names": self.old_names, "new_names": self.new_names,
                  "dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
                  "postprocessing": "native independent branches; no cross-branch NMS or top-k",
                  "input": "identical preprocessed NCHW tensors; application owns preprocessing",
                  "head_options": {name: {key: getattr(branch[-1], key) for key in ("max_det", "dynamic", "xyxy")}
                                   for name, branch in (("old", self.old_branch), ("new", self.new_branch))}}
        # Validate portable architecture metadata before writing any weights.
        json.dumps(config, allow_nan=False)
        weights = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in self.state_dict().items()}
        save_file(weights, str(weights_path))
        config["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return config_path

    @classmethod
    def load_bundle(cls, directory, device="cpu"):
        from safetensors.torch import load_file
        from ultralytics.nn.tasks import DetectionModel
        import ultralytics
        directory = Path(directory)
        config = json.loads((directory / "branches.json").read_text(encoding="utf-8"))
        if config.get("schema_version") != cls.SCHEMA_VERSION or config.get("architecture") != "frozen_yolo26_shared_backbone":
            raise ValueError("Unsupported research bundle schema.")
        if config.get("ultralytics_version") != ultralytics.__version__:
            raise ValueError("Use the recorded Ultralytics version to preserve native branch behavior.")
        weights_path = directory / "branches.safetensors"
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != config.get("weights_sha256"):
            raise ValueError("Research bundle weight checksum differs.")
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(config.get("dtype"))
        if dtype is None:
            raise ValueError("Unsupported bundle floating-point dtype.")
        models = []
        for name in ("old", "new"):
            cfg = deepcopy(config[f"{name}_config"])
            model = DetectionModel(cfg, ch=cfg.get("channels", 3), verbose=False).to(dtype=dtype)
            model.names = dict(enumerate(config[f"{name}_names"]))
            for key, value in config["head_options"][name].items():
                setattr(model.model[-1], key, value)
            models.append(model)
        boundary = len(models[0].yaml["backbone"])
        if boundary != config["boundary"]:
            raise ValueError("Backbone boundary differs from the saved manifest.")
        # Temporary builders need matching backbones; the one saved backbone is
        # loaded below, once, under shared_backbone.*.
        models[1].load_state_dict(_backbone_state(models[0], boundary), strict=False)
        result = cls(*models)
        result.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        return result.to(device=device).eval()
