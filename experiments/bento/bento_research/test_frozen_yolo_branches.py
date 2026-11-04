"""CPU-only canaries using random native YOLO26n; no downloaded weights."""
from copy import deepcopy
import json

import pytest
import torch

from frozen_yolo_branches import FrozenYOLO26Branches, make_new_branch_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous_deterministic)
    torch.set_num_threads(previous_threads)


def models():
    from ultralytics.nn.tasks import DetectionModel
    old = DetectionModel("yolo26n.yaml", nc=80, verbose=False).cpu().eval()
    old.names = {index: f"old_{index}" for index in range(80)}
    new, audit = make_new_branch_model(old, ["staple", "vegetable", "egg", "soy", "meat", "seafood"])
    return old, new, audit


def equal_tree(expected, actual):
    assert type(expected) is type(actual)
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual), f"Tensor changed: {expected.shape}"
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            equal_tree(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            equal_tree(left, right)
    else:
        assert expected == actual


def frozen_state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            if name.startswith(("shared_backbone.", "old_branch."))}


def test_native_raw_equivalence_and_new_training_preserves_old_path():
    from ultralytics.utils import DEFAULT_CFG_DICT
    from types import SimpleNamespace
    old, new, audit = models()
    reference = deepcopy(old).eval()
    new.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    criterion = new.init_criterion()
    bundle = FrozenYOLO26Branches(old, new)
    assert audit["copied_tensor_count"] > 0 and len(audit["new_class_projection_prefixes"]) == 6
    assert bundle.boundary == 11
    assert all(not p.requires_grad for p in bundle.shared_backbone.parameters())
    assert all(not p.requires_grad for p in bundle.old_branch.parameters())
    assert all(p.requires_grad for p in bundle.new_branch.parameters())
    assert not {id(p) for p in bundle.old_branch.parameters()} & {id(p) for p in bundle.new_branch.parameters()}
    # One serialized backbone; no hidden full source model retained as a child.
    assert set(dict(bundle.named_children())) == {"shared_backbone", "old_branch", "new_branch"}
    for size in ((64, 64), (96, 64)):
        images = torch.rand(2, 3, *size)
        with torch.no_grad():
            expected = reference(images)
            actual = bundle(images)["old"]
        equal_tree(expected, actual)
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = reference(images)
    before = frozen_state(bundle)
    new_before = {name: value.clone() for name, value in bundle.new_branch.state_dict().items()}
    bundle.train()
    assert all(not module.training for module in bundle.shared_backbone.modules())
    assert all(not module.training for module in bundle.old_branch.modules())
    batch = {"img": images, "batch_idx": torch.tensor([0, 1]), "cls": torch.tensor([[0.], [5.]]),
             "bboxes": torch.tensor([[.5, .5, .5, .5], [.4, .4, .3, .3]])}
    optimizer = torch.optim.AdamW(bundle.new_parameters(), lr=1e-4)
    loss, _ = criterion(bundle.forward_new(images), batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(p.grad is None for p in bundle.shared_backbone.parameters())
    assert all(p.grad is None for p in bundle.old_branch.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bundle.new_parameters())
    optimizer.step()
    assert any(not torch.equal(value, bundle.new_branch.state_dict()[name]) for name, value in new_before.items())
    equal_tree(before, frozen_state(bundle))
    # Even train() on the outer module must keep the old native eval output.
    equal_tree(expected, bundle.forward_old(images))


def test_in_place_new_feature_mutation_cannot_change_old_output():
    old, new, _ = models()
    bundle = FrozenYOLO26Branches(old, new).eval()
    images = torch.rand(1, 3, 64, 64)
    before = bundle.forward_old(images)
    shared = {}
    def capture_shared_output(_module, _args, output):
        shared["output"] = output
        shared["before"] = output.clone()
    def mutate_new_input(_module, args):
        args[0].add_(100)
    backbone_handle = bundle.shared_backbone[-1].register_forward_hook(capture_shared_output)
    handle = bundle.new_branch[0].register_forward_pre_hook(mutate_new_input)
    try:
        equal_tree(before, bundle(images)["old"])
        assert torch.equal(shared["before"], shared["output"])
        equal_tree(before, bundle.forward_old(images))
    finally:
        handle.remove()
        backbone_handle.remove()


def test_reject_changed_backbone_and_shared_trainable_parameters():
    old, new, _ = models()
    with torch.no_grad():
        next(new.parameters()).add_(.001)
    with pytest.raises(ValueError, match="backbone differs"):
        FrozenYOLO26Branches(old, new)
    with pytest.raises(ValueError, match="share parameter"):
        FrozenYOLO26Branches(old, old)


def test_same_class_count_different_names_resets_only_output_semantics():
    old, _, _ = models()
    before = {name: tensor.clone() for name, tensor in old.state_dict().items()}
    new, audit = make_new_branch_model(old, [f"different_{index}" for index in range(80)])
    equal_tree(before, dict(old.state_dict()))
    assert len(audit["new_class_projection_prefixes"]) == 6
    for prefix in audit["new_class_projection_prefixes"]:
        assert not torch.equal(old.state_dict()[prefix + "weight"], new.state_dict()[prefix + "weight"])
    assert torch.equal(next(old.parameters()), next(new.parameters()))


def test_safetensors_roundtrip_retains_independent_native_outputs(tmp_path):
    old, new, _ = models()
    old.model[-1].max_det = 13
    new.model[-1].max_det = 7
    bundle = FrozenYOLO26Branches(old, new).eval()
    images = torch.rand(1, 3, 64, 64)
    expected = bundle(images)
    path = bundle.save_bundle(tmp_path / "bundle")
    config = json.loads(path.read_text())
    assert config["old_names"] == bundle.old_names and config["new_names"] == bundle.new_names
    restored = FrozenYOLO26Branches.load_bundle(path.parent)
    equal_tree(expected, restored(images))
    assert expected["old"][0].shape[1] == 13
    assert expected["new"][0].shape[1] == 7
    assert next(restored.parameters()).device.type == "cpu"
    with pytest.raises(FileExistsError):
        bundle.save_bundle(path.parent)
    payload = path.parent / "branches.safetensors"
    content = bytearray(payload.read_bytes())
    content[-1] ^= 1
    payload.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        FrozenYOLO26Branches.load_bundle(path.parent)
