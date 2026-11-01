"""Single and independent dual-branch detectors used by graph nodes."""
import json
from .dual_branch import MANIFEST, branch_request, is_bundle, mapped_predictions, validate_bundle


def create_workflow_detector(model_id, params, device, checkpoint_path=None, model_weight_path=None):
    from .inference import create_detector
    if not is_bundle(checkpoint_path):
        return create_detector(model_id, params, device, checkpoint_path, model_weight_path)
    root = validate_bundle(checkpoint_path, model_id)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    request = {"model_id": model_id, "params": params}
    branches = []
    for name in ("original", "new"):
        resolved = branch_request(request, root, name)
        predictor = create_detector(model_id, resolved["params"], device,
                                    resolved.get("checkpoint_path"), resolved.get("model_weight_path"))
        branches.append((name, predictor, resolved["params"]))

    def predict(image, inference_params=None):
        # Each branch keeps its own scores and vocabulary. Never apply joint
        # softmax/NMS across an unchanged original and a fine-tuned branch.
        return [row for name, predictor, selected in branches
                for row in mapped_predictions(predictor(image, selected), manifest, name)]
    return predict
