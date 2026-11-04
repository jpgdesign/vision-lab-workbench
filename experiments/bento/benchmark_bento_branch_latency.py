"""Benchmark serial full YOLO26 models against the shared-backbone research bundle.

No downloads or training. Model loading, image disk I/O, parameter checks and
warmup are excluded. Timed end-to-end latency includes identical square 640
letterboxing, RGB conversion, device transfer, both branches and independent box
decoding back to CPU. Forward-only latency is reported separately. This is a
batch-one, serial benchmark, not camera/HTTP performance or an accuracy test.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from bento_research.common import ROOT, EXPERIMENT, digest, now, save


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path, default=ROOT / "outputs/model_weights/yolo26x/yolo26x.pt")
    parser.add_argument("--new-checkpoint", type=Path, default=EXPERIMENT / "yolo26x_frozen_branch/train/weights/best.pt")
    parser.add_argument("--bundle", type=Path, default=EXPERIMENT / "yolo26x_frozen_branch/preservation/bundle", help="Directory containing branches.json and branches.safetensors")
    parser.add_argument("--images", type=Path, default=ROOT / "outputs/vision_training_starter/images", help="Image directory, or one image; default is the ten COCO canaries")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "yolo26x_frozen_branch/branch_latency.json", help="New JSON report path; existing reports are not overwritten")
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed image pairs per phase")
    parser.add_argument("--repeats", type=int, default=3, help="Measured complete passes over the image list")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads, identical in both phases")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("warmup must be nonnegative; repeats and threads must be positive")
    return args


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(torch, device):
    gc.collect()
    synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(torch, device)


def state_fingerprint(entries):
    """Verify weight/buffer identity across serialization layouts, outside timing."""
    hasher = hashlib.sha256()
    for name, value in sorted(entries):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode()
        data = tensor.numpy().tobytes()
        hasher.update(len(header).to_bytes(8, "big")); hasher.update(header)
        hasher.update(len(data).to_bytes(8, "big")); hasher.update(data)
    return hasher.hexdigest()


def native_fingerprints(model, boundary):
    backbone, branch = [], []
    for key, value in model.state_dict().items():
        prefix, index, suffix = key.split(".", 2)
        if prefix != "model":
            raise ValueError(f"Unexpected native state key: {key}")
        index = int(index)
        if index < boundary:
            backbone.append((f"{index}.{suffix}", value))
        else:
            branch.append((f"{index-boundary}.{suffix}", value))
    return state_fingerprint(backbone), state_fingerprint(branch)


def class_names(model):
    return [model.names[i] if i in model.names else model.names[str(i)] for i in range(len(model.names))]


def head_options(head):
    return {key: getattr(head, key) for key in ("max_det", "dynamic", "xyxy", "export", "format")}


def load_native(args, device):
    from ultralytics import YOLO
    old = YOLO(str(args.old_checkpoint)).model.float().eval()
    new = YOLO(str(args.new_checkpoint)).model.float().eval()
    boundary = len(old.yaml["backbone"])
    if boundary != 11 or len(new.yaml["backbone"]) != boundary:
        raise ValueError("Expected the same 11-module YOLO26 backbone")
    for model in (old, new):
        if model.yaml.get("end2end") is not True or model.yaml.get("reg_max") != 1:
            raise ValueError("Expected native YOLO26 detection models")
        if model.model[-1].cv2 is None or model.model[-1].export:
            raise ValueError("Use unfused, non-export native models in both phases")
    old_backbone, old_branch = native_fingerprints(old, boundary)
    new_backbone, new_branch = native_fingerprints(new, boundary)
    if old_backbone != new_backbone:
        raise ValueError("Native new checkpoint has changed backbone tensors")
    metadata = {"names": {"old": class_names(old), "new": class_names(new)},
                "state_sha256": {"shared_backbone": old_backbone, "old_branch": old_branch, "new_branch": new_branch},
                "head_options": {"old": head_options(old.model[-1]), "new": head_options(new.model[-1])},
                "parameter_count": sum(p.numel() for p in old.parameters()) + sum(p.numel() for p in new.parameters())}
    old.to(device); new.to(device)
    def run(tensor):
        # Deliberately serial; both complete models recompute their backbone.
        return {"old": old(tensor), "new": new(tensor)}
    return run, metadata


def load_shared(args, device):
    from bento_research.frozen_yolo_branches import FrozenYOLO26Branches
    model = FrozenYOLO26Branches.load_bundle(args.bundle, device="cpu").float().eval()
    metadata = {"names": {"old": model.old_names, "new": model.new_names},
                "state_sha256": {name: state_fingerprint(getattr(model, name).state_dict().items())
                                 for name in ("shared_backbone", "old_branch", "new_branch")},
                "head_options": {"old": head_options(model.old_branch[-1]), "new": head_options(model.new_branch[-1])},
                "parameter_count": sum(p.numel() for p in model.parameters())}
    model.to(device)
    return model, metadata


def one_sample(torch, np, letterbox, decode, runner, original, device):
    start = time.perf_counter()
    image = letterbox(image=original)
    array = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.
    synchronize(torch, device)
    after_preprocess = time.perf_counter()
    outputs = runner(tensor)
    synchronize(torch, device)
    after_forward = time.perf_counter()
    old_boxes, new_boxes = decode(outputs["old"], original), decode(outputs["new"], original)
    synchronize(torch, device)
    end = time.perf_counter()
    # Only scalar metadata survive the call; no image/prediction tensors are
    # retained across samples or across the isolated model-loading phases.
    return {"preprocess_ms": 1000 * (after_preprocess-start),
            "forward_ms": 1000 * (after_forward-after_preprocess),
            "decode_ms": 1000 * (end-after_forward),
            "end_to_end_ms": 1000 * (end-start),
            "old_detections": len(old_boxes), "new_detections": len(new_boxes)}


def distribution(np, values):
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": float(min(values)), "max_ms": float(max(values))}


def run_phase(name, loader, args, samples, expected=None):
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    # Use exactly the same decode helper as the preservation canary.
    from verify_bento_frozen_bundle import decode
    device = torch.device(args.device)
    cleanup(torch, device)
    allocated_before_load = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    runner = None
    try:
        started = time.perf_counter()
        runner, metadata = loader(args, device)
        synchronize(torch, device)
        loading_seconds = time.perf_counter() - started
        if expected is not None:
            for key in ("names", "state_sha256", "head_options"):
                if metadata[key] != expected[key]:
                    raise ValueError(f"Benchmark would compare different models: {key} does not match")
        allocated_after_load = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True)
        with torch.inference_mode():
            for index in range(args.warmup):
                one_sample(torch, np, letterbox, decode, runner, samples[index % len(samples)][1], device)
            synchronize(torch, device)
            gc.collect()
            # Keep the warmed allocator cache. Reset peaks only after warmup;
            # emptying it here would turn the first measured sample cold again.
            if device.type == "cuda":
                steady_allocated = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                steady_allocated = None
            measurements = []
            for repeat in range(args.repeats):
                for path, original in samples:
                    measurement = one_sample(torch, np, letterbox, decode, runner, original, device)
                    measurements.append({"repeat": repeat + 1, "image": path.name, **measurement})
                print(json.dumps({"phase": name, "completed_repeat": repeat + 1, "samples": len(measurements)}), flush=True)
            synchronize(torch, device)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        timings = {key: distribution(np, [row[key] for row in measurements])
                   for key in ("preprocess_ms", "forward_ms", "decode_ms", "end_to_end_ms")}
        count = len(measurements)
        return {"phase": name, "sample_count": count, "warmup_pairs_excluded": args.warmup,
                "model_loading_and_checks_seconds_excluded": loading_seconds, "model": metadata,
                "timing": timings,
                "throughput": {"end_to_end_input_images_per_second": 1000 * count / sum(row["end_to_end_ms"] for row in measurements),
                               "forward_only_input_images_per_second": 1000 * count / sum(row["forward_ms"] for row in measurements),
                               "unit": "one input image processed by both branches; serial batch size 1"},
                "cuda_memory": {"allocated_before_phase_load_bytes": allocated_before_load,
                                "allocated_after_model_load_bytes": allocated_after_load,
                                "allocated_after_warmup_bytes": steady_allocated,
                                "peak_allocated_during_measurement_bytes": peak,
                                "peak_increment_over_warm_resident_bytes": None if peak is None else peak-steady_allocated,
                                "scope": "PyTorch allocated tensors in this process; excludes CUDA driver/context and other processes" if peak is not None else "Not available for CPU; no CUDA memory measurement"},
                "samples": measurements}
    finally:
        # runner is the only owner of the loaded models. Drop it before the next
        # phase so native references cannot inflate shared-model VRAM results.
        runner = None
        cleanup(torch, device)


def main():
    args = arguments()
    # Heavy dependencies and CUDA initialization occur after --help exits.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import cv2
    import torch
    import ultralytics
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Select cpu or an available cuda:N device")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.output.exists():
        raise FileExistsError("Use a new --output JSON path to preserve previous benchmark results")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) if args.images.is_dir() else [args.images]
    if not paths:
        raise ValueError("No benchmark images were found")
    samples, image_records = [], []
    for path in paths:
        original = cv2.imread(str(path))
        if original is None:
            raise ValueError(f"Cannot decode benchmark image: {path}")
        samples.append((path, original))
        image_records.append({"path": str(path.resolve()), "sha256": digest(path), "height": original.shape[0], "width": original.shape[1]})
    source_paths = {"old_checkpoint": args.old_checkpoint, "new_checkpoint": args.new_checkpoint,
                    "bundle_config": args.bundle / "branches.json", "bundle_weights": args.bundle / "branches.safetensors"}
    sources = {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in source_paths.items()}
    report = {"created_at": now(), "status": "running", "sources": sources, "images": image_records,
              "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda_runtime": torch.version.cuda},
              "protocol": {"precision": "FP32; TF32 disabled; no Conv/BN fusion", "batch_size": 1,
                           "deterministic_algorithms": True,
                           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                           "preprocessing": "640x640 square LetterBox, auto=False, scaleup=True, BGR to RGB, NCHW /255",
                           "end_to_end_includes": ["letterbox and tensor preparation", "input transfer", "both branch forwards", "independent confidence filtering and scale_boxes", "output transfer to CPU", "synchronization"],
                           "excludes": ["model loading", "state/checksum checks", "disk image reading and image decoding", "warmup", "HTTP/camera/UI", "file writing"],
                           "postprocessing": "same preservation-canary decode; confidence 0.001, max_det100 per branch, no extra NMS or shared top-k",
                           "warmup_image_pairs_per_phase": args.warmup, "repeats": args.repeats, "torch_cpu_threads": args.threads,
                           "phase_order": ["native_two_full_models_serial", "shared_frozen_backbone"],
                           "cuda_synchronized_at_each_timing_boundary": device.type == "cuda",
                           "memory_peaks_reset_after_warmup": device.type == "cuda",
                           "percentile_method": "numpy.percentile default linear interpolation"},
              "limitations": ["Performance on these inputs/device/settings only; not a camera FPS guarantee",
                              "Fixed phase order can be affected by machine load, thermals or other GPU processes",
                              "Source weights, class order and head settings are checked; this benchmark does not repeat accuracy/raw-output verification"],
              "phases": []}
    partial = args.output.with_name(args.output.stem + ".partial.json")
    try:
        native = run_phase("native_two_full_models_serial", load_native, args, samples)
        report["phases"].append(native); save(partial, report)
        shared = run_phase("shared_frozen_backbone", load_shared, args, samples, expected=native["model"])
        report["phases"].append(shared)
        if any(digest(path) != sources[name]["sha256"] for name, path in source_paths.items()):
            raise RuntimeError("A source checkpoint or bundle changed during the benchmark")
        report["source_files_unchanged"] = True
        report["model_identity_checks"] = {key: native["model"][key] == shared["model"][key] for key in ("names", "state_sha256", "head_options")}
        report["comparison"] = {"mean_end_to_end_speedup": native["timing"]["end_to_end_ms"]["mean_ms"] / shared["timing"]["end_to_end_ms"]["mean_ms"],
                                "mean_forward_speedup": native["timing"]["forward_ms"]["mean_ms"] / shared["timing"]["forward_ms"]["mean_ms"],
                                "parameters_saved": native["model"]["parameter_count"] - shared["model"]["parameter_count"],
                                "cuda_peak_allocated_reduction_bytes": None if device.type != "cuda" else native["cuda_memory"]["peak_allocated_during_measurement_bytes"] - shared["cuda_memory"]["peak_allocated_during_measurement_bytes"],
                                "per_sample_detection_counts_equal": [(x["old_detections"], x["new_detections"]) for x in native["samples"]] == [(x["old_detections"], x["new_detections"]) for x in shared["samples"]]}
        report.update(status="completed", completed_at=now())
        save(partial, report)
        save(args.output, report)
        print(json.dumps({"status": "completed", "report": str(args.output.resolve()), "comparison": report["comparison"]}), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(partial, report)
        raise


if __name__ == "__main__":
    main()
