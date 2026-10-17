"""Isolated training/live tasks and a serial, reusable inference process."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


def configure_environment(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings").mkdir(exist_ok=True)
    (output / "mpl").mkdir(exist_ok=True)
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_AUTOINSTALL": "false",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true",
                       "YOLO_CONFIG_DIR": str(output / "settings"), "MPLCONFIGDIR": str(output / "mpl")}.items():
        os.environ[key] = value


def configure_torch():
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def execute_job(request_path, output, *, inference_runner=None):
    """Finish every event and artifact before returning a completion payload."""
    configure_environment(output)

    def emit(event):
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        configure_torch()
        emit({"event": "progress", "progress": 0.01, "message": "載入本機模型與參數"})
        if request["kind"] == "inference":
            if inference_runner is None:
                from .inference import run_inference
                inference_runner = run_inference
            result = inference_runner(request, output, emit)
        elif request["kind"] == "training":
            from .training import run_training
            result = run_training(request, output, emit)
        elif request["kind"] == "live":
            from .live_inference import run_live
            result = run_live(request, output, emit)
        else:
            raise ValueError("不支援的工作類型")
        payload = {"status": "completed", "result": result}
        emit({"event": "progress", "progress": 1.0, "message": "完成"})
        code = 0
    except Exception as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPU 顯存不足。請降低 batch size／解析度，或使用只訓練分類頭。原始錯誤：" + message[:1000]
        payload = {"status": "failed", "error": message}
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        emit({"event": "log", "message": message})
        code = 1
    return payload, code


def publish_result(output, payload):
    tmp = output / "result.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(output / "result.json")
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01)


@contextmanager
def job_log(path):
    """Keep Python and native-library stdout/stderr in the current job only."""
    saved = []
    with path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(log.fileno(), descriptor)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            log.flush()
            for descriptor, original in saved:
                os.dup2(original, descriptor)
                os.close(original)


@contextmanager
def service_stdio():
    # Third-party libraries retain their initial logging StreamHandler stream.
    # These streams remain open across jobs, while job_log switches fd 1/2.
    # closefd=False also leaves the process's original descriptors intact.
    with open(1, "w", encoding="utf-8", buffering=1, closefd=False) as stdout:
        with open(2, "w", encoding="utf-8", buffering=1, closefd=False) as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield


def service_paths(command):
    """Accept only manager-created jobs within this service's runtime."""
    if not isinstance(command, dict):
        raise ValueError("Invalid inference command")
    for key in ("job_id", "dispatch_token"):
        if not isinstance(command.get(key), str) or not re.fullmatch(r"[a-f0-9]{32}", command[key]):
            raise ValueError("Invalid inference command identity")
    root = (Path(os.environ["VISION_WORKBENCH_RUNTIME"]).resolve() / "jobs").resolve()
    output = Path(command["output_dir"])
    request = Path(command["request_path"])
    expected = root / command["job_id"]
    if (not output.is_absolute() or output.is_symlink() or output.resolve() != expected
            or not output.is_dir() or request.is_symlink()
            or request.resolve() != expected / "request.json" or not request.is_file()):
        raise ValueError("Invalid inference command paths")
    payload = json.loads(request.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("id") != command["job_id"] or payload.get("kind") != "inference":
        raise ValueError("Inference service accepts only matching inference jobs")
    return request.resolve(), output.resolve()


def run_inference_service(stream=None, *, cache_factory=None, inference_runner=None):
    """Read one request at a time; atomically publish completion after cleanup."""
    stream = stream if stream is not None else sys.stdin
    cache = None

    def infer(request, output, emit):
        nonlocal cache, cache_factory, inference_runner
        if cache_factory is None:
            from .inference_cache import InferenceModelCache
            cache_factory = lambda: InferenceModelCache(max_entries=4)
        if cache is None:
            cache = cache_factory()
        if inference_runner is None:
            from .inference import run_inference
            inference_runner = run_inference
        return inference_runner(request, output, emit, model_cache=cache)

    with service_stdio():
        try:
            while True:
                line = stream.readline(64 * 1024 + 1)
                if not line:
                    return 0
                if len(line) > 64 * 1024 or not line.endswith("\n"):
                    raise ValueError("Invalid inference command length")
                command = json.loads(line)
                request_path, output = service_paths(command)
                with job_log(output / "worker.log"):
                    payload, _ = execute_job(request_path, output, inference_runner=infer)
                payload.update(job_id=command["job_id"], dispatch_token=command["dispatch_token"])
                # Publication is the last operation on this job. The manager may
                # immediately dispatch the next job or terminate this process.
                publish_result(output, payload)
                del payload, command
        finally:
            if cache is not None:
                cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--inference-service", action="store_true")
    args = parser.parse_args()
    if args.inference_service:
        if args.request or args.output:
            parser.error("--inference-service does not accept --request or --output")
        return run_inference_service()
    if not args.request or not args.output:
        parser.error("--request and --output are required for an isolated job")
    output = Path(args.output).resolve()
    payload, code = execute_job(args.request, output)
    publish_result(output, payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
