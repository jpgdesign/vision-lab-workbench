"""One isolated GPU task. The API owns cancellation and the serial queue."""
import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings").mkdir(exist_ok=True)
    (output / "mpl").mkdir(exist_ok=True)
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_AUTOINSTALL": "false",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true",
                       "YOLO_CONFIG_DIR": str(output / "settings"), "MPLCONFIGDIR": str(output / "mpl")}.items():
        os.environ[key] = value

    def emit(event):
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        emit({"event": "progress", "progress": 0.01, "message": "載入本機模型與參數"})
        if request["kind"] == "inference":
            from .inference import run_inference
            result = run_inference(request, output, emit)
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
    tmp = output / "result.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(output / "result.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
