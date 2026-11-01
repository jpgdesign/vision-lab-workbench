"""Exercise actual OpenCV encoders: odd CV outputs must not lose edge pixels."""
import shutil

import cv2
import numpy as np
import pytest

from backend import inference, workflow_runtime


@pytest.mark.parametrize("width,height", [(83, 55), (84, 55), (83, 56), (84, 56)])
def test_odd_crop_video_retains_last_content_row_and_column(tmp_path, monkeypatch, width, height):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 80))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.full((80, 100, 3), 255, dtype=np.uint8))
    finally:
        writer.release()
    # MJPG is the lossy stage that used to truncate odd dimensions. Keep its
    # real bytes for decoding; this test does not depend on a terminal ffmpeg.
    monkeypatch.setattr(inference, "h264_video", lambda raw, target: shutil.copyfile(raw, target))
    monkeypatch.setattr(workflow_runtime, "render_output", lambda frame, *_: frame["image"])
    graph = {"version": 1, "nodes": [{"id": "source", "type": "source"},
        {"id": "crop", "type": "cv_crop", "params": {"roi": [0, 0, width / 100, height / 80]}},
        {"id": "output", "type": "output"}], "edges": [
        {"source": "source", "sourceHandle": "frame", "target": "crop", "targetHandle": "frame"},
        {"source": "crop", "sourceHandle": "frame", "target": "output", "targetHandle": "frame"}]}
    result = workflow_runtime.run_workflow({"id": "odd-crop", "workflow_graph": graph,
        "media_path": str(source), "params": {"frame_stride": 1, "max_frames": 2}}, tmp_path / "output", lambda _: None)
    encoded_width, encoded_height = width + width % 2, height + height % 2
    assert result["output_dimensions"] == {"content": [width, height], "encoded": [encoded_width, encoded_height],
        "padding_right": width % 2, "padding_bottom": height % 2}
    assert result["frames"] == 2
    capture = cv2.VideoCapture(result["media_path"])
    try:
        for _ in range(2):
            success, image = capture.read()
            assert success
            assert image.shape[:2] == (encoded_height, encoded_width)
            # The last original column/row, including their intersection, are
            # still white; a crop-to-even implementation cannot pass this.
            assert image[:height, width - 1].mean() > 230
            assert image[height - 1, :width].mean() > 230
            assert image[height - 1, width - 1].mean() > 200
            if width % 2:
                assert image[:height, width].mean() < 35
            if height % 2:
                assert image[height, :width].mean() < 35
        assert capture.read()[0] is False
    finally:
        capture.release()
