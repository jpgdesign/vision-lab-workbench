"""Shared live-camera bounds; deliberately free of HTTP/model dependencies."""

MAX_CAPTURE_FPS = 30
DEFAULT_CAPTURE_FPS = 30
# A complete 16-second action window at 30 FPS needs 481 timestamped frames.
# Keep a small scheduling margin while retaining a fixed disk/memory bound.
FRAME_RING_SIZE = 512
