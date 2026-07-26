from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


class CameraStream:
    """Threaded camera frame grabber."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 60) -> None:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. "
                "On macOS grant camera access in System Settings, Privacy and "
                "Security, Camera, then restart."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError("Camera opened but returns no frames.")

        self._frame: np.ndarray = frame
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.failed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def shape(self) -> tuple[int, int]:
        """Frame height and width."""
        return self._frame.shape[0], self._frame.shape[1]

    @property
    def source_fps(self) -> float:
        """Camera reported FPS."""
        v = self.cap.get(cv2.CAP_PROP_FPS)
        return float(v) if v and v > 0 else 0.0

    def _loop(self) -> None:
        """Background capture loop."""
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.failed = True
                break
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self) -> tuple[np.ndarray, int]:
        """Latest frame and sequence."""
        with self._lock:
            return self._frame, self._seq

    def wait_next(self, last_seq: int, timeout: float = 1.0) -> tuple[np.ndarray, int]:
        """Wait for fresh frame."""
        deadline = time.perf_counter() + timeout
        while True:
            frame, seq = self.read()
            if seq != last_seq or self.failed or time.perf_counter() > deadline:
                return frame, seq
            time.sleep(0.001)

    def release(self) -> None:
        """Stop thread and release."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
