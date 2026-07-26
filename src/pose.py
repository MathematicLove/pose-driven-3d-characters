"""Human pose tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smoothing import LandmarkFilter


def pick_device() -> str:
    """Best available inference device."""
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except (AttributeError, RuntimeError):
        return "cpu"
    return "cpu"


NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "l eye", "r eye", "l ear", "r ear",
    "l shoulder", "r shoulder", "l elbow", "r elbow", "l wrist", "r wrist",
    "l hip", "r hip", "l knee", "r knee", "l ankle", "r ankle",
)

SKELETON: tuple[tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER),
)

FACE = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)


@dataclass
class Pose:
    """One detected person."""

    keypoints: np.ndarray
    scores: np.ndarray
    box: np.ndarray
    confidence: float = 0.0
    track_id: int = -1
    filter: LandmarkFilter | None = field(default=None, repr=False)

    @property
    def center(self) -> np.ndarray:
        """Box center point."""
        return np.array([(self.box[0] + self.box[2]) * 0.5,
                         (self.box[1] + self.box[3]) * 0.5])

    @property
    def height(self) -> float:
        """Bounding box height."""
        return float(max(self.box[3] - self.box[1], 1.0))

    def visible(self, index: int, threshold: float = 0.35) -> bool:
        """One keypoint is reliable."""
        return bool(self.scores[index] >= threshold)

    def point(self, *indices: int, threshold: float = 0.35) -> np.ndarray | None:
        """Mean of the reliable keypoints."""
        good = [self.keypoints[i] for i in indices if self.visible(i, threshold)]
        if not good:
            return None
        return np.mean(good, axis=0)


class PoseTracker:
    """Body keypoints from a YOLO pose model."""

    def __init__(self, model_path: str | Path, imgsz: int = 640, conf: float = 0.35,
                 max_people: int = 4, smooth: bool = True) -> None:
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")
        self.device = pick_device()
        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.conf = conf
        self.max_people = max_people
        self.smooth = smooth
        self._prev: list[Pose] = []
        self._filters: dict[int, LandmarkFilter] = {}
        self._next_id = 0

    def __call__(self, frame: np.ndarray, dt: float = 1 / 30) -> list[Pose]:
        """Detect people in one frame."""
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            self._prev = []
            return []

        xy = result.keypoints.xy.cpu().numpy().astype(np.float64)
        scores = result.keypoints.conf
        scores = (np.ones(xy.shape[:2]) if scores is None
                  else scores.cpu().numpy().astype(np.float64))
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
        confs = result.boxes.conf.cpu().numpy().astype(np.float64)

        poses = [Pose(keypoints=xy[i], scores=scores[i], box=boxes[i],
                      confidence=float(confs[i]))
                 for i in range(min(len(xy), self.max_people))]
        poses.sort(key=lambda p: p.height, reverse=True)
        self._track(poses)
        if self.smooth:
            self._smooth(poses, dt)
        self._prev = poses
        return poses

    def _track(self, poses: list[Pose]) -> None:
        """Keep ids stable between frames."""
        taken: set[int] = set()
        for p in poses:
            c = p.center
            best, best_d = None, float("inf")
            for old in self._prev:
                if old.track_id in taken:
                    continue
                d = float(np.linalg.norm(c - old.center))
                if d < best_d:
                    best, best_d = old, d
            if best is not None and best_d < max(best.height * 0.6, 60.0):
                p.track_id = best.track_id
            else:
                p.track_id = self._next_id
                self._next_id += 1
            taken.add(p.track_id)
        alive = {p.track_id for p in poses}
        for tid in [t for t in self._filters if t not in alive]:
            self._filters.pop(tid)

    def _smooth(self, poses: list[Pose], dt: float) -> None:
        """Filter keypoints per track."""
        for p in poses:
            flt = self._filters.get(p.track_id)
            if flt is None:
                flt = LandmarkFilter(min_cutoff=0.7, beta=0.02)
                self._filters[p.track_id] = flt
            p.keypoints = flt(p.keypoints, dt)
            p.filter = flt
