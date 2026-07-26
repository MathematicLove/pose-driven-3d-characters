"""Text, panels and the pose skeleton overlay."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pose import FACE, KEYPOINT_NAMES, SKELETON

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

_POSE_LIMB = (255, 210, 90)
_POSE_FACE = (190, 240, 170)
_POSE_JOINT = (255, 255, 255)
_POSE_WEAK = (110, 90, 90)


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    """Find available font."""
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load font at size."""
    path = _font_path()
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=256)
def _render_text(text: str, size: int,
                 color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize and cache text."""
    font = _font(size)
    dummy = ImageDraw.Draw(Image.new("L", (1, 1)))
    box = dummy.textbbox((0, 0), text, font=font)
    w = max(1, box[2] - box[0] + 4)
    h = max(1, box[3] - box[1] + 4)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, font=font, fill=255)
    alpha = np.array(img, dtype=np.float32) / 255.0
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :] = color
    return rgb, alpha


def put_text(frame: np.ndarray, text: str, org: tuple[int, int], size: int = 18,
             color: tuple[int, int, int] = (240, 240, 240)) -> int:
    """Draw text string."""
    if not text:
        return size + 6
    rgb, alpha = _render_text(text, size, color)
    h, w = alpha.shape
    x, y = org
    fh, fw = frame.shape[:2]
    if x >= fw or y >= fh:
        return h
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return h
    sub_a = alpha[y0 - y:y1 - y, x0 - x:x1 - x][:, :, None]
    sub_c = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - sub_a) + sub_c * sub_a).astype(np.uint8)
    return h


def panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Dark translucent backing."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, roi)


def draw_pose(frame: np.ndarray, pose, threshold: float = 0.35,
              labels: bool = False) -> None:
    """Draw one body skeleton with its keypoints."""
    kp = pose.keypoints
    ok = pose.scores >= threshold
    # Scale the strokes with the frame so the skeleton stays readable when the
    # camera view is shrunk into a corner inset.
    k = max(frame.shape[1] / 1280.0, 0.35)
    thin = max(int(round(2 * k)), 1)
    halo = thin + max(int(round(3 * k)), 2)

    for a, b in SKELETON:
        pa, pb = tuple(kp[a].astype(int)), tuple(kp[b].astype(int))
        if ok[a] and ok[b]:
            color = _POSE_FACE if (a in FACE and b in FACE) else _POSE_LIMB
            cv2.line(frame, pa, pb, (25, 25, 30), halo, cv2.LINE_AA)
            cv2.line(frame, pa, pb, color, thin, cv2.LINE_AA)
        elif ok[a] or ok[b]:
            cv2.line(frame, pa, pb, _POSE_WEAK, max(thin - 1, 1), cv2.LINE_AA)

    for i in range(len(kp)):
        p = tuple(kp[i].astype(int))
        if not ok[i]:
            cv2.circle(frame, p, max(int(round(2 * k)), 1), _POSE_WEAK, -1, cv2.LINE_AA)
            continue
        radius = max(int(round((3 if i in FACE else 5) * k)), 2)
        cv2.circle(frame, p, radius + max(int(round(2 * k)), 1), (25, 25, 30), -1, cv2.LINE_AA)
        cv2.circle(frame, p, radius, _POSE_JOINT, -1, cv2.LINE_AA)
        if labels and i not in FACE:
            put_text(frame, KEYPOINT_NAMES[i], (p[0] + 8, p[1] - 8), 13, (200, 230, 255))


def draw_help(frame: np.ndarray, lines: tuple[str, ...]) -> None:
    """Draw help overlay."""
    h, w = frame.shape[:2]
    bw, bh = min(w - 60, 720), 20 + len(lines) * 26
    x, y = (w - bw) // 2, (h - bh) // 2
    panel(frame, x, y, bw, bh, 0.78)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (90, 90, 90), 1, cv2.LINE_AA)
    cy = y + 12
    for line in lines:
        bold = line and not line.startswith(" ")
        put_text(frame, line, (x + 22, cy), 17 if bold else 16,
                 (255, 255, 255) if bold else (215, 215, 215))
        cy += 26
