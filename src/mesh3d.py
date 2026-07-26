"""Triangle rasterizer for skinned character meshes.

Every per-triangle quantity — projection, culling, lighting, colour — is
computed for the whole mesh in one batched numpy pass, leaving the inner loop
with nothing but a single OpenCV polygon fill per triangle. Shading a triangle
pixel by pixel in Python instead costs about thirty times as much, which is the
difference between a few frames per second and a usable one.

Depth is resolved by drawing back to front rather than with a z-buffer: for a
character mesh the sort handles the cases that matter, such as an arm crossing
the chest, and it needs no per-pixel work.
"""

from __future__ import annotations

import cv2
import numpy as np

_LIGHT = np.array([0.4, -0.6, -0.9])
_LIGHT /= np.linalg.norm(_LIGHT)
_FILL = np.array([-0.5, 0.2, -0.8])
_FILL /= np.linalg.norm(_FILL)
_AMBIENT = 0.30
_FILL_GAIN = 0.22
_RIM_GAIN = 0.40
_SPEC_GAIN = 0.28
_SPEC_POWER = 24.0
_HALF = _LIGHT + np.array([0.0, 0.0, 1.0])
_HALF /= np.linalg.norm(_HALF)


def _rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Build a yaw/pitch/roll rotation matrix."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Part:
    """One skinned, drawable mesh piece in world space."""

    def __init__(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        triangles: np.ndarray,
        *,
        color: tuple[int, int, int] = (185, 180, 175),
        vertex_colors: np.ndarray | None = None,
        uv: np.ndarray | None = None,
        texture: np.ndarray | None = None,
    ) -> None:
        self.positions = positions
        self.normals = normals
        self.triangles = triangles
        self.color = color
        self.vertex_colors = vertex_colors
        self.uv = uv
        self.texture = texture


class MeshRenderer:
    """Painter's-algorithm rasterizer with batched shading."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def render(
        self,
        background: np.ndarray,
        parts: list[Part],
        *,
        origin: tuple[float, float],
        focal: float,
        distance: float,
        yaw: float,
        pitch: float,
        roll: float = 0.0,
    ) -> np.ndarray:
        """Draw skinned parts over the background."""
        out = background if background.flags.writeable else background.copy()
        h, w = out.shape[:2]
        rot = _rotation(yaw, pitch, roll)
        cx, cy = origin

        for part in parts:
            tris = part.triangles
            if part.positions.shape[0] == 0 or tris.shape[0] == 0:
                continue

            cam = part.positions @ rot.T
            z = cam[:, 2] + distance
            safe_z = np.maximum(z, 1e-3)
            screen = np.empty((len(cam), 2))
            screen[:, 0] = focal * cam[:, 0] / safe_z + cx
            screen[:, 1] = -focal * cam[:, 1] / safe_z + cy

            pts = screen[tris]
            depth = z[tris].mean(axis=1)
            # Clockwise on screen means the face turns away from the camera.
            winding = ((pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
                       - (pts[:, 2, 0] - pts[:, 0, 0]) * (pts[:, 1, 1] - pts[:, 0, 1]))
            lo = pts.min(axis=1)
            hi = pts.max(axis=1)
            visible = (
                (winding < 0) & (depth > 0) & np.isfinite(depth)
                & (hi[:, 0] >= 0) & (lo[:, 0] < w) & (hi[:, 1] >= 0) & (lo[:, 1] < h)
                # Sub-pixel triangles cannot show anything the neighbours miss.
                & ((hi[:, 0] - lo[:, 0]) >= 0.7) & ((hi[:, 1] - lo[:, 1]) >= 0.7)
            )
            keep = np.flatnonzero(visible)
            if keep.size == 0:
                continue

            # Sort first, then shade, so colours already come out in draw order.
            order = keep[np.argsort(-depth[keep])]
            poly = np.round(screen[tris[order]]).astype(np.int32)
            palette = self._colors(part, tris[order], cam, rot)

            for i in range(len(order)):
                cv2.fillConvexPoly(out, poly[i], palette[i], cv2.LINE_8)
        return out

    @staticmethod
    def _colors(part: Part, tris: np.ndarray, cam: np.ndarray,
                rot: np.ndarray) -> list[tuple[int, int, int]]:
        """Flat colour per triangle: unlit base times a light factor."""
        if part.normals.size:
            n = (part.normals @ rot.T)[tris].mean(axis=1)
        else:
            v = cam[tris]
            n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        n_len = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(n_len > 1e-9, n_len, 1.0)
        n = np.where(n[:, 2:3] > 0, -n, n)

        shade = (_AMBIENT
                 + (1.0 - _AMBIENT) * np.maximum(-(n @ _LIGHT), 0.0)
                 + _FILL_GAIN * np.maximum(-(n @ _FILL), 0.0)
                 + _RIM_GAIN * np.maximum(1.0 - np.abs(n[:, 2]), 0.0) ** 2.2
                 + _SPEC_GAIN * np.maximum(-(n @ _HALF), 0.0) ** _SPEC_POWER)

        if part.texture is not None and part.uv is not None:
            tex = part.texture
            th, tw = tex.shape[:2]
            uv = part.uv[tris].mean(axis=1)
            u = np.clip(np.mod(uv[:, 0], 1.0) * (tw - 1), 0, tw - 1).astype(np.int32)
            v = np.clip(np.mod(uv[:, 1], 1.0) * (th - 1), 0, th - 1).astype(np.int32)
            base = tex[v, u].astype(np.float64)
        elif part.vertex_colors is not None:
            base = part.vertex_colors[tris].mean(axis=1)
        else:
            base = np.broadcast_to(np.asarray(part.color, dtype=np.float64), (len(tris), 3))

        lit = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)
        # OpenCV wants plain ints; one tolist beats per-triangle conversion.
        return [tuple(c) for c in lit.tolist()]
