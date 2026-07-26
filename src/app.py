"""Drive a rigged 3D character with body pose keypoints from a camera."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

import mesh3d
import ui
from camera import CameraStream
from character import MODEL_FILES, TRIANGLE_BUDGET, Character
from mesh3d import MeshRenderer
from pose import PoseTracker
from smoothing import ScalarEMA

WINDOW = "Pose Driven 3D Characters"
STAGE_TOP = (58, 46, 38)
STAGE_BOTTOM = (24, 20, 17)
STAGE_MARGIN = 0.06
RENDER_SCALE = 1.0
# Cost is set by triangle count, not by pixels, so the quality lever cycled with
# F is how finely the mesh is kept: these scale the loader's triangle budget.
DETAIL_LEVELS = (1.0, 0.45, 0.2)
DETAIL_LABELS = ("high", "medium", "fast")
SHADOW_COLOR = (10, 8, 7)
SHADOW_ALPHA = 0.55
# How fast the torso turn follows its estimate. Low, because shoulder width is
# the noisiest thing the tracker reports and the turn reads as a whole-body move.
TURN_SMOOTHING = 0.15
VIEWS = ("full3d", "pip_cam", "pip_3d")
VIEW_LABELS = {
    "full3d": "3D CHARACTER",
    "pip_cam": "3D CHARACTER + camera inset",
    "pip_3d": "CAMERA + character inset",
}
PIP_RATIO = 0.30
MODEL_NAMES = list(MODEL_FILES.keys())

HELP_LINES_POSE = (
    "POSES",
    "  Stand so your shoulders, hips, arms and legs are visible",
    "  The character faces you and copies your pose in real time",
    "  Raise your right arm and its right arm goes up",
    "  Turn your shoulders to turn its torso",
    "  Keypoints and skeleton are drawn on the camera image",
    "",
    "KEYS",
    "  K cycle view (3D only / cam inset / 3D inset)",
    "  M next model    N previous model",
    "  F render quality (lower it if the frame rate drops)",
    "  L keypoint names",
    "  Y / P yaw / pitch character view    R reset view",
    "  D debug    H help    Q quit",
)


class _Scaled(NamedTuple):
    """Keypoints resized for a smaller canvas, for drawing only."""

    keypoints: np.ndarray
    scores: np.ndarray


class PoseApp:
    """Body-pose driven 3D character viewer."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cam = CameraStream(args.camera, args.width, args.height)
        self.h, self.w = self.cam.shape
        print(f"[i] camera: {self.w}x{self.h} @ {self.cam.source_fps:.0f} FPS", flush=True)

        self.tracker = PoseTracker(args.pose_model, imgsz=args.pose_size, conf=args.pose_conf)
        print(f"[i] pose device: {self.tracker.device}", flush=True)

        self.renderer = MeshRenderer(self.w, self.h)
        self.model_index = MODEL_NAMES.index(getattr(args, "model", MODEL_NAMES[0]))
        self.character: Character | None = None
        self.yaw, self.pitch = 0.0, 0.0
        self.detail_index = 0
        self._load_character()

        self.view_index = 0
        self.turn = 0.0
        self.turn_s = ScalarEMA(TURN_SMOOTHING, 0.0)
        self._stage_cache: np.ndarray | None = None
        self.show_help = False
        self.show_debug = args.debug
        self.show_labels = False
        self.toast_text = ""
        self.toast_until = 0.0
        self.fps = 0.0
        self._last_t = time.perf_counter()
        self._no_pose_since: float | None = None

    def _load_character(self) -> None:
        """(Re)load the active character model."""
        name = MODEL_NAMES[self.model_index]
        budget = int(TRIANGLE_BUDGET * DETAIL_LEVELS[self.detail_index])
        self.character = Character.load(name, mirror=not self.args.no_mirror,
                                        budget=budget)
        self.yaw, self.pitch = self.character.default_yaw, 0.0
        self.toast(f"Model: {name}")

    def toast(self, text: str, seconds: float = 2.0) -> None:
        """Show transient message."""
        self.toast_text = text
        self.toast_until = time.time() + seconds

    def next_model(self, step: int) -> None:
        """Switch to another character model."""
        self.model_index = (self.model_index + step) % len(MODEL_NAMES)
        self._load_character()

    def reset_view(self) -> None:
        """Back to this model's default camera angle."""
        self.yaw = self.character.default_yaw if self.character else 0.0
        self.pitch = 0.0

    def render_character(self, canvas_shape: tuple[int, int], pose) -> np.ndarray:
        """Render the character posed by one detected person, or the rest pose.

        The software rasterizer costs time per covered pixel, so the figure is
        drawn at a reduced resolution and scaled up. Smooth-shaded geometry
        survives that far better than the frame rate survives full resolution.
        """
        out_h, out_w = canvas_shape
        assert self.character is not None
        ch = self.character
        h = max(int(out_h * RENDER_SCALE), 64)
        w = max(int(out_w * RENDER_SCALE), 64)

        # Ease the torso turn toward its estimate, and back to square when the
        # person is lost, so a noisy shoulder measurement cannot make the figure
        # twitch from side to side.
        self.turn = self.turn_s(ch.torso_yaw(pose))
        rots = ch.local_rotations(pose, torso_yaw=self.turn) if pose is not None else {}
        parts = ch.skin(ch.pose_globals(rots))

        rot = mesh3d._rotation(self.yaw, self.pitch)
        distance = 3.0
        mid_y, height, half_w = ch.framing(parts, rot, distance)
        # Fit the projected figure into the frame with margin, on both axes.
        margin = 1.0 - 2 * STAGE_MARGIN
        focal = min(h * margin / height, w * 0.5 * margin / half_w)

        cx = w * 0.5
        cy = h * 0.5 + focal * mid_y
        out = self._stage(h, w)
        self._ground_shadow(out, parts, rot, origin=(cx, cy), focal=focal, distance=distance)
        out = self.renderer.render(
            out, parts,
            origin=(cx, cy), focal=focal, distance=distance,
            yaw=self.yaw, pitch=self.pitch,
        )
        if (h, w) != (out_h, out_w):
            out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return out

    def _stage(self, h: int, w: int) -> np.ndarray:
        """Vertical studio-gradient backdrop, built once per size."""
        if self._stage_cache is None or self._stage_cache.shape[:2] != (h, w):
            top = np.array(STAGE_TOP, dtype=np.float32)
            bottom = np.array(STAGE_BOTTOM, dtype=np.float32)
            t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
            column = top * (1.0 - t) + bottom * t
            self._stage_cache = np.repeat(column[:, None, :], w, axis=1).astype(np.uint8)
        return self._stage_cache.copy()

    def _ground_shadow(self, out: np.ndarray, parts, rot: np.ndarray, *,
                       origin, focal, distance) -> None:
        """Soft elliptical contact shadow under the figure."""
        cam = np.concatenate([p.positions for p in parts], axis=0) @ rot.T
        z = np.maximum(cam[:, 2] + distance, 1e-3)
        px, py = cam[:, 0] / z, cam[:, 1] / z
        cx, cy = origin
        # Sit the ellipse at the lowest projected point, spanning the silhouette.
        sx = int(cx + focal * float(np.median(px)))
        sy = int(cy - focal * float(np.percentile(py, 0.5)))
        rx = max(int(focal * float(np.percentile(np.abs(px), 98)) * 0.95), 6)
        ry = max(int(rx * 0.22), 3)

        # Work inside the ellipse's own bounding box: blurring the whole frame
        # to soften a shape this small was costing more than drawing the figure.
        pad = max(rx // 4, 4)
        x0, y0 = max(sx - rx - pad, 0), max(sy - ry - pad, 0)
        x1, y1 = min(sx + rx + pad, out.shape[1]), min(sy + ry + pad, out.shape[0])
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        roi = out[y0:y1, x0:x1]
        layer = roi.copy()
        cv2.ellipse(layer, (sx - x0, sy - y0), (rx, ry), 0, 0, 360,
                    SHADOW_COLOR, -1, cv2.LINE_AA)
        blur = max(3, (pad // 2) * 2 + 1)
        cv2.GaussianBlur(layer, (blur, blur), 0, dst=layer)
        cv2.addWeighted(layer, SHADOW_ALPHA, roi, 1.0 - SHADOW_ALPHA, 0.0, roi)

    def compose(self, frame: np.ndarray, poses: list) -> np.ndarray:
        """Build the output frame for the current view mode."""
        view = VIEWS[self.view_index]
        pose = poses[0] if poses else None
        char_img = self.render_character((self.h, self.w), pose)

        if view == "full3d":
            return char_img

        threshold = self.character.pose_scores_threshold if self.character else 0.35
        if view == "pip_cam":
            # The camera goes in the inset: shrink first, then draw the skeleton
            # at that size so its strokes stay proportional instead of vanishing.
            pw, ph = int(self.w * PIP_RATIO), int(self.h * PIP_RATIO)
            cam = cv2.resize(frame, (pw, ph))
            k = np.array([pw / self.w, ph / self.h])
            for p in poses:
                ui.draw_pose(cam, _Scaled(p.keypoints * k, p.scores), threshold=threshold)
            out = char_img
            self._blit_pip(out, cam, resize=False)
        else:
            out = frame.copy()
            for p in poses:
                ui.draw_pose(out, p, threshold=threshold, labels=self.show_labels)
            self._blit_pip(out, char_img)
        return out

    def _blit_pip(self, out: np.ndarray, source: np.ndarray, resize: bool = True) -> None:
        """Draw a picture-in-picture inset bottom-left."""
        h, w = out.shape[:2]
        pw, ph = int(w * PIP_RATIO), int(h * PIP_RATIO)
        inset = cv2.resize(source, (pw, ph)) if resize else source
        ph, pw = inset.shape[:2]
        x0, y0 = 16, h - ph - 16
        cv2.rectangle(out, (x0 - 3, y0 - 3), (x0 + pw + 3, y0 + ph + 3), (210, 210, 210), 2, cv2.LINE_AA)
        out[y0:y0 + ph, x0:x0 + pw] = inset

    def render(self, frame: np.ndarray, poses: list) -> np.ndarray:
        """Compose the full HUD frame."""
        pose = poses[0] if poses else None
        out = self.compose(frame, poses)

        h, w = out.shape[:2]
        ui.panel(out, 0, 0, w, 60, 0.5)
        ui.put_text(out, VIEW_LABELS[VIEWS[self.view_index]], (16, 10), 20, (255, 255, 255))
        name = MODEL_NAMES[self.model_index]
        status = f"{name}   {'tracking' if pose is not None else 'no person detected'}"
        ui.put_text(out, status, (16, 38), 16,
                    (140, 230, 140) if pose is not None else (200, 160, 100))

        right = f"{self.fps:.0f} FPS   {self.tracker.device.upper()}"
        rw = ui._render_text(right, 16, (200, 200, 200))[0].shape[1]
        ui.put_text(out, right, (w - rw - 16, 14), 16, (200, 200, 200))
        ui.put_text(out, "H help", (w - rw - 16, 38), 15, (140, 140, 140))

        if self.toast_text and time.time() < self.toast_until:
            tw = ui._render_text(self.toast_text, 18, (255, 255, 255))[0].shape[1]
            ui.panel(out, w // 2 - tw // 2 - 14, 74, tw + 28, 36, 0.65)
            ui.put_text(out, self.toast_text, (w // 2 - tw // 2, 82), 18, (255, 255, 255))

        if self.show_debug and pose is not None:
            tris = sum(len(p.triangles) for p in self.character.model.primitives)
            ui.panel(out, 12, h - 128, 300, 108, 0.6)
            ui.put_text(out, f"score {pose.confidence:.2f}   tris {tris}",
                        (20, h - 120), 15, (220, 220, 220))
            ui.put_text(out, f"view yaw {np.degrees(self.yaw):+.0f} pitch "
                             f"{np.degrees(self.pitch):+.0f}", (20, h - 98), 15, (220, 220, 220))
            ui.put_text(out, f"torso turn {np.degrees(self.turn):+.0f} deg",
                        (20, h - 76), 15, (150, 230, 255))
            ui.put_text(out, f"detail {DETAIL_LABELS[self.detail_index]}",
                        (20, h - 54), 15, (220, 220, 220))
        if self.show_help:
            ui.draw_help(out, HELP_LINES_POSE)
        return out

    def handle_key(self, key: int) -> bool:
        """Handle one keypress."""
        if key in (ord("q"), 27):
            return False
        if key == ord("k"):
            self.view_index = (self.view_index + 1) % len(VIEWS)
            self.toast(VIEW_LABELS[VIEWS[self.view_index]])
        elif key == ord("m"):
            self.next_model(1)
        elif key == ord("n"):
            self.next_model(-1)
        elif key == ord("y"):
            self.yaw += 0.25
        elif key == ord("p"):
            self.pitch = float(np.clip(self.pitch + 0.2, -1.2, 1.2))
        elif key == ord("r"):
            self.reset_view()
            self.toast("View reset")
        elif key == ord("f"):
            self.detail_index = (self.detail_index + 1) % len(DETAIL_LEVELS)
            self._load_character()
            self.toast(f"Detail: {DETAIL_LABELS[self.detail_index]}")
        elif key == ord("l"):
            self.show_labels = not self.show_labels
            self.toast("Keypoint names " + ("on" if self.show_labels else "off"))
        elif key == ord("d"):
            self.show_debug = not self.show_debug
            self.toast("Debug " + ("on" if self.show_debug else "off"))
        elif key == ord("h"):
            self.show_help = not self.show_help
        return True

    def run(self) -> None:
        """Run the main loop."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.w, self.h)
        print("[i] window open. H help, Q quit.", flush=True)
        self.toast("Stand back so your full body is visible")

        reason = "loop finished"
        frames = 0
        seq = -1
        while True:
            frame, seq = self.cam.wait_next(seq)
            if self.cam.failed:
                reason = "camera stopped delivering frames"
                break
            if not self.args.no_mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[:2] != (self.h, self.w):
                frame = cv2.resize(frame, (self.w, self.h))

            now = time.perf_counter()
            dt = max(now - self._last_t, 1e-4)
            self._last_t = now
            self.fps += 0.12 * (1.0 / dt - self.fps)

            poses = self.tracker(frame, dt)
            out = self.render(frame, poses)
            cv2.imshow(WINDOW, out)

            key = cv2.waitKey(1) & 0xFF
            frames += 1
            if frames % 150 == 0:
                print(f"[i] frames: {frames}  {self.fps:.1f} FPS  people: {len(poses)}", flush=True)

            if key != 255 and not self.handle_key(key):
                reason = f"quit key pressed ({key})"
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                reason = "window closed"
                break

        print(f"[i] stopped: {reason} (frames processed: {frames})", flush=True)
        self.cam.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Drive a rigged 3D character with your body pose")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--pose-model", default=str(root / "models" / "yolo26s-pose.pt"))
    p.add_argument("--pose-size", type=int, default=640)
    p.add_argument("--pose-conf", type=float, default=0.35)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", choices=MODEL_NAMES, default=MODEL_NAMES[0],
                   help="character to start with")
    p.add_argument("--no-mirror", action="store_true",
                   help="do not flip the camera image")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    try:
        PoseApp(args).run()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[i] stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
