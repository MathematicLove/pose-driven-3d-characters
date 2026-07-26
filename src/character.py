"""Retarget body pose keypoints onto a rigged glTF character."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import decimate
import gltf
import pose as posemod
from mesh3d import Part

# Triangles the software rasterizer can pose and draw at an interactive rate.
TRIANGLE_BUDGET = 8000

TARGET_HEIGHT = 1.6
_UP = np.array([0.0, 1.0, 0.0])
_EYE3 = np.eye(3)

# Body proportions as multiples of torso length (shoulders to hips), used to
# tell "limb is short because it points at the camera" from "limb is short
# because the person is small". Rough human averages are good enough here.
_CHAIN_LENGTHS: dict[str, float] = {
    "arm_l_1": 0.62, "arm_r_1": 0.62,
    "arm_l_2": 0.58, "arm_r_2": 0.58,
    "leg_l_1": 0.84, "leg_r_1": 0.84,
    "leg_l_2": 0.80, "leg_r_2": 0.80,
}
# Which way a foreshortened limb is assumed to bend: arms reach forward toward
# the camera, knees bend backward.
_CHAIN_DEPTH_SIGN: dict[str, float] = {
    "arm_l_1": 1.0, "arm_r_1": 1.0,
    "arm_l_2": 1.0, "arm_r_2": 1.0,
    "leg_l_1": -0.5, "leg_r_1": -0.5,
    "leg_l_2": -0.5, "leg_r_2": -0.5,
}
SHOULDER_WIDTH_RATIO = 0.78
TORSO_UNIT_RATIO = 0.30
# Shoulders must narrow to this fraction of their expected width before the body
# counts as turned, so ordinary build differences do not read as rotation.
TURN_DEADZONE = 0.82
TURN_GAIN = 0.55
# A single camera cannot tell a deep turn from a shrug, so the torso spin is
# capped well short of a right angle to keep the figure upright and readable.
TURN_LIMIT = np.radians(50.0)
# Foreshortening depth is only an estimate from average limb proportions, so it
# is applied at part strength; at full strength joints visibly overshoot.
DEPTH_DAMPING = 0.55

# Per-chain angle multipliers, kept at 1.0 so every bone lands exactly on the
# direction its keypoints describe. Amplifying the swing was a workaround for a
# retargeting bug that misaimed limbs; with the aim correct, any gain above 1.0
# just pushes joints off the pose it is supposed to copy.
_CHAIN_GAIN: dict[str, float] = {}

# Body-part tints (BGR) for models that ship without a texture, keyed by the
# chain a vertex's dominant bone belongs to: teal shirt, denim legs, skin limbs.
_PART_COLORS: dict[str, tuple[int, int, int]] = {
    "spine_lower": (150, 128, 60),
    "spine_upper": (168, 146, 72),
    "neck": (165, 195, 225),
    "arm_l_1": (168, 146, 72),
    "arm_l_2": (165, 195, 225),
    "arm_r_1": (168, 146, 72),
    "arm_r_2": (165, 195, 225),
    "leg_l_1": (120, 96, 74),
    "leg_l_2": (134, 108, 84),
    "leg_r_1": (120, 96, 74),
    "leg_r_2": (134, 108, 84),
}
_SKIN_COLOR = (165, 195, 225)
_DEFAULT_PART_COLOR = (168, 146, 72)

# Bone name -> (start keypoint(s), end keypoint(s)) driving its rest direction.
# A bone's rotation swings its rest-pose direction (parent joint -> this joint,
# in model space) onto the direction implied by the matched pose keypoints.
_CHAINS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "spine_lower": ((posemod.LEFT_HIP, posemod.RIGHT_HIP),
                    (posemod.LEFT_SHOULDER, posemod.RIGHT_SHOULDER)),
    "spine_upper": ((posemod.LEFT_HIP, posemod.RIGHT_HIP),
                    (posemod.LEFT_SHOULDER, posemod.RIGHT_SHOULDER)),
    "neck": ((posemod.LEFT_SHOULDER, posemod.RIGHT_SHOULDER), (posemod.NOSE,)),
    "arm_l_1": ((posemod.LEFT_SHOULDER,), (posemod.LEFT_ELBOW,)),
    "arm_l_2": ((posemod.LEFT_ELBOW,), (posemod.LEFT_WRIST,)),
    "arm_r_1": ((posemod.RIGHT_SHOULDER,), (posemod.RIGHT_ELBOW,)),
    "arm_r_2": ((posemod.RIGHT_ELBOW,), (posemod.RIGHT_WRIST,)),
    "leg_l_1": ((posemod.LEFT_HIP,), (posemod.LEFT_KNEE,)),
    "leg_l_2": ((posemod.LEFT_KNEE,), (posemod.LEFT_ANKLE,)),
    "leg_r_1": ((posemod.RIGHT_HIP,), (posemod.RIGHT_KNEE,)),
    "leg_r_2": ((posemod.RIGHT_KNEE,), (posemod.RIGHT_ANKLE,)),
}

# Chain to read keypoints from when the camera image is mirrored: a bone on the
# model's left is driven by the keypoints of the body's other side.
_MIRRORED_CHAINS: dict[str, str] = {
    key: (key.replace("_l_", "_r_") if "_l_" in key
          else key.replace("_r_", "_l_") if "_r_" in key else key)
    for key in _CHAINS
}

# Per-model bone-name -> chain key. Names come from gltf node names.
_BONE_MAPS: dict[str, dict[str, str]] = {
    "RiggedFigure.glb": {
        "torso_joint_1": "spine_lower",
        "torso_joint_2": "spine_upper",
        "neck_joint_1": "neck",
        "arm_joint_L_1": "arm_l_1",
        "arm_joint_L_2": "arm_l_2",
        "arm_joint_R_1": "arm_r_1",
        "arm_joint_R_2": "arm_r_2",
        "leg_joint_L_1": "leg_l_1",
        "leg_joint_L_2": "leg_l_2",
        "leg_joint_R_1": "leg_r_1",
        "leg_joint_R_2": "leg_r_2",
    },
    "CesiumMan.glb": {
        "Skeleton_torso_joint_1": "spine_lower",
        "Skeleton_torso_joint_2": "spine_upper",
        "Skeleton_neck_joint_1": "neck",
        "Skeleton_arm_joint_L__4_": "arm_l_1",
        "Skeleton_arm_joint_L__3_": "arm_l_2",
        "Skeleton_arm_joint_R": "arm_r_1",
        "Skeleton_arm_joint_R__2_": "arm_r_2",
        "leg_joint_L_1": "leg_l_1",
        "leg_joint_L_2": "leg_l_2",
        "leg_joint_R_1": "leg_r_1",
        "leg_joint_R_2": "leg_r_2",
    },
    # Standard Mixamo rig. "Arm" is the upper arm, "ForeArm" the lower.
    "Michelle.glb": {
        "mixamorig:Spine": "spine_lower",
        "mixamorig:Spine1": "spine_upper",
        "mixamorig:Neck": "neck",
        "mixamorig:LeftArm": "arm_l_1",
        "mixamorig:LeftForeArm": "arm_l_2",
        "mixamorig:RightArm": "arm_r_1",
        "mixamorig:RightForeArm": "arm_r_2",
        "mixamorig:LeftUpLeg": "leg_l_1",
        "mixamorig:LeftLeg": "leg_l_2",
        "mixamorig:RightUpLeg": "leg_r_1",
        "mixamorig:RightLeg": "leg_r_2",
    },
}

MODEL_FILES: dict[str, str] = {
    "Michelle": "Michelle.glb",
    "Rigged Figure": "RiggedFigure.glb",
    "Cesium Man": "CesiumMan.glb",
}

# Where a pose's screen-right axis points in each model's own space. These rigs
# all face the camera, so screen-right is model +x.
_DEFAULT_RIGHT_AXIS = np.array([1.0, 0.0, 0.0])

# Default view yaw per model, so each one is first shown from a flattering angle.
DEFAULT_YAW: dict[str, float] = {}


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix swinging unit vector a onto unit vector b."""
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, ortho)
        v /= max(np.linalg.norm(v), 1e-9)
        return 2.0 * np.outer(v, v) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _orthonormal_inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a rotation-like matrix, tolerating a uniform scale."""
    scale = float(np.linalg.norm(m[:, 0]))
    if scale < 1e-9:
        return np.eye(3)
    # A pure rotation inverts by transposing; a uniformly scaled one needs the
    # scale divided out twice, once for the transpose and once for the scale.
    return m.T / (scale * scale)


def _axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix of `theta` radians about `axis`."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.eye(3)
    x, y, z = axis / norm
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _scale_rotation(rot: np.ndarray, gain: float) -> np.ndarray:
    """Same rotation axis, angle multiplied by gain."""
    angle = float(np.arccos(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-6:
        return rot
    axis = np.array([rot[2, 1] - rot[1, 2],
                     rot[0, 2] - rot[2, 0],
                     rot[1, 0] - rot[0, 1]])
    if float(np.linalg.norm(axis)) < 1e-9:
        return rot
    return _axis_rotation(axis, float(np.clip(angle * gain, -np.pi, np.pi)))


@dataclass
class Character:
    """A loaded, riggable character model."""

    model: gltf.Gltf
    joint_index: dict[str, int]
    bone_map: dict[str, str]
    scale: float
    root_lift: float
    vertex_colors: list[np.ndarray | None]
    right_axis: np.ndarray
    default_yaw: float = 0.0
    pose_scores_threshold: float = 0.35
    # The camera frame is mirrored for a natural on-screen experience, so a
    # keypoint's screen side is the opposite of the body side it belongs to.
    # Swapping the horizontal axis makes the character face the viewer and
    # raise the same arm the viewer sees themselves raise.
    mirror: bool = True

    @classmethod
    def load(cls, name: str, mirror: bool = True,
             budget: int = TRIANGLE_BUDGET) -> "Character":
        """Load a character by its display name."""
        filename = MODEL_FILES[name]
        root = Path(__file__).resolve().parent.parent
        model = gltf.load(root / "models" / "characters" / filename)
        model.primitives = decimate.to_budget(model.primitives, max(budget, 200))
        joint_index = {model.nodes[j].name: j for j in model.joints}
        bone_map = _BONE_MAPS[filename]

        character = cls(model=model, joint_index=joint_index, bone_map=bone_map,
                        scale=1.0, root_lift=0.0, vertex_colors=[],
                        right_axis=_DEFAULT_RIGHT_AXIS,
                        default_yaw=DEFAULT_YAW.get(filename, 0.0),
                        mirror=mirror)
        character.vertex_colors = character._build_vertex_colors()
        character._fit_to_unit_height()
        return character

    def _fit_to_unit_height(self) -> None:
        """Scale/lift from the skinned rest pose, whose bind space may be rotated."""
        rest = np.concatenate(
            [p.positions for p in self.skin(self.pose_globals({}))], axis=0)
        height = float(rest[:, 1].max() - rest[:, 1].min())
        self.scale = TARGET_HEIGHT / max(height, 1e-6)
        self.root_lift = -float(rest[:, 1].min()) * self.scale

    def _build_vertex_colors(self) -> list[np.ndarray | None]:
        """Per-vertex tints for untextured primitives, from each vertex's dominant bone."""
        chain_of_joint = {
            self.joint_index[name]: chain
            for name, chain in self.bone_map.items() if name in self.joint_index
        }
        colors: list[np.ndarray | None] = []
        for prim in self.model.primitives:
            if prim.texture is not None and prim.uv is not None:
                colors.append(None)
                continue
            dominant = prim.joints[np.arange(len(prim.joints)),
                                   np.argmax(prim.weights, axis=1)]
            out = np.empty((len(prim.positions), 3), dtype=np.float64)
            for vi, jid in enumerate(dominant):
                node = self.model.joints[int(jid)]
                chain = self._nearest_chain(node, chain_of_joint)
                out[vi] = _PART_COLORS.get(chain, _DEFAULT_PART_COLOR)
            head = self._head_mask(prim.positions)
            out[head] = _SKIN_COLOR
            colors.append(out)
        return colors

    def _nearest_chain(self, node: int, chain_of_joint: dict[int, str]) -> str | None:
        """Chain key of the closest mapped ancestor of one joint node."""
        seen = 0
        while node is not None and seen < 32:
            if node in chain_of_joint:
                return chain_of_joint[node]
            node = self.model.nodes[node].parent
            seen += 1
        return None

    def _head_mask(self, positions: np.ndarray) -> np.ndarray:
        """Vertices in the top slice of the model, tinted as skin."""
        top = float(positions[:, 1].max())
        span = top - float(positions[:, 1].min())
        return positions[:, 1] > top - 0.12 * max(span, 1e-6)

    def _rest_dir(self, joint_idx: int) -> np.ndarray | None:
        """Rest-pose axis of the bone starting at one joint, in model space.

        Measured from the joint toward its child, which is the segment the pose
        keypoints describe (hip to knee, shoulder to elbow). Using the incoming
        direction from the parent instead bakes the joint's own splay into the
        fit, and rotating that onto a vertical target swings the limb across the
        body — visibly crossed legs.
        """
        this_pos = self.model.globals[joint_idx][:3, 3]
        child = self._rest_child(joint_idx)
        if child is not None:
            d = self.model.globals[child][:3, 3] - this_pos
            if float(np.linalg.norm(d)) > 1e-9:
                return d / float(np.linalg.norm(d))
        # Leaf bone: fall back to the direction it was reached from.
        parent = self.model.nodes[joint_idx].parent
        if parent is None:
            return None
        d = this_pos - self.model.globals[parent][:3, 3]
        norm = float(np.linalg.norm(d))
        return d / norm if norm > 1e-9 else None

    def _rest_child(self, joint_idx: int) -> int | None:
        """The skinned child that continues this bone's chain, if any."""
        joints = set(self.model.joints)
        children = [c for c in self.model.nodes[joint_idx].children if c in joints]
        if not children:
            return None
        # Follow the longest segment: on hands and feet several children fan out,
        # and the farthest one is the limb's continuation.
        origin = self.model.globals[joint_idx][:3, 3]
        return max(children,
                   key=lambda c: float(np.linalg.norm(self.model.globals[c][:3, 3] - origin)))

    def _target_dir(self, pose: posemod.Pose, chain_key: str,
                    unit: float) -> np.ndarray | None:
        """Direction implied by keypoints for one chain, in this model's space.

        A single camera gives no true depth, but a limb that looks shorter than
        its own reference length must be angled toward or away from the lens.
        Recovering that component lets the figure reach forward and turn instead
        of staying pinned to a flat plane.
        """
        source = _MIRRORED_CHAINS[chain_key] if self.mirror else chain_key
        starts, ends = _CHAINS[source]
        p0 = pose.point(*starts, threshold=self.pose_scores_threshold)
        p1 = pose.point(*ends, threshold=self.pose_scores_threshold)
        if p0 is None or p1 is None:
            return None
        d2 = p1 - p0
        norm = float(np.linalg.norm(d2))
        if norm < 1e-6:
            return None

        # Image space is +x right, +y down; the model's up is +y. Mirroring
        # flips the horizontal component so the figure faces the viewer.
        dx = -d2[0] if self.mirror else d2[0]
        flat = self.right_axis * dx - _UP * d2[1]
        expected = _CHAIN_LENGTHS.get(chain_key, 0.0) * unit
        depth = 0.0
        if expected > 1e-6:
            # Foreshortening: length missing from the image plane must lie along
            # the view axis. Damped, because limb proportions are approximate and
            # an over-eager depth term visibly wrenches the joint out of place.
            ratio = float(np.clip(norm / expected, 0.0, 1.0))
            depth = np.sqrt(max(1.0 - ratio * ratio, 0.0)) * expected
            depth *= _CHAIN_DEPTH_SIGN.get(chain_key, 1.0) * DEPTH_DAMPING
        target = flat + np.array([0.0, 0.0, 1.0]) * depth
        n = float(np.linalg.norm(target))
        return target / n if n > 1e-9 else None

    def _torso_frame(self, pose: posemod.Pose) -> tuple[float, float]:
        """Torso yaw (radians) and a body unit length, from shoulders and hips.

        Shoulders that look narrower than the body's own proportions mean the
        person is turned away from the camera; the sign of that turn comes from
        which shoulder sits closer to the body midline.
        """
        ls = pose.keypoints[posemod.LEFT_SHOULDER]
        rs = pose.keypoints[posemod.RIGHT_SHOULDER]
        shoulders = pose.point(posemod.LEFT_SHOULDER, posemod.RIGHT_SHOULDER,
                               threshold=self.pose_scores_threshold)
        hips = pose.point(posemod.LEFT_HIP, posemod.RIGHT_HIP,
                          threshold=self.pose_scores_threshold)
        if shoulders is None or hips is None:
            return 0.0, max(pose.height * TORSO_UNIT_RATIO, 1e-6)

        # Torso length is the most reliable scale reference: it barely changes
        # with limb motion, unlike the bounding box.
        unit = max(float(np.linalg.norm(shoulders - hips)), 1e-6)
        if not (pose.visible(posemod.LEFT_SHOULDER, self.pose_scores_threshold)
                and pose.visible(posemod.RIGHT_SHOULDER, self.pose_scores_threshold)):
            return 0.0, unit

        # Only a clear narrowing counts as a turn. Shoulder width varies enough
        # between people that a small shortfall is build, not rotation, and
        # treating it as rotation injects depth into every limb of a face-on pose.
        width = abs(float(ls[0] - rs[0]))
        ratio = width / (unit * SHOULDER_WIDTH_RATIO)
        if ratio >= TURN_DEADZONE:
            return 0.0, unit
        yaw = float(np.arccos(np.clip(ratio / TURN_DEADZONE, 0.0, 1.0)))
        # Facing sign from which shoulder sits left on screen, flipped when the
        # image is mirrored so it matches the sides the bones are driven from.
        facing = ls[0] >= rs[0]
        if self.mirror:
            facing = not facing
        return (yaw if facing else -yaw), unit

    def torso_yaw(self, pose: posemod.Pose | None) -> float:
        """Estimated torso turn in radians, positive when turning one way."""
        return 0.0 if pose is None else self._torso_frame(pose)[0]

    def local_rotations(self, pose: posemod.Pose | None,
                        torso_yaw: float | None = None) -> dict[int, np.ndarray]:
        """Per-joint local rotations that aim each driven bone at its target.

        A bone's direction comes from its parent's frame: the child's offset is
        fixed in that frame, so the bone points along `A.rot @ offset`, where A
        is the parent's animated global transform. Wanting that to equal the
        target while the same offset points along `G.rot @ offset` in bind pose
        gives, for the bone's own local rotation:

            R = A_parent.rot^-1 @ X @ G_parent.rot,   X = swing(rest -> target)

        Both accumulated rotations are needed, animated *and* bind. Using the
        animated one on both sides is only right when the bind rotation is
        identity, which it never is once a rig has an axis-conversion root or
        any nested joint, and it misaims every limb by up to ~20 degrees.
        """
        if pose is None:
            return {}
        chain_of_joint = {
            self.joint_index[name]: chain
            for name, chain in self.bone_map.items() if name in self.joint_index
        }
        measured_yaw, unit = self._torso_frame(pose)
        # A caller tracking the turn over time can pass a steadier value in;
        # the raw per-frame estimate follows keypoint noise.
        if torso_yaw is None:
            torso_yaw = measured_yaw
        # Turning the body is a spin about the up axis, which direction fitting
        # cannot see: a bone rotating about its own axis still points the same
        # way. It is folded into every bone's world swing, which yaws the whole
        # figure. It has to go on all of them, not just the spine: each driven
        # bone's orientation works out to `X @ bind` no matter what its parent
        # does, so spinning one alone is undone by the next one down and only
        # twitches the handful of vertices weighted to it.
        #
        # One camera cannot say how a limb is really angled in depth, only where
        # it projects, so yawing everything is the honest reading of a turn: the
        # figure faces where the shoulders say, and the limbs follow the body.
        spin = float(np.clip(torso_yaw * TURN_GAIN, -TURN_LIMIT, TURN_LIMIT))
        spin_rot = _axis_rotation(_UP, spin) if abs(spin) > 1e-3 else None

        nodes = self.model.nodes
        rots: dict[int, np.ndarray] = {}
        animated: dict[int, np.ndarray] = {}
        queue = [i for i in range(len(nodes)) if nodes[i].parent is None]
        while queue:
            i = queue.pop(0)
            parent = nodes[i].parent
            a_parent = animated.get(parent, _EYE3) if parent is not None else _EYE3
            g_parent = (self.model.globals[parent][:3, :3]
                        if parent is not None else _EYE3)

            local_rot = np.eye(3)
            chain = chain_of_joint.get(i)
            if chain is not None:
                rest = self._rest_dir(i)
                target = self._target_dir(pose, chain, unit)
                if rest is not None and target is not None:
                    world_swing = _rotation_between(rest, target)
                    gain = _CHAIN_GAIN.get(chain, 1.0)
                    if gain != 1.0:
                        world_swing = _scale_rotation(world_swing, gain)
                    if spin_rot is not None:
                        world_swing = spin_rot @ world_swing
                    local_rot = _orthonormal_inverse(a_parent) @ world_swing @ g_parent
                    rots[i] = local_rot
            animated[i] = a_parent @ local_rot @ nodes[i].matrix[:3, :3]
            queue.extend(nodes[i].children)
        return rots

    def pose_hip_center(self, pose: posemod.Pose) -> np.ndarray | None:
        """Pose-space hip midpoint (image xy)."""
        return pose.point(posemod.LEFT_HIP, posemod.RIGHT_HIP,
                          threshold=self.pose_scores_threshold)

    def skin(self, joint_globals: dict[int, np.ndarray]) -> list[Part]:
        """Skinned world-space drawable parts, one per primitive."""
        out: list[Part] = []
        for pi, prim in enumerate(self.model.primitives):
            n = prim.positions.shape[0]
            acc_pos = np.zeros((n, 3), dtype=np.float64)
            acc_norm = np.zeros((n, 3), dtype=np.float64)
            for k in range(prim.joints.shape[1]):
                w = prim.weights[:, k]
                if not np.any(w > 0):
                    continue
                joint_ids = prim.joints[:, k]
                for jid in np.unique(joint_ids):
                    mask = joint_ids == jid
                    ww = w[mask]
                    if not np.any(ww > 0):
                        continue
                    real_joint = self.model.joints[jid]
                    skin_mat = joint_globals[real_joint] @ self.model.inverse_bind[jid]
                    pos_h = np.concatenate(
                        [prim.positions[mask], np.ones((mask.sum(), 1))], axis=1)
                    world = pos_h @ skin_mat.T
                    acc_pos[mask] += world[:, :3] * ww[:, None]
                    if prim.normals is not None:
                        nrm = prim.normals[mask] @ skin_mat[:3, :3].T
                        acc_norm[mask] += nrm * ww[:, None]
            out.append(Part(
                positions=acc_pos * self.scale,
                normals=acc_norm,
                triangles=prim.triangles,
                color=prim.color,
                vertex_colors=self.vertex_colors[pi],
                uv=prim.uv,
                texture=prim.texture,
            ))
        return out

    def pose_globals(self, rots: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Recompute global joint transforms with extra local rotations applied."""
        nodes = self.model.nodes
        globals_: dict[int, np.ndarray] = {}
        order = [i for i in range(len(nodes)) if nodes[i].parent is None]
        while order:
            i = order.pop(0)
            local = nodes[i].matrix.copy()
            if i in rots:
                t = local[:3, 3].copy()
                local[:3, :3] = rots[i] @ local[:3, :3]
                local[:3, 3] = t
            parent = nodes[i].parent
            base = globals_[parent] if parent is not None else np.eye(4)
            globals_[i] = base @ local
            order.extend(nodes[i].children)
        return globals_

    def framing(self, parts: list[Part], rotation: np.ndarray,
                distance: float) -> tuple[float, float, float]:
        """Focal length and screen center that fit the posed figure in a frame.

        Returns (focal_per_unit, mid_y_over_z, half_width_over_z): the extents
        are measured after perspective division, because a model with depth
        projects far wider than its flat extent suggests. Percentiles rather
        than extremes, so one stretched vertex from a bad keypoint cannot
        shrink the whole figure.
        """
        pts = np.concatenate([p.positions for p in parts], axis=0) @ rotation.T
        z = np.maximum(pts[:, 2] + distance, 1e-3)
        # Unit-focal projected coordinates; screen = focal * these.
        px = pts[:, 0] / z
        py = pts[:, 1] / z
        lo, hi = (float(v) for v in np.percentile(py, (0.5, 99.5)))
        half_w = float(np.percentile(np.abs(px), 99.5))
        return (lo + hi) * 0.5, max(hi - lo, 1e-6), max(half_w, 1e-6)
