"""Minimal glTF binary reader."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942

_DTYPES = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Primitive:
    """One drawable mesh piece."""

    positions: np.ndarray
    normals: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    triangles: np.ndarray
    color: tuple[int, int, int] = (185, 180, 175)
    uv: np.ndarray | None = None
    texture: np.ndarray | None = None


@dataclass
class Node:
    """One scene node."""

    name: str
    matrix: np.ndarray
    children: list[int] = field(default_factory=list)
    parent: int | None = None


@dataclass
class Gltf:
    """Loaded model."""

    nodes: list[Node]
    primitives: list[Primitive]
    joints: list[int]
    inverse_bind: np.ndarray
    globals: np.ndarray


def _node_matrix(node: dict) -> np.ndarray:
    """Local transform of one node."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        m[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag(node["scale"])
    if "translation" in node:
        m[:3, 3] = node["translation"]
    return m


def _decode_image(js: dict, blob: bytes, image_index: int) -> np.ndarray | None:
    """Decode one embedded glTF image to a BGR array."""
    images = js.get("images") or []
    if not 0 <= image_index < len(images):
        return None
    image = images[image_index]
    if "bufferView" not in image:
        return None
    view = js["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    raw = np.frombuffer(blob, dtype=np.uint8, count=view["byteLength"], offset=start)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return decoded if decoded is not None and decoded.size else None


def _material_texture(js: dict, blob: bytes, material_index: int) -> np.ndarray | None:
    """Base-color texture of one material, if it has an embedded one."""
    pbr = js["materials"][material_index].get("pbrMetallicRoughness", {})
    ref = pbr.get("baseColorTexture")
    if ref is None:
        return None
    textures = js.get("textures") or []
    if not 0 <= ref.get("index", -1) < len(textures):
        return None
    source = textures[ref["index"]].get("source")
    return None if source is None else _decode_image(js, blob, source)


class _Reader:
    """Accessor decoder."""

    def __init__(self, js: dict, blob: bytes) -> None:
        self.js = js
        self.blob = blob

    def __call__(self, index: int) -> np.ndarray:
        """Read one accessor."""
        acc = self.js["accessors"][index]
        dtype = _DTYPES[acc["componentType"]]
        ncomp = _COUNTS[acc["type"]]
        count = acc["count"]
        if "bufferView" not in acc:
            return np.zeros((count, ncomp), dtype=np.float64)
        view = self.js["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or np.dtype(dtype).itemsize * ncomp
        item = np.dtype(dtype).itemsize * ncomp
        if stride == item:
            raw = np.frombuffer(self.blob, dtype=dtype, count=count * ncomp, offset=start)
            out = raw.reshape(count, ncomp)
        else:
            rows = [np.frombuffer(self.blob, dtype=dtype, count=ncomp, offset=start + i * stride)
                    for i in range(count)]
            out = np.stack(rows)
        return out.astype(np.float64) if dtype == np.float32 else out


def load(path: str | Path) -> Gltf:
    """Read a glb file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Character model not found: {path}")
    data = path.read_bytes()
    magic, _version, total = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a binary glTF: {path}")

    js: dict | None = None
    blob = b""
    off = 12
    while off < total:
        length, kind = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + length]
        if kind == _JSON_CHUNK:
            js = json.loads(chunk)
        elif kind == _BIN_CHUNK:
            blob = bytes(chunk)
        off += 8 + length + ((4 - length % 4) % 4 if length % 4 else 0)
    if js is None:
        raise ValueError(f"No json chunk in {path}")

    read = _Reader(js, blob)
    nodes = [Node(name=n.get("name", f"node{i}"), matrix=_node_matrix(n),
                  children=list(n.get("children", []))) for i, n in enumerate(js["nodes"])]
    for i, node in enumerate(nodes):
        for c in node.children:
            nodes[c].parent = i

    globals_ = np.tile(np.eye(4), (len(nodes), 1, 1))
    order = [i for i in range(len(nodes)) if nodes[i].parent is None]
    while order:
        i = order.pop(0)
        parent = nodes[i].parent
        base = globals_[parent] if parent is not None else np.eye(4)
        globals_[i] = base @ nodes[i].matrix
        order.extend(nodes[i].children)

    skin = js["skins"][0]
    joints = list(skin["joints"])
    inverse_bind = read(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)

    primitives: list[Primitive] = []
    for mesh in js.get("meshes", []):
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                continue
            pos = read(attrs["POSITION"])
            normals = read(attrs["NORMAL"]) if "NORMAL" in attrs else np.zeros_like(pos)
            jnt = read(attrs["JOINTS_0"]).astype(np.int32)
            wts = read(attrs["WEIGHTS_0"]).astype(np.float64)
            wsum = wts.sum(axis=1, keepdims=True)
            wts = wts / np.where(wsum > 1e-9, wsum, 1.0)
            if "indices" in prim:
                idx = read(prim["indices"]).astype(np.int64).reshape(-1)
            else:
                idx = np.arange(len(pos), dtype=np.int64)
            color = (185, 180, 175)
            texture = None
            mat = prim.get("material")
            if mat is not None:
                pbr = js["materials"][mat].get("pbrMetallicRoughness", {})
                base = pbr.get("baseColorFactor")
                if base:
                    color = tuple(int(np.clip(c * 255, 0, 255)) for c in base[2::-1])
                texture = _material_texture(js, blob, mat)
            uv = read(attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else None
            if texture is None:
                uv = None
            primitives.append(Primitive(positions=pos, normals=normals, joints=jnt,
                                        weights=wts, triangles=idx.reshape(-1, 3),
                                        color=color, uv=uv, texture=texture))
    return Gltf(nodes=nodes, primitives=primitives, joints=joints,
                inverse_bind=inverse_bind, globals=globals_)
