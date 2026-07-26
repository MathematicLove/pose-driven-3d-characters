"""Mesh simplification for the software rasterizer.

The rasterizer costs roughly a fixed amount of Python work per triangle, so a
film-quality character mesh has to be thinned before it can animate in real
time. Vertex clustering is used rather than edge-collapse: it is a few lines,
runs once at load, and carries skinning weights and UVs along untouched, which
matters because these meshes are posed every frame.
"""

from __future__ import annotations

import numpy as np

import gltf

# Resolution of the texture-space grid that keeps atlas islands from merging.
_UV_CELLS = 16


def _weld(prim: gltf.Primitive, cells: int) -> gltf.Primitive:
    """Snap vertices to a grid of `cells` per axis and drop collapsed triangles."""
    pos = prim.positions
    lo = pos.min(axis=0)
    span = np.maximum(pos.max(axis=0) - lo, 1e-9)
    # Grid coordinate per vertex, then one representative vertex per cell.
    grid = np.floor((pos - lo) / span * cells).astype(np.int64)
    grid = np.clip(grid, 0, cells - 1)
    keys = (grid[:, 0] * cells + grid[:, 1]) * cells + grid[:, 2]
    # Vertices that a nearby grid cell would merge but that belong to different
    # bones must stay apart, or an arm gets welded to the chest it rests against.
    dominant = prim.joints[np.arange(len(prim.joints)), np.argmax(prim.weights, axis=1)]
    keys = keys * (int(dominant.max()) + 1) + dominant.astype(np.int64)
    if prim.uv is not None:
        # Keep texture-atlas islands apart too. Neighbours straddling a UV seam
        # look adjacent in space but map to unrelated pixels, and collapsing
        # them together is what speckles small details like hair and shoes.
        uv_cell = np.clip(np.floor(np.mod(prim.uv, 1.0) * _UV_CELLS), 0, _UV_CELLS - 1)
        uv_key = uv_cell[:, 0].astype(np.int64) * _UV_CELLS + uv_cell[:, 1].astype(np.int64)
        keys = keys * (_UV_CELLS * _UV_CELLS) + uv_key
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)

    tris = inverse[prim.triangles]
    # A triangle whose corners landed in the same cell has no area left.
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[keep]
    if tris.shape[0] == 0:
        return prim

    # Every surviving attribute is taken from the same representative vertex, so
    # position, normal and UV stay mutually consistent.
    merged_pos = pos[first]
    merged_nrm = (prim.normals[first] if prim.normals is not None and prim.normals.size
                  else np.zeros((len(first), 3)))

    # UVs come from one representative vertex, never averaged: neighbours in
    # space can sit far apart in the texture atlas, and a mean of two such
    # coordinates points at unrelated pixels, which speckles the face and shoes.
    merged_uv = None if prim.uv is None else prim.uv[first]

    return gltf.Primitive(
        positions=merged_pos,
        normals=merged_nrm,
        # Joints and weights are taken from one representative vertex per cell;
        # averaging them would blend unrelated bones and tear the mesh.
        joints=prim.joints[first],
        weights=prim.weights[first],
        triangles=tris,
        color=prim.color,
        uv=merged_uv,
        texture=prim.texture,
    )


def to_budget(primitives: list[gltf.Primitive], budget: int) -> list[gltf.Primitive]:
    """Thin primitives until their combined triangle count fits the budget."""
    total = sum(len(p.triangles) for p in primitives)
    if total <= budget:
        return primitives

    out = list(primitives)
    # Search downward for the coarsest grid that still meets the budget; each
    # pass is cheap and this runs once per model load.
    for cells in (128, 96, 72, 56, 44, 34, 26, 20, 16, 12):
        candidate = [_weld(p, cells) for p in primitives]
        if sum(len(p.triangles) for p in candidate) <= budget:
            return candidate
        out = candidate
    return out
