import numpy as np

import decimate
import gltf


def grid_primitive(n=30, with_uv=False):
    """A flat n x n grid of quads (2 triangles each) bound to a single bone."""
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    pos = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    idx = np.arange(n * n).reshape(n, n)
    tris = []
    for r in range(n - 1):
        for c in range(n - 1):
            a, b, d, e = idx[r, c], idx[r, c + 1], idx[r + 1, c], idx[r + 1, c + 1]
            tris += [[a, b, d], [b, e, d]]
    count = n * n
    return gltf.Primitive(
        positions=pos,
        normals=np.tile([0.0, 0.0, 1.0], (count, 1)),
        joints=np.zeros((count, 4), dtype=np.int64),
        weights=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        triangles=np.array(tris, dtype=np.int64),
        uv=pos[:, :2].copy() if with_uv else None,
    )


def total_triangles(prims):
    return sum(len(p.triangles) for p in prims)


def test_under_budget_is_returned_untouched():
    prims = [grid_primitive(10)]
    assert decimate.to_budget(prims, budget=10_000) is prims


def test_reduces_triangles_toward_budget():
    prim = grid_primitive(40)
    before = len(prim.triangles)
    out = decimate.to_budget([prim], budget=before // 4)
    assert total_triangles(out) < before


def test_attributes_stay_consistent():
    out = decimate.to_budget([grid_primitive(40)], budget=300)[0]
    n = len(out.positions)
    assert len(out.normals) == len(out.joints) == len(out.weights) == n
    assert out.triangles.max() < n
    assert out.triangles.min() >= 0


def test_no_degenerate_triangles_remain():
    out = decimate.to_budget([grid_primitive(40)], budget=300)[0]
    t = out.triangles
    assert np.all(t[:, 0] != t[:, 1])
    assert np.all(t[:, 1] != t[:, 2])
    assert np.all(t[:, 0] != t[:, 2])


def test_uv_is_carried_along():
    out = decimate.to_budget([grid_primitive(40, with_uv=True)], budget=300)[0]
    assert out.uv is not None and len(out.uv) == len(out.positions)


def test_different_bones_are_not_welded_together():
    prim = grid_primitive(20)
    # Alternate the dominant bone so neighbouring vertices belong to different bones.
    prim.joints[:, 0] = np.arange(len(prim.joints)) % 2
    welded = decimate._weld(prim, cells=4)
    plain = decimate._weld(grid_primitive(20), cells=4)
    assert len(welded.positions) > len(plain.positions)


def test_weld_returns_input_if_everything_collapses():
    prim = grid_primitive(5)
    assert decimate._weld(prim, cells=1) is prim
