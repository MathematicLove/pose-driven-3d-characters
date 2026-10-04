import numpy as np
import pytest

from character import (
    _axis_rotation,
    _orthonormal_inverse,
    _rotation_between,
    _scale_rotation,
)
from gltf import _node_matrix
from mesh3d import _rotation


def is_rotation(m):
    return np.allclose(m @ m.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(m), 1.0)


class TestRotationBetween:
    def test_maps_a_onto_b(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 1.0])
        r = _rotation_between(a, b)
        assert is_rotation(r)
        assert np.allclose(r @ a, b / np.linalg.norm(b))

    def test_same_vector_is_identity(self):
        v = np.array([0.0, 2.0, 0.0])
        assert np.allclose(_rotation_between(v, v), np.eye(3))

    def test_opposite_vectors_flip(self):
        a = np.array([0.0, 0.0, 1.0])
        r = _rotation_between(a, -a)
        assert np.allclose(r @ a, -a)
        assert np.allclose(r @ r.T, np.eye(3))

    def test_opposite_along_x_uses_other_axis(self):
        a = np.array([1.0, 0.0, 0.0])
        assert np.allclose(_rotation_between(a, -a) @ a, -a)


class TestOrthonormalInverse:
    def test_rotation_inverse_is_transpose(self):
        r = _rotation(0.4, -0.7, 0.2)
        assert np.allclose(_orthonormal_inverse(r) @ r, np.eye(3))

    def test_uniform_scale_is_removed(self):
        r = 2.5 * _rotation(0.3, 0.2)
        assert np.allclose(_orthonormal_inverse(r) @ r, np.eye(3))

    def test_degenerate_matrix_returns_identity(self):
        assert np.allclose(_orthonormal_inverse(np.zeros((3, 3))), np.eye(3))


class TestAxisRotation:
    def test_quarter_turn_about_z(self):
        r = _axis_rotation(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        assert np.allclose(r @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def test_axis_is_normalized(self):
        a = _axis_rotation(np.array([0.0, 0.0, 5.0]), 0.8)
        b = _axis_rotation(np.array([0.0, 0.0, 1.0]), 0.8)
        assert np.allclose(a, b)

    def test_zero_axis_is_identity(self):
        assert np.allclose(_axis_rotation(np.zeros(3), 1.0), np.eye(3))

    def test_result_is_a_rotation(self):
        assert is_rotation(_axis_rotation(np.array([1.0, 2.0, 3.0]), 1.1))


class TestScaleRotation:
    def test_gain_two_doubles_angle(self):
        axis = np.array([0.0, 1.0, 0.0])
        r = _axis_rotation(axis, 0.4)
        assert np.allclose(_scale_rotation(r, 2.0), _axis_rotation(axis, 0.8))

    def test_gain_zero_gives_identity(self):
        r = _axis_rotation(np.array([1.0, 0.0, 0.0]), 0.5)
        assert np.allclose(_scale_rotation(r, 0.0), np.eye(3))

    def test_identity_is_returned_unchanged(self):
        assert np.allclose(_scale_rotation(np.eye(3), 3.0), np.eye(3))

    def test_angle_is_clamped(self):
        r = _axis_rotation(np.array([0.0, 0.0, 1.0]), 2.0)
        assert is_rotation(_scale_rotation(r, 10.0))


class TestMeshRotation:
    def test_zero_angles_is_identity(self):
        assert np.allclose(_rotation(0.0, 0.0, 0.0), np.eye(3))

    @pytest.mark.parametrize("angles", [(0.3, 0.1, 0.0), (1.2, -0.8, 0.5), (3.0, 2.0, -1.0)])
    def test_is_a_rotation(self, angles):
        assert is_rotation(_rotation(*angles))

    def test_yaw_turns_around_y(self):
        r = _rotation(np.pi / 2, 0.0)
        assert np.allclose(r @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])
        assert np.allclose(r @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])


class TestNodeMatrix:
    def test_empty_node_is_identity(self):
        assert np.allclose(_node_matrix({}), np.eye(4))

    def test_translation(self):
        m = _node_matrix({"translation": [1, 2, 3]})
        assert np.allclose(m[:3, 3], [1, 2, 3])

    def test_scale(self):
        m = _node_matrix({"scale": [2, 3, 4]})
        assert np.allclose(np.diag(m)[:3], [2, 3, 4])

    def test_identity_quaternion(self):
        assert np.allclose(_node_matrix({"rotation": [0, 0, 0, 1]}), np.eye(4))

    def test_quaternion_quarter_turn_about_z(self):
        s = np.sqrt(0.5)
        m = _node_matrix({"rotation": [0, 0, s, s]})
        assert np.allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0])

    def test_explicit_matrix_is_column_major(self):
        col_major = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]
        m = _node_matrix({"matrix": col_major})
        assert np.allclose(m[:3, 3], [5, 6, 7])
