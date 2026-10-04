import numpy as np
import pytest

from smoothing import LandmarkFilter, OneEuroFilter, ScalarEMA, _LowPass


class TestLowPass:
    def test_first_sample_passes_through(self):
        f = _LowPass()
        assert np.allclose(f(np.array([3.0, 4.0]), 0.5), [3.0, 4.0])

    def test_alpha_one_follows_input(self):
        f = _LowPass()
        f(np.array([0.0]), 1.0)
        assert f(np.array([10.0]), 1.0)[0] == 10.0

    def test_blends_with_previous(self):
        f = _LowPass()
        f(np.array([0.0]), 0.5)
        assert f(np.array([10.0]), 0.5)[0] == pytest.approx(5.0)

    def test_reset_forgets_state(self):
        f = _LowPass()
        f(np.array([1.0]), 0.5)
        f.reset()
        assert f(np.array([9.0]), 0.1)[0] == 9.0


class TestOneEuroFilter:
    def test_alpha_is_between_zero_and_one_and_grows_with_cutoff(self):
        low = OneEuroFilter._alpha(0.5, 30.0)
        high = OneEuroFilter._alpha(5.0, 30.0)
        assert 0.0 < low < high < 1.0

    def test_constant_signal_stays_constant(self):
        f = OneEuroFilter()
        for _ in range(20):
            out = f([100.0, 200.0])
        assert np.allclose(out, [100.0, 200.0])

    def test_first_point_is_unchanged(self):
        assert np.allclose(OneEuroFilter()([5.0, 6.0]), [5.0, 6.0])

    def test_jitter_is_reduced(self):
        rng = np.random.default_rng(1)
        f = OneEuroFilter()
        noisy = 50 + rng.normal(0, 3, size=(200, 2))
        out = np.array([f(p, dt=1 / 30) for p in noisy])
        assert out[50:].std() < noisy[50:].std()

    def test_dt_updates_frequency(self):
        f = OneEuroFilter()
        f([0.0, 0.0], dt=0.5)
        assert f.freq == pytest.approx(2.0)

    def test_reset(self):
        f = OneEuroFilter()
        f([1.0, 1.0])
        f.reset()
        assert np.allclose(f([8.0, 8.0]), [8.0, 8.0])


class TestLandmarkFilter:
    def test_first_frame_unchanged(self):
        pts = np.arange(34, dtype=float).reshape(17, 2)
        assert np.allclose(LandmarkFilter()(pts), pts)

    def test_constant_pose_stays_constant(self):
        pts = np.random.default_rng(2).random((17, 2)) * 100
        f = LandmarkFilter()
        for _ in range(15):
            out = f(pts)
        assert np.allclose(out, pts)

    def test_shape_change_resets_state(self):
        f = LandmarkFilter()
        f(np.zeros((17, 2)))
        new = np.ones((17, 3)) * 4
        assert np.allclose(f(new), new)

    def test_output_keeps_shape(self):
        out = LandmarkFilter()(np.zeros((17, 3)))
        assert out.shape == (17, 3)


class TestScalarEMA:
    def test_moves_toward_target(self):
        ema = ScalarEMA(alpha=0.5, value=0.0)
        assert ema(10.0) == pytest.approx(5.0)
        assert ema(10.0) == pytest.approx(7.5)

    def test_converges(self):
        ema = ScalarEMA(alpha=0.35)
        for _ in range(100):
            ema(3.0)
        assert ema.value == pytest.approx(3.0, abs=1e-6)

    def test_set(self):
        ema = ScalarEMA()
        ema.set(4.0)
        assert ema.value == 4.0
