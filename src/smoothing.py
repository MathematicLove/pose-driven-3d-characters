from __future__ import annotations

import math

import numpy as np


class _LowPass:
    """Exponential low-pass filter."""

    def __init__(self) -> None:
        self.y: np.ndarray | None = None

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Filter one sample."""
        if self.y is None:
            self.y = x.astype(np.float64)
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y

    def reset(self) -> None:
        """Clear filter state."""
        self.y = None


class OneEuroFilter:
    """One-euro cursor filter."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Smoothing factor from cutoff."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        """Clear filter state."""
        self._x.reset()
        self._dx.reset()
        self._prev = None

    def __call__(self, point, dt: float | None = None) -> np.ndarray:
        """Filter one point."""
        x = np.asarray(point, dtype=np.float64)
        if dt is not None and dt > 1e-6:
            self.freq = 1.0 / dt

        prev = self._prev if self._prev is not None else x
        dx = (x - prev) * self.freq
        self._prev = x

        edx = self._dx(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(edx))
        return self._x(x, self._alpha(cutoff, self.freq))


class LandmarkFilter:
    """One-euro filter for landmarks."""

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.03) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev: np.ndarray | None = None
        self.freq = 30.0

    def reset(self) -> None:
        """Clear filter state."""
        self._x.reset()
        self._dx.reset()
        self._prev = None

    def __call__(self, points: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Filter all keypoints."""
        x = np.asarray(points, dtype=np.float64)
        if dt is not None and dt > 1e-6:
            self.freq = 1.0 / dt
        if self._prev is None or self._prev.shape != x.shape:
            self.reset()
            self._prev = x
            return self._x(x, 1.0)

        dx = (x - self._prev) * self.freq
        self._prev = x
        edx = self._dx(dx, OneEuroFilter._alpha(self.d_cutoff, self.freq))
        speed = float(np.linalg.norm(edx, axis=-1).mean())
        cutoff = self.min_cutoff + self.beta * speed
        return self._x(x, OneEuroFilter._alpha(cutoff, self.freq))

    d_cutoff = 1.0


class ScalarEMA:
    """Scalar exponential moving average."""

    def __init__(self, alpha: float = 0.35, value: float = 0.0) -> None:
        self.alpha = alpha
        self.value = value

    def __call__(self, target: float) -> float:
        """Advance toward target."""
        self.value += self.alpha * (target - self.value)
        return self.value

    def set(self, value: float) -> None:
        """Force current value."""
        self.value = value
