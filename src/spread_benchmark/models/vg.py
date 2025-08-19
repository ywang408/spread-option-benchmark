from __future__ import annotations
import numpy as np
from .base import SpreadModel


def _logG(z: np.ndarray, a_plus: float, a_minus: float) -> np.ndarray:
    # = -log(1 - i z/a+) - log(1 + i z/a-)
    return -np.log1p(-1j * z / a_plus) - np.log1p(1j * z / a_minus)


class AlphaVGModel(SpreadModel):
    """α–Variance-Gamma joint characteristic function (driftless)."""

    def __init__(
        self,
        lam: float,
        alpha: float,
        a_plus: float,
        a_minus: float,
        *,
        S1: float,
        S2: float,
        T: float,
        r: float = 0.0,
        q1: float = 0.0,
        q2: float = 0.0,
    ):
        super().__init__(S1=S1, S2=S2, T=T, r=r, q1=q1, q2=q2)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)

    def phi(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        lam, alpha = self.lam, self.alpha
        a_plus, a_minus, T = self.a_plus, self.a_minus, self.T

        expo = (lam * alpha) * _logG(u1 + u2, a_plus, a_minus)
        expo += (
            lam
            * (1.0 - alpha)
            * (_logG(u1, a_plus, a_minus) + _logG(u2, a_plus, a_minus))
        )
        return np.exp(T * expo)
