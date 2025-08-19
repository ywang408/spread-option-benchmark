from __future__ import annotations
import numpy as np
from .base import SpreadModel

class GBMModel(SpreadModel):
    """Bivariate GBM diffusion under Q (drift handled via forwards/discount)."""

    def __init__(
        self,
        sigma1: float,
        sigma2: float,
        rho: float,
        *,
        S1: float,
        S2: float,
        T: float,
        r: float = 0.0,
        q1: float = 0.0,
        q2: float = 0.0,
    ):
        super().__init__(S1=S1, S2=S2, T=T, r=r, q1=q1, q2=q2)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.rho = float(rho)

    def phi(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        u1 = np.asarray(u1, dtype=np.complex128)
        u2 = np.asarray(u2, dtype=np.complex128)
        s1, s2, rho, T = self.sigma1, self.sigma2, self.rho, self.T
        quad = -0.5 * ((s1**2) * u1 * u1 + (s2**2) * u2 * u2 + 2.0 * rho * s1 * s2 * u1 * u2)
        return np.exp(T * quad)
