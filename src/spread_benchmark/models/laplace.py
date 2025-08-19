from __future__ import annotations
import numpy as np
from .base import SpreadModel

class LaplaceJumpsModel(SpreadModel):
    """Diffusion + Laplace (double-exponential) jumps (driftless)."""

    def __init__(
        self,
        sigma1: float,
        sigma2: float,
        rho: float,
        lam_common: float,
        mu_common: tuple[float, float],
        sigma_common: tuple[float, float],
        rho_y: float,
        lam_id1: float,
        lam_id2: float,
        mu_id1: float,
        mu_id2: float,
        sigma_id1: float,
        sigma_id2: float,
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
        self.lam_c = float(lam_common)
        self.mu_c1, self.mu_c2 = map(float, mu_common)
        self.xi_c1, self.xi_c2 = map(float, sigma_common)
        self.rho_y = float(rho_y)
        self.lam1 = float(lam_id1)
        self.lam2 = float(lam_id2)
        self.mu1 = float(mu_id1)
        self.mu2 = float(mu_id2)
        self.xi1 = float(sigma_id1)
        self.xi2 = float(sigma_id2)

    @staticmethod
    def _laplace_char_1d(u: np.ndarray, alpha: float, xi: float) -> np.ndarray:
        """φ(u) = 1 / (1 - i α u + ½ ξ² u²)"""
        return 1.0 / (1.0 - 1j * alpha * u + 0.5 * (xi**2) * u * u)

    def phi(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        u1 = np.asarray(u1, dtype=np.complex128)
        u2 = np.asarray(u2, dtype=np.complex128)
        T = self.T

        # Diffusion (driftless)
        s1, s2, rho = self.sigma1, self.sigma2, self.rho
        diff = -0.5 * (
            (s1**2) * u1 * u1 + (s2**2) * u2 * u2 + 2.0 * rho * s1 * s2 * u1 * u2
        )

        # Idiosyncratic Laplace jumps: λ (φ(u) - 1)
        j1 = self._laplace_char_1d(u1, self.mu1, self.xi1)
        j2 = self._laplace_char_1d(u2, self.mu2, self.xi2)
        idio = self.lam1 * (j1 - 1.0) + self.lam2 * (j2 - 1.0)

        # Common bivariate Laplace jump:
        denom_common = (
            1.0
            - 1j * (self.mu_c1 * u1 + self.mu_c2 * u2)
            + 0.5 * (
                (self.xi_c1**2) * u1 * u1
                + (self.xi_c2**2) * u2 * u2
                + 2.0 * self.rho_y * self.xi_c1 * self.xi_c2 * u1 * u2
            )
        )
        common = 1.0 / denom_common
        jumps = idio + self.lam_c * (common - 1.0)

        return np.exp(T * (diff + jumps))
