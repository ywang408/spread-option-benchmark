from __future__ import annotations
import numpy as np
from .base import SpreadModel

class MertonNormalJumpsModel(SpreadModel):
    """Diffusion + (common & idiosyncratic) normally-distributed jumps (driftless)."""

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

    def phi(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        u1 = np.asarray(u1, dtype=np.complex128)
        u2 = np.asarray(u2, dtype=np.complex128)
        T = self.T

        # Diffusion exponent (driftless)
        s1, s2, rho = self.sigma1, self.sigma2, self.rho
        diff = -0.5 * (
            (s1**2) * u1 * u1 + (s2**2) * u2 * u2 + 2.0 * rho * s1 * s2 * u1 * u2
        )

        # Idiosyncratic normal jumps: λ (exp(i u μ - ½ ξ^2 u^2) - 1)
        idio1 = self.lam1 * (np.exp(1j * u1 * self.mu1 - 0.5 * (self.xi1**2) * u1 * u1) - 1.0)
        idio2 = self.lam2 * (np.exp(1j * u2 * self.mu2 - 0.5 * (self.xi2**2) * u2 * u2) - 1.0)

        # Common bivariate normal jump
        mu_c1, mu_c2 = self.mu_c1, self.mu_c2
        xi_c1, xi_c2, rho_y = self.xi_c1, self.xi_c2, self.rho_y
        common = np.exp(
            1j * (u1 * mu_c1 + u2 * mu_c2)
            - 0.5 * (
                (xi_c1**2) * u1 * u1
                + (xi_c2**2) * u2 * u2
                + 2.0 * rho_y * xi_c1 * xi_c2 * u1 * u2
            )
        )

        jumps = idio1 + idio2 + self.lam_c * (common - 1.0)
        return np.exp(T * (diff + jumps))
