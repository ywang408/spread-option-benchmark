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

        def _compute_jump_compensator(mu: float, xi: float) -> float:
            return np.exp(mu + 0.5 * xi**2) - 1.0

        def _compute_characteristic_exponent(u: np.ndarray, mu: float, xi: float) -> np.ndarray:
            return np.exp(1j * u * mu - 0.5 * xi**2 * u**2)

        # Quadratic forms for u1 and u2
        u1_sq, u2_sq = u1**2, u2**2
        u1u2 = u1 * u2

        # Diffusion component: bivariate normal
        diffusion = -0.5 * (
            self.sigma1**2 * u1_sq + self.sigma2**2 * u2_sq + 
            2.0 * self.rho * self.sigma1 * self.sigma2 * u1u2
        )

        # Jump components
        idio_jump1 = self.lam1 * (_compute_characteristic_exponent(u1, self.mu1, self.xi1) - 1.0)
        idio_jump2 = self.lam2 * (_compute_characteristic_exponent(u2, self.mu2, self.xi2) - 1.0)
        
        common_jump_inner = np.exp(
            1j * (u1 * self.mu_c1 + u2 * self.mu_c2) - 0.5 * (
                self.xi_c1**2 * u1_sq + self.xi_c2**2 * u2_sq + 
                2.0 * self.rho_y * self.xi_c1 * self.xi_c2 * u1u2
            )
        )
        common_jump = self.lam_c * (common_jump_inner - 1.0)
        
        jumps = idio_jump1 + idio_jump2 + common_jump

        # Risk-neutral drift with jump compensators
        k_common = (_compute_jump_compensator(self.mu_c1, self.xi_c1),
                   _compute_jump_compensator(self.mu_c2, self.xi_c2))
        k_idio = (_compute_jump_compensator(self.mu1, self.xi1),
                 _compute_jump_compensator(self.mu2, self.xi2))

        drift_terms = (
            self.r - self.q1 - 0.5 * self.sigma1**2 - self.lam_c * k_common[0] - self.lam1 * k_idio[0],
            self.r - self.q2 - 0.5 * self.sigma2**2 - self.lam_c * k_common[1] - self.lam2 * k_idio[1]
        )
        drift = 1j * (u1 * drift_terms[0] + u2 * drift_terms[1])

        return np.exp(self.T * (drift + diffusion + jumps))
