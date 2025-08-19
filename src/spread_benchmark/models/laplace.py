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
        """
        Characteristic function for bivariate log-returns under diffusion + Laplace jumps.
        
        Returns φ(u1, u2) = E[exp(i*u1*X1 + i*u2*X2)] where X1, X2 are log-returns.
        """
        u1 = np.asarray(u1, dtype=np.complex128)
        u2 = np.asarray(u2, dtype=np.complex128)

        def laplace_char_2d(mu1: float, mu2: float, xi1: float, xi2: float, rho: float) -> np.ndarray:
            """Bivariate Laplace characteristic function φ_c(u1,u2)"""
            denominator = (
                1.0 - 1j * (mu1 * u1 + mu2 * u2) 
                + 0.5 * (xi1**2 * u1**2 + xi2**2 * u2**2 + 2.0 * rho * xi1 * xi2 * u1 * u2)
            )
            return 1.0 / denominator

        def jump_compensation(mu: float, xi: float) -> float:
            """Compute E[e^Y] - 1 for jump compensation at u = -i"""
            return 1.0 / (1.0 - mu - 0.5 * xi**2) - 1.0

        def diffusion_term() -> np.ndarray:
            """Compute the diffusion contribution to the characteristic function"""
            return -0.5 * (
                self.sigma1**2 * u1**2 + 
                self.sigma2**2 * u2**2 + 
                2.0 * self.rho * self.sigma1 * self.sigma2 * u1 * u2
            )

        def idiosyncratic_jumps() -> np.ndarray:
            """Compute idiosyncratic jump contributions λ_i (φ_i(u_i) - 1)"""
            char_1 = self._laplace_char_1d(u1, self.mu1, self.xi1)
            char_2 = self._laplace_char_1d(u2, self.mu2, self.xi2)
            return self.lam1 * (char_1 - 1.0) + self.lam2 * (char_2 - 1.0)

        def common_jumps() -> np.ndarray:
            """Compute common jump contribution λ_c (φ_c(u1,u2) - 1)"""
            char_common = laplace_char_2d(self.mu_c1, self.mu_c2, self.xi_c1, self.xi_c2, self.rho_y)
            return self.lam_c * (char_common - 1.0)

        def martingale_drift() -> np.ndarray:
            """Compute drift terms to enforce martingale property"""
            k_c1 = jump_compensation(self.mu_c1, self.xi_c1)
            k_c2 = jump_compensation(self.mu_c2, self.xi_c2)
            k_z1 = jump_compensation(self.mu1, self.xi1)
            k_z2 = jump_compensation(self.mu2, self.xi2)

            gamma1 = (self.r - self.q1) - 0.5 * self.sigma1**2 - self.lam_c * k_c1 - self.lam1 * k_z1
            gamma2 = (self.r - self.q2) - 0.5 * self.sigma2**2 - self.lam_c * k_c2 - self.lam2 * k_z2
            
            return 1j * (gamma1 * u1 + gamma2 * u2)

        diffusion = diffusion_term()
        idiosyncratic_jumps_term = idiosyncratic_jumps()
        common_jumps_term = common_jumps()
        martingale_drift_term = martingale_drift()

        exponent = self.T * (martingale_drift_term + diffusion + idiosyncratic_jumps_term + common_jumps_term)
        return np.exp(exponent)
