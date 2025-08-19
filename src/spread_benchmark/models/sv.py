from __future__ import annotations
import numpy as np
from .base import SpreadModel

class HestonSVModel(SpreadModel):
    """Bivariate Heston-type stochastic volatility (shared variance factor)."""

    def __init__(
        self,
        sigma1: float,
        sigma2: float,
        sigmav: float,
        rho12: float,
        rho1v: float,
        rho2v: float,
        v0: float,
        kappa: float,
        mu: float,
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
        self.sigmav = float(sigmav)
        self.rho12 = float(rho12)
        self.rho1v = float(rho1v)
        self.rho2v = float(rho2v)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.mu = float(mu)

    def phi(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        u1 = np.asarray(u1, dtype=np.complex128)
        u2 = np.asarray(u2, dtype=np.complex128)

        r, q1, q2 = self.r, self.q1, self.q2
        s1, s2 = self.sigma1, self.sigma2
        rv = self.sigmav
        rho12, rho1v, rho2v = self.rho12, self.rho1v, self.rho2v
        v0, kappa, mu, T = self.v0, self.kappa, self.mu, self.T

        # zeta = -0.5 * [ s1^2 u1(u1+i) + s2^2 u2(u2+i) + 2 rho12 s1 s2 u1 u2 ]
        zeta = -0.5 * (
            (s1**2) * u1 * (u1 + 1j) +
            (s2**2) * u2 * (u2 + 1j) +
            2.0 * rho12 * s1 * s2 * u1 * u2
        )

        gamma = kappa - 1j * rv * (rho1v * s1 * u1 + rho2v * s2 * u2)
        theta = np.sqrt(gamma * gamma - 2.0 * (rv**2) * zeta)

        exp_neg_theta_T = np.exp(-theta * T)
        tmp = 2.0 * theta - (theta - gamma) * (1.0 - exp_neg_theta_T)

        term1 = (2.0 * zeta) * (1.0 - exp_neg_theta_T) / tmp * v0
        term2 = 1j * ((r - q1) * u1 + (r - q2) * u2) * T
        term3 = -kappa * mu / (rv**2) * (2.0 * np.log(tmp / (2.0 * theta)) + (theta - gamma) * T)

        return np.exp(term1 + term2 + term3)
