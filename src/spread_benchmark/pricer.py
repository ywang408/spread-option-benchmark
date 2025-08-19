from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy import pi
from .quadrature import gl_square_nodes_weights
from .utils import kahan_sum_real
from .payoff import payoff_transform

if TYPE_CHECKING:
    from .models.base import SpreadModel


class SpreadPricer:
    """Spread option pricer using FFT-based integration with adaptive quadrature.
    
    Parameters
    ----------
    model : SpreadModel
        The underlying joint characteristic function model.
    omega : tuple[float, float], default (-3.0, 1.0)
        Integration contour parameters (w1, w2).
    """
    
    def __init__(self, model: 'SpreadModel', omega: tuple[float, float] = (-3.0, 1.0)) -> None:
        self.model = model
        self.omega = tuple(map(float, omega))

    def _price_fixed(self, K: float, U: float, n: int) -> float:
        nodes, weights = gl_square_nodes_weights(n, U)
        W = np.outer(weights, weights)

        w1, w2 = self.omega
        u1 = nodes[:, None] + 1j * w1
        u2 = nodes[None, :] + 1j * w2
        U1 = u1 + 0.0 * u2
        U2 = u2 + 0.0 * u1

        b1 = np.log(self.model.S1 / K)
        b2 = np.log(self.model.S2 / K)

        phi = self.model.phi(U1, U2)
        integ = np.exp(1j * (U1 * b1 + U2 * b2)) * phi * payoff_transform(U1, U2)
        const = self.model.discount() * K / (2.0 * pi) ** 2
        return const * kahan_sum_real(W * integ.real)

    def price(self, K: float, tol: float = 1e-9, U0: float = 40.0, h_max: float = 0.02, info: bool = False) -> float | tuple[float, dict]:
        # Step size from oscillations using log-moneyness
        b1 = np.log(self.model.S1 / K)
        b2 = np.log(self.model.S2 / K)
        B = max(abs(b1), abs(b2), abs(b1 + b2), 1e-6)
        h_tgt = min(np.pi / (12.0 * B), h_max)

        U = float(U0)
        n = int(np.ceil(2 * U / h_tgt))
        n += n % 2

        disc_budget = 0.25 * tol
        tail_budget = 0.75 * tol

        v1 = self._price_fixed(K, U, n)
        v2 = self._price_fixed(K, U, 2 * n)
        while abs(v2 - v1) > disc_budget:
            n *= 2
            v1, v2 = v2, self._price_fixed(K, U, 2 * n)

        base = v2
        while True:
            U2 = 1.5 * U
            n2 = int(np.ceil(2 * U2 / h_tgt))
            n2 += n2 % 2
            vU = self._price_fixed(K, U2, n2)
            if abs(vU - base) < tail_budget:
                if info:
                    return vU, {
                        "U": U2,
                        "n": n2,
                        "h": 2 * U2 / n2,
                        "omega": self.omega,
                        "disc_diff": abs(v2 - v1),
                        "tail_diff": abs(vU - base),
                        "T": self.model.T,
                        "r": self.model.r,
                        "q1": self.model.q1,
                        "q2": self.model.q2,
                        "S1": self.model.S1,
                        "S2": self.model.S2,
                    }
                return vU
            U, n, base = U2, n2, vU

    def price_strikes(self, Ks: np.ndarray, tol: float = 1e-9, **kwargs) -> np.ndarray:
        Ks = np.asarray(Ks, dtype=float)
        out = np.empty_like(Ks)
        for i, K in enumerate(Ks):
            out[i] = self.price(K, tol=tol, **kwargs)
        return out
