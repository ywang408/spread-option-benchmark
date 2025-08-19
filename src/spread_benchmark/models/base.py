from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SpreadModel(ABC):
    """Base class for joint characteristic functions φ(u1, u2; T).

    The model owns market state (S1, S2, r, q1, q2) and maturity T.
    """

    def __init__(
        self,
        S1: float,
        S2: float,
        T: float,
        r: float = 0.0,
        q1: float = 0.0,
        q2: float = 0.0,
    ):
        self.S1 = float(S1)
        self.S2 = float(S2)
        self.T = float(T)
        self.r = float(r)
        self.q1 = float(q1)
        self.q2 = float(q2)

    # --- Convenience for pricing layer ---
    def discount(self) -> float:
        return float(np.exp(-self.r * self.T))

    def forwards(self) -> tuple[float, float]:
        """Return forward prices (F1, F2) = (S1*exp((r-q1)*T), S2*exp((r-q2)*T))."""
        F1 = self.S1 * np.exp((self.r - self.q1) * self.T)
        F2 = self.S2 * np.exp((self.r - self.q2) * self.T)
        return float(F1), float(F2)

    def set_market(
        self,
        *,
        S1: float | None = None,
        S2: float | None = None,
        T: float | None = None,
        r: float | None = None,
        q1: float | None = None,
        q2: float | None = None,
    ) -> None:
        for param, value in [
            ("S1", S1),
            ("S2", S2),
            ("T", T),
            ("r", r),
            ("q1", q1),
            ("q2", q2),
        ]:
            if value is not None:
                setattr(self, param, float(value))

    @abstractmethod
    def phi(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """Return φ(u1, u2; self.T) on a complex grid."""
        raise NotImplementedError
