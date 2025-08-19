from __future__ import annotations
import numpy as np

# ---- special functions (stable) ----
try:
    # SciPy is fast and vectorized; supports complex input
    from scipy.special import loggamma  # complex log Γ
except Exception:  # fallback to mpmath (slower)
    import mpmath as mp

    def loggamma(z: np.ndarray) -> np.ndarray:
        # vectorized wrapper returning a numpy array
        zf = np.asarray(z, dtype=np.complex128)
        out = np.empty_like(zf)
        it = np.nditer(zf, flags=["multi_index", "refs_ok"])
        for v in it:
            out[it.multi_index] = complex(mp.loggamma(complex(v)))
        return out


def payoff_transform(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Hurd–Zhou payoff transform (stable via log-gamma).

    P̂(u1,u2) = Γ(i(u1+u2)-1) Γ(-iu2) / Γ(1+iu1)
    """
    return np.exp(
        loggamma(1j * (u1 + u2) - 1.0) + loggamma(-1j * u2) - loggamma(1.0 + 1j * u1)
    )
