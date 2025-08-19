from __future__ import annotations
import numpy as np


def kahan_sum_real(a: np.ndarray) -> float:
    """Kahan compensated sum over real-valued array a."""
    s = 0.0
    c = 0.0
    af = np.ravel(np.asarray(a, dtype=np.float64))
    for x in af:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return float(s)
