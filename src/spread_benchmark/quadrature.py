from __future__ import annotations
import numpy as np


def gl_square_nodes_weights(n: int, U: float):
    """Gauss–Legendre nodes/weights on [-U, U].

    Returns
    -------
    nodes : (n,) ndarray
        Nodes mapped from [-1,1] to [-U,U].
    weights : (n,) ndarray
        Weights scaled by U.
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    return U * x, U * w
