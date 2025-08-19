import time
import numpy as np
from spread_benchmark import AlphaVGModel, SpreadPricer


def run():
    model = AlphaVGModel(
        lam=10.0,
        alpha=0.4,
        a_plus=20.4499,
        a_minus=24.4499,
        S1=100.0,
        S2=96.0,
        T=1.0,
        r=0.1,
        q1=0,
        q2=0,
    )
    pricer = SpreadPricer(model)

    Ks = np.arange(0.4, 4.1, 0.4, dtype=float)

    for K in Ks:
        t0 = time.perf_counter()
        v = pricer.price(K, tol=1e-6, U0=40.0, h_max=0.02, info=False)
        dt = time.perf_counter() - t0
        print(f"{dt} used, {K}: {v}")


if __name__ == "__main__":
    run()
