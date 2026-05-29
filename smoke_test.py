#!/usr/bin/env python3
"""Fast local smoke test for the 2D benchmark generator.

Part A shows the integral *gradually converging* as the node count ``n`` grows
at a fixed box, measured against the committed benchmark value.
Part B exercises the resumable gen_all plumbing (per-case JSON write, skip on
restart, CSV assembly) at a loose tol so it runs in seconds.

Run:  uv run python smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from spread_benchmark import SpreadPricer  # noqa: E402
from spread_benchmark import generate as gen  # noqa: E402

# Committed benchmark values at the K_bench=2.0 strike (results/2d/*_benchmark.csv).
BENCH_K2 = {"gbm_2d": 7.5423238958548104, "vg_2d": 9.727457905410768}


def convergence_table(model: str, K: float = 2.0, U: float = 40.0):
    pricer = SpreadPricer(gen.build_model(model), omega=gen.OMEGA)
    bench = BENCH_K2[model]
    print(f"\n{model}  K={K}  (fixed U={U}); benchmark={bench!r}")
    print(f"  {'n':>6}  {'price':>22}  {'|price-bench|':>14}")
    prev = None
    for n in (128, 256, 512, 1024, 2048):
        v = pricer._price_fixed(K, U, n)
        err = abs(v - bench)
        trend = "" if prev is None else ("  v" if err < prev else "  ^!")
        print(f"  {n:>6}  {v:>22.16f}  {err:>14.3e}{trend}")
        prev = err


def plumbing_test():
    print("\n--- resume / assemble plumbing (tol=1e-6, gbm_2d K=2.0) ---")
    # fixed dir so a second run demonstrates cross-invocation resume (and is fast)
    out = os.path.join(tempfile.gettempdir(), "spread_bench_smoke")
    os.makedirs(out, exist_ok=True)
    r1 = gen.gen_case("gbm_2d", 2.0, tol=1e-6, out_dir=out)
    cp = gen.case_path(out, "gbm_2d", 2.0)
    assert os.path.exists(cp), "case JSON not written"
    # second call must hit the cache (resume), not recompute
    r2 = gen.gen_case("gbm_2d", 2.0, tol=1e-6, out_dir=out)
    assert r2["price"] == r1["price"], "cached read differs"
    csv_path = gen.assemble("gbm_2d", out)
    lines = open(csv_path).read().splitlines()
    print(f"  case file : {os.path.basename(cp)}")
    print(f"  assembled : {csv_path}")
    print("  csv head  :", lines[0])
    print("  csv row   :", lines[1][:90], "...")
    print(f"  price={r1['price']!r}  eps_obs={r1['eps_obs']:.2e}  met_tol={r1['met_tol']}")


if __name__ == "__main__":
    for m in ("gbm_2d", "vg_2d"):
        convergence_table(m)
    plumbing_test()
    print("\nsmoke OK")
