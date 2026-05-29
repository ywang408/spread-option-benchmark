#!/usr/bin/env python3
"""Thin CLI wrapper around :mod:`spread_benchmark.generate`.

    python gen_all.py --out-dir ./benchmarks/2d
    python gen_all.py --model gbm_2d --strike 2.0 --out-dir ./benchmarks/2d
    python gen_all.py --tol 1e-10 --out-dir ./benchmarks/2d

After ``pip install`` the same entry points are importable directly:

    from spread_benchmark import gen_all, gen_2d, gen_case
"""
try:
    from spread_benchmark.generate import main
except ModuleNotFoundError:  # running from a clone without installing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from spread_benchmark.generate import main

if __name__ == "__main__":
    main()
