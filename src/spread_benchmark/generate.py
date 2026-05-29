"""Resumable, Drive-backed 2D spread-option benchmark generation.

Importable after ``pip install`` (e.g. on Colab):

    from spread_benchmark import gen_all, gen_2d, gen_case
    gen_all(out_dir="/content/drive/MyDrive/spread_benchmark/2d")
    gen_2d("/content/drive/MyDrive/spread_benchmark/2d", models=["vg_2d"])
    gen_case("gbm_2d", 2.0, out_dir="/content/drive/MyDrive/spread_benchmark/2d")

Each ``(model, strike)`` case is priced with the stop-rule adaptive quadrature
(:meth:`SpreadPricer.price_benchmark`) and written to its own JSON file under
``<out_dir>/_cases/`` the moment it finishes; a restart re-validates and skips
completed cases, so an interrupted session loses no work. After each case the
per-model ``<out_dir>/<model>_benchmark.csv`` is rebuilt -- the format the v3
error tables and convergence/error figures consume (columns ``model, K, price``
plus the recorded-precision diagnostics).

Only 2D is generated here: the committed 3D benchmarks already record their
achieved precision (``recorded_precision``/``eps_obs``) and are not regenerated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone

from .models.gbm import GBMModel
from .models.sv import HestonSVModel
from .models.vg import AlphaVGModel
from .models.merton import MertonNormalJumpsModel
from .models.laplace import LaplaceJumpsModel
from .pricer import SpreadPricer

# --------------------------- benchmark definition ---------------------------
# Fixed contour / adaptive box and target tol. These define the benchmark; the
# stop rule records the achieved precision per case when tol cannot be met.
OMEGA = (-3.0, 1.0)
U0 = 40.0
H_MAX = 0.02
DEFAULT_TOL = 1e-11

STRIKES = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]

# Model parameters (mirror config/processes.toml in the main repo).
PARAMS = {
    "gbm_2d": {"type": "GBM2d", "s0": [100.0, 96.0], "r": 0.1, "q": [0.05, 0.05],
               "sigma": [0.2, 0.1], "rho": 0.5, "tau": 1.0},
    "sv_2d": {"type": "SV2d", "s0": [100.0, 96.0], "r": 0.1, "q": [0.05, 0.05],
              "sigma": [1.0, 0.5], "sigma_v": 0.05, "rho": 0.5,
              "rho_v": [-0.5, 0.25], "v0": 0.04, "kappa": 1.0, "mu": 0.04,
              "tau": 1.0},
    "vg_2d": {"type": "VG2d", "s0": [100.0, 96.0], "r": 0.1,
              "a": [20.4499, 24.4499], "alpha": 0.4, "lambda": 10.0, "tau": 1.0},
    "merton_2d": {"type": "Merton2d", "s0": [100.0, 96.0], "r": 0.1,
                  "q": [0.03, 0.05], "sigma": [0.15, 0.1], "lambda0": 0.2,
                  "lambda": [0.2, 0.1], "alpha": [0.06, 0.03],
                  "alpha_j": [0.02, -0.07], "xi": [0.03, 0.09],
                  "xi_j": [0.06, 0.01], "rho": 0.5, "rho_j": -0.8, "tau": 1.0},
    "lj_2d": {"type": "LJump2d", "s0": [100.0, 96.0], "r": 0.1,
              "q": [0.03, 0.05], "sigma": [0.15, 0.1], "lambda0": 0.2,
              "lambda": [0.2, 0.1], "alpha": [0.06, 0.03],
              "alpha_j": [0.02, -0.07], "xi": [0.03, 0.09],
              "xi_j": [0.06, 0.01], "rho": 0.5, "rho_j": -0.8, "tau": 1.0},
}
MODELS = list(PARAMS)

CSV_COLUMNS = [
    "model", "K", "price", "elapsed_sec", "tol", "eps_obs",
    "recorded_precision", "met_tol", "dominant", "disc_err", "tail_err",
    "price_base", "price_tail", "U", "n", "h", "U_tail", "n_tail", "h_tail",
    "disc_iters", "tail_iters", "disc_stalled", "tail_stalled",
    "omega0", "omega1", "param_hash", "timestamp",
]


def build_model(name):
    """Construct a SpreadModel for a model name in :data:`PARAMS`."""
    p = PARAMS[name]
    t = p["type"]
    s1, s2 = p["s0"]
    q = p.get("q", [0.0, 0.0])
    mk = dict(S1=s1, S2=s2, T=p["tau"], r=p["r"], q1=q[0], q2=q[1])
    if t == "GBM2d":
        return GBMModel(sigma1=p["sigma"][0], sigma2=p["sigma"][1], rho=p["rho"], **mk)
    if t == "VG2d":
        return AlphaVGModel(lam=p["lambda"], alpha=p["alpha"],
                            a_plus=p["a"][0], a_minus=p["a"][1], **mk)
    if t == "SV2d":
        return HestonSVModel(
            sigma1=p["sigma"][0], sigma2=p["sigma"][1], sigmav=p["sigma_v"],
            rho12=p["rho"], rho1v=p["rho_v"][0], rho2v=p["rho_v"][1],
            v0=p["v0"], kappa=p["kappa"], mu=p["mu"], **mk)
    cls = MertonNormalJumpsModel if t == "Merton2d" else LaplaceJumpsModel
    return cls(
        sigma1=p["sigma"][0], sigma2=p["sigma"][1], rho=p["rho"],
        lam_common=p["lambda0"], mu_common=tuple(p["alpha"]),
        sigma_common=tuple(p["xi"]), rho_y=p["rho_j"],
        lam_id1=p["lambda"][0], lam_id2=p["lambda"][1],
        mu_id1=p["alpha_j"][0], mu_id2=p["alpha_j"][1],
        sigma_id1=p["xi_j"][0], sigma_id2=p["xi_j"][1], **mk)


def param_hash(name, tol):
    """Identity of a case's inputs; a restart recomputes if these change."""
    blob = json.dumps({"p": PARAMS[name], "omega": OMEGA, "U0": U0,
                       "h_max": H_MAX, "tol": tol}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _cases_dir(out_dir):
    return os.path.join(out_dir, "_cases")


def case_path(out_dir, name, K):
    return os.path.join(_cases_dir(out_dir), f"{name}_K={K:g}.json")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # some FUSE-mounted Drives reject fsync
    os.replace(tmp, path)


def _load_valid_case(path, expect_hash):
    """Return a cached row only if it is complete and matches the inputs."""
    try:
        with open(path) as fh:
            row = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("param_hash") != expect_hash or "price" not in row:
        return None
    return row


def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return repr(v)   # full round-trip precision
    return v


def gen_case(name, K, *, out_dir, tol=DEFAULT_TOL, force=False, verbose=True):
    """Price one case unless a valid file for the same inputs already exists."""
    K = float(K)
    cp = case_path(out_dir, name, K)
    h = param_hash(name, tol)
    if not force:
        cached = _load_valid_case(cp, h)
        if cached is not None:
            if verbose:
                print(f"[skip] {name} K={K} (cached)", flush=True)
            return cached

    pricer = SpreadPricer(build_model(name), omega=OMEGA)
    t0 = time.perf_counter()
    res = pricer.price_benchmark(K, tol=tol, U0=U0, h_max=H_MAX)
    row = res.as_row()
    row.update(
        model=name, K=K, price=res.price_tail,
        elapsed_sec=time.perf_counter() - t0,
        omega0=OMEGA[0], omega1=OMEGA[1], param_hash=h,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(cp, row)
    if verbose:
        flag = "ok" if res.met_tol else f"STOP eps={res.eps_obs:.2e}"
        print(f"[done] {name} K={K}: {res.price_tail:.12f} "
              f"({row['elapsed_sec']:.1f}s, {flag})", flush=True)
    return row


def assemble(name, out_dir):
    """Rebuild <model>_benchmark.csv from that model's case files (sorted by K)."""
    rows = []
    for K in STRIKES:
        cp = case_path(out_dir, name, K)
        if os.path.exists(cp):
            try:
                with open(cp) as fh:
                    rows.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                pass
    if not rows:
        return None
    rows.sort(key=lambda r: r["K"])
    out = os.path.join(out_dir, f"{name}_benchmark.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in CSV_COLUMNS})
    return out


def gen_2d(out_dir, models=None, strikes=None, tol=DEFAULT_TOL):
    """Generate every requested case, resumable, assembling CSVs as it goes."""
    models = models or MODELS
    strikes = STRIKES if strikes is None else strikes
    os.makedirs(_cases_dir(out_dir), exist_ok=True)
    for name in models:
        for K in strikes:
            gen_case(name, K, out_dir=out_dir, tol=tol)
            assemble(name, out_dir)  # keep the CSV live for safe interruption
        print(f"  -> {assemble(name, out_dir)}", flush=True)


def gen_all(out_dir="./benchmarks/2d", tol=DEFAULT_TOL):
    """All 2D benchmarks. (3D already records its tol and is not regenerated.)"""
    gen_2d(out_dir, tol=tol)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="spread-benchmark",
        description="Generate resumable 2D spread-option benchmarks.")
    ap.add_argument("--out-dir", default="./benchmarks/2d")
    ap.add_argument("--models", nargs="+", default=None, choices=MODELS)
    ap.add_argument("--model", default=None, choices=MODELS)
    ap.add_argument("--strike", type=float, default=None)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.model and args.strike is not None:
        gen_case(args.model, args.strike, out_dir=args.out_dir, tol=args.tol,
                 force=args.force)
        assemble(args.model, args.out_dir)
    else:
        gen_2d(args.out_dir, models=args.models, tol=args.tol)


if __name__ == "__main__":
    main()
