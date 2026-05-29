"""Adaptive box/grid refinement with a stop rule and recorded precision.

Shared by the 2D spread and the nD basket benchmark drivers. Given a
fixed-grid quadrature callable ``price_fixed(U, n) -> float`` and the
oscillation bound ``B`` (max over subset-sums of |log-moneyness|), it

  1. refines the node count ``n`` at a fixed box ``[-U, U]^d`` until the
     Richardson-style indicator ``|v(2n) - v(n)|`` falls under the
     discretization budget, and
  2. grows the box ``U <- rho * U`` (rescaling ``n`` to hold the effective
     spacing ``h ~ h_tgt`` constant) until ``|v(U') - v(U)|`` falls under the
     tail budget.

Unlike a plain ``while`` loop, each phase has a **stop rule**: it gives up
after ``max_*_iters`` refinements, or earlier if the indicator stops
decreasing (the round-off floor has been reached). When the target ``tol`` is
not met, the achieved precision ``eps_obs`` is recorded per case rather than
looping forever.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable


@dataclass
class AdaptiveResult:
    price_base: float       # value at the box before the final growth
    price_tail: float       # value at the final (largest) box -> reported price
    disc_err: float         # |v(2n) - v(n)| at fixed U (discretization indicator)
    tail_err: float         # |v(U') - v(U)| (truncation indicator)
    eps_obs: float          # max(disc_err, tail_err): achieved precision
    met_tol: bool           # eps_obs <= tol
    recorded_precision: float   # tol if met else eps_obs (the binding accuracy)
    dominant: str           # 'disc' or 'tail' (larger error source)
    tol: float
    U: float
    n: int
    h: float
    U_tail: float
    n_tail: int
    h_tail: float
    disc_iters: int
    tail_iters: int
    disc_stalled: bool      # disc phase quit on the round-off floor, not tol
    tail_stalled: bool      # tail phase quit on the round-off floor, not tol

    def as_row(self) -> dict:
        return asdict(self)


def _even_ceil(x: float) -> int:
    n = int(math.ceil(x))
    n += n % 2
    return max(n, 2)


def adaptive_price(
    price_fixed: Callable[[float, int], float],
    B: float,
    *,
    tol: float = 1e-11,
    U0: float = 40.0,
    h_max: float = 0.02,
    rho: float = 1.5,
    disc_frac: float = 0.25,
    max_disc_iters: int = 12,
    max_tail_iters: int = 16,
) -> AdaptiveResult:
    B = max(abs(B), 1e-6)
    h_tgt = min(math.pi / (12.0 * B), h_max)
    disc_budget = disc_frac * tol
    tail_budget = (1.0 - disc_frac) * tol

    # ----- discretization control at the fixed initial box -----
    U = float(U0)
    n = _even_ceil(2.0 * U / h_tgt)
    v1 = price_fixed(U, n)
    v2 = price_fixed(U, 2 * n)
    disc_err = abs(v2 - v1)
    disc_iters = 0
    disc_stalled = False
    while disc_err > disc_budget and disc_iters < max_disc_iters:
        n *= 2
        v_new = price_fixed(U, 2 * n)
        new_err = abs(v_new - v2)
        v2 = v_new
        disc_iters += 1
        if new_err >= disc_err:          # round-off floor: refining no longer helps
            disc_err = new_err
            disc_stalled = True
            break
        disc_err = new_err
    n_fine = 2 * n

    # ----- tail control: grow the box at ~constant spacing -----
    # The reported pair is always (value at the previous box, value at the
    # current box); each iteration records it, so break and loop-exhaust both
    # leave the right values behind.
    v_prev, U_prev, n_prev = v2, U, n_fine
    tail_err = math.inf
    prev_tail_err = math.inf
    tail_iters = 0
    tail_stalled = False
    price_base, U_base, n_base = v2, U, n_fine
    price_tail, U_tail, n_tail = v2, U, n_fine

    while tail_iters < max_tail_iters:
        U_cur = rho * U_prev
        n_cur = _even_ceil(2.0 * U_cur / h_tgt)
        v_cur = price_fixed(U_cur, n_cur)
        tail_err = abs(v_cur - v_prev)
        tail_iters += 1

        price_base, U_base, n_base = v_prev, U_prev, n_prev
        price_tail, U_tail, n_tail = v_cur, U_cur, n_cur

        if tail_err <= tail_budget:
            break
        if tail_err >= prev_tail_err:                # round-off floor reached
            tail_stalled = True
            break
        prev_tail_err = tail_err
        v_prev, U_prev, n_prev = v_cur, U_cur, n_cur

    eps_obs = max(disc_err, tail_err)
    met = eps_obs <= tol
    return AdaptiveResult(
        price_base=price_base,
        price_tail=price_tail,
        disc_err=disc_err,
        tail_err=tail_err,
        eps_obs=eps_obs,
        met_tol=met,
        recorded_precision=(tol if met else eps_obs),
        dominant=("disc" if disc_err >= tail_err else "tail"),
        tol=tol,
        U=U_base,
        n=n_base,
        h=2.0 * U_base / n_base,
        U_tail=U_tail,
        n_tail=n_tail,
        h_tail=2.0 * U_tail / n_tail,
        disc_iters=disc_iters,
        tail_iters=tail_iters,
        disc_stalled=disc_stalled,
        tail_stalled=tail_stalled,
    )
