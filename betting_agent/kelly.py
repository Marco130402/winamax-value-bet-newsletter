"""
Portfolio Kelly stake sizing.

Finds the fraction of bankroll to bet on each value bet by maximising expected
log-bankroll growth across all simultaneous bets:

    max  E[ log(1 + Σ fᵢ · rᵢ) ]

where rᵢ = +bᵢ (win, prob pᵢ) or −1 (lose, prob 1−pᵢ), and bᵢ = odds − 1.

For n bets this enumerates 2ⁿ scenarios — fast for n ≤ 15.

The raw optimal fractions are then:
  1. Scaled by KELLY_FRACTION (e.g. ×0.25 for quarter Kelly)
  2. Capped per bet at MAX_SINGLE_BET
  3. Rescaled proportionally if total > MAX_TOTAL_EXPOSURE
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from .config import KELLY_FRACTION, MAX_SINGLE_BET, MAX_TOTAL_EXPOSURE

log = logging.getLogger(__name__)


def compute_kelly_stakes(bets: list[dict]) -> list[float]:
    """
    Return a list of recommended bankroll fractions (0–1), one per bet,
    in the same order as the input list.

    Returns all-zeros if optimisation fails.
    """
    n = len(bets)
    if n == 0:
        return []

    probs  = np.array([b["model_prob"] / 100.0 for b in bets])
    b_vals = np.array([b["winamax_odd"] - 1.0   for b in bets])  # net win per unit

    # ── Build all 2ⁿ outcome scenarios ───────────────────────────────────────
    scenario_probs   = np.zeros(2 ** n)
    scenario_returns = np.zeros((2 ** n, n))

    for mask in range(2 ** n):
        prob = 1.0
        for i in range(n):
            if mask & (1 << i):
                prob *= probs[i]
                scenario_returns[mask, i] = b_vals[i]
            else:
                prob *= (1 - probs[i])
                scenario_returns[mask, i] = -1.0
        scenario_probs[mask] = prob

    # ── Objective: negative expected log-growth ───────────────────────────────
    def neg_elg(fracs: np.ndarray) -> float:
        net_returns = 1.0 + scenario_returns @ fracs   # shape: (2ⁿ,)
        if np.any(net_returns <= 0):
            return 1e10
        return -float(np.dot(scenario_probs, np.log(net_returns)))

    def neg_elg_grad(fracs: np.ndarray) -> np.ndarray:
        net_returns = 1.0 + scenario_returns @ fracs
        if np.any(net_returns <= 0):
            return np.zeros(n)
        weights = scenario_probs / net_returns          # shape: (2ⁿ,)
        return -scenario_returns.T @ weights            # shape: (n,)

    # ── Optimise unconstrained (just non-negativity) ──────────────────────────
    # We'll apply fraction and caps afterwards, which is cleaner than
    # baking them into the optimiser bounds.
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 0.01)

    result = minimize(
        neg_elg,
        x0,
        jac=neg_elg_grad,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success and result.fun > 1e9:
        log.warning("Kelly optimisation failed: %s", result.message)
        return [0.0] * n

    raw_fracs = result.x.clip(0)

    # ── Apply quarter Kelly ───────────────────────────────────────────────────
    fracs = raw_fracs * KELLY_FRACTION

    # ── Cap each bet ─────────────────────────────────────────────────────────
    fracs = np.minimum(fracs, MAX_SINGLE_BET)

    # ── Cap total exposure (proportional rescale) ─────────────────────────────
    total = fracs.sum()
    if total > MAX_TOTAL_EXPOSURE:
        fracs = fracs * (MAX_TOTAL_EXPOSURE / total)

    log.info(
        "Kelly stakes: %s | total exposure: %.1f%%",
        [f"{f*100:.2f}%" for f in fracs],
        fracs.sum() * 100,
    )

    return [round(float(f), 5) for f in fracs]
