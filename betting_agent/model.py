"""
Poisson / Dixon-Coles football outcome model.

Goals scored by each team are modelled as independent Poisson processes:

    lambda_home = exp(mu + attack_i + defence_j + home_adv)
    lambda_away = exp(mu + attack_j + defence_i)

The Dixon-Coles (1997) tau correction adjusts probabilities for low-scoring
outcomes (total goals ≤ 2) where the independence assumption underperforms.

Matches are time-weighted with w = exp(-xi * days_ago) so that recent
results have more influence than old ones.
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .config import CACHE_DIR, DIXON_COLES_XI, MAX_GOALS, MIN_MATCHES_FOR_MODEL

log = logging.getLogger(__name__)

_PARAMS_CACHE = CACHE_DIR / "model_params.json"


def _tau(g_h: int, g_a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles correction factor for scorelines where both goals ≤ 1."""
    if g_h == 0 and g_a == 0:
        return 1.0 - lam_h * lam_a * rho
    if g_h == 1 and g_a == 0:
        return 1.0 + lam_a * rho
    if g_h == 0 and g_a == 1:
        return 1.0 + lam_h * rho
    if g_h == 1 and g_a == 1:
        return 1.0 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Return (max_goals+1)×(max_goals+1) matrix of scoreline probabilities."""
    size = max_goals + 1
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mat[i, j] = (
                poisson.pmf(i, lam_h)
                * poisson.pmf(j, lam_a)
                * _tau(i, j, lam_h, lam_a, rho)
            )
    # Renormalise (tau can shift the total slightly)
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


class PoissonModel:
    """Dixon-Coles Poisson model with temporal weighting."""

    def __init__(self) -> None:
        self._teams: list[str] = []
        self._team_index: dict[str, int] = {}
        self._params: np.ndarray | None = None
        self._n_matches: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the model to historical results.

        Parameters
        ----------
        df : DataFrame with columns date, home_team, away_team, home_goals, away_goals
        """
        if len(df) < MIN_MATCHES_FOR_MODEL:
            raise ValueError(
                f"Need at least {MIN_MATCHES_FOR_MODEL} matches; got {len(df)}."
            )

        self._n_matches = len(df)
        self._teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_index = {t: i for i, t in enumerate(self._teams)}
        n = len(self._teams)

        # Try loading cached params if data hasn't changed
        if self._load_cache(n):
            return

        today = datetime.now(tz=timezone.utc).date()
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["days_ago"] = df["date"].apply(lambda d: (today - d).days)
        df["weight"] = np.exp(-DIXON_COLES_XI * df["days_ago"])

        # Parameter layout:
        # [attack_0 … attack_{n-1}, defence_0 … defence_{n-1}, home_adv, mu, rho]
        x0 = np.concatenate([
            np.zeros(n),   # attack params (sum-to-zero constraint applied via identifiability)
            np.zeros(n),   # defence params
            [0.3],         # home_adv
            [0.3],         # mu
            [-0.1],        # rho
        ])

        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(df,),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        if not result.success:
            log.warning("Model optimisation did not fully converge: %s", result.message)

        self._params = result.x
        # Enforce sum-to-zero on attack params for identifiability
        self._params[:n] -= self._params[:n].mean()

        self._save_cache(n)
        log.info("Model fitted on %d matches, %d teams.", len(df), n)

    def predict(self, home_team: str, away_team: str) -> dict[str, float] | None:
        """
        Predict win/draw/loss probabilities for a match.

        Returns None if either team was not seen during training.
        """
        if self._params is None:
            raise RuntimeError("Model has not been fitted yet.")

        if home_team not in self._team_index or away_team not in self._team_index:
            log.debug("Unknown team(s): %s / %s", home_team, away_team)
            return None

        lam_h, lam_a = self._lambdas(home_team, away_team)
        n = len(self._teams)
        rho = self._params[2 * n + 2]

        mat = _score_matrix(lam_h, lam_a, rho)
        home_win = float(np.tril(mat, -1).sum())  # rows > cols  (home scored more)
        away_win = float(np.triu(mat, 1).sum())   # rows < cols
        draw = float(np.trace(mat))

        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _lambdas(self, home: str, away: str) -> tuple[float, float]:
        n = len(self._teams)
        hi, ai = self._team_index[home], self._team_index[away]
        attack = self._params[:n]
        defence = self._params[n : 2 * n]
        home_adv = self._params[2 * n]
        mu = self._params[2 * n + 1]
        lam_h = np.exp(mu + attack[hi] + defence[ai] + home_adv)
        lam_a = np.exp(mu + attack[ai] + defence[hi])
        return float(lam_h), float(lam_a)

    def _neg_log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        n = len(self._teams)
        attack = params[:n]
        defence = params[n : 2 * n]
        home_adv = params[2 * n]
        mu = params[2 * n + 1]
        rho = params[2 * n + 2]

        nll = 0.0
        for row in df.itertuples(index=False):
            hi = self._team_index[row.home_team]
            ai = self._team_index[row.away_team]
            lam_h = np.exp(mu + attack[hi] + defence[ai] + home_adv)
            lam_a = np.exp(mu + attack[ai] + defence[hi])
            g_h, g_a = int(row.home_goals), int(row.away_goals)
            tau = _tau(g_h, g_a, lam_h, lam_a, rho)
            if tau <= 0:
                nll += 1e6 * row.weight
                continue
            ll = (
                poisson.logpmf(g_h, lam_h)
                + poisson.logpmf(g_a, lam_a)
                + np.log(tau)
            )
            nll -= row.weight * ll
        return nll

    def _save_cache(self, n_teams: int) -> None:
        data = {
            "n_matches": self._n_matches,
            "teams": self._teams,
            "params": self._params.tolist(),
        }
        with open(_PARAMS_CACHE, "w") as f:
            json.dump(data, f)

    def _load_cache(self, n_teams: int) -> bool:
        """Return True if cached params were loaded successfully."""
        if not _PARAMS_CACHE.exists():
            return False
        try:
            with open(_PARAMS_CACHE) as f:
                data = json.load(f)
            if data["n_matches"] != self._n_matches or data["teams"] != self._teams:
                return False
            self._params = np.array(data["params"])
            log.info("Loaded model params from cache (%d matches).", self._n_matches)
            return True
        except Exception as exc:
            log.warning("Could not load model cache: %s", exc)
            return False
