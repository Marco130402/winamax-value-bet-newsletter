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

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .config import CACHE_DIR, DIXON_COLES_XI, MAX_GOALS, MIN_MATCHES_FOR_MODEL

log = logging.getLogger(__name__)

def _params_cache_path(league_name: str) -> Path:
    safe = league_name.lower().replace(" ", "_")
    return CACHE_DIR / f"model_params_{safe}.json"


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

    def __init__(self, league_name: str = "default") -> None:
        self._league_name = league_name
        self._teams: list[str] = []
        self._team_index: dict[str, int] = {}
        self._params: np.ndarray | None = None
        self._n_matches: int = 0
        self._xg_matches: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the model to historical results.

        Parameters
        ----------
        df : DataFrame with columns date, home_team, away_team, home_goals, away_goals.
             If home_xg/away_xg columns are present, uses rounded xG instead of
             actual goals for the Poisson likelihood (better signal, less noise).
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

        # Use xG when available (better measure of underlying performance than actual goals)
        # pandas itertuples() silently renames columns with leading underscores,
        # so use non-underscore names for the goals columns used in the NLL.
        if "home_xg" in df.columns:
            df["fit_g_h"] = df["home_xg"].fillna(df["home_goals"]).round().clip(0, MAX_GOALS).astype(int)
            df["fit_g_a"] = df["away_xg"].fillna(df["away_goals"]).round().clip(0, MAX_GOALS).astype(int)
            self._xg_matches = int(df["home_xg"].notna().sum())
            log.info("  Using xG for %d/%d matches (%.0f%%).",
                     self._xg_matches, len(df), self._xg_matches / len(df) * 100)
        else:
            df["fit_g_h"] = df["home_goals"]
            df["fit_g_a"] = df["away_goals"]
            self._xg_matches = 0

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

    def predict(
        self,
        home_team: str,
        away_team: str,
        injury_adjustments: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """
        Predict win/draw/loss probabilities for a match.

        Parameters
        ----------
        home_team, away_team : canonical team names
        injury_adjustments : optional dict with keys
            home_attack, away_attack, home_defence, away_defence (all floats).
            See injury_fetcher.compute_injury_adjustments() for details.

        Returns None if either team was not seen during training.
        """
        if self._params is None:
            raise RuntimeError("Model has not been fitted yet.")

        if home_team not in self._team_index or away_team not in self._team_index:
            log.debug("Unknown team(s): %s / %s", home_team, away_team)
            return None

        lam_h, lam_a = self._lambdas(home_team, away_team)

        if injury_adjustments:
            lam_h *= injury_adjustments.get("home_attack", 1.0)
            lam_h *= injury_adjustments.get("away_defence", 1.0)
            lam_a *= injury_adjustments.get("away_attack", 1.0)
            lam_a *= injury_adjustments.get("home_defence", 1.0)

        n = len(self._teams)
        rho = self._params[2 * n + 2]

        mat = _score_matrix(lam_h, lam_a, rho)
        home_win = float(np.tril(mat, -1).sum())  # rows > cols  (home scored more)
        away_win = float(np.triu(mat, 1).sum())   # rows < cols
        draw = float(np.trace(mat))

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "exp_home_goals": round(lam_h, 2),
            "exp_away_goals": round(lam_a, 2),
        }

    def predict_ou(
        self,
        home_team: str,
        away_team: str,
        threshold: float = 2.5,
        injury_adjustments: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """
        Predict Over/Under probabilities for total goals in a match.

        Returns {"over": float, "under": float} or None if either team is unknown.
        Uses the same score matrix (with Dixon-Coles tau correction) as predict().
        """
        if self._params is None:
            raise RuntimeError("Model has not been fitted yet.")
        if home_team not in self._team_index or away_team not in self._team_index:
            return None

        lam_h, lam_a = self._lambdas(home_team, away_team)

        if injury_adjustments:
            lam_h *= injury_adjustments.get("home_attack", 1.0)
            lam_h *= injury_adjustments.get("away_defence", 1.0)
            lam_a *= injury_adjustments.get("away_attack", 1.0)
            lam_a *= injury_adjustments.get("home_defence", 1.0)

        n = len(self._teams)
        rho = self._params[2 * n + 2]
        mat = _score_matrix(lam_h, lam_a, rho)

        # Sum all scorelines where total goals exceeds the threshold
        size = mat.shape[0]
        over_prob = sum(
            mat[i, j]
            for i in range(size)
            for j in range(size)
            if i + j > threshold
        )
        over_prob = float(np.clip(over_prob, 0.0, 1.0))
        return {"over": over_prob, "under": 1.0 - over_prob}

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
            g_h, g_a = int(row.fit_g_h), int(row.fit_g_a)
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
            "xg_matches": self._xg_matches,
            "teams": self._teams,
            "params": self._params.tolist(),
        }
        with open(_params_cache_path(self._league_name), "w") as f:
            json.dump(data, f)

    def _load_cache(self, n_teams: int) -> bool:
        """Return True if cached params were loaded successfully."""
        cache_path = _params_cache_path(self._league_name)
        if not cache_path.exists():
            return False
        try:
            with open(cache_path) as f:
                data = json.load(f)
            if (data["n_matches"] != self._n_matches
                    or data["teams"] != self._teams
                    or data.get("xg_matches", -1) != self._xg_matches):
                return False
            self._params = np.array(data["params"])
            log.info(
                "Loaded model params from cache (%d matches, %d with xG).",
                self._n_matches, self._xg_matches,
            )
            return True
        except Exception as exc:
            log.warning("Could not load model cache: %s", exc)
            return False
