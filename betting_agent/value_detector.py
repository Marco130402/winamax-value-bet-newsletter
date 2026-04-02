"""
Identify value bets by combining two signals:

1. Model EV — the Poisson/Dixon-Coles model estimates the true probability of
   each outcome.  If that probability implies an expected value above
   EV_THRESHOLD against Winamax's decimal odds, the bet passes the model gate.

2. Consensus comparison — Winamax's overround-normalised implied probability is
   compared against the consensus market's normalised implied probability.  If
   Winamax lags the consensus by at least CONSENSUS_DIFF_THRESHOLD, the bet
   passes the consensus gate.

A bet is reported only when BOTH gates pass.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from .config import (
    CONSENSUS_DIFF_THRESHOLD,
    EV_THRESHOLD,
    MAX_BETS,
    UPCOMING_DAYS,
)
from .model import PoissonModel
from .odds_fetcher import get_consensus_odds, get_winamax_odds

log = logging.getLogger(__name__)

_OUTCOMES = [
    ("home_win", "home_odds", "consensus_home_odds", "Home Win"),
    ("draw",     "draw_odds", "consensus_draw_odds",  "Draw"),
    ("away_win", "away_odds", "consensus_away_odds",  "Away Win"),
]


def _fair_probs(home_odd: float, draw_odd: float, away_odd: float) -> tuple[float, float, float]:
    """Convert decimal odds to overround-normalised implied probabilities."""
    raw = (1 / home_odd, 1 / draw_odd, 1 / away_odd)
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


def _ev(true_prob: float, decimal_odd: float) -> float:
    """Expected value as a fraction (multiply by 100 for %)."""
    return true_prob * decimal_odd - 1.0


def _is_upcoming(commence_time: str) -> bool:
    """Return True if match starts within UPCOMING_DAYS from now."""
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        delta = dt - datetime.now(tz=timezone.utc)
        return 0 <= delta.total_seconds() <= UPCOMING_DAYS * 86400
    except Exception:
        return True  # include if we can't parse


def analyze_match(
    winamax_row: pd.Series,
    consensus_row: pd.Series,
    model_probs: dict[str, float] | None,
) -> list[dict]:
    """
    Evaluate all three outcomes for a single match.

    Returns a (possibly empty) list of qualifying value-bet dicts.
    """
    w_fair_home, w_fair_draw, w_fair_away = _fair_probs(
        winamax_row["home_odds"],
        winamax_row["draw_odds"],
        winamax_row["away_odds"],
    )
    c_fair_home, c_fair_draw, c_fair_away = _fair_probs(
        consensus_row["consensus_home_odds"],
        consensus_row["consensus_draw_odds"],
        consensus_row["consensus_away_odds"],
    )

    winamax_fair = {
        "home_win": w_fair_home,
        "draw": w_fair_draw,
        "away_win": w_fair_away,
    }
    consensus_fair = {
        "home_win": c_fair_home,
        "draw": c_fair_draw,
        "away_win": c_fair_away,
    }

    bets = []
    for outcome_key, wm_odd_col, con_odd_col, outcome_label in _OUTCOMES:
        winamax_odd = winamax_row[wm_odd_col]
        winamax_implied = winamax_fair[outcome_key]
        consensus_implied = consensus_fair[outcome_key]
        consensus_edge = consensus_implied - winamax_implied

        # ── Gate 2: consensus comparison ────────────────────────────────────
        if consensus_edge < CONSENSUS_DIFF_THRESHOLD:
            continue

        # ── Gate 1: model EV ─────────────────────────────────────────────────
        if model_probs is None:
            # Consensus-only fallback: we still report if consensus edge is strong
            model_ev = None
            model_prob = None
        else:
            model_prob = model_probs[outcome_key]
            model_ev = _ev(model_prob, winamax_odd)
            if model_ev < EV_THRESHOLD:
                continue

        commence = winamax_row.get("commence_time", "")
        try:
            match_date = datetime.fromisoformat(
                commence.replace("Z", "+00:00")
            ).strftime("%a %d %b")
        except Exception:
            match_date = commence[:10]

        bets.append(
            {
                "match": f"{winamax_row['home_team']} vs {winamax_row['away_team']}",
                "home_team": winamax_row["home_team"],
                "away_team": winamax_row["away_team"],
                "league": winamax_row["league"],
                "date": match_date,
                "outcome": outcome_label,
                "winamax_odd": round(winamax_odd, 2),
                "model_prob": round(model_prob * 100, 1) if model_prob is not None else None,
                "model_ev_pct": round(model_ev * 100, 1) if model_ev is not None else None,
                "consensus_implied_pct": round(consensus_implied * 100, 1),
                "winamax_implied_pct": round(winamax_implied * 100, 1),
                "consensus_edge_pct": round(consensus_edge * 100, 1),
                "model_available": model_probs is not None,
            }
        )
    return bets


def find_all_value_bets(
    odds_df: pd.DataFrame,
    models: dict[str, PoissonModel],
) -> list[dict]:
    """
    Scan all upcoming matches and return up to MAX_BETS qualifying value bets,
    sorted by model EV descending (or by consensus edge when model unavailable).
    """
    if odds_df.empty:
        log.warning("No odds data — cannot detect value bets.")
        return []

    winamax_df = get_winamax_odds(odds_df)
    consensus_df = get_consensus_odds(odds_df)

    if winamax_df.empty:
        log.warning("Winamax not found in the odds data for any league.")
        return []

    merged = winamax_df.merge(consensus_df, on="match_id", suffixes=("", "_con"))
    if merged.empty:
        log.warning("No matches have both Winamax and consensus odds.")
        return []

    all_bets: list[dict] = []

    for _, row in merged.iterrows():
        if not _is_upcoming(row.get("commence_time", "")):
            continue

        league = row["league"]
        home = row["home_team"]
        away = row["away_team"]

        model = models.get(league)
        model_probs = None
        if model is not None:
            model_probs = model.predict(home, away)
            if model_probs is None:
                log.debug("Unknown team in model for %s vs %s — using consensus-only.", home, away)

        # Build a clean Series with the right column names for analyze_match
        winamax_series = row[["home_team", "away_team", "league",
                               "commence_time", "home_odds", "draw_odds", "away_odds"]]

        consensus_series = row[["consensus_home_odds", "consensus_draw_odds", "consensus_away_odds"]]

        bets = analyze_match(winamax_series, consensus_series, model_probs)
        all_bets.extend(bets)

    # Sort: model bets by EV desc; consensus-only bets by consensus edge desc
    model_bets = [b for b in all_bets if b["model_available"]]
    consensus_only = [b for b in all_bets if not b["model_available"]]

    model_bets.sort(key=lambda b: b["model_ev_pct"] or 0, reverse=True)
    consensus_only.sort(key=lambda b: b["consensus_edge_pct"], reverse=True)

    combined = model_bets + consensus_only
    log.info("Found %d value bets (%d model, %d consensus-only).",
             len(combined), len(model_bets), len(consensus_only))
    return combined[:MAX_BETS]
