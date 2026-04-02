"""
Identify value bets using the Poisson/Dixon-Coles model as the primary signal.

A bet qualifies when model EV > EV_THRESHOLD against Winamax's decimal odds.
Consensus market data is fetched and displayed in the newsletter as context
(showing how much the broader market agrees or disagrees) but does not gate
which bets are reported.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .config import EV_THRESHOLD, MAX_BETS, UPCOMING_DAYS
from .injury_fetcher import compute_injury_adjustments
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
        return True


def analyze_match(
    winamax_row: pd.Series,
    consensus_row: pd.Series | None,
    model_probs: dict[str, float] | None,
) -> list[dict]:
    """
    Evaluate all three outcomes for a single match.

    The model EV is the sole qualifier. Consensus data is attached as context
    when available but does not affect whether a bet is reported.

    Returns a (possibly empty) list of qualifying value-bet dicts.
    """
    if model_probs is None:
        return []

    w_fair_home, w_fair_draw, w_fair_away = _fair_probs(
        winamax_row["home_odds"],
        winamax_row["draw_odds"],
        winamax_row["away_odds"],
    )
    winamax_fair = {"home_win": w_fair_home, "draw": w_fair_draw, "away_win": w_fair_away}

    # Consensus implied probs — informational only
    consensus_fair: dict[str, float] = {}
    if consensus_row is not None:
        c_h, c_d, c_a = _fair_probs(
            consensus_row["consensus_home_odds"],
            consensus_row["consensus_draw_odds"],
            consensus_row["consensus_away_odds"],
        )
        consensus_fair = {"home_win": c_h, "draw": c_d, "away_win": c_a}

    bets = []
    for outcome_key, wm_odd_col, con_odd_col, outcome_label in _OUTCOMES:
        winamax_odd = winamax_row[wm_odd_col]
        model_prob = model_probs[outcome_key]
        model_ev = _ev(model_prob, winamax_odd)

        if model_ev < EV_THRESHOLD:
            continue

        winamax_implied = winamax_fair[outcome_key]
        consensus_implied = consensus_fair.get(outcome_key)
        consensus_edge = (
            round((consensus_implied - winamax_implied) * 100, 1)
            if consensus_implied is not None
            else None
        )

        commence = winamax_row.get("commence_time", "")
        try:
            match_date = datetime.fromisoformat(
                commence.replace("Z", "+00:00")
            ).strftime("%a %d %b")
        except Exception:
            match_date = commence[:10]

        exp_goals = round(
            model_probs.get("exp_home_goals", 0) + model_probs.get("exp_away_goals", 0), 1
        )
        bets.append(
            {
                "match": f"{winamax_row['home_team']} vs {winamax_row['away_team']}",
                "home_team": winamax_row["home_team"],
                "away_team": winamax_row["away_team"],
                "league": winamax_row["league"],
                "date": match_date,
                "outcome": outcome_label,
                "winamax_odd": round(winamax_odd, 2),
                "model_prob": round(model_prob * 100, 1),
                "model_ev_pct": round(model_ev * 100, 1),
                "winamax_implied_pct": round(winamax_implied * 100, 1),
                "consensus_implied_pct": round(consensus_implied * 100, 1) if consensus_implied else None,
                "consensus_edge_pct": consensus_edge,
                "exp_goals": exp_goals,
            }
        )
    return bets


def _analyze_ou(
    row: pd.Series,
    model: PoissonModel,
    injury_adj: dict[str, float] | None,
    model_probs: dict[str, float] | None = None,
) -> list[dict]:
    """Check Over/Under 2.5 for value bets on a single match row from totals_df."""
    home = row["home_team"]
    away = row["away_team"]

    ou_probs = model.predict_ou(home, away, injury_adjustments=injury_adj)
    if ou_probs is None:
        return []

    commence = row.get("commence_time", "")
    try:
        match_date = datetime.fromisoformat(
            commence.replace("Z", "+00:00")
        ).strftime("%a %d %b")
    except Exception:
        match_date = commence[:10]

    bets = []
    for direction, odd_col, bm_col, label in [
        ("over",  "over_odds",  "over_bookmaker",  "Over 2.5"),
        ("under", "under_odds", "under_bookmaker", "Under 2.5"),
    ]:
        odd = row.get(odd_col)
        if not odd or pd.isna(odd):
            continue
        prob = ou_probs[direction]
        ev = _ev(prob, float(odd))
        if ev < EV_THRESHOLD:
            continue
        exp_goals = round(
            (model_probs or {}).get("exp_home_goals", 0)
            + (model_probs or {}).get("exp_away_goals", 0),
            1,
        )
        bets.append({
            "match":                  f"{home} vs {away}",
            "home_team":              home,
            "away_team":              away,
            "league":                 row["league"],
            "date":                   match_date,
            "outcome":                label,
            "winamax_odd":            round(float(odd), 2),
            "bookmaker":              row.get(bm_col, ""),
            "model_prob":             round(prob * 100, 1),
            "model_ev_pct":           round(ev * 100, 1),
            "winamax_implied_pct":    round(1 / float(odd) * 100, 1),
            "consensus_implied_pct":  None,
            "consensus_edge_pct":     None,
            "exp_goals":              exp_goals,
        })
    return bets


def find_all_value_bets(
    odds_df: pd.DataFrame,
    models: dict[str, PoissonModel],
    totals_df: pd.DataFrame | None = None,
    injury_api_key: str | None = None,
) -> list[dict]:
    """
    Scan all upcoming matches and return up to MAX_BETS qualifying value bets,
    sorted by model EV descending.

    Checks both 1X2 outcomes (from odds_df) and Over/Under 2.5 (from totals_df).
    Injury adjustments are always applied for supported leagues (no API key needed
    for PL / Ligue 1 / La Liga / Bundesliga 2 / Eredivisie).
    """
    if odds_df.empty:
        log.warning("No odds data — cannot detect value bets.")
        return []

    winamax_df = get_winamax_odds(odds_df)
    consensus_df = get_consensus_odds(odds_df)

    if winamax_df.empty:
        log.warning("Winamax not found in the odds data for any league.")
        return []

    # Left-join so we keep all Winamax matches even if no consensus data exists
    merged = winamax_df.merge(consensus_df, on="match_id", how="left", suffixes=("", "_con"))

    # Build a fast lookup for totals odds: match_id → row
    totals_lookup: dict[str, pd.Series] = {}
    if totals_df is not None and not totals_df.empty:
        for _, tr in totals_df.iterrows():
            totals_lookup[tr["match_id"]] = tr

    all_bets: list[dict] = []

    for _, row in merged.iterrows():
        if not _is_upcoming(row.get("commence_time", "")):
            continue

        league = row["league"]
        home = row["home_team"]
        away = row["away_team"]

        model = models.get(league)
        if model is None:
            continue

        # Always compute injury adjustments (scraper-based leagues need no API key)
        injury_adj = compute_injury_adjustments(league, home, away, injury_api_key)

        # ── 1X2 bets ──────────────────────────────────────────────────────────
        model_probs = model.predict(home, away, injury_adjustments=injury_adj)
        if model_probs is None:
            log.debug("Unknown team(s) in model: %s / %s", home, away)
        else:
            consensus_series = None
            if pd.notna(row.get("consensus_home_odds")):
                consensus_series = row[["consensus_home_odds", "consensus_draw_odds", "consensus_away_odds"]]

            winamax_series = row[["home_team", "away_team", "league",
                                   "commence_time", "home_odds", "draw_odds", "away_odds"]]
            all_bets.extend(analyze_match(winamax_series, consensus_series, model_probs))

        # ── Over/Under 2.5 bets ───────────────────────────────────────────────
        totals_row = totals_lookup.get(row["match_id"])
        if totals_row is not None:
            all_bets.extend(_analyze_ou(totals_row, model, injury_adj, model_probs))

    all_bets.sort(key=lambda b: b["model_ev_pct"], reverse=True)
    log.info("Found %d value bet(s) (%d 1X2, %d O/U).",
             len(all_bets),
             sum(1 for b in all_bets if b["outcome"] in ("Home Win", "Draw", "Away Win")),
             sum(1 for b in all_bets if "2.5" in b["outcome"]))
    return all_bets[:MAX_BETS]
