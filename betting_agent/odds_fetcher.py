"""Fetch upcoming match odds from The Odds API."""

import logging

import pandas as pd
import requests

from .config import BOOKMAKER_WINAMAX, BOOKMAKERS_CONSENSUS, LEAGUES, normalize_team

log = logging.getLogger(__name__)

_BASE = "https://api.the-odds-api.com/v4"


def _get_odds(odds_key: str, api_key: str) -> list[dict]:
    url = f"{_BASE}/sports/{odds_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    log.info(
        "The Odds API [%s]: used=%s remaining=%s",
        odds_key,
        used,
        remaining,
    )
    if int(remaining if remaining != "?" else 999) < 20:
        log.warning("The Odds API quota almost exhausted! remaining=%s", remaining)
    resp.raise_for_status()
    return resp.json()


def _parse_events(events: list[dict], league_name: str) -> pd.DataFrame:
    rows = []
    for event in events:
        home = normalize_team(event["home_team"])
        away = normalize_team(event["away_team"])
        match_id = event["id"]
        commence = event["commence_time"]

        for bm in event.get("bookmakers", []):
            bm_key = bm["key"].lower()
            if bm_key not in BOOKMAKERS_CONSENSUS and bm_key != BOOKMAKER_WINAMAX:
                continue
            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                home_odds = outcomes.get(event["home_team"]) or outcomes.get(home)
                away_odds = outcomes.get(event["away_team"]) or outcomes.get(away)
                draw_odds = outcomes.get("Draw")
                if not (home_odds and away_odds and draw_odds):
                    continue
                rows.append(
                    {
                        "match_id": match_id,
                        "league": league_name,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence,
                        "bookmaker": bm_key,
                        "home_odds": float(home_odds),
                        "draw_odds": float(draw_odds),
                        "away_odds": float(away_odds),
                    }
                )
    return pd.DataFrame(rows)


def fetch_all_leagues(api_key: str) -> pd.DataFrame:
    """Return a flat DataFrame of odds for all leagues from relevant bookmakers."""
    frames = []
    for league_name, cfg in LEAGUES.items():
        events = _get_odds(cfg["odds_key"], api_key)
        if not events:
            log.warning("No events returned for %s.", league_name)
            continue
        df = _parse_events(events, league_name)
        log.info("Parsed %d bookmaker-match rows for %s.", len(df), league_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_winamax_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Winamax rows only (one row per match)."""
    return df[df["bookmaker"] == BOOKMAKER_WINAMAX].copy()


def get_consensus_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average odds across consensus bookmakers, return one row per match.

    Only matches where at least one consensus bookmaker has posted odds are returned.
    """
    consensus = df[df["bookmaker"].isin(BOOKMAKERS_CONSENSUS)]
    if consensus.empty:
        return pd.DataFrame()
    agg = (
        consensus.groupby("match_id")
        .agg(
            league=("league", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            commence_time=("commence_time", "first"),
            consensus_home_odds=("home_odds", "mean"),
            consensus_draw_odds=("draw_odds", "mean"),
            consensus_away_odds=("away_odds", "mean"),
        )
        .reset_index()
    )
    return agg
