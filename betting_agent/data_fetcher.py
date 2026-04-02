"""Fetch and cache historical match results from football-data.org."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import CACHE_DIR, LEAGUES, normalize_team
from .xg_fetcher import get_xg_data

log = logging.getLogger(__name__)

_FD_BASE = "https://api.football-data.org/v4"
_CACHE_MAX_AGE_DAYS = 3
_RATE_LIMIT_SLEEP = 6.5   # 10 req/min limit → sleep ≥6s between calls
_SEASONS_ON_COLD_START = 3  # how many completed seasons to back-fill


def _fd_get(path: str, api_key: str) -> dict:
    """GET a football-data.org endpoint; raise on error."""
    url = f"{_FD_BASE}{path}"
    resp = requests.get(url, headers={"X-Auth-Token": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_matches(raw_matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw_matches:
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_goals = ft.get("home")
        away_goals = ft.get("away")
        if home_goals is None or away_goals is None:
            continue
        utc_date = m.get("utcDate", "")[:10]
        rows.append(
            {
                "date": utc_date,
                "home_team": normalize_team(m["homeTeam"]["name"]),
                "away_team": normalize_team(m["awayTeam"]["name"]),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_season(fd_code: str, season_year: int, api_key: str) -> pd.DataFrame | None:
    """
    Fetch all finished matches for a given season year.
    Returns None if the season is not accessible on the current API plan (403).
    """
    log.info("Fetching %s season %d from football-data.org …", fd_code, season_year)
    path = f"/competitions/{fd_code}/matches?season={season_year}&status=FINISHED"
    try:
        data = _fd_get(path, api_key)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            log.warning(
                "Season %d for %s is not accessible on the free tier — skipping.",
                season_year, fd_code,
            )
            time.sleep(_RATE_LIMIT_SLEEP)
            return None
        raise
    matches = data.get("matches", [])
    time.sleep(_RATE_LIMIT_SLEEP)
    return _parse_matches(matches)


def _cache_path(league_name: str) -> Path:
    safe = league_name.lower().replace(" ", "_")
    return CACHE_DIR / f"{safe}_results.csv"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(tz=timezone.utc) - mtime) < timedelta(days=_CACHE_MAX_AGE_DAYS)


def _merge_xg(df: pd.DataFrame, league_name: str, seasons: list[int]) -> pd.DataFrame:
    """Merge xG data from Understat into the match DataFrame (left join on date+teams)."""
    xg_df = get_xg_data(league_name, seasons)
    if xg_df.empty:
        return df

    # Drop stale xG columns before re-merging
    df = df.drop(columns=["home_xg", "away_xg"], errors="ignore")

    df["_date_key"] = df["date"].dt.strftime("%Y-%m-%d")
    xg_df["_date_key"] = xg_df["date"].dt.strftime("%Y-%m-%d")

    df = df.merge(
        xg_df[["_date_key", "home_team", "away_team", "home_xg", "away_xg"]],
        on=["_date_key", "home_team", "away_team"],
        how="left",
    ).drop(columns=["_date_key"])

    xg_count = int(df["home_xg"].notna().sum())
    log.info("  [%s] xG enrichment: %d/%d matches.", league_name, xg_count, len(df))
    return df


def load_or_update_cache(league_name: str, fd_code: str, api_key: str) -> pd.DataFrame:
    """Return historical results DataFrame for a league, updating cache if stale."""
    path = _cache_path(league_name)

    if _cache_is_fresh(path):
        log.info("Cache for %s is fresh — loading from disk.", league_name)
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    now = datetime.now(tz=timezone.utc)
    current_season = now.year if now.month >= 7 else now.year - 1
    seasons = list(range(current_season - _SEASONS_ON_COLD_START, current_season + 1))

    if not path.exists():
        # Cold start: attempt to fetch past seasons + current, skip any blocked by the free tier
        log.info("Cold start for %s — attempting %d seasons.", league_name, _SEASONS_ON_COLD_START + 1)
        frames = []
        for yr in seasons:
            result = _fetch_season(fd_code, yr, api_key)
            if result is not None and not result.empty:
                frames.append(result)
        if not frames:
            log.error("No season data retrieved for %s — check your API plan.", league_name)
            return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
        df = pd.concat(frames, ignore_index=True).drop_duplicates()
    else:
        # Incremental update: re-fetch only the current season
        log.info("Incremental update for %s (current season %d).", league_name, current_season)
        existing = pd.read_csv(path, parse_dates=["date"])
        current = _fetch_season(fd_code, current_season, api_key)
        df = (
            pd.concat([existing, current], ignore_index=True)
            .drop_duplicates(subset=["date", "home_team", "away_team"])
        )

    df = df.sort_values("date").reset_index(drop=True)
    df = _merge_xg(df, league_name, seasons)
    df.to_csv(path, index=False)
    log.info("Saved %d matches for %s to cache.", len(df), league_name)
    return df


def get_all_historical_data(api_key: str) -> dict[str, pd.DataFrame]:
    """Fetch/load historical results for all configured leagues."""
    return {
        league_name: load_or_update_cache(league_name, cfg["fd_code"], api_key)
        for league_name, cfg in LEAGUES.items()
    }
