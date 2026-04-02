"""
Fetch xG (expected goals) data from Understat.com.

Supported leagues: Premier League, Ligue 1, La Liga (the three covered by Understat).
Bundesliga 2 and Eredivisie are not on Understat — those matches fall back to
actual goals in the model.

Data is cached per league/season in data/cache/xg_{league}_{season}.csv.
Past seasons are cached indefinitely; the current season re-fetches after 3 days.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import CACHE_DIR, UNDERSTAT_LEAGUE_MAP, normalize_team

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_BASE_URL = "https://understat.com/league"
_CACHE_MAX_AGE_DAYS = 3


def _cache_path(league_name: str, season: int) -> Path:
    safe = league_name.lower().replace(" ", "_")
    return CACHE_DIR / f"xg_{safe}_{season}.csv"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(tz=timezone.utc) - mtime) < timedelta(days=_CACHE_MAX_AGE_DAYS)


def _fetch_understat_season(understat_key: str, season: int) -> list[dict]:
    """Fetch raw match data from Understat for a league/season."""
    url = f"{_BASE_URL}/{understat_key}/{season}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Understat fetch failed for %s/%d: %s", understat_key, season, exc)
        return []

    # Data is embedded as: var datesData = JSON.parse('...')
    match = re.search(
        r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)",
        resp.text,
        re.DOTALL,
    )
    if not match:
        log.warning(
            "Could not find datesData on Understat page %s/%d — page structure may have changed.",
            understat_key, season,
        )
        return []

    try:
        # The string uses unicode_escape encoding (\x5B, \u0022, etc.)
        raw_json = match.group(1).encode("raw_unicode_escape").decode("unicode_escape")
        return json.loads(raw_json)
    except Exception as exc:
        log.warning("Failed to parse Understat JSON for %s/%d: %s", understat_key, season, exc)
        return []


def _parse_xg_data(raw: list[dict]) -> pd.DataFrame:
    """Convert raw Understat match list to a clean DataFrame."""
    rows = []
    for m in raw:
        if not m.get("isResult"):
            continue  # skip upcoming fixtures
        try:
            rows.append({
                "date":      m["datetime"][:10],
                "home_team": normalize_team(m["h"]["title"]),
                "away_team": normalize_team(m["a"]["title"]),
                "home_xg":   float(m["xG"]["h"]),
                "away_xg":   float(m["xG"]["a"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_xg", "away_xg"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_xg_data(league_name: str, seasons: list[int]) -> pd.DataFrame:
    """
    Return a DataFrame with home_xg/away_xg for all available seasons.

    Returns an empty DataFrame if the league is not supported by Understat
    (e.g. Bundesliga 2, Eredivisie).
    """
    understat_key = UNDERSTAT_LEAGUE_MAP.get(league_name)
    if not understat_key:
        return pd.DataFrame()

    now = datetime.now(tz=timezone.utc)
    current_season = now.year if now.month >= 7 else now.year - 1

    frames = []
    for season in seasons:
        cache = _cache_path(league_name, season)
        is_current = season == current_season

        # Past seasons: cache forever. Current season: re-fetch after 3 days.
        if cache.exists() and (not is_current or _cache_is_fresh(cache)):
            frames.append(pd.read_csv(cache, parse_dates=["date"]))
            continue

        raw = _fetch_understat_season(understat_key, season)
        if not raw:
            if cache.exists():
                log.warning("Understat fetch failed — using stale cache for %s/%d.", league_name, season)
                frames.append(pd.read_csv(cache, parse_dates=["date"]))
            continue

        df = _parse_xg_data(raw)
        if not df.empty:
            df.to_csv(cache, index=False)
            log.info("Cached %d xG rows for %s/%d.", len(df), league_name, season)
            frames.append(df)

        time.sleep(1.0)  # be polite to Understat

    if not frames:
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "home_team", "away_team"])
    )
    return combined.sort_values("date").reset_index(drop=True)
