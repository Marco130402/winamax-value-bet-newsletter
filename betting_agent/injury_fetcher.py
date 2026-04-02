"""
Fetch injury/suspension data from API-Football and compute lambda adjustment
factors for the Poisson model.

Adjustment logic
----------------
For each team in an upcoming fixture we:

1. Fetch the list of injured/suspended players for that fixture.
2. Look up each absent player's current-season stats:
      contribution = goals_per90 + ASSIST_WEIGHT * assists_per90   (attackers/mids)
      concede_rate = goals_conceded_per90                           (defenders / GK)
3. Sum contributions across absent *outfield* players → attacking lambda reduction.
4. Sum concede rates across absent *defenders/GK* → opposing lambda boost.
5. Apply safety floor so lambda never drops below 60% of its base value.

API-Football free tier: 100 requests/day — ample for our use case
(3 leagues × ~5 fixtures/round × 2 teams = ~30 requests max).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from functools import lru_cache

import requests

from .fpl_fetcher import compute_pl_injury_adjustments
from .sportsgambler_scraper import compute_scraped_injury_adjustments
from .config import (
    ASSIST_WEIGHT,
    DEF_CONCEDE_WEIGHT,
    LEAGUES,
    MAX_LAMBDA_REDUCTION,
    normalize_team,
)

log = logging.getLogger(__name__)

_BASE = "https://v3.football.api-sports.io"
_POSITION_DEFENDERS = {"Defender", "Goalkeeper"}
_POSITION_ATTACKERS = {"Forward", "Midfielder"}


def _headers(api_key: str) -> dict:
    return {"x-apisports-key": api_key}


def _get(path: str, params: dict, api_key: str) -> dict:
    resp = requests.get(
        f"{_BASE}{path}",
        headers=_headers(api_key),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
    log.debug("API-Football [%s] remaining=%s", path, remaining)
    if remaining != "?" and int(remaining) < 10:
        log.warning("API-Football quota nearly exhausted! remaining=%s", remaining)
    return resp.json()


# ── Player season stats (cached per season to avoid re-fetching) ──────────────

@lru_cache(maxsize=256)
def _player_stats(player_id: int, league_id: int, season: int, api_key: str) -> dict:
    """Return current-season stats dict for a player (cached)."""
    data = _get(
        "/players",
        {"id": player_id, "league": league_id, "season": season},
        api_key,
    )
    try:
        stats = data["response"][0]["statistics"][0]
        games = stats.get("games", {})
        goals = stats.get("goals", {})
        minutes = games.get("minutes") or 0
        goals_scored = goals.get("total") or 0
        assists = goals.get("assists") or 0
        goals_conceded = goals.get("conceded") or 0
        position = games.get("position", "")
        per90 = 90 / minutes if minutes > 0 else 0
        return {
            "position": position,
            "goals_per90": goals_scored * per90,
            "assists_per90": assists * per90,
            "conceded_per90": goals_conceded * per90,
            "minutes": minutes,
        }
    except (IndexError, KeyError, TypeError):
        return {
            "position": "",
            "goals_per90": 0.0,
            "assists_per90": 0.0,
            "conceded_per90": 0.0,
            "minutes": 0,
        }


def _current_season() -> int:
    now = datetime.now(tz=timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


# ── Fixture injury lookup ─────────────────────────────────────────────────────

def _get_fixture_id(league_id: int, home_team_name: str, away_team_name: str, api_key: str) -> int | None:
    """Find API-Football fixture ID for an upcoming match."""
    today = datetime.now(tz=timezone.utc).date()
    from_date = today.isoformat()
    to_date = (today + timedelta(days=8)).isoformat()
    data = _get(
        "/fixtures",
        {
            "league": league_id,
            "season": _current_season(),
            "from": from_date,
            "to": to_date,
            "status": "NS",  # Not Started
        },
        api_key,
    )
    for fixture in data.get("response", []):
        home = normalize_team(fixture["teams"]["home"]["name"])
        away = normalize_team(fixture["teams"]["away"]["name"])
        if home == home_team_name and away == away_team_name:
            return fixture["fixture"]["id"]
    return None


def _get_injuries_for_fixture(fixture_id: int, api_key: str) -> list[dict]:
    """Return raw injury/suspension list for a fixture."""
    data = _get("/injuries", {"fixture": fixture_id}, api_key)
    return data.get("response", [])


# ── Lambda adjustment computation ─────────────────────────────────────────────

def compute_injury_adjustments(
    league_name: str,
    home_team: str,
    away_team: str,
    api_key: str,
) -> dict[str, float]:
    """
    Return lambda adjustment multipliers for home and away teams.

    Returns:
        {
          "home_attack": float,   # multiplier for lambda_home (≤ 1.0)
          "away_attack": float,   # multiplier for lambda_away (≤ 1.0)
          "home_defence": float,  # multiplier for lambda_away due to home defence (≥ 1.0)
          "away_defence": float,  # multiplier for lambda_home due to away defence (≥ 1.0)
        }
    All default to 1.0 (no adjustment) on any error.
    """
    result = {
        "home_attack": 1.0,
        "away_attack": 1.0,
        "home_defence": 1.0,
        "away_defence": 1.0,
    }

    # Premier League: use free FPL API
    if league_name == "Premier League":
        return compute_pl_injury_adjustments(home_team, away_team)

    # Ligue 1 / La Liga: scrape sportsgambler.com (free, no key needed)
    if league_name in ("Ligue 1", "La Liga"):
        return compute_scraped_injury_adjustments(league_name, home_team, away_team)

    # Fallback: API-Football (requires paid plan for current season)
    if not api_key:
        return result

    league_cfg = LEAGUES.get(league_name)
    if not league_cfg:
        return result

    league_id = league_cfg["apifootball_id"]
    season = _current_season()

    try:
        fixture_id = _get_fixture_id(league_id, home_team, away_team, api_key)
        if fixture_id is None:
            log.debug("No fixture ID found for %s vs %s — skipping injury adjustment.", home_team, away_team)
            return result

        injuries = _get_injuries_for_fixture(fixture_id, api_key)
        if not injuries:
            return result

        # Group by team
        by_team: dict[str, list[dict]] = {home_team: [], away_team: []}
        for entry in injuries:
            team_name = normalize_team(entry["team"]["name"])
            if team_name in by_team:
                by_team[team_name].append(entry)

        for team_name, absent_players in by_team.items():
            attack_reduction = 0.0
            defence_increase = 0.0

            for player_entry in absent_players:
                player_id = player_entry["player"]["id"]
                stats = _player_stats(player_id, league_id, season, api_key)

                # Players with < 90 min this season have no meaningful contribution
                if stats["minutes"] < 90:
                    continue

                position = stats["position"]
                if position in _POSITION_ATTACKERS:
                    contribution = (
                        stats["goals_per90"]
                        + ASSIST_WEIGHT * stats["assists_per90"]
                    )
                    attack_reduction += contribution
                elif position in _POSITION_DEFENDERS:
                    defence_increase += stats["conceded_per90"] * DEF_CONCEDE_WEIGHT

            # Convert reductions to multipliers
            attack_mult = max(1.0 - MAX_LAMBDA_REDUCTION, 1.0 - attack_reduction)
            defence_mult = 1.0 + min(defence_increase, 0.30)  # cap boost at +30%

            if team_name == home_team:
                result["home_attack"] = round(attack_mult, 4)
                result["away_defence"] = round(defence_mult, 4)
            else:
                result["away_attack"] = round(attack_mult, 4)
                result["home_defence"] = round(defence_mult, 4)

        log.info(
            "Injury adjustments for %s vs %s: %s",
            home_team, away_team, result,
        )

    except Exception as exc:
        log.warning("Injury fetch failed for %s vs %s: %s", home_team, away_team, exc)

    return result
