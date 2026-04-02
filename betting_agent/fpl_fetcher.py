"""
Fetch Premier League injury/availability data from the official FPL API.

No API key required. Used as a free fallback for the Premier League when
API-Football's paid plan is not available.

Player availability is read from:
  - status: 'a' available, 'd' doubtful, 'i' injured, 's' suspended, 'u' unavailable
  - chance_of_playing_next_round: 0-100 or null

The absence probability drives a partial lambda adjustment:
  - 0 % chance (or null + not available) → fully absent
  - 25 % chance → 75 % weight applied to contribution
  - 50 % chance → 50 % weight
  - 75 % chance → 25 % weight (still likely to play, small adjustment)
  - 100 % / status 'a' → ignored
"""

from __future__ import annotations

import logging

import requests

from .config import ASSIST_WEIGHT, DEF_CONCEDE_WEIGHT, MAX_LAMBDA_REDUCTION, normalize_team

log = logging.getLogger(__name__)

_FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# FPL team name → canonical name (football-data.org spellings)
_FPL_TEAM_MAP: dict[str, str] = {
    "Arsenal":        "Arsenal FC",
    "Aston Villa":    "Aston Villa FC",
    "Bournemouth":    "AFC Bournemouth",
    "Brentford":      "Brentford FC",
    "Brighton":       "Brighton & Hove Albion FC",
    "Chelsea":        "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton":        "Everton FC",
    "Fulham":         "Fulham FC",
    "Ipswich":        "Ipswich Town FC",
    "Leicester":      "Leicester City FC",
    "Liverpool":      "Liverpool FC",
    "Man City":       "Manchester City FC",
    "Man Utd":        "Manchester United FC",
    "Newcastle":      "Newcastle United FC",
    "Nott'm Forest":  "Nottingham Forest FC",
    "Southampton":    "Southampton FC",
    "Spurs":          "Tottenham Hotspur FC",
    "West Ham":       "West Ham United FC",
    "Wolves":         "Wolverhampton Wanderers FC",
}

# FPL element_type → role
_GK  = 1
_DEF = 2
_MID = 3
_FWD = 4


def _absence_probability(status: str, chance: int | None) -> float:
    """Return probability (0–1) that a player will NOT play next round."""
    if status == "a":
        return 0.0
    if status in ("i", "s", "u"):
        return 1.0
    # Doubtful — use chance_of_playing inverted
    if chance is None:
        return 1.0
    return 1.0 - chance / 100.0


def get_pl_unavailable_players() -> dict[str, list[dict]]:
    """
    Return a dict mapping canonical PL team name → list of unavailable/doubtful
    player dicts with their contribution scores and absence probability.

    Returns {} on any network error.
    """
    try:
        resp = requests.get(_FPL_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("FPL API fetch failed: %s", exc)
        return {}

    # Build id → canonical team name mapping
    team_map: dict[int, str] = {}
    for team in data.get("teams", []):
        canonical = _FPL_TEAM_MAP.get(team["name"])
        if canonical:
            team_map[team["id"]] = canonical

    result: dict[str, list[dict]] = {}

    for player in data.get("elements", []):
        status = player.get("status", "a")
        chance = player.get("chance_of_playing_next_round")
        absence_prob = _absence_probability(status, chance)

        if absence_prob == 0.0:
            continue  # fully available

        team_id = player.get("team")
        team_name = team_map.get(team_id)
        if not team_name:
            continue  # team not in our leagues

        minutes = player.get("minutes", 0) or 0
        if minutes < 90:
            continue  # no meaningful current-season data

        el_type = player.get("element_type")
        goals_per90 = player.get("goals_scored", 0) / (minutes / 90)
        assists_per90 = player.get("assists", 0) / (minutes / 90)
        conceded_per90 = player.get("goals_conceded_per_90") or 0.0

        if el_type in (_MID, _FWD):
            contribution = goals_per90 + ASSIST_WEIGHT * assists_per90
            role = "attack"
        else:  # GK / DEF
            contribution = conceded_per90 * DEF_CONCEDE_WEIGHT
            role = "defence"

        if contribution == 0.0:
            continue

        entry = {
            "name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "role": role,
            "contribution": contribution,
            "absence_prob": absence_prob,
            "status": status,
            "news": player.get("news", ""),
        }

        result.setdefault(team_name, []).append(entry)

    log.info("FPL: found unavailable/doubtful players at %d PL clubs.", len(result))
    return result


def compute_pl_injury_adjustments(home_team: str, away_team: str) -> dict[str, float]:
    """
    Compute lambda adjustment multipliers for a PL match using FPL data.

    Same return format as injury_fetcher.compute_injury_adjustments():
        home_attack, away_attack, home_defence, away_defence
    """
    result = {
        "home_attack": 1.0,
        "away_attack": 1.0,
        "home_defence": 1.0,
        "away_defence": 1.0,
    }

    unavailable = get_pl_unavailable_players()
    if not unavailable:
        return result

    for team_name, adj_key_attack, adj_key_defence in [
        (home_team, "home_attack", "away_defence"),
        (away_team, "away_attack", "home_defence"),
    ]:
        players = unavailable.get(team_name, [])
        attack_reduction = 0.0
        defence_increase = 0.0

        for p in players:
            weighted = p["contribution"] * p["absence_prob"]
            if p["role"] == "attack":
                attack_reduction += weighted
                log.debug(
                    "FPL absent (%s): %s — attack contrib %.3f × %.0f%% absence",
                    team_name, p["name"], p["contribution"], p["absence_prob"] * 100,
                )
            else:
                defence_increase += weighted
                log.debug(
                    "FPL absent (%s): %s — defence contrib %.3f × %.0f%% absence",
                    team_name, p["name"], p["contribution"], p["absence_prob"] * 100,
                )

        result[adj_key_attack] = round(
            max(1.0 - MAX_LAMBDA_REDUCTION, 1.0 - attack_reduction), 4
        )
        result[adj_key_defence] = round(
            1.0 + min(defence_increase, 0.30), 4
        )

    log.info("FPL adjustments for %s vs %s: %s", home_team, away_team, result)
    return result
