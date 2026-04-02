"""
Scrape injury and suspension data from sportsgambler.com for Ligue 1 and La Liga.

No API key required. Data is server-side rendered and includes player name,
position, season goals/assists, and absence type per team.

Absence types:
  injury-plus        → confirmed injury   (100% absent)
  redcard            → suspended           (100% absent)
  injury-questionmark → doubtful           (50% absent)

Flags a WARNING in the pipeline output when scraping returns no data for a team,
so misconfigurations or site changes are visible immediately.
"""

from __future__ import annotations

import logging
import time

import requests
from bs4 import BeautifulSoup

from .config import ASSIST_WEIGHT, DEF_CONCEDE_WEIGHT, MAX_LAMBDA_REDUCTION, normalize_team

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_LEAGUE_URLS = {
    "Ligue 1":  "https://www.sportsgambler.com/injuries/football/france-ligue-1/",
    "La Liga":  "https://www.sportsgambler.com/injuries/football/spain-la-liga/",
}

# CSS class on .inj-type span → absence probability
_TYPE_ABSENCE: dict[str, float] = {
    "injury-plus":         1.0,   # confirmed injury
    "redcard":             1.0,   # suspended
    "injury-questionmark": 0.5,   # doubtful
}

# Short position code → role
_ATTACK_POSITIONS  = {"F", "AM", "M", "W", "AML", "AMR", "AMC"}
_DEFENCE_POSITIONS = {"D", "GK", "DC", "DL", "DR", "DM"}


def _absence_prob_from_classes(classes: list[str]) -> float:
    for cls in classes:
        if cls in _TYPE_ABSENCE:
            return _TYPE_ABSENCE[cls]
    return 0.0


def _parse_int(text: str) -> int:
    try:
        return int(text.strip().replace("-", "0") or 0)
    except ValueError:
        return 0


def _scrape_league(league_name: str) -> dict[str, list[dict]]:
    """
    Return {canonical_team_name: [player_dict, ...]} for all absent/doubtful
    players in the league.

    Logs a WARNING (visible in pipeline output) if the page cannot be fetched
    or returns no data.
    """
    url = _LEAGUE_URLS[league_name]
    result: dict[str, list[dict]] = {}

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        log.warning(
            "⚠️  [%s] Injury scrape FAILED — could not fetch %s: %s",
            league_name, url, exc,
        )
        return result

    soup = BeautifulSoup(resp.text, "lxml")
    blocks = soup.select(".injury-block")

    if not blocks:
        log.warning(
            "⚠️  [%s] Injury scrape returned NO data — site HTML may have changed (%s)",
            league_name, url,
        )
        return result

    for block in blocks:
        team_el = block.select_one("h3.injuries-title a")
        if not team_el:
            continue
        raw_team = team_el.get_text(strip=True)
        team_name = normalize_team(raw_team)

        players: list[dict] = []
        for row in block.select(".inj-row"):
            container = row.select_one(".inj-container")
            if not container:
                continue

            inj_type_el = container.select_one(".inj-type")
            if not inj_type_el:
                continue

            type_classes = inj_type_el.get("class", [])
            absence_prob = _absence_prob_from_classes(type_classes)
            if absence_prob == 0.0:
                continue

            name_el  = container.select_one(".inj-player")
            pos_el   = container.select_one(".inj-position")
            games_el = container.select_one(".inj-game")
            goals_el = container.select_one(".inj-goals")
            asst_el  = container.select_one(".inj-assist")
            info_el  = container.select_one(".inj-info")

            name   = name_el.get_text(strip=True)  if name_el  else "Unknown"
            pos    = pos_el.get_text(strip=True)    if pos_el   else ""
            games  = _parse_int(games_el.get_text() if games_el else "0")
            goals  = _parse_int(goals_el.get_text() if goals_el else "0")
            assists= _parse_int(asst_el.get_text()  if asst_el  else "0")
            info   = info_el.get_text(strip=True)   if info_el  else ""

            if games < 1:
                continue  # no meaningful data

            # Contribution: treat games ≈ 90-min appearances (slight underestimate)
            goals_per90   = goals   / games
            assists_per90 = assists / games

            pos_upper = pos.upper()
            if pos_upper in _ATTACK_POSITIONS:
                contribution = goals_per90 + ASSIST_WEIGHT * assists_per90
                role = "attack"
            elif pos_upper in _DEFENCE_POSITIONS:
                contribution = DEF_CONCEDE_WEIGHT  # use fixed weight (no conceded data)
                role = "defence"
            else:
                continue  # unknown position — skip rather than guess

            players.append({
                "name":         name,
                "position":     pos,
                "role":         role,
                "contribution": contribution,
                "absence_prob": absence_prob,
                "info":         info,
            })

        if players:
            result[team_name] = players

    scraped_teams = len(result)
    total_absent  = sum(len(v) for v in result.values())

    if scraped_teams == 0:
        log.warning(
            "⚠️  [%s] Injury scrape returned 0 teams with absent players — "
            "check %s manually before placing bets.",
            league_name, url,
        )
    else:
        log.info(
            "[%s] Scraped %d absent/doubtful players across %d clubs.",
            league_name, total_absent, scraped_teams,
        )

    return result


# Module-level cache so we only scrape each league once per pipeline run
_cache: dict[str, dict[str, list[dict]]] = {}


def get_absent_players(league_name: str) -> dict[str, list[dict]]:
    """Return cached scrape result for the league (scraped once per process)."""
    if league_name not in _cache:
        _cache[league_name] = _scrape_league(league_name)
        time.sleep(1.5)  # be polite between league requests
    return _cache[league_name]


def compute_scraped_injury_adjustments(
    league_name: str,
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    """
    Return lambda multipliers for a match using scraped injury/suspension data.
    Same return format as injury_fetcher.compute_injury_adjustments().
    """
    result = {
        "home_attack": 1.0,
        "away_attack": 1.0,
        "home_defence": 1.0,
        "away_defence": 1.0,
    }

    absent = get_absent_players(league_name)

    for team_name, atk_key, def_key in [
        (home_team, "home_attack", "away_defence"),
        (away_team, "away_attack", "home_defence"),
    ]:
        players = absent.get(team_name, [])
        if not players:
            continue

        attack_reduction = 0.0
        defence_increase = 0.0

        for p in players:
            weighted = p["contribution"] * p["absence_prob"]
            if p["role"] == "attack":
                attack_reduction += weighted
                log.debug(
                    "Absent (%s): %s [%s] %.0f%% — atk_contrib=%.3f",
                    team_name, p["name"], p["info"], p["absence_prob"] * 100, p["contribution"],
                )
            else:
                defence_increase += weighted
                log.debug(
                    "Absent (%s): %s [%s] %.0f%% — def_contrib=%.3f",
                    team_name, p["name"], p["info"], p["absence_prob"] * 100, p["contribution"],
                )

        result[atk_key] = round(
            max(1.0 - MAX_LAMBDA_REDUCTION, 1.0 - attack_reduction), 4
        )
        result[def_key] = round(
            1.0 + min(defence_increase, 0.30), 4
        )

    log.info("Injury adjustments [%s] %s vs %s: %s", league_name, home_team, away_team, result)
    return result
