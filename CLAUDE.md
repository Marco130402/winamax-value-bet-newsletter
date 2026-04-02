# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full pipeline (dry run — no Telegram send)
python main.py --dry-run

# Send the newsletter to Telegram
python main.py

# Start the weekly scheduler (fires every Friday at 08:00)
python scheduler.py

# Record match results interactively
python main.py --record-result

# Print ROI performance report
python main.py --report
```

## Architecture

The pipeline runs in five sequential steps (see `main.py:run_pipeline`):

1. **Historical data** (`betting_agent/data_fetcher.py`) — fetches finished match results from football-data.org and caches them as CSVs in `data/cache/`. Incremental on subsequent runs (re-fetches current season only). Free tier blocks seasons older than ~3 years (handles 403 gracefully).

2. **Poisson model** (`betting_agent/model.py`) — fits a Dixon-Coles Poisson model per league. Parameters are cached to `data/cache/model_params_{league}.json` keyed by match count; reloaded without re-fitting if data hasn't changed. Fitting takes ~2 min per league on cold start. Accepts optional `injury_adjustments` dict in `predict()` that multiplies lambda_home/lambda_away.

3. **Odds fetch** (`betting_agent/odds_fetcher.py`) — calls The Odds API once per league, returns a flat DataFrame of all bookmaker odds. `BOOKMAKER_WINAMAX = "winamax_fr"` (not `"winamax"` — the API uses region-suffixed keys). Consensus pool is defined in `config.BOOKMAKERS_CONSENSUS`.

4. **Value detection** (`betting_agent/value_detector.py`) — merges Winamax and consensus rows on `match_id`, then for each of 3 outcomes (home/draw/away) checks model EV > `EV_THRESHOLD`. Model EV is the **sole qualifier**; consensus implied prob is attached as context in the newsletter but does not gate bets. Left-joins so Winamax matches without consensus data are still included. Returns top `MAX_BETS` sorted by EV.

4b. **Kelly stake sizing** (`betting_agent/kelly.py`) — computes optimal bankroll fractions by maximising E[log(1 + Σ fᵢ·rᵢ)] across all 2^n outcome scenarios via scipy L-BFGS-B. Raw fractions are then scaled by `KELLY_FRACTION` (0.25), capped per bet at `MAX_SINGLE_BET` (5%), and proportionally rescaled if the total exceeds `MAX_TOTAL_EXPOSURE` (25%). Returns a list of fractions in bet order; each bet gets a `kelly_stake_pct` field.

5. **Newsletter + tracker** (`betting_agent/newsletter.py`, `betting_agent/tracker.py`, `betting_agent/telegram_sender.py`) — formats HTML (shows per-bet stake and total exposure), logs bets to `data/bets_log.csv` (including `kelly_stake_pct`), splits into ≤4096-char chunks and POSTs to Telegram Bot API. `--report` prints both flat-stake and Kelly-weighted ROI.

**Injury adjustment** (`betting_agent/injury_fetcher.py`) is optional (skipped if `API_FOOTBALL_KEY` not set, but operates without it for the two main sources). Routes by league:
- **Premier League** → `betting_agent/fpl_fetcher.py`: official FPL API (`fantasy.premierleague.com/api/bootstrap-static/`), no key required.
- **Ligue 1 / La Liga** → `betting_agent/sportsgambler_scraper.py`: scrapes `sportsgambler.com/injuries/football/{league}/`, logs a warning if the fetch returns 0 players.
- **Other leagues** → API-Football (paid; not currently used).

## Key constants (`betting_agent/config.py`)

| Constant | Default | Effect |
|---|---|---|
| `EV_THRESHOLD` | 0.05 | Minimum model expected value to flag a bet (sole qualifier) |
| `DIXON_COLES_XI` | 0.0018 | Time-decay rate; half-life ≈385 days |
| `MAX_BETS` | 10 | Max bets per newsletter |
| `BOOKMAKER_WINAMAX` | `"winamax_fr"` | The Odds API key for Winamax |
| `KELLY_FRACTION` | 0.25 | Scale raw Kelly fractions by this (quarter Kelly) |
| `MAX_SINGLE_BET` | 0.05 | Hard cap per bet (5% of bankroll) |
| `MAX_TOTAL_EXPOSURE` | 0.25 | Hard cap on total weekly exposure (25% of bankroll) |

## Environment variables (`.env`)

```
FOOTBALL_DATA_API_KEY   # football-data.org — historical results
THE_ODDS_API_KEY        # the-odds-api.com — live odds (500 credits/month free; ~7 used/week)
API_FOOTBALL_KEY        # api-football.com — injuries (optional; PL uses FPL API, L1/LL use scraper instead)
TELEGRAM_BOT_TOKEN      # from @BotFather
TELEGRAM_CHAT_ID        # numeric personal chat ID (get from /getUpdates)
```

## Python version

Runs on Python 3.9. All type hints use `from __future__ import annotations` for `X | Y` union syntax compatibility.

## Data flow notes

- Team names are normalised via `config.normalize_team()` at every ingestion point (football-data.org names are canonical).
- The model and odds fetcher use the same canonical names, so `model.predict(home, away)` will return `None` for teams not seen in training (promoted clubs) — `value_detector` falls back to consensus-only for those matches.
- `data/cache/` is gitignored. On a fresh clone, the first `--dry-run` bootstraps all data (~3–5 min).
