"""
Weekly Winamax value-bet newsletter pipeline.

Usage:
    python main.py                 # full run (fetches data, sends Telegram message)
    python main.py --dry-run       # run without sending to Telegram (prints to stdout)
    python main.py --report        # print ROI performance report
    python main.py --record-result # interactively record match results
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from betting_agent.config import MIN_MATCHES_FOR_MODEL
from betting_agent.data_fetcher import get_all_historical_data
from betting_agent.kelly import compute_kelly_stakes
from betting_agent.model import PoissonModel
from betting_agent.newsletter import format_newsletter
from betting_agent.odds_fetcher import fetch_all_leagues
from betting_agent.telegram_sender import send_message
from betting_agent.tracker import log_bets, print_report, record_result_interactive
from betting_agent.value_detector import find_all_value_bets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        log.error("Missing required environment variable: %s", key)
        sys.exit(1)
    return val


def run_pipeline(dry_run: bool = False) -> None:
    load_dotenv()

    fd_api_key = _require_env("FOOTBALL_DATA_API_KEY")
    odds_api_key = _require_env("THE_ODDS_API_KEY")
    injury_api_key = os.getenv("API_FOOTBALL_KEY")  # optional — skipped if not set
    tg_token = _require_env("TELEGRAM_BOT_TOKEN")
    tg_chat_id = _require_env("TELEGRAM_CHAT_ID")
    if not injury_api_key:
        log.info("API_FOOTBALL_KEY not set — injury adjustment disabled.")

    run_date = datetime.now(tz=timezone.utc).strftime("%-d %B %Y")
    log.info("=== Pipeline start — %s ===", run_date)

    # ── 1. Historical data ────────────────────────────────────────────────────
    log.info("Step 1/5: Fetching historical match results …")
    historical = get_all_historical_data(fd_api_key)
    for lg, df in historical.items():
        log.info("  %s: %d matches in cache.", lg, len(df))

    # ── 2. Train Poisson models ───────────────────────────────────────────────
    log.info("Step 2/5: Fitting Poisson models …")
    models: dict[str, PoissonModel] = {}
    for league_name, df in historical.items():
        if len(df) < MIN_MATCHES_FOR_MODEL:
            log.warning(
                "Not enough data for %s (%d matches) — skipping model, will use consensus-only.",
                league_name, len(df),
            )
            continue
        model = PoissonModel(league_name=league_name)
        try:
            model.fit(df)
            models[league_name] = model
            log.info("  %s: model ready.", league_name)
        except Exception as exc:
            log.error("  %s: model fitting failed — %s", league_name, exc)

    # ── 3. Fetch upcoming odds ────────────────────────────────────────────────
    log.info("Step 3/5: Fetching upcoming odds from The Odds API …")
    h2h_df, totals_df = fetch_all_leagues(odds_api_key)
    if h2h_df.empty:
        log.error("No odds data retrieved. Aborting.")
        return
    log.info("  %d h2h rows, %d O/U rows fetched.", len(h2h_df), len(totals_df))

    # ── 4. Detect value bets ──────────────────────────────────────────────────
    log.info("Step 4/5: Detecting value bets …")
    value_bets = find_all_value_bets(
        h2h_df, models, totals_df=totals_df, injury_api_key=injury_api_key
    )
    log.info("  %d value bet(s) found.", len(value_bets))

    # ── 4b. Kelly portfolio sizing ────────────────────────────────────────────
    if value_bets:
        stakes = compute_kelly_stakes(value_bets)
        for bet, stake in zip(value_bets, stakes):
            bet["kelly_stake_pct"] = round(stake * 100, 2)
    else:
        for bet in value_bets:
            bet["kelly_stake_pct"] = 0.0

    # ── 5. Format, log, and send newsletter ──────────────────────────────────
    log.info("Step 5/5: Sending newsletter …")
    message = format_newsletter(value_bets, run_date)
    log_bets(value_bets, run_date)

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN — newsletter not sent to Telegram.")
        print("=" * 60 + "\n")
        print(message)
        return

    send_message(tg_token, tg_chat_id, message)
    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Winamax value-bet newsletter pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print newsletter without sending to Telegram.")
    parser.add_argument("--report", action="store_true", help="Print ROI performance report.")
    parser.add_argument("--record-result", action="store_true", help="Interactively record match results.")
    args = parser.parse_args()

    if args.report:
        load_dotenv()
        print_report()
    elif args.record_result:
        load_dotenv()
        record_result_interactive()
    else:
        run_pipeline(dry_run=args.dry_run)
