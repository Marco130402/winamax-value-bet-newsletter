"""
Performance tracker — logs recommended bets and computes ROI.

Usage
-----
Log bets automatically (called from main.py after each run):
    tracker.log_bets(value_bets, run_date)

Record results manually (after matches are played):
    python main.py --record-result

Print ROI report:
    python main.py --report

CSV schema (data/bets_log.csv)
-------------------------------
week_date, match, league, date, outcome, winamax_odd,
model_prob, model_ev_pct, consensus_edge_pct,
result      # "W", "L", "D (void)", "" = not yet recorded
"""

import csv
import logging
from pathlib import Path

from .config import BASE_DIR

log = logging.getLogger(__name__)

_LOG_PATH = BASE_DIR / "data" / "bets_log.csv"
_FIELDNAMES = [
    "week_date",
    "match",
    "league",
    "date",
    "outcome",
    "winamax_odd",
    "model_prob",
    "model_ev_pct",
    "consensus_edge_pct",
    "kelly_stake_pct",
    "result",
]


def _ensure_log() -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        with open(_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()


def log_bets(value_bets: list[dict], week_date: str) -> None:
    """Append this week's recommended bets to the log (result left blank)."""
    _ensure_log()
    with open(_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        for bet in value_bets:
            writer.writerow(
                {
                    "week_date": week_date,
                    "match": bet["match"],
                    "league": bet["league"],
                    "date": bet["date"],
                    "outcome": bet["outcome"],
                    "winamax_odd": bet["winamax_odd"],
                    "model_prob": bet.get("model_prob", ""),
                    "model_ev_pct": bet.get("model_ev_pct", ""),
                    "consensus_edge_pct": bet.get("consensus_edge_pct", ""),
                    "kelly_stake_pct": bet.get("kelly_stake_pct", ""),
                    "result": "",
                }
            )
    log.info("Logged %d bets to %s.", len(value_bets), _LOG_PATH)


def print_report() -> None:
    """Print flat-stake P&L and cumulative ROI from the bets log."""
    _ensure_log()
    rows = []
    with open(_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No bets logged yet.")
        return

    settled = [r for r in rows if r["result"] in ("W", "L")]
    pending = [r for r in rows if not r["result"]]

    total_bets = len(settled)
    wins   = [r for r in settled if r["result"] == "W"]
    losses = [r for r in settled if r["result"] == "L"]

    # Flat-stake ROI (1 unit per bet)
    flat_profit = sum(float(r["winamax_odd"]) - 1.0 for r in wins) - len(losses)
    flat_roi    = (flat_profit / total_bets * 100) if total_bets > 0 else 0.0

    # Kelly-weighted ROI (stake = kelly_stake_pct % of bankroll)
    kelly_pnl = 0.0
    kelly_staked = 0.0
    for r in settled:
        try:
            stake = float(r.get("kelly_stake_pct") or 0) / 100
        except ValueError:
            stake = 0.0
        kelly_staked += stake
        if r["result"] == "W":
            kelly_pnl += stake * (float(r["winamax_odd"]) - 1.0)
        else:
            kelly_pnl -= stake
    kelly_roi = (kelly_pnl / kelly_staked * 100) if kelly_staked > 0 else 0.0

    print("\n" + "=" * 50)
    print("  WINAMAX VALUE BET — PERFORMANCE REPORT")
    print("=" * 50)
    print(f"  Settled bets   : {total_bets}")
    print(f"  Wins           : {len(wins)}")
    print(f"  Losses         : {len(losses)}")
    print(f"  Win rate       : {len(wins)/total_bets*100:.1f}%" if total_bets else "  Win rate       : —")
    print(f"  Flat P&L       : {flat_profit:+.2f} units  (ROI {flat_roi:+.1f}%)")
    print(f"  Kelly P&L      : {kelly_pnl:+.3f} bankroll (ROI {kelly_roi:+.1f}%)")
    print(f"  Pending bets   : {len(pending)}")
    print("=" * 50 + "\n")

    if pending:
        print("Pending (result not recorded):")
        for r in pending:
            print(f"  [{r['week_date']}] {r['match']} — {r['outcome']} @ {r['winamax_odd']}")
        print()


def record_result_interactive() -> None:
    """Interactive CLI to record results for pending bets."""
    _ensure_log()
    rows = []
    with open(_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    pending = [(i, r) for i, r in enumerate(rows) if not r["result"]]
    if not pending:
        print("No pending bets to record.")
        return

    print(f"\n{len(pending)} pending bet(s) to record:\n")
    for idx, (row_i, row) in enumerate(pending):
        print(
            f"  [{idx + 1}] {row['match']} — {row['outcome']} @ {row['winamax_odd']} ({row['date']})"
        )
        result = input("       Result (W/L, or Enter to skip): ").strip().upper()
        if result in ("W", "L"):
            rows[row_i]["result"] = result

    with open(_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print("\nResults saved.\n")
