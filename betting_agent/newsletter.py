"""Format the weekly value-bet newsletter as a Telegram HTML string."""

from .config import KELLY_FRACTION, LEAGUES, MAX_SINGLE_BET, MAX_TOTAL_EXPOSURE

_LEAGUE_EMOJI = {
    "Ligue 1": "🇫🇷",
    "La Liga": "🇪🇸",
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Bundesliga 2": "🇩🇪",
    "Eredivisie": "🇳🇱",
}


def _format_bet(bet: dict) -> str:
    consensus_line = ""
    if bet.get("consensus_implied_pct") is not None:
        edge = bet.get("consensus_edge_pct")
        edge_str = f" | Market edge: {edge:+.1f}pp" if edge is not None else ""
        consensus_line = f"\nMarket: {bet['consensus_implied_pct']}% implied{edge_str}"

    stake = bet.get("kelly_stake_pct")
    stake_line = f"\n📐 Stake: <b>{stake:.2f}% of bankroll</b>" if stake else ""

    exp_goals = bet.get("exp_goals")
    goals_line = f"\n⚽ Model expects: <b>{exp_goals} goals total</b>" if exp_goals else ""

    bookmaker = bet.get("bookmaker") or "Winamax"
    # Convert raw API key to readable name (e.g. "pinnacle" → "Pinnacle")
    bookmaker_display = bookmaker.replace("_", " ").title()

    return (
        f"<b>{bet['match']}</b> — {bet['date']}\n"
        f"Outcome: <b>{bet['outcome']}</b> @ {bet['winamax_odd']} ({bookmaker_display})\n"
        f"Model prob: {bet['model_prob']}% | EV: <b>+{bet['model_ev_pct']}%</b>"
        + consensus_line
        + goals_line
        + stake_line
    )


def format_newsletter(value_bets: list[dict], run_date: str) -> str:
    if not value_bets:
        return (
            f"<b>Winamax Value Bets — {run_date}</b>\n\n"
            "No value bets found this week across Ligue 1, La Liga, and the Premier League.\n\n"
            "<i>Data via The Odds API + football-data.org. Not financial advice.</i>"
        )

    header = (
        f"<b>Winamax Value Bets — {run_date}</b>\n\n"
        "<b>Methodology</b>\n"
        "Poisson/Dixon-Coles model EV ≥ 5%. "
        f"Stakes: {int(KELLY_FRACTION * 100)}% Kelly, "
        f"max {int(MAX_SINGLE_BET * 100)}% per bet, "
        f"max {int(MAX_TOTAL_EXPOSURE * 100)}% total.\n"
    )

    # Group bets by league, preserving the league order from config
    by_league: dict[str, list[dict]] = {lg: [] for lg in LEAGUES}
    for bet in value_bets:
        lg = bet["league"]
        if lg in by_league:
            by_league[lg].append(bet)

    sections = []
    for league_name, bets in by_league.items():
        if not bets:
            continue
        emoji = _LEAGUE_EMOJI.get(league_name, "⚽")
        section = f"{emoji} <b>{league_name}</b>\n\n"
        section += "\n\n".join(_format_bet(b) for b in bets)
        sections.append(section)

    total_stake = sum(b.get("kelly_stake_pct", 0) for b in value_bets)
    total = len(value_bets)
    footer = (
        f"\n\n<b>Total exposure: {total_stake:.1f}% of bankroll</b>\n"
        f"<i>{total} value bet{'s' if total != 1 else ''} | "
        "The Odds API + football-data.org | Not financial advice.</i>"
    )

    body = "\n\n".join(sections)
    return header + "\n" + body + footer
