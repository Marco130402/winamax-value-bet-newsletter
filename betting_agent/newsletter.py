"""Format the weekly value-bet newsletter as a Telegram HTML string."""

from .config import LEAGUES

_LEAGUE_EMOJI = {
    "Ligue 1": "🇫🇷",
    "La Liga": "🇪🇸",
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
}


def _format_bet(bet: dict) -> str:
    model_line = ""
    if bet["model_available"] and bet["model_prob"] is not None:
        model_line = (
            f"Model prob: {bet['model_prob']}% | "
            f"Model EV: <b>+{bet['model_ev_pct']}%</b>\n"
        )
    else:
        model_line = "<i>Model: N/A (consensus-only signal)</i>\n"

    return (
        f"<b>{bet['match']}</b> — {bet['date']}\n"
        f"Outcome: <b>{bet['outcome']}</b> @ {bet['winamax_odd']} (Winamax)\n"
        + model_line
        + f"Consensus: {bet['consensus_implied_pct']}% | "
        f"Winamax: {bet['winamax_implied_pct']}% | "
        f"Edge: {bet['consensus_edge_pct']}pp"
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
        "Bets must pass BOTH: (1) Poisson/Dixon-Coles model EV ≥ 4%, "
        "(2) Winamax implied prob lags market consensus by ≥ 5pp.\n"
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

    total = len(value_bets)
    n_model = sum(1 for b in value_bets if b["model_available"])
    footer = (
        f"\n\n<i>{total} value bet{'s' if total != 1 else ''} found "
        f"({n_model} model-confirmed, {total - n_model} consensus-only). "
        "Data via The Odds API + football-data.org.</i>\n"
        "<i>Not financial advice. Bet responsibly.</i>"
    )

    body = "\n\n".join(sections)
    return header + "\n" + body + footer
