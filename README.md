# Winamax Value Bet Newsletter

A Python tool that identifies mispriced odds on [Winamax](https://www.winamax.fr/) for Ligue 1, La Liga, and the Premier League, then sends a weekly Telegram newsletter every Friday.

## How it works

Value bets are flagged only when **both** signals agree:

1. **Statistical model** — A [Dixon-Coles](https://academic.oup.com/jrsssc/article-abstract/46/2/265/6990546) Poisson model trained on the last 2+ seasons of historical results estimates the true win/draw/loss probability for each match. A bet qualifies if the model's expected value against Winamax's odds exceeds **4%**.

2. **Market consensus** — Odds from Bet365, Unibet, Betclic, and Pinnacle are averaged into a consensus. A bet qualifies if Winamax's implied probability lags the consensus by at least **5 percentage points**.

The dual-gate approach reduces noise: the model catches structural edges, the consensus comparison confirms the market agrees Winamax is mispricing.

## Data sources

| Source | Used for | Free tier |
|---|---|---|
| [football-data.org](https://www.football-data.org/) | Historical match results (model training) | 10 req/min |
| [The Odds API](https://the-odds-api.com/) | Live odds for Winamax + consensus books | 500 credits/month (~13 used/month) |

Historical odds from The Odds API are **not** needed — the model trains on scores only.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
FOOTBALL_DATA_API_KEY=...   # football-data.org free account
THE_ODDS_API_KEY=...        # the-odds-api.com free account
TELEGRAM_BOT_TOKEN=...      # from @BotFather → /newbot
TELEGRAM_CHAT_ID=...        # see below
```

**Getting your Telegram chat ID:** Send any message to your bot, then open:
```
https://api.telegram.org/bot{YOUR_TOKEN}/getUpdates
```
The `chat.id` field in the response is your chat ID.

### 3. Test locally (dry run)

```bash
python main.py --dry-run
```

This fetches all data and prints the newsletter to stdout without sending it to Telegram. The first run bootstraps ~2 seasons of historical data per league (~30 seconds due to API rate limiting).

### 4. Start the scheduler

```bash
python scheduler.py
```

Sends the newsletter every **Friday at 08:00** local time. Run in the background with:

```bash
nohup python scheduler.py > logs/scheduler.log 2>&1 &
```

## Project structure

```
betting_agent/
├── main.py                    # Pipeline entrypoint
├── scheduler.py               # Weekly Friday trigger
├── betting_agent/
│   ├── config.py              # Leagues, thresholds, team name map
│   ├── data_fetcher.py        # Historical results from football-data.org
│   ├── model.py               # Poisson/Dixon-Coles model
│   ├── odds_fetcher.py        # The Odds API (Winamax + consensus books)
│   ├── value_detector.py      # Dual-gate EV + consensus filtering
│   ├── newsletter.py          # Telegram HTML formatter
│   └── telegram_sender.py     # Telegram Bot API delivery
└── data/cache/                # Auto-generated (gitignored)
    ├── ligue_1_results.csv
    ├── la_liga_results.csv
    ├── premier_league_results.csv
    └── model_params.json
```

## Model details

Goals are modelled as independent Poisson processes:

```
λ_home = exp(μ + attack_i + defence_j + home_advantage)
λ_away = exp(μ + attack_j + defence_i)
```

Parameters are estimated by maximising the time-weighted log-likelihood using `scipy.optimize.minimize` (L-BFGS-B). Matches are weighted by `exp(-0.0018 × days_ago)`, giving a half-life of ~385 days so recent form matters more than old results.

The Dixon-Coles correction adjusts probabilities for low-scoring outcomes (0-0, 1-0, 0-1, 1-1), which the raw Poisson model systematically mis-estimates.

Win/draw/loss probabilities are computed by summing over an 8×8 scoreline matrix (covers 99.97% of real outcomes).

## Key parameters

| Parameter | Value | Rationale |
|---|---|---|
| EV threshold | 4% | Below this, noise dominates |
| Consensus gap | 5pp | Meaningful mispricing signal |
| Time decay (ξ) | 0.0018 | ~385-day half-life (standard in literature) |
| Max bets/newsletter | 10 | Keeps the message readable |

## Planned enhancements (Phase 2)

- **Injury/suspension adjustment** — integrate [API-Football](https://www.api-football.com/) to apply a lambda adjustment based on current-season goal contributions of absent players. Friday morning timing is ideal as most team news is confirmed by then.

## Disclaimer

This tool is for informational and educational purposes. It does not constitute financial advice. Bet responsibly.
