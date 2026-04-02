from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Leagues ───────────────────────────────────────────────────────────────────
# fd_code  : football-data.org competition code
# odds_key : The Odds API sport key
LEAGUES = {
    "Ligue 1": {
        "fd_code": "FL1",
        "odds_key": "soccer_france_ligue_one",
        "apifootball_id": 61,   # API-Football league ID
    },
    "La Liga": {
        "fd_code": "PD",
        "odds_key": "soccer_spain_la_liga",
        "apifootball_id": 140,
    },
    "Premier League": {
        "fd_code": "PL",
        "odds_key": "soccer_epl",
        "apifootball_id": 39,
    },
    "Bundesliga 2": {
        "fd_code": "BL2",
        "odds_key": "soccer_germany_bundesliga2",
        "apifootball_id": 78,
    },
    "Eredivisie": {
        "fd_code": "DED",
        "odds_key": "soccer_netherlands_eredivisie",
        "apifootball_id": 88,
    },
}

# ── Understat xG data ─────────────────────────────────────────────────────────
# Maps league name → Understat league key (only leagues covered by Understat)
UNDERSTAT_LEAGUE_MAP: dict[str, str] = {
    "Premier League": "EPL",
    "Ligue 1":        "Ligue_1",
    "La Liga":        "La_liga",
}

# ── Injury model weights ──────────────────────────────────────────────────────
# Combined contribution score = goals_per90 + ASSIST_WEIGHT * assists_per90
ASSIST_WEIGHT = 0.6         # assists count 60% vs a goal (industry-standard proxy)
# Maximum lambda reduction from absences — prevents absurd adjustments
MAX_LAMBDA_REDUCTION = 0.40  # lambda floored at 60% of base value
# Defensive: absent key CB/GK increases opponent's lambda_away (or lambda_home)
DEF_CONCEDE_WEIGHT = 0.15   # fraction of opponent lambda increase per key defender out

# ── Bookmakers ────────────────────────────────────────────────────────────────
# Winamax is intentionally excluded from the consensus pool so we can compare
# its odds against the rest of the market.
BOOKMAKERS_CONSENSUS = [
    "pinnacle",       # sharpest market — best reference for true probability
    "unibet_fr",
    "unibet_nl",
    "unibet_se",
    "betclic_fr",
    "williamhill",
    "marathonbet",
    "betsson",
]
BOOKMAKER_WINAMAX = "winamax_fr"

# ── Value detection thresholds ────────────────────────────────────────────────
EV_THRESHOLD = 0.05            # minimum expected value (5%) from the Poisson model
# Consensus is shown in the newsletter as context but does not gate bets

# ── Kelly staking ─────────────────────────────────────────────────────────────
KELLY_FRACTION   = 0.25   # use quarter Kelly (scales full Kelly down by 75%)
MAX_SINGLE_BET   = 0.05   # cap any single bet at 5% of bankroll
MAX_TOTAL_EXPOSURE = 0.25 # cap total weekly exposure at 25% of bankroll

# ── Model parameters ──────────────────────────────────────────────────────────
DIXON_COLES_XI = 0.0018        # time-decay rate (half-life ≈385 days)
MAX_GOALS = 8                  # score matrix dimension (covers 99.97% of outcomes)
MIN_MATCHES_FOR_MODEL = 30     # cold-start guard: skip model if fewer matches cached

# ── Scheduler / pipeline ──────────────────────────────────────────────────────
UPCOMING_DAYS = 8              # fetch odds for matches up to 8 days ahead
MAX_BETS = 10                  # maximum value bets to include in the newsletter

# ── Team name normalisation ───────────────────────────────────────────────────
# Maps every known variant → canonical name used internally.
# Canonical names are the football-data.org spellings.
TEAM_NAME_MAP: dict[str, str] = {
    # Ligue 1
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Paris SG": "Paris Saint-Germain",
    "PSG": "Paris Saint-Germain",
    "Olympique de Marseille": "Olympique Marseille",
    "Marseille": "Olympique Marseille",
    "Olympique Lyonnais": "Olympique Lyonnais",
    "Lyon": "Olympique Lyonnais",
    "AS Monaco FC": "AS Monaco",
    "Monaco": "AS Monaco",
    "Stade Rennais FC": "Stade Rennais FC 1901",
    "Rennes": "Stade Rennais FC 1901",
    "RC Lens": "Lens",
    "Lens": "Lens",
    "LOSC Lille": "Lille OSC",
    "Lille": "Lille OSC",
    "OGC Nice": "OGC Nice",
    "Nice": "OGC Nice",
    "Stade de Reims": "Stade de Reims",
    "Reims": "Stade de Reims",
    "RC Strasbourg Alsace": "RC Strasbourg Alsace",
    "Strasbourg": "RC Strasbourg Alsace",
    "Toulouse FC": "Toulouse FC",
    "Toulouse": "Toulouse FC",
    "Montpellier HSC": "Montpellier HSC",
    "Montpellier": "Montpellier HSC",
    "FC Nantes": "FC Nantes",
    "Nantes": "FC Nantes",
    "Girondins de Bordeaux": "Girondins de Bordeaux",
    "Bordeaux": "Girondins de Bordeaux",
    "Angers SCO": "Angers SCO",
    "Angers": "Angers SCO",
    "Clermont Foot 63": "Clermont Foot 63",
    "Clermont": "Clermont Foot 63",
    "FC Lorient": "FC Lorient",
    "Lorient": "FC Lorient",
    "Havre AC": "Le Havre AC",
    "Le Havre": "Le Havre AC",
    "Stade Brestois 29": "Stade Brestois 29",
    "Brest": "Stade Brestois 29",
    "AJ Auxerre": "AJ Auxerre",
    "Auxerre": "AJ Auxerre",
    "Saint-Etienne": "AS Saint-Étienne",
    "AS Saint-Etienne": "AS Saint-Étienne",
    # La Liga
    "FC Barcelona": "FC Barcelona",
    "Barcelona": "FC Barcelona",
    "Real Madrid CF": "Real Madrid CF",
    "Real Madrid": "Real Madrid CF",
    "Club Atlético de Madrid": "Club Atlético de Madrid",
    "Atletico Madrid": "Club Atlético de Madrid",
    "Atlético de Madrid": "Club Atlético de Madrid",
    "Sevilla FC": "Sevilla FC",
    "Sevilla": "Sevilla FC",
    "Real Betis Balompié": "Real Betis Balompié",
    "Real Betis": "Real Betis Balompié",
    "Real Sociedad de Fútbol": "Real Sociedad de Fútbol",
    "Real Sociedad": "Real Sociedad de Fútbol",
    "Athletic Club": "Athletic Club",
    "Athletic Bilbao": "Athletic Club",
    "Villarreal CF": "Villarreal CF",
    "Villarreal": "Villarreal CF",
    "Celta de Vigo": "Celta de Vigo",
    "Celta Vigo": "Celta de Vigo",
    "Rayo Vallecano de Madrid": "Rayo Vallecano de Madrid",
    "Rayo Vallecano": "Rayo Vallecano de Madrid",
    "Getafe CF": "Getafe CF",
    "Getafe": "Getafe CF",
    "UD Almería": "UD Almería",
    "Almeria": "UD Almería",
    "Osasuna": "CA Osasuna",
    "CA Osasuna": "CA Osasuna",
    "RCD Mallorca": "RCD Mallorca",
    "Mallorca": "RCD Mallorca",
    "Girona FC": "Girona FC",
    "Girona": "Girona FC",
    "Deportivo Alavés": "Deportivo Alavés",
    "Alaves": "Deportivo Alavés",
    "Valencia CF": "Valencia CF",
    "Valencia": "Valencia CF",
    "UD Las Palmas": "UD Las Palmas",
    "Las Palmas": "UD Las Palmas",
    "RCD Espanyol de Barcelona": "RCD Espanyol de Barcelona",
    "Espanyol": "RCD Espanyol de Barcelona",
    "Leganés": "CD Leganés",
    "CD Leganes": "CD Leganés",
    # Premier League
    "Manchester City FC": "Manchester City FC",
    "Manchester City": "Manchester City FC",
    "Arsenal FC": "Arsenal FC",
    "Arsenal": "Arsenal FC",
    "Liverpool FC": "Liverpool FC",
    "Liverpool": "Liverpool FC",
    "Chelsea FC": "Chelsea FC",
    "Chelsea": "Chelsea FC",
    "Tottenham Hotspur FC": "Tottenham Hotspur FC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Spurs": "Tottenham Hotspur FC",
    "Manchester United FC": "Manchester United FC",
    "Manchester United": "Manchester United FC",
    "Newcastle United FC": "Newcastle United FC",
    "Newcastle United": "Newcastle United FC",
    "West Ham United FC": "West Ham United FC",
    "West Ham United": "West Ham United FC",
    "West Ham": "West Ham United FC",
    "Aston Villa FC": "Aston Villa FC",
    "Aston Villa": "Aston Villa FC",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Brentford FC": "Brentford FC",
    "Brentford": "Brentford FC",
    "Fulham FC": "Fulham FC",
    "Fulham": "Fulham FC",
    "Crystal Palace FC": "Crystal Palace FC",
    "Crystal Palace": "Crystal Palace FC",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers FC",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Nottingham Forest FC": "Nottingham Forest FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Everton FC": "Everton FC",
    "Everton": "Everton FC",
    "Leicester City FC": "Leicester City FC",
    "Leicester City": "Leicester City FC",
    "Leicester": "Leicester City FC",
    "Ipswich Town FC": "Ipswich Town FC",
    "Ipswich Town": "Ipswich Town FC",
    "Ipswich": "Ipswich Town FC",
    "Southampton FC": "Southampton FC",
    "Southampton": "Southampton FC",
    "AFC Bournemouth": "AFC Bournemouth",
    "Bournemouth": "AFC Bournemouth",
    "Luton Town FC": "Luton Town FC",
    "Luton Town": "Luton Town FC",
    "Luton": "Luton Town FC",
    "Burnley FC": "Burnley FC",
    "Burnley": "Burnley FC",
    "Sheffield United FC": "Sheffield United FC",
    "Sheffield United": "Sheffield United FC",
    # Bundesliga 2
    "Hamburger SV": "Hamburger SV",
    "Hamburg": "Hamburger SV",
    "SpVgg Greuther Fürth": "SpVgg Greuther Fürth",
    "Greuther Fürth": "SpVgg Greuther Fürth",
    "FC Schalke 04": "FC Schalke 04",
    "Schalke 04": "FC Schalke 04",
    "Schalke": "FC Schalke 04",
    "Hannover 96": "Hannover 96",
    "Hannover": "Hannover 96",
    "1. FC Kaiserslautern": "1. FC Kaiserslautern",
    "Kaiserslautern": "1. FC Kaiserslautern",
    "SV Darmstadt 98": "SV Darmstadt 98",
    "Darmstadt 98": "SV Darmstadt 98",
    "Darmstadt": "SV Darmstadt 98",
    "1. FC Nürnberg": "1. FC Nürnberg",
    "Nürnberg": "1. FC Nürnberg",
    "Nuremberg": "1. FC Nürnberg",
    "Fortuna Düsseldorf": "Fortuna Düsseldorf",
    "Düsseldorf": "Fortuna Düsseldorf",
    "Dusseldorf": "Fortuna Düsseldorf",
    "Karlsruher SC": "Karlsruher SC",
    "Karlsruhe": "Karlsruher SC",
    "SC Paderborn 07": "SC Paderborn 07",
    "Paderborn": "SC Paderborn 07",
    "1. FC Magdeburg": "1. FC Magdeburg",
    "Magdeburg": "1. FC Magdeburg",
    "Hertha BSC": "Hertha BSC",
    "Hertha Berlin": "Hertha BSC",
    "Hertha": "Hertha BSC",
    "SSV Ulm 1846": "SSV Ulm 1846",
    "Ulm": "SSV Ulm 1846",
    "SV Elversberg": "SV Elversberg",
    "Elversberg": "SV Elversberg",
    "1. FC Köln": "1. FC Köln",
    "FC Köln": "1. FC Köln",
    "Köln": "1. FC Köln",
    "Cologne": "1. FC Köln",
    "VfL Bochum": "VfL Bochum",
    "Bochum": "VfL Bochum",
    "Eintracht Braunschweig": "Eintracht Braunschweig",
    "Braunschweig": "Eintracht Braunschweig",
    "Preußen Münster": "Preußen Münster",
    "Münster": "Preußen Münster",
    "SSV Jahn Regensburg": "SSV Jahn Regensburg",
    "Regensburg": "SSV Jahn Regensburg",
    "SV 07 Elversberg": "SV Elversberg",
    # Eredivisie
    "Ajax": "Ajax",
    "AFC Ajax": "Ajax",
    "PSV Eindhoven": "PSV Eindhoven",
    "PSV": "PSV Eindhoven",
    "Feyenoord": "Feyenoord",
    "AZ": "AZ Alkmaar",
    "AZ Alkmaar": "AZ Alkmaar",
    "FC Utrecht": "FC Utrecht",
    "Utrecht": "FC Utrecht",
    "FC Groningen": "FC Groningen",
    "Groningen": "FC Groningen",
    "SC Heerenveen": "SC Heerenveen",
    "Heerenveen": "SC Heerenveen",
    "Sparta Rotterdam": "Sparta Rotterdam",
    "Sparta": "Sparta Rotterdam",
    "Heracles Almelo": "Heracles Almelo",
    "Heracles": "Heracles Almelo",
    "NEC Nijmegen": "NEC Nijmegen",
    "NEC": "NEC Nijmegen",
    "RKC Waalwijk": "RKC Waalwijk",
    "RKC": "RKC Waalwijk",
    "Almere City FC": "Almere City FC",
    "Almere City": "Almere City FC",
    "Almere": "Almere City FC",
    "FC Twente": "FC Twente",
    "Twente": "FC Twente",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "PEC Zwolle": "PEC Zwolle",
    "Zwolle": "PEC Zwolle",
    "Willem II": "Willem II",
    "NAC Breda": "NAC Breda",
    "NAC": "NAC Breda",
    "Vitesse": "Vitesse",
    "SBV Vitesse": "Vitesse",
    "FC Volendam": "FC Volendam",
    "Volendam": "FC Volendam",
    "Excelsior": "Excelsior",
    "SBV Excelsior": "Excelsior",
    "Fortuna Sittard": "Fortuna Sittard",
}


def normalize_team(name: str) -> str:
    """Return canonical team name, or the original if not in the map."""
    return TEAM_NAME_MAP.get(name, name)
