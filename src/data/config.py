# config.py
# Official colors for visualization (Matplotlib/Seaborn)
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#F91536",
    "McLaren": "#F58020",
    "Mercedes": "#6CD3BF",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Haas F1 Team": "#B6BABD",
    "Williams": "#37BEDD",
    "RB": "#6692FF",
    # 🆕 NEW TEAMS 2026
    "Audi": "#4C4C4C",  # Replaces Kick Sauber (Audi Grey/German Silver)
    "Cadillac": "#DDC575",  # NEW ENTRY (Cadillac Gold/V-Series)
    # 🕰️ LEGACY / SYNONYMS (Keep for historical compatibility)
    "AlphaTauri": "#5E8FAA",
    "Alfa Romeo": "#C92D4B",
    "Racing Bulls": "#6692FF",
    "Kick Sauber": "#52E252",  # Kept only to avoid errors in old, unprocessed data.
}

# Mapping abbreviations to full names
TEAM_MAPPING = {
    "RBR": "Red Bull Racing",
    "MER": "Mercedes",
    "FER": "Ferrari",
    "MCL": "McLaren",
    "AST": "Aston Martin",
    "ALP": "Alpine",
    "WIL": "Williams",
    "RB": "RB",
    "HAA": "Haas",
    "AUD": "Audi",  # Novo
    "CAD": "Cadillac",  # Novo
}

# Mapping teammates for strategy simulation.
# Key: Pilot -> Value: Companion
teammate_map = {
    # Red Bull Racing (Verstappen + Hadjar)
    "VER": "HAD",
    "HAD": "VER",
    # Mercedes (Russell + Antonelli)
    "RUS": "ANT",
    "ANT": "RUS",
    # Ferrari (Leclerc + Hamilton)
    "LEC": "HAM",
    "HAM": "LEC",
    # McLaren (Norris + Piastri)
    "NOR": "PIA",
    "PIA": "NOR",
    # Aston Martin (Alonso + Stroll)
    "ALO": "STR",
    "STR": "ALO",
    # Alpine (Gasly + Colapinto)
    "GAS": "COL",
    "COL": "GAS",
    # Williams (Albon + Sainz)
    "ALB": "SAI",
    "SAI": "ALB",
    # RB / VCARB (Lawson + Lindblad)
    "LAW": "LIN",
    "LIN": "LAW",
    # Audi (Hulkenberg + Bortoleto)
    "HUL": "BOR",
    "BOR": "HUL",
    # Haas (Ocon + Bearman)
    "OCO": "BEA",
    "BEA": "OCO",
    # Cadillac (Bottas + Perez)
    "BOT": "PER",
    "PER": "BOT",
}