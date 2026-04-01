"""
race_pace_overview.py
Analyzes long-run stints from FP2 to estimate race pace (base pace + degradation)
for each driver/team. Saves results to CSV for downstream use.
"""

import os
import warnings

import fastf1 as ff1
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# Configuration  —  update before each race weekend

CURRENT_YEAR = 2026
EVENT_NAME   = "Japan Grand Prix"

CACHE_PATH       = "/workspaces/drs_data/cache"
ROOT_PREDICTIONS = f"/workspaces/drs_data/outputs_predictions/{CURRENT_YEAR}"
PATH_OUTPUT      = os.path.join(ROOT_PREDICTIONS, EVENT_NAME.replace(" ", "_"))

# Stint filtering thresholds
MIN_STINT_LENGTH = 5   # minimum laps to qualify as a long run
MIN_CLEAN_LAPS   = 3   # minimum accurate laps needed to fit a regression

# Regression sanity-check bounds
VALID_INTERCEPT_RANGE = (60, 200)   # base pace in seconds (realistic lap time window)
VALID_SLOPE_RANGE     = (-5, 5)     # degradation in seconds per lap

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


# Setup

def setup() -> None:
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    ff1.Cache.enable_cache(CACHE_PATH)
    print(f"Output directory: {PATH_OUTPUT}")
    print(f"Cache enabled at: {CACHE_PATH}")


# Data loading

def load_fp2() -> ff1.core.Session:
    """
    Loads the FP2 session for the configured event.
    Exits gracefully if the session does not exist (Sprint weekend).
    """
    print(f"Loading FP2 for {EVENT_NAME} {CURRENT_YEAR}...")
    try:
        session = ff1.get_session(CURRENT_YEAR, EVENT_NAME, "FP2")
        session.load(laps=True)

        if session.laps is None or session.laps.empty:
            raise ValueError("No lap data found in FP2 session.")

        print("FP2 session loaded.")
        return session

    except ValueError as exc:
        raise SystemExit(
            f"[EXIT] Could not load FP2 for {EVENT_NAME}. "
            "This is likely a Sprint weekend — race pace analysis unavailable.\n"
            f"Detail: {exc}"
        )


# Stint analysis

def analyse_stints(session: ff1.core.Session) -> pd.DataFrame:
    """
    Iterates over all stints in FP2, fits a linear regression of lap time vs
    stint lap number, and returns a DataFrame of valid long-run stints.
    """
    print("Analysing long-run stints...")
    rows = []

    for stint_num in session.laps["Stint"].unique():
        try:
            stint = session.laps[session.laps["Stint"] == stint_num]

            if len(stint) < MIN_STINT_LENGTH:
                continue

            clean = stint.loc[stint["IsAccurate"]].copy()

            if len(clean) < MIN_CLEAN_LAPS:
                continue

            compound = clean["Compound"].iloc[0]
            if compound not in DRY_COMPOUNDS:
                continue

            # Relative lap number within the stint (starts at 1)
            clean["StintLapNumber"] = (
                clean["LapNumber"] - clean["LapNumber"].min() + 1
            )

            y = clean["LapTime"].dt.total_seconds()
            X = clean[["StintLapNumber"]]

            reg = LinearRegression().fit(X, y)
            slope     = reg.coef_[0]
            intercept = reg.intercept_

            # Discard physically implausible regressions
            lo, hi = VALID_INTERCEPT_RANGE
            sl, sh = VALID_SLOPE_RANGE
            if not (lo < intercept < hi) or not (sl < slope < sh):
                continue

            rows.append({
                "Driver":                clean["Driver"].iloc[0],
                "Team":                  clean["Team"].iloc[0],
                "Compound":              compound,
                "Clean_Laps":            len(clean),
                "Base_Pace_Seconds":     round(intercept, 4),  # estimated lap time at lap 1
                "Pace_Degradation_Slope": round(slope, 4),     # seconds gained/lost per lap
            })

        except Exception:
            continue   # skip any single stint that fails silently

    if not rows:
        raise SystemExit("[EXIT] No valid long-run stints found in this FP2 session.")

    return pd.DataFrame(rows)


# Rebranding

def apply_rebranding(df: pd.DataFrame) -> pd.DataFrame:
    """Normalises team names to reflect 2026 rebranding and mergers."""
    # Sauber lineage → Audi
    sauber_lineage = ["Sauber", "Kick Sauber", "Alfa Romeo", "Alfa Romeo Racing"]
    df.loc[df["Team"].isin(sauber_lineage), "Team"] = "Audi"

    # Williams name variations → Williams
    df.loc[df["Team"].str.contains("Williams", case=False, na=False), "Team"] = "Williams"

    # Red Bull junior team lineage → RB
    rb_lineage = ["AlphaTauri", "Scuderia AlphaTauri", "Racing Bulls"]
    df.loc[df["Team"].isin(rb_lineage), "Team"] = "RB"

    return df


# Output

def save_results(df: pd.DataFrame) -> None:
    filename = f"{CURRENT_YEAR}_{EVENT_NAME.replace(' ', '_')}_race_pace_data.csv"
    path = os.path.join(PATH_OUTPUT, filename)
    df.to_csv(path, index=False)
    print(f"Race pace data saved to: {path}")
    print(f"Teams in file: {sorted(df['Team'].unique())}")


# Entry point

def main() -> None:
    setup()

    session    = load_fp2()
    df         = analyse_stints(session)
    df         = apply_rebranding(df)

    # Final quality filter (belt-and-suspenders after regression bounds check)
    lo, hi = VALID_INTERCEPT_RANGE
    sl, sh = VALID_SLOPE_RANGE
    df = df[df["Base_Pace_Seconds"].between(lo, hi)]
    df = df[df["Pace_Degradation_Slope"].between(sl, sh)]

    print(f"{len(df)} valid stints ready for analysis.")
    save_results(df)


if __name__ == "__main__":
    main()