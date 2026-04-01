"""
collect_qualifying_data.py
Collects data from F1 free practice and qualifying sessions (2023–2026)
using the FastF1 library, applies team rebranding, and saves to CSV.
"""

import os

import fastf1 as ff1
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Path configuration

CACHE_PATH = "/drs_data/cache"
DATA_PATH  = "/drs_data/data"
OUTPUT_FILE = f"{DATA_PATH}/master_qualifying_data_2023-2026.csv"

YEARS = [2023, 2024, 2025, 2026]


# Helpers

def setup_directories() -> None:
    """Create the cache and data directories, if necessary."""
    for path in (CACHE_PATH, DATA_PATH):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory created: {path}")

    ff1.Cache.enable_cache(CACHE_PATH)
    print(f"Cache enabled in: {CACHE_PATH}")


def get_fastest_time(session: ff1.core.Session, driver_abbr: str) -> float:
    """Returns a driver's fastest lap time in seconds."""
    try:
        laps = session.laps.pick_drivers(driver_abbr)
        if laps.empty:
            return np.nan
        return laps.pick_fastest()["LapTime"].total_seconds()
    except Exception:
        return np.nan


def apply_rebranding(df: pd.DataFrame) -> pd.DataFrame:
    """
   Normalizes team names to reflect rebrandings and mergers:
    - Sauber lineage → Audi
    - Williams variations → Williams
    - AlphaTauri / Racing Bulls lineage → RB
    """
    print("Applying team rebranding...")

    # Audi 
    sauber_lineage = ["Sauber", "Kick Sauber", "Alfa Romeo", "Alfa Romeo Racing"]
    df.loc[df["Team"].isin(sauber_lineage), "Team"] = "Audi"

    # Williams 
    mask_williams = df["Team"].str.contains("Williams", case=False, na=False)
    df.loc[mask_williams, "Team"] = "Williams"

    # RB / Racing Bulls
    rb_lineage = ["AlphaTauri", "Scuderia AlphaTauri", "Racing Bulls"]
    df.loc[df["Team"].isin(rb_lineage), "Team"] = "RB"

    return df


# Main collection

def collect_data(years: list[int]) -> pd.DataFrame:
    """
   It iterates over years and events, collects FP1/FP2/FP3 times and position]
   in the standings for each driver.
    """
    all_rows: list[dict] = []

    for year in tqdm(years, desc="Anos"):
        schedule = ff1.get_event_schedule(
            year, include_testing=False, backend="ergast"
        )

        for _, event in tqdm(
            schedule.iterrows(),
            total=len(schedule),
            desc=f"Corridas {year}",
            leave=False,
        ):
            # Skip future events
            if event["EventDate"].tz_localize("UTC") > pd.Timestamp.now(tz="UTC"):
                continue

            try:
                fp1   = ff1.get_session(year, event["EventName"], "FP1")
                fp2   = ff1.get_session(year, event["EventName"], "FP2")
                fp3   = ff1.get_session(year, event["EventName"], "FP3")
                quali = ff1.get_session(year, event["EventName"], "Q")

                for session in (fp1, fp2, fp3, quali):
                    session.load()

                quali_results = quali.results
                if quali_results is None:
                    continue

                for _, driver in quali_results.iterrows():
                    driver_abbr = driver["Abbreviation"]

                    all_rows.append(
                        {
                            "Year":           year,
                            "Event":          event["EventName"],
                            "Driver":         driver_abbr,
                            "Team":           driver["TeamName"],
                            "FP1_Time":       get_fastest_time(fp1, driver_abbr),
                            "FP2_Time":       get_fastest_time(fp2, driver_abbr),
                            "FP3_Time":       get_fastest_time(fp3, driver_abbr),
                            "Quali_Position": int(driver["Position"]),
                        }
                    )

            except Exception as exc:
                print(f"[ERRO] {year} – {event['EventName']}: {exc}")

    return pd.DataFrame(all_rows)


# Entry point

def main() -> None:
    setup_directories()

    print("Starting data collection...")
    df = collect_data(YEARS)

    df = apply_rebranding(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved file: {OUTPUT_FILE}")
    print("-" * 50)
    print("Teams in the final file:")
    print(sorted(df["Team"].unique()))
    print("-" * 50)
    df.info()


if __name__ == "__main__":
    main()