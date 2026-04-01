"""
data_enrichment.py
Enriches the qualifying data CSV with driver championship points and position
before each race, fetched from the Jolpica (Ergast) API.
"""

import time
import warnings

import fastf1 as ff1
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")


# Configuration

INPUT_FILE  = "/workspaces/drs_data/data/master_qualifying_data_2023-2026.csv"
OUTPUT_FILE = "/workspaces/drs_data/data/master_qualifying_data_enriched.csv"
CACHE_PATH  = "/workspaces/f1-project/cache"

API_DELAY   = 0.2   # seconds between API calls
API_TIMEOUT = 5     # timeout by request


# Helpers

def get_standings_before_round(year: int, round_num: int) -> dict:
    """
    Fetches driver standings BEFORE the given `round_num`.
    Returns dict {driver_code: {Points, Wins, Champ_Pos}} or {} on error.
    """
    if round_num <= 1:
        return {}   # Start of season: everyone at 0 points

    prev_round = round_num - 1
    url = (
        f"http://api.jolpi.ca/ergast/f1/{year}/{prev_round}/driverStandings.json"
    )

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"][0][
            "DriverStandings"
        ]

        return {
            item["Driver"]["code"]: {
                "Points":   float(item["points"]),
                "Wins":     int(item["wins"]),
                "Champ_Pos": int(item["position"]),
            }
            for item in standings_list
        }

    except (KeyError, IndexError):
        print(f"  [WARNING] No standings data found for {year} Round {prev_round}")
        return {}
    except Exception as exc:
        print(f"  [ERROR] API {year} R{prev_round}: {exc}")
        return {}


def map_round_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a RoundNumber column to the DataFrame using the FastF1 calendar."""
    print("Mapping events to round numbers via FastF1...")
    ff1.Cache.enable_cache(CACHE_PATH)

    event_to_round: dict[tuple, int] = {}

    for year in df["Year"].unique():
        schedule = ff1.get_event_schedule(year, include_testing=False)
        for _, row in schedule.iterrows():
            event_to_round[(year, row["EventName"])] = row["RoundNumber"]

    df["RoundNumber"] = df.apply(
        lambda x: event_to_round.get((x["Year"], x["Event"]), 0), axis=1
    )
    print("Round numbers mapped.")
    return df


def enrich_with_standings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Driver_Points, Driver_Wins and Championship_Pos columns
    with championship data prior to each race.
    """
    print("Fetching standings from Jolpica...")

    df["Driver_Points"]    = 0.0
    df["Driver_Wins"]      = 0
    df["Championship_Pos"] = 0

    unique_races = (
        df[["Year", "RoundNumber"]]
        .drop_duplicates()
        .sort_values(["Year", "RoundNumber"])
    )

    for _, race in tqdm(unique_races.iterrows(), total=len(unique_races), desc="Rounds"):
        year      = int(race["Year"])
        round_num = int(race["RoundNumber"])

        if round_num == 0:
            continue

        stats = get_standings_before_round(year, round_num)
        mask  = (df["Year"] == year) & (df["RoundNumber"] == round_num)

        for idx in df[mask].index:
            driver_code = df.loc[idx, "Driver"]

            if driver_code in stats:
                df.loc[idx, "Driver_Points"]    = stats[driver_code]["Points"]
                df.loc[idx, "Driver_Wins"]      = stats[driver_code]["Wins"]
                df.loc[idx, "Championship_Pos"] = stats[driver_code]["Champ_Pos"]
            else:
                # Rookie or first race of the season
                df.loc[idx, "Driver_Points"]    = 0.0
                df.loc[idx, "Driver_Wins"]      = 0
                df.loc[idx, "Championship_Pos"] = 20

        time.sleep(API_DELAY)

    return df


# Entry point

def main() -> None:
    # 1. Load data
    print(f"Loading: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"{len(df)} rows loaded.")
    except FileNotFoundError:
        print("Input file not found.")
        raise

    # 2. Map round numbers
    df = map_round_numbers(df)

    # 3. Enrich with standings
    df = enrich_with_standings(df)

    # 4. Save result
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFile saved: {OUTPUT_FILE}")
    print("\nSample of enriched data:")
    print(
        df[["Year", "Event", "Driver", "Driver_Points", "Championship_Pos"]].sample(5)
    )


if __name__ == "__main__":
    main()