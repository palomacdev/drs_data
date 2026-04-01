"""
race_simulator.py
F1 2026 Race Strategy Simulator — M30 Engine
Simulates lap-by-lap strategy for a given Grand Prix using FastF1 qualifying data
and optional FP2 race pace data.

Race and track settings need to be updated for each simulated race. Additionally, tire strategies and performance need to be updated.

Updates are made based on FIA data and historical track data (tire type and performance).
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm
import fastf1 as ff1

warnings.filterwarnings("ignore")

ff1.Cache.enable_cache("/workspaces/drs_data/cache")


# ==============================================================================
# SECTION 1: RACE & TRACK CONFIGURATION
# ==============================================================================

# --- Folder Paths ---
ROOT_PREDICTIONS = "/workspaces/drs_data/outputs_predictions/2026"
PATH_DATA = "/workspaces/drs_data/data"

# --- Race Configuration (Suzuka) ---
ANO_ATUAL = 2026
NOME_EVENTO = "Japan Grand Prix"
TOTAL_LAPS = 53          # 5.807 km x 53 = 307.771 km
PIT_STOP_TIME = 22       # Pit lane entry/exit is challenging (crosses the track)

# --- Weather Configuration ---
RACE_DATE = "2026-03-29"  # Race day: Sunday, March 29, 2026
TRACK_LAT = 34.8431       # Suzuka International Racing Course — Latitude
TRACK_LON = 136.5408      # Suzuka International Racing Course — Longitude

# --- Build output folder ---
race_folder_name = f"{NOME_EVENTO.replace(' ', '_')}"
PATH_PREDICTIONS = os.path.join(ROOT_PREDICTIONS, race_folder_name)

if not os.path.exists(PATH_PREDICTIONS):
    os.makedirs(PATH_PREDICTIONS)
    print(f"📂 Output folder created: {PATH_PREDICTIONS}")
else:
    print(f"📂 Saving files to: {PATH_PREDICTIONS}")


# ==============================================================================
# SECTION 1.1: WEATHER FORECAST INTEGRATION
# ==============================================================================

def get_race_weather_forecast(lat: float, lon: float, race_date_str: str) -> float:
    """
    Fetches an hourly temperature forecast from Open-Meteo for the race location
    and returns the approximate track temperature on race day.

    Returns:
        float: Estimated track temperature in °C (defaults to 25.0 on failure).
    """
    print(f"🌤️ Fetching weather forecast for {race_date_str}...")
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "forecast_days": 7,
            "timezone": "auto",
        }
        response = requests.get(url, params=params)
        data = response.json()

        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]

        race_temps = [t_val for t, t_val in zip(times, temps) if race_date_str in t]

        if not race_temps:
            print("⚠️ Weather data not found for the race date. Using default 25°C.")
            return 25.0

        # Use the minimum daily temperature + 2°C as a conservative track estimate
        avg_temp = min(race_temps) + 2
        print(f"🌡️ Forecasted Track Temperature: ~{avg_temp:.1f}°C")
        return avg_temp

    except Exception as e:
        print(f"⚠️ Weather API error: {e}. Using default 25°C.")
        return 25.0


track_temp = get_race_weather_forecast(TRACK_LAT, TRACK_LON, RACE_DATE)

# No weather-based tire penalty applied at this track
HARD_TIRE_PENALTY = 0.0


# ==============================================================================
# SECTION 2: STRATEGY DEFINITIONS
# ==============================================================================

STRATEGIES = {
    # Standard 1-stop (Medium → Hard)
    # Preferred by top teams — Suzuka suits a long C1 final stint
    "C2-C1 (1-Stop Standard)": [
        ('MEDIUM', 22, 27),   # Ideal window ~24 laps (longer than Shanghai)
        ('HARD',   53, 53),   # Final stint ~28 laps — C1 is extremely durable
    ],

    # Aggressive undercut (Soft → Hard)
    # For P4–P8 drivers looking to attack early (window: L15–L20)
    "C3-C1 (1-Stop Undercut)": [
        ('SOFT', 16, 21),     # C3 holds better than C4 — stint ~18 laps
        ('HARD', 53, 53),     # Long final stint of 32–37 laps (C1 is a tank)
    ],

    # Overcut strategy (Hard → Medium)
    # For drivers starting P11+ wanting to avoid early traffic
    "C1-C2 (1-Stop Overcut)": [
        ('HARD',   28, 33),   # Very long first stint (C1 nearly indestructible)
        ('MEDIUM', 53, 53),   # Fresh tire push to the finish
    ],

    # Ultra-conservative (Hard → Hard)
    # For backmarker teams or recovery drives after an incident
    "C1-C1 (1-Stop Safe)": [
        ('HARD', 25, 29),     # Mid-race swap
        ('HARD', 53, 53),     # Near-zero degradation risk, but no performance gain
    ],
}


# ==============================================================================
# SECTION 3: TIRE PERFORMANCE MODEL (DYNAMIC)
# ==============================================================================

TIRE_PERFORMANCE = {
    'SOFT': {   # Pirelli C3
        'base_pace_adjustment': -0.55,  # Slower than C4 reference (was -0.65)
        'deg_factor':           1.60,   # Less degradation than C4 (was 1.85)
        'warm_up_laps':         1.8,    # Slightly slower to reach operating window
        'cliff_lap':            20,     # Cliff onset later than C4 (was lap 16)
        'notes': 'Good for undercut but less aggressive than C4',
    },

    'MEDIUM': {  # Pirelli C2
        'base_pace_adjustment': 0.0,    # Reference compound (always 0.0)
        'deg_factor':           0.85,   # Less degradation than C3 (was 1.0)
        'warm_up_laps':         2.2,    # Moderate warm-up time
        'cliff_lap':            None,   # No cliff — linear degradation profile
        'notes': 'Workhorse compound — very predictable',
    },

    'HARD': {   # Pirelli C1
        'base_pace_adjustment': 0.50,   # Slower than C2 reference (was 0.45)
        'deg_factor':           0.40,   # Extremely durable (was 0.50)
        'warm_up_laps':         3.5,    # Slow to reach temperature (was 3.0)
        'cliff_lap':            None,   # Virtually indestructible
        'notes': 'Tank mode — can easily run 35+ laps',
    },
}

print(f"Configuration loaded for: {NOME_EVENTO} {ANO_ATUAL}")


# ==============================================================================
# SECTION 1B: LOAD & CLEAN QUALIFYING GRID DATA
# ==============================================================================

print(f"🏎️ Fetching official qualifying grid from FastF1 for {NOME_EVENTO} {ANO_ATUAL}...")

try:
    session_q = ff1.get_session(ANO_ATUAL, NOME_EVENTO, 'Q')
    session_q.load(telemetry=False, weather=False, messages=False)
    results = session_q.results

    grid_data = []
    for _, row in results.iterrows():
        grid_data.append({
            "Driver":             row["Abbreviation"],
            "Team":               row["TeamName"],
            "Grid_Position":      int(row["Position"]),
            # 'Predicted_Position' kept for backward compatibility with simulation blocks
            "Predicted_Position": int(row["Position"]),
        })

    df_grid_clean = pd.DataFrame(grid_data)
    df_grid_clean = df_grid_clean.sort_values(by="Grid_Position").reset_index(drop=True)
    print(f"✅ Official FIA grid loaded successfully! ({len(df_grid_clean)} drivers on the grid)")

except Exception as e:
    print(f"🚨 ERROR loading FastF1 data: {e}")
    print("The M30 simulator requires the official grid to run.")
    raise


# --- Load M25 Output (Race Pace) ---
pace_file = os.path.join(
    PATH_PREDICTIONS,
    f"{ANO_ATUAL}_{NOME_EVENTO.replace(' ', '_')}_race_pace_data.csv",
)
try:
    df_pace_raw = pd.read_csv(pace_file)
    print(f"✅ M25 race pace data loaded. ({len(df_pace_raw)} stints found)")
except FileNotFoundError:
    print(f"⚠️ WARNING: Race pace file not found: {pace_file}")
    print("This is normal for Sprint weekends or if M25 was not run. Falling back to skill adjustment.")
    df_pace_raw = pd.DataFrame(
        columns=["Driver", "Team", "Compound", "Base_Pace_Seconds", "Pace_Degradation_Slope"]
    )


# ==============================================================================
# SECTION 2B: IMPUTE MISSING PACE DATA (v6 — Noise Filter)
# ==============================================================================

print("Imputing missing pace data and applying dynamic skill adjustments...")

# --- Simulation Parameters ---
MAX_REALISTIC_DEG_PER_LAP      = 0.5
DEFAULT_DEG_IF_CAPPED          = 0.1
MAX_REALISTIC_PACE_SECONDS     = 200
MIN_REALISTIC_PACE_SECONDS     = 60
SKILL_ADJUSTMENT_FACTOR_PER_POS = 0.05
REALISTIC_PACE_WINDOW_SECONDS  = 3.0   # Filters out outlier / noise laps

# 1. Build base simulation DataFrame from the qualifying grid
df_sim = df_grid_clean[["Driver", "Team", "Predicted_Position"]].copy()

# 2. Merge sparse FP2 pace data (Medium compound only)
if df_pace_raw.empty:
    print("⚠️ No FP2 data available (Sprint Weekend?). Skipping pace merge.")
    df_pace_medium = pd.DataFrame(
        columns=["Driver", "Base_Pace_Seconds", "Pace_Degradation_Slope"]
    )
else:
    if "Compound" in df_pace_raw.columns:
        df_pace_medium = df_pace_raw[df_pace_raw["Compound"] == "MEDIUM"].copy()
    else:
        df_pace_medium = pd.DataFrame(
            columns=["Driver", "Base_Pace_Seconds", "Pace_Degradation_Slope"]
        )

df_pace_reference_raw = pd.merge(
    df_sim,
    df_pace_medium,
    on=["Driver"],
    how="left",
    suffixes=("", "_drop"),
)
df_pace_reference_raw = df_pace_reference_raw.loc[
    :, ~df_pace_reference_raw.columns.str.endswith("_drop")
]

# 3. Keep only valid, realistic FP2 reference data
df_pace_reference = df_pace_reference_raw.dropna(subset=["Base_Pace_Seconds"])
df_pace_reference = df_pace_reference[
    df_pace_reference["Base_Pace_Seconds"].between(
        MIN_REALISTIC_PACE_SECONDS, MAX_REALISTIC_PACE_SECONDS
    )
    & df_pace_reference["Pace_Degradation_Slope"].between(-1, MAX_REALISTIC_DEG_PER_LAP)
].copy()

# 4. Calculate global averages from available FP2 data
if df_pace_reference.empty:
    print("⚠️ No realistic FP2 pace data found. Using safe default values.")
    global_avg_pace = 90.0
    global_avg_deg  = 0.08
else:
    global_avg_pace = df_pace_reference["Base_Pace_Seconds"].mean()
    global_avg_deg  = df_pace_reference["Pace_Degradation_Slope"].mean()
    print(
        f"Global averages from FP2 (Medium): "
        f"Pace = {global_avg_pace:.3f}s, Degradation = {global_avg_deg:.3f}s/lap"
    )

# 5. Set default imputed values for all drivers
df_sim["Base_Pace_MEDIUM"] = global_avg_pace
df_sim["Deg_Slope_MEDIUM"] = global_avg_deg

# 6. Dynamic skill adjustment — better qualifying position = faster base pace
MEDIAN_POSITION    = 10.5
pace_upper_bound   = global_avg_pace + REALISTIC_PACE_WINDOW_SECONDS
pace_lower_bound   = global_avg_pace - REALISTIC_PACE_WINDOW_SECONDS


def apply_dynamic_skill_adjustment(row: pd.Series) -> float:
    """
    Returns the driver's base pace in seconds, using real FP2 data where
    available and within the realistic window, otherwise applying a
    position-based skill adjustment relative to the global average.
    """
    real_data_row = df_pace_reference[df_pace_reference["Driver"] == row["Driver"]]

    if not real_data_row.empty:
        real_pace = real_data_row.iloc[0]["Base_Pace_Seconds"]
        if pace_lower_bound <= real_pace <= pace_upper_bound:
            return real_pace

    # Skill-based adjustment: front runners faster, backmarkers slower
    pos = row["Predicted_Position"]
    pace_adjustment = (MEDIAN_POSITION - pos) * SKILL_ADJUSTMENT_FACTOR_PER_POS
    return global_avg_pace - pace_adjustment


df_sim["Base_Pace_MEDIUM"] = df_sim.apply(apply_dynamic_skill_adjustment, axis=1)

# 7. Clip degradation to realistic bounds
df_sim["Deg_Slope_MEDIUM"] = df_sim["Deg_Slope_MEDIUM"].clip(0.01, MAX_REALISTIC_DEG_PER_LAP)

print("\n--- Final Simulation Input Data (Preview) ---")
print(df_sim[["Driver", "Predicted_Position", "Base_Pace_MEDIUM"]].head())


# ==============================================================================
# SECTION 3: SIMULATION ENGINE (2026 MOM — Manual Override Mode)
# ==============================================================================

print("\nInitializing simulation engine with MOM (Manual Override Mode) logic...")

# --- MOM Configuration ---
# MOM makes overtaking easier by reducing the time lost in traffic.
# Pre-DRS era: 0.20s per position. MOM era: 0.15s per position.
TRAFFIC_PENALTY_PER_POS = 0.15

# Penalty for new power unit manufacturers (e.g. Audi/Cadillac) that may
# suffer energy recovery inefficiency ("de-rating") in their debut season.
NEW_ENGINE_PENALTY = 0.08  # Average seconds lost per lap due to energy management

final_race_results = []

# --- Main simulation loop ---
for index, driver_stats in tqdm(
    df_sim.iterrows(), total=len(df_sim), desc="Simulating Drivers"
):
    driver             = driver_stats["Driver"]
    team               = driver_stats["Team"]
    predicted_start_pos = driver_stats["Predicted_Position"]
    base_pace_medium   = driver_stats["Base_Pace_MEDIUM"]
    base_deg_medium    = driver_stats["Deg_Slope_MEDIUM"]

    best_strategy_name = ""
    best_total_time    = float("inf")

    # --- Strategy selection loop ---
    for strategy_name, strategy_stints in STRATEGIES.items():

        # Initial traffic penalty (reduced by MOM effect)
        total_time = (predicted_start_pos - 1) * TRAFFIC_PENALTY_PER_POS

        current_laps_on_tire  = 0
        current_stint_index   = 0
        current_compound      = strategy_stints[current_stint_index][0]

        # --- Lap-by-lap simulation ---
        for lap in range(1, TOTAL_LAPS + 1):
            stint_info = strategy_stints[current_stint_index]
            _, min_pit_lap, max_pit_lap = stint_info

            # Pit stop logic: change tires at the minimum pit lap of each stint
            if lap == min_pit_lap and current_stint_index < len(strategy_stints) - 1:
                total_time            += PIT_STOP_TIME
                current_stint_index   += 1
                current_compound       = strategy_stints[current_stint_index][0]
                current_laps_on_tire   = 0
                continue

            # Calculate lap time using weather-adjusted tire performance model
            tire_adj        = TIRE_PERFORMANCE[current_compound]["base_pace_adjustment"]
            tire_deg_factor = TIRE_PERFORMANCE[current_compound]["deg_factor"]

            stint_base_pace = base_pace_medium + tire_adj
            stint_deg_slope = base_deg_medium * tire_deg_factor

            lap_time = stint_base_pace + (stint_deg_slope * current_laps_on_tire)

            # Apply new power unit penalty for debut manufacturers
            if team in ["Audi", "Cadillac", "AUD", "CAD"]:
                lap_time += NEW_ENGINE_PENALTY

            total_time           += lap_time
            current_laps_on_tire += 1

        # Keep the fastest strategy found so far
        if total_time < best_total_time:
            best_total_time    = total_time
            best_strategy_name = strategy_name

    final_race_results.append({
        "Driver":              driver,
        "Team":                team,
        "Start_Pos":           predicted_start_pos,
        "Best_Strategy":       best_strategy_name,
        "Total_Time_Seconds":  best_total_time,
    })

print("✅ Simulation complete (MOM logic applied).")


# ==============================================================================
# SECTION 4: RESULTS & OUTPUT
# ==============================================================================

print("\n--- FINAL RACE RESULTS (STRATEGY SIMULATION) ---")

df_final_results = pd.DataFrame(final_race_results)
df_final_results = df_final_results.sort_values(by="Total_Time_Seconds").reset_index(drop=True)

# Gap to race leader
df_final_results["Gap_to_P1"] = (
    df_final_results["Total_Time_Seconds"] - df_final_results["Total_Time_Seconds"].min()
)

# Assign finishing positions
df_final_results["Final_Position"] = range(1, len(df_final_results) + 1)

# --- Podium ---
print("\n--- 🏆 PODIUM 🏆 ---")
print(
    df_final_results.head(3)[
        ["Final_Position", "Driver", "Team", "Best_Strategy", "Total_Time_Seconds"]
    ]
)

# --- Full grid ---
print("\n--- Full Grid ---")
print(
    df_final_results[
        ["Final_Position", "Driver", "Team", "Start_Pos", "Best_Strategy", "Gap_to_P1"]
    ].to_string(
        formatters={"Gap_to_P1": lambda x: f"+{x:.2f}s" if x > 0 else f"{x:.2f}s"}
    )
)

# --- Save to CSV ---
output_csv_path = os.path.join(
    PATH_PREDICTIONS,
    f"{ANO_ATUAL}_{NOME_EVENTO.replace(' ', '_')}_RaceSimulation.csv",
)
df_final_results.to_csv(output_csv_path, index=False)
print(f"\n✅ Simulation results saved to: {output_csv_path}")