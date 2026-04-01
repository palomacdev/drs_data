"""
predict_qualifying.py
Loads live F1 practice session data, engineers features, and predicts
qualifying positions using the trained XGBoost model.
Generates SHAP explainability charts for the predicted pole sitter.
"""

import os
import warnings

import fastf1 as ff1
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


# Configuration  —  update these before each race weekend

CURRENT_YEAR  = 2026
EVENT_NAME    = "Japan Grand Prix"

PATH_DATA     = "/drs_data/data"
PATH_MODELS   = "/drs_data/models"
PATH_CACHE    = "/drs_data/cache"

ROOT_OUTPUTS  = f"/drs_data/outputs_predictions/{CURRENT_YEAR}"
PATH_OUTPUT   = os.path.join(ROOT_OUTPUTS, EVENT_NAME.replace(" ", "_"))

MODEL_PATH    = f"{PATH_MODELS}/xgb_model_quali_v1.pkl"
FEATURES_PATH = f"{PATH_MODELS}/xgb_features_quali_v1.pkl"
HISTORY_FILE  = f"{PATH_DATA}/master_qualifying_data_enriched.csv"

# Default championship position for drivers not yet in standings
DEFAULT_CHAMP_POS = 21

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari":         "#F91536",
    "McLaren":         "#F58020",
    "Mercedes":        "#6CD3BF",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Haas F1 Team":    "#B6BABD",
    "Williams":        "#37BEDD",
    "RB":              "#6692FF",
    # 2026 new entries
    "Audi":            "#4C4C4C",
    "Cadillac":        "#DDC575",
    # Legacy / aliases kept for historical data compatibility
    "AlphaTauri":      "#5E8FAA",
    "Alfa Romeo":      "#C92D4B",
    "Racing Bulls":    "#6692FF",
    "Kick Sauber":     "#52E252",
}

TRACK_STATS = {
    "Bahrain Grand Prix":        {"Downforce": 3, "Overtaking": 4, "Type": "Permanent"},
    "Saudi Arabian Grand Prix":  {"Downforce": 2, "Overtaking": 4, "Type": "Street"},
    "Australian Grand Prix":     {"Downforce": 4, "Overtaking": 3, "Type": "Street_Hybrid"},
    "Azerbaijan Grand Prix":     {"Downforce": 2, "Overtaking": 5, "Type": "Street"},
    "Miami Grand Prix":          {"Downforce": 3, "Overtaking": 3, "Type": "Street"},
    "Monaco Grand Prix":         {"Downforce": 5, "Overtaking": 1, "Type": "Street"},
    "Spanish Grand Prix":        {"Downforce": 4, "Overtaking": 3, "Type": "Permanent"},
    "Canadian Grand Prix":       {"Downforce": 2, "Overtaking": 4, "Type": "Permanent"},
    "Austrian Grand Prix":       {"Downforce": 3, "Overtaking": 4, "Type": "Permanent"},
    "British Grand Prix":        {"Downforce": 4, "Overtaking": 4, "Type": "Permanent"},
    "Hungarian Grand Prix":      {"Downforce": 5, "Overtaking": 2, "Type": "Permanent"},
    "Belgian Grand Prix":        {"Downforce": 2, "Overtaking": 4, "Type": "Permanent"},
    "Dutch Grand Prix":          {"Downforce": 5, "Overtaking": 2, "Type": "Permanent"},
    "Italian Grand Prix":        {"Downforce": 1, "Overtaking": 4, "Type": "Permanent"},  # Monza
    "Singapore Grand Prix":      {"Downforce": 5, "Overtaking": 2, "Type": "Street"},
    "Japanese Grand Prix":       {"Downforce": 5, "Overtaking": 3, "Type": "Permanent"},
    "Qatar Grand Prix":          {"Downforce": 4, "Overtaking": 3, "Type": "Permanent"},
    "United States Grand Prix":  {"Downforce": 4, "Overtaking": 4, "Type": "Permanent"},
    "Mexico City Grand Prix":    {"Downforce": 5, "Overtaking": 3, "Type": "Permanent"},
    "São Paulo Grand Prix":      {"Downforce": 3, "Overtaking": 4, "Type": "Permanent"},
    "Las Vegas Grand Prix":      {"Downforce": 1, "Overtaking": 5, "Type": "Street"},
    "Abu Dhabi Grand Prix":      {"Downforce": 3, "Overtaking": 3, "Type": "Permanent"},
    "Chinese Grand Prix":        {"Downforce": 4, "Overtaking": 4, "Type": "Permanent"},
    "Emilia Romagna Grand Prix": {"Downforce": 4, "Overtaking": 2, "Type": "Permanent"},
}


# Setup

def setup_output_dir() -> None:
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    print(f"Output directory: {PATH_OUTPUT}")


def load_model_and_features() -> tuple:
    print("Loading model and feature list...")
    try:
        model    = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        print("Model and features loaded.")
        return model, features
    except FileNotFoundError:
        print(f"[ERROR] Model files not found in: {PATH_MODELS}")
        raise


def load_history() -> pd.DataFrame:
    print(f"Loading historical data from: {HISTORY_FILE}")
    try:
        df = pd.read_csv(HISTORY_FILE)
        print(f"Historical data loaded: {len(df)} rows.")
        return df
    except FileNotFoundError:
        print("[ERROR] Historical enriched CSV not found.")
        raise


# Live session loading

def get_fastest_time(session: ff1.core.Session, driver_abbr: str) -> float:
    """Returns the fastest lap time (seconds) for a driver in a session."""
    try:
        return (
            session.laps.pick_driver(driver_abbr)
            .pick_fastest()["LapTime"]
            .total_seconds()
        )
    except Exception:
        return np.nan


def load_live_sessions() -> tuple:
    """
    Loads FP1/FP2/FP3 from FastF1.
    Detects Sprint weekends automatically: if FP2 is missing, FP1 is reused.
    """
    ff1.Cache.enable_cache(PATH_CACHE)
    print(f"Loading live session data for {EVENT_NAME} {CURRENT_YEAR}...")

    try:
        fp1 = ff1.get_session(CURRENT_YEAR, EVENT_NAME, "FP1")
        fp1.load(laps=True)

        try:
            fp2 = ff1.get_session(CURRENT_YEAR, EVENT_NAME, "FP2")
            fp2.load(laps=True)
            fp3 = ff1.get_session(CURRENT_YEAR, EVENT_NAME, "FP3")
            fp3.load(laps=True)
            print("Sessions loaded: FP1, FP2, FP3 (standard weekend).")
        except ValueError:
            # Sprint weekend: FP2/FP3 don't exist
            print("Sprint weekend detected — FP2/FP3 not found. Using FP1 for all sessions.")
            fp2 = fp1
            fp3 = fp1

        return fp1, fp2, fp3

    except Exception as exc:
        print(f"[ERROR] Failed to load FP1: {exc}")
        raise


def build_live_dataframe(fp1, fp2, fp3) -> pd.DataFrame:
    """Creates a raw DataFrame with FP times for all drivers in the current event."""
    # Use the first available session results to get the driver list
    driver_list = next(
        (s.results for s in [fp1, fp2, fp3] if s.results is not None), None
    )

    if driver_list is None:
        raise ValueError("Could not retrieve driver list from any practice session.")

    rows = []
    for _, driver in driver_list.iterrows():
        abbr = driver["Abbreviation"]
        rows.append({
            "Year":             CURRENT_YEAR,
            "Event":            EVENT_NAME,
            "Driver":           abbr,
            "Team":             driver["TeamName"],
            "FP1_Time":         get_fastest_time(fp1, abbr),
            "FP2_Time":         get_fastest_time(fp2, abbr),
            "FP3_Time":         get_fastest_time(fp3, abbr),
            "Quali_Position":   np.nan,  # unknown until qualifying runs
        })

    return pd.DataFrame(rows)


# Championship standings

def get_current_standings(year: int) -> dict:
    """Fetches the latest driver standings from the Jolpica API."""
    url = f"http://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
    try:
        data = requests.get(url, timeout=5).json()
        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
        return {
            item["Driver"]["code"]: {
                "Points": float(item["points"]),
                "Wins":   int(item["wins"]),
                "Pos":    int(item["position"]),
            }
            for item in standings
        }
    except Exception as exc:
        print(f"[WARNING] Jolpica API error: {exc}. Defaulting to zero standings.")
        return {}


def apply_current_standings(df: pd.DataFrame, standings: dict) -> pd.DataFrame:
    """Fills championship columns for the current race rows only."""
    mask = (df["Event"] == EVENT_NAME) & (df["Year"] == CURRENT_YEAR)

    for idx in df[mask].index:
        code = df.loc[idx, "Driver"]
        if code in standings:
            df.loc[idx, "Driver_Points"]    = standings[code]["Points"]
            df.loc[idx, "Driver_Wins"]      = standings[code]["Wins"]
            df.loc[idx, "Championship_Pos"] = standings[code]["Pos"]
        else:
            # Reserve driver or first appearance — default to back of field
            df.loc[idx, "Driver_Points"]    = 0.0
            df.loc[idx, "Driver_Wins"]      = 0
            df.loc[idx, "Championship_Pos"] = DEFAULT_CHAMP_POS

    return df


# Feature engineering

def apply_track_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Downforce, Overtaking and Type columns based on the TRACK_STATS dict."""
    def _get(row):
        s = TRACK_STATS.get(row["Event"], {"Downforce": 3, "Overtaking": 3, "Type": "Permanent"})
        return pd.Series([s["Downforce"], s["Overtaking"], s["Type"]])

    df[["Track_Downforce", "Track_Overtaking", "Track_Type"]] = df.apply(_get, axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all derived features (gaps, ranks, rolling history, interactions)."""

    # Relative gaps and ranks per session
    for session in ["FP1_Time", "FP2_Time", "FP3_Time"]:
        p1 = df.groupby(["Year", "Event"])[session].transform("min")
        df[f"Gap_{session}"]        = df[session] - p1
        df[f"Rank_{session}"]       = df.groupby(["Year", "Event"])[session].rank()
        df[f"Percentile_{session}"] = df.groupby(["Year", "Event"])[session].rank(pct=True)

    # Gap aggregations
    gap_cols = ["Gap_FP1_Time", "Gap_FP2_Time", "Gap_FP3_Time"]
    df["Avg_Gap"]  = df[gap_cols].mean(axis=1)
    df["Min_Gap"]  = df[gap_cols].min(axis=1)

    # Rank aggregations
    rank_cols = ["Rank_FP1_Time", "Rank_FP2_Time", "Rank_FP3_Time"]
    df["Avg_Rank"]   = df[rank_cols].mean(axis=1)
    df["Best_Rank"]  = df[rank_cols].min(axis=1)
    df["Worst_Rank"] = df[rank_cols].max(axis=1)

    # Percentile aggregations
    df["Avg_Percentile"] = df[
        ["Percentile_FP1_Time", "Percentile_FP2_Time", "Percentile_FP3_Time"]
    ].mean(axis=1)

    # Driver rolling history
    df = df.sort_values(["Driver", "Year", "Event"])
    for window in [3, 5, 10]:
        df[f"Driver_Last{window}_Avg"] = df.groupby("Driver")["Quali_Position"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f"Driver_Last{window}_Best"] = df.groupby("Driver")["Quali_Position"].transform(
            lambda x: x.rolling(window=window, min_periods=1).min().shift(1)
        )

    df["Driver_Recent_Trend"] = df["Driver_Last5_Avg"] - df["Driver_Last10_Avg"]
    df["Driver_Std"] = df.groupby("Driver")["Quali_Position"].transform(
        lambda x: x.expanding().std().shift(1)
    )

    # Team rolling history
    df = df.sort_values(["Team", "Year", "Event"])
    for window in [3, 5]:
        df[f"Team_Last{window}_Avg"] = df.groupby("Team")["Quali_Position"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Circuit-specific history
    df["Driver_Track_Avg"]  = df.groupby(["Driver", "Event"])["Quali_Position"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["Driver_Track_Best"] = df.groupby(["Driver", "Event"])["Quali_Position"].transform(
        lambda x: x.expanding().min().shift(1)
    )
    df["Team_Track_Avg"] = df.groupby(["Team", "Event"])["Quali_Position"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Interaction features
    df["Gap_x_DriverAvg"] = df["Avg_Gap"]  * df["Driver_Last5_Avg"]
    df["Rank_x_TeamAvg"]  = df["Avg_Rank"] * df["Team_Last5_Avg"]

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fills remaining NaN values using per-event medians, then global medians."""
    # Raw session times
    for col in ["FP1_Time", "FP2_Time", "FP3_Time"]:
        df[col] = df.groupby(["Year", "Event"])[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Derived per-session columns
    derived_cols = [c for c in df.columns if any(p in c for p in ["Gap_", "Rank_", "Percentile_"])]
    for col in derived_cols:
        df[col] = df.groupby(["Year", "Event"])[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Remaining numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in ["Year", "Quali_Position"] and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    remaining = df.isnull().sum().sum()
    print(f"Remaining missing values after imputation: {remaining}")
    return df


# Prediction

def build_prediction_input(
    df_model: pd.DataFrame, features: list
) -> pd.DataFrame:
    """
    Aligns the current race slice with the model's expected feature columns.
    Missing dummy columns (unseen teams/drivers) are filled with 0.
    """
    current_slice = df_model[
        (df_model["Year"] == CURRENT_YEAR) & (df_model["Event"] == EVENT_NAME)
    ].copy()

    X = pd.DataFrame(columns=features)
    for col in features:
        X[col] = current_slice[col] if col in current_slice.columns else 0

    X = X.fillna(0)
    return X


def run_prediction(model, X: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    """Runs the model and returns a sorted results DataFrame."""
    raw_scores = model.predict(X)

    results = df_info[["Driver", "Team"]].copy()
    results["Predicted_Score"]    = raw_scores
    results["Predicted_Position"] = (
        results["Predicted_Score"].rank(method="first").astype(int)
    )
    results = results.sort_values("Predicted_Position")

    print("\n" + "=" * 40)
    print(f"QUALIFYING PREDICTION — {EVENT_NAME} {CURRENT_YEAR}")
    print("=" * 40)
    print(results[["Predicted_Position", "Driver", "Team"]].to_string(index=False))

    return results, raw_scores


def save_predictions(results: pd.DataFrame) -> None:
    filename = f"{CURRENT_YEAR}_{EVENT_NAME.replace(' ', '_')}_prediction.csv"
    path = os.path.join(PATH_OUTPUT, filename)
    results.to_csv(path, index=False)
    print(f"\nPredictions saved to: {path}")


# SHAP explainability

def run_shap_analysis(
    model, X: pd.DataFrame, df_predict_encoded: pd.DataFrame, raw_scores: np.ndarray
) -> None:
    """Computes SHAP values and plots waterfall + bar charts for the predicted pole sitter."""
    print("\nComputing SHAP values...")
    explainer   = shap.Explainer(model)
    shap_values = explainer(X)

    pole_idx   = int(np.argmin(raw_scores))
    pole_score = raw_scores[pole_idx]

    # Identify pole driver from dummy columns
    driver_cols = [
        c for c in df_predict_encoded.columns
        if c.startswith("Driver_") and len(c) == 10
    ]
    pole_row  = df_predict_encoded.iloc[pole_idx]
    pole_code = next(
        (c.replace("Driver_", "") for c in driver_cols if pole_row[c] == 1),
        "Unknown"
    )

    print(f"\nPredicted pole sitter: {pole_code} (raw score: {pole_score:.4f})")

    # --- Waterfall chart ---
    fig, _ = plt.subplots(figsize=(14, 10))
    shap.plots.waterfall(shap_values[pole_idx], show=False)
    plt.title(
        f"SHAP Analysis — Why was {pole_code} predicted as Pole?\n{EVENT_NAME} {CURRENT_YEAR}",
        fontsize=16, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    waterfall_path = os.path.join(PATH_OUTPUT, f"shap_waterfall_{pole_code}.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Waterfall chart saved to: {waterfall_path}")

    # --- Feature importance summary ---
    importance   = shap_values[pole_idx].values
    feat_names   = shap_values[pole_idx].feature_names
    feat_values  = shap_values[pole_idx].data
    top_indices  = abs(importance).argsort()[-10:][::-1]

    print("\n" + "=" * 70)
    print("TOP 10 FEATURES  |  Negative impact = pulls toward P1")
    print("=" * 70)

    for i, idx in enumerate(top_indices, 1):
        impact    = importance[idx]
        direction = "HELPS  (pulls to P1)" if impact < 0 else "HURTS  (pushes back)"
        print(
            f"{i:2d}. {direction:<25s} | {feat_names[idx]:<30s} "
            f"value={feat_values[idx]:.4f}  SHAP={impact:+.4f}"
        )

    helps = sum(1 for v in importance[top_indices] if v < 0)
    hurts = sum(1 for v in importance[top_indices] if v > 0)
    total = importance.sum()

    print(f"\nHelping features (top 10): {helps}  |  Hurting features: {hurts}")
    print(f"Overall SHAP balance: {total:.4f}", end="  →  ")
    print("Strong pole candidate!" if total < 0 else "Not a clear favourite.")

    # --- Bar chart ---
    top_vals  = importance[top_indices]
    top_names = [feat_names[i] for i in top_indices]
    colors    = ["#00C853" if v < 0 else "#D32F2F" for v in top_vals]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_vals)), top_vals, color=colors, alpha=0.8, edgecolor="black")
    plt.yticks(range(len(top_vals)), top_names, fontsize=10)
    plt.xlabel("SHAP Impact  (negative = helps | positive = hurts)", fontsize=11, fontweight="bold")
    plt.title(
        f"Top 10 Features — {pole_code} — {EVENT_NAME} {CURRENT_YEAR}",
        fontsize=14, fontweight="bold",
    )
    plt.axvline(x=0, color="black", linewidth=1.2)
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.legend(
        handles=[
            Patch(facecolor="#00C853", label="Helps reach P1"),
            Patch(facecolor="#D32F2F", label="Pushes back"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    bar_path = os.path.join(PATH_OUTPUT, f"shap_bar_{pole_code}.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Bar chart saved to: {bar_path}")

    # --- Top 5 summary ---
    sorted_idx = np.argsort(raw_scores)
    medals = {1: "GOLD", 2: "SILVER", 3: "BRONZE"}

    print("\n" + "=" * 70)
    print(f"TOP 5 PREDICTED — {EVENT_NAME} {CURRENT_YEAR}")
    print("=" * 70)

    for pos, idx in enumerate(sorted_idx[:5], 1):
        row  = df_predict_encoded.iloc[idx]
        code = next(
            (c.replace("Driver_", "") for c in driver_cols if row[c] == 1),
            "???"
        )
        medal = f"[{medals[pos]}]" if pos in medals else "      "
        print(f"  P{pos} {medal}  {code:<6s}  score={raw_scores[idx]:.4f}")

    print("=" * 70)


# Entry point

def main() -> None:
    setup_output_dir()

    # 1. Load model and history
    model, features = load_model_and_features()
    df_history      = load_history()

    # 2. Load live practice data
    fp1, fp2, fp3 = load_live_sessions()
    df_live       = build_live_dataframe(fp1, fp2, fp3)

    # 3. Merge history + live rows
    df_combined = pd.concat([df_history, df_live], ignore_index=True)
    df_combined["Quali_Position"] = pd.to_numeric(
        df_combined["Quali_Position"], errors="coerce"
    )
    print(f"Combined dataset: {len(df_combined)} rows")

    # 4. Feature engineering
    print("Running feature engineering...")
    df = df_combined.copy()
    df = apply_track_metadata(df)

    standings = get_current_standings(CURRENT_YEAR)
    df = apply_current_standings(df, standings)

    df = build_features(df)
    df = impute_missing(df)

    # Snapshot of current-race rows (before encoding) for result labelling
    df_current_info = df[
        (df["Year"] == CURRENT_YEAR) & (df["Event"] == EVENT_NAME)
    ].copy()

    # 5. Encode categoricals and build model input
    df_encoded      = pd.get_dummies(df, columns=["Team", "Driver"], drop_first=True)
    df_current_enc  = df_encoded[
        (df_encoded["Year"] == CURRENT_YEAR) & (df_encoded["Event"] == EVENT_NAME)
    ].copy()

    X = build_prediction_input(df_encoded, features)
    print(f"Ready to predict {len(X)} drivers.")

    # 6. Predict
    results, raw_scores = run_prediction(model, X, df_current_info)

    # 7. Save predictions
    save_predictions(results)

    # 8. SHAP analysis
    run_shap_analysis(model, X, df_current_enc, raw_scores)

    print("\nAll done.")


if __name__ == "__main__":
    main()