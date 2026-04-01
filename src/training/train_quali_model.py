"""
train_quali_model.py
Trains an XGBoost model to predict F1 qualifying positions using
practice session data and championship standings.
Saves the trained model, feature list, SHAP explainer, and evaluation metrics.
"""

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# Configuration

INPUT_FILE      = "/drs_data/data/master_qualifying_data_enriched.csv"
MODEL_PATH      = "/drs_data/models/xgb_model_quali_v1.pkl"
FEATURES_PATH   = "/drs_data/models/xgb_features_quali_v1.pkl"
EXPLAINER_PATH  = "/drs_data/models/shap_explainer_v1.pkl"
METRICS_PATH    = "/drs_data/models/metrics_quali_v1.json"

# Cold-start handicap position for rookie/new teams
HANDICAP_POS = 19.0

# Track metadata: Downforce 1 (low) to 5 (high), Overtaking 1 (hard) to 5 (easy)
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

# Top features used for training
TOP_FEATURES = [
    "Rank_x_TeamAvg", "Avg_Rank", "Avg_Percentile", "Gap_x_DriverAvg",
    "Driver_Last10_Avg", "Best_Rank", "Rank_FP3_Time", "Percentile_FP3_Time",
    "Min_Gap", "Avg_Gap", "Driver_Last3_Avg", "Driver_Last5_Avg",
    "Team_Last5_Avg", "Worst_Rank", "Driver_Last10_Best", "Driver_Last5_Best",
    "Gap_FP3_Time", "Rank_FP2_Time", "Percentile_FP2_Time", "Gap_FP2_Time",
    "Track_Downforce", "Track_Overtaking",
    "Driver_Points", "Driver_Wins", "Championship_Pos",
]

# XGBoost hyperparameter search grid
PARAM_GRID = {
    "n_estimators":     [100, 300, 500],
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [3, 5, 7],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}


# Feature engineering

def apply_track_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Adds track metadata columns (Downforce, Overtaking, Type) to the DataFrame."""
    print("Applying track metadata (physics-awareness)...")

    def _get_stats(row):
        stats = TRACK_STATS.get(
            row["Event"], {"Downforce": 3, "Overtaking": 3, "Type": "Permanent"}
        )
        return pd.Series([stats["Downforce"], stats["Overtaking"], stats["Type"]])

    df[["Track_Downforce", "Track_Overtaking", "Track_Type"]] = df.apply(
        _get_stats, axis=1
    )
    print("Track metadata applied.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all derived features from raw practice session times."""

    # Relative gaps and ranks within each session
    for session in ["FP1_Time", "FP2_Time", "FP3_Time"]:
        p1_time = df.groupby(["Year", "Event"])[session].transform("min")
        df[f"Gap_{session}"]        = df[session] - p1_time
        df[f"Rank_{session}"]       = df.groupby(["Year", "Event"])[session].rank()
        df[f"Percentile_{session}"] = df.groupby(["Year", "Event"])[session].rank(pct=True)

    # Gap aggregations
    gap_cols = ["Gap_FP1_Time", "Gap_FP2_Time", "Gap_FP3_Time"]
    df["Avg_Gap"] = df[gap_cols].mean(axis=1)
    df["Min_Gap"] = df[gap_cols].min(axis=1)
    df["Max_Gap"] = df[gap_cols].max(axis=1)
    df["Std_Gap"] = df[gap_cols].std(axis=1)

    # Rank aggregations
    rank_cols = ["Rank_FP1_Time", "Rank_FP2_Time", "Rank_FP3_Time"]
    df["Avg_Rank"]   = df[rank_cols].mean(axis=1)
    df["Best_Rank"]  = df[rank_cols].min(axis=1)
    df["Worst_Rank"] = df[rank_cols].max(axis=1)

    # Percentile aggregations
    df["Avg_Percentile"] = df[
        ["Percentile_FP1_Time", "Percentile_FP2_Time", "Percentile_FP3_Time"]
    ].mean(axis=1)

    # Consistency metrics
    df["Gap_Range"] = df["Max_Gap"] - df["Min_Gap"]
    df["Rank_Range"] = df["Worst_Rank"] - df["Best_Rank"]
    df["Gap_CV"]    = df["Std_Gap"] / (df["Avg_Gap"].abs() + 0.001)  # coefficient of variation

    # Session-to-session trends (negative = improvement)
    df["Trend_Gap"]  = df["Gap_FP3_Time"]  - df["Gap_FP1_Time"]
    df["Trend_Rank"] = df["Rank_FP3_Time"] - df["Rank_FP1_Time"]

    # Absolute time aggregations
    fp_cols = ["FP1_Time", "FP2_Time", "FP3_Time"]
    df["Avg_FP_Time"] = df[fp_cols].mean(axis=1)
    df["Min_FP_Time"] = df[fp_cols].min(axis=1)

    # Driver rolling history
    df = df.sort_values(["Driver", "Year", "Event"])
    for window in [3, 5, 10]:
        df[f"Driver_Last{window}_Avg"] = df.groupby("Driver")["Quali_Position"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        df[f"Driver_Last{window}_Best"] = df.groupby("Driver")["Quali_Position"].transform(
            lambda x: x.rolling(window=window, min_periods=1).min().shift(1)
        )

    # Recent trend: last 5 vs last 10 races
    df["Driver_Recent_Trend"] = df["Driver_Last5_Avg"] - df["Driver_Last10_Avg"]

    # Driver consistency (expanding std)
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


def apply_cold_start_handicap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills NaN historical features for new/rookie teams (e.g. Cadillac)
    with a back-of-grid penalty to prevent the model from assuming global average.
    """
    rookie_teams = ["Cadillac"]
    print(f"\nChecking cold-start handicap for: {rookie_teams}")

    mask_rookie = df["Team"].isin(rookie_teams)

    if mask_rookie.sum() == 0:
        print("  -> No new teams found in this dataset (expected for historical data).")
        return df

    print(f"  -> Found {mask_rookie.sum()} entries for new teams.")

    cols_to_fix = [
        "Driver_Last10_Avg", "Driver_Last3_Avg", "Driver_Last5_Avg",
        "Team_Last5_Avg", "Team_Last3_Avg",
        "Driver_Last10_Best", "Driver_Last5_Best",
        "Driver_Recent_Trend",
    ]

    for col in cols_to_fix:
        if col in df.columns:
            nans = df.loc[mask_rookie, col].isna().sum()
            if nans > 0:
                df.loc[mask_rookie & df[col].isna(), col] = HANDICAP_POS
                print(f"    -> Imputed P{HANDICAP_POS} in '{col}' for {nans} rows.")

    # Trend should be neutral for rookies, not penalized
    if "Driver_Recent_Trend" in df.columns:
        df.loc[mask_rookie & df["Driver_Recent_Trend"].isna(), "Driver_Recent_Trend"] = 0.0

    return df


# Training

def train_model(X_train, y_train) -> tuple:
    """Runs GridSearchCV over XGBoost and returns the best model."""
    print("Starting XGBoost training with GridSearchCV...")

    xgb_model = XGBRegressor(
        objective="reg:absoluteerror",  # optimise directly for MAE
        n_jobs=-1,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=1,
        n_jobs=-1,
    )

    print("Optimising hyperparameters (this may take a while)...")
    grid_search.fit(X_train, y_train)

    print(f"\nTraining complete.")
    print(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_


# Evaluation

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """
    Evaluates the model on train and test sets.
    Prints results to terminal and returns a metrics dict.
    """
    pred_train = np.round(np.clip(model.predict(X_train), 1, 20))
    pred_test  = np.round(np.clip(model.predict(X_test),  1, 20))

    metrics = {
        "train": {
            "r2":  round(r2_score(y_train, pred_train), 4),
            "mae": round(mean_absolute_error(y_train, pred_train), 4),
        },
        "test": {
            "r2":  round(r2_score(y_test, pred_test), 4),
            "mae": round(mean_absolute_error(y_test, pred_test), 4),
        },
    }

    print("\n--- TRAIN ---")
    print(f"  R²  : {metrics['train']['r2']}")
    print(f"  MAE : {metrics['train']['mae']} positions")

    print("\n--- TEST ---")
    print(f"  R²  : {metrics['test']['r2']}")
    print(f"  MAE : {metrics['test']['mae']} positions")

    return metrics


def save_metrics(metrics: dict, best_params: dict) -> None:
    """Saves evaluation metrics and best hyperparameters to a JSON file."""
    output = {
        "model":       "XGBRegressor",
        "version":     "v1",
        "best_params": best_params,
        "metrics":     metrics,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMetrics saved to: {METRICS_PATH}")


# SHAP explainability

def run_shap(model, X_test: pd.DataFrame) -> None:
    """Computes SHAP values and plots a beeswarm summary for the test set."""
    print("\nComputing SHAP values...")
    shap.initjs()

    explainer   = shap.Explainer(model)
    shap_values = explainer(X_test)

    plt.title("Feature Impact on Qualifying Position", fontsize=14)
    shap.plots.beeswarm(shap_values, max_display=15)

    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"SHAP explainer saved to: {EXPLAINER_PATH}")


# Entry point

def main() -> None:
    # 1. Load data
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["Quali_Position"] = df["Quali_Position"].astype(int)
    print(f"Enriched dataset loaded: {len(df)} rows.")

    # 2. Feature engineering
    df = apply_track_stats(df)
    df = build_features(df)
    df = apply_cold_start_handicap(df)

    # 3. Encode categorical columns
    print("\nEncoding Team and Driver columns...")
    df_model = pd.get_dummies(df, columns=["Team", "Driver"], drop_first=True)

    # 4. Select valid features
    valid_features = [col for col in TOP_FEATURES if col in df_model.columns]
    print(f"{len(valid_features)} features selected for the model.")

    X = df_model[valid_features]
    y = df_model["Quali_Position"]

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 6. Train
    best_model, best_params = train_model(X_train, y_train)

    # 7. Evaluate and save metrics
    metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    save_metrics(metrics, best_params)

    # 8. Save model and features
    joblib.dump(best_model,    MODEL_PATH)
    joblib.dump(valid_features, FEATURES_PATH)
    print(f"\nModel saved to:    {MODEL_PATH}")
    print(f"Features saved to: {FEATURES_PATH}")

    # 9. SHAP explainability
    run_shap(best_model, X_test)


if __name__ == "__main__":
    main()