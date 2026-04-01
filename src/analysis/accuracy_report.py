"""
accuracy_report.py
F1 2026 — Prediction Accuracy Analyzer
Compares M18 (Qualifying) or M30 (Race) predictions against official FastF1 results.
Saves a human-readable Markdown report and a structured JSON file to the output folder.
"""

import json
import os
import warnings
from datetime import datetime

import fastf1 as ff1
import pandas as pd

warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

ANO_ATUAL   = 2026
NOME_EVENTO = "Japan Grand Prix"

# MODE: "QUALI" 
#       "RACE"  
ANALYSIS_MODE = "QUALI"

# --- Paths ---
ROOT_PREDICTIONS = "/drs_data/outputs_predictions/2026"
CACHE_PATH       = "/drs_data/cache"

ff1.Cache.enable_cache(CACHE_PATH)
print(f"Cache enabled at: {CACHE_PATH}")

race_folder_name = NOME_EVENTO.replace(" ", "_")
PATH_PREDICTIONS = os.path.join(ROOT_PREDICTIONS, race_folder_name)

if not os.path.exists(PATH_PREDICTIONS):
    os.makedirs(PATH_PREDICTIONS)
    print(f"📂 Output folder created: {PATH_PREDICTIONS}")
else:
    print(f"📂 Saving files to: {PATH_PREDICTIONS}")


# ==============================================================================
# SECTION 2: LOAD PREDICTION DATA
# ==============================================================================

print(f"\n--- Analyzing {ANALYSIS_MODE} predictions for {NOME_EVENTO} {ANO_ATUAL} ---")

try:
    if ANALYSIS_MODE == "QUALI":
        suffix       = "_prediction.csv"
        col_target   = "Predicted_Position"
        session_code = "Q"
    elif ANALYSIS_MODE == "RACE":
        suffix       = "_RaceSimulation.csv"
        col_target   = "Final_Position"
        session_code = "R"
    else:
        raise ValueError(f"Invalid ANALYSIS_MODE: '{ANALYSIS_MODE}'. Use 'QUALI' or 'RACE'.")

    filename  = f"{ANO_ATUAL}_{NOME_EVENTO.replace(' ', '_')}{suffix}"
    file_path = os.path.join(PATH_PREDICTIONS, filename)

    if not os.path.exists(file_path):
        print(f"🚨 CRITICAL ERROR: File not found — {filename}")
        print(f"Available files in {PATH_PREDICTIONS}:")
        print(os.listdir(PATH_PREDICTIONS))
        raise FileNotFoundError(f"Prediction data file not found: {file_path}")

    df_pred = pd.read_csv(file_path)
    print(f"✅ Prediction data loaded: {file_path}")

    # Normalize the position column name to 'Predicted_Pos'
    if col_target in df_pred.columns:
        df_pred.rename(columns={col_target: "Predicted_Pos"}, inplace=True)
    elif "Predicted_Pos" not in df_pred.columns:
        raise KeyError(
            f"Column '{col_target}' not found in CSV. "
            f"Available columns: {list(df_pred.columns)}"
        )

except Exception as e:
    print(f"🚨 Error during configuration or file loading: {e}")
    raise


# ==============================================================================
# SECTION 3: LOAD ACTUAL RESULTS FROM FASTF1
# ==============================================================================

print("Loading actual session results from FastF1...")

try:
    session = ff1.get_session(ANO_ATUAL, NOME_EVENTO, session_code)
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    if session.results is None or session.results.empty:
        raise ValueError("Session results are not available in FastF1.")

    df_actual = session.results[["Abbreviation", "Position", "Status"]].copy()
    df_actual.rename(
        columns={"Abbreviation": "Driver", "Position": "Actual_Pos"}, inplace=True
    )

    # Handle missing positions (Ergast API outage or DNFs)
    if df_actual["Actual_Pos"].isna().all():
        print("⚠️ Official positions unavailable (Ergast issue). Using session finishing order.")
        df_actual["Actual_Pos"] = range(1, len(df_actual) + 1)
    else:
        max_finished = df_actual["Actual_Pos"].max()
        if pd.isna(max_finished):
            max_finished = 0
        mask_nan = df_actual["Actual_Pos"].isna()
        num_nan  = mask_nan.sum()
        if num_nan > 0:
            df_actual.loc[mask_nan, "Actual_Pos"] = range(
                int(max_finished) + 1, int(max_finished) + num_nan + 1
            )

    print(f"✅ Actual results loaded: {len(df_actual)} drivers found.")
    winner = df_actual.iloc[0]
    print(f"🏆 Race winner: {winner['Driver']} (P{int(winner['Actual_Pos'])})")

except Exception as e:
    print(f"🚨 ERROR loading FastF1 data: {e}")
    raise


# ==============================================================================
# SECTION 4: MERGE & CALCULATE METRICS
# ==============================================================================

# Normalize driver column name
if "Driver" not in df_pred.columns and "Abbreviation" in df_pred.columns:
    df_pred.rename(columns={"Abbreviation": "Driver"}, inplace=True)

# Smart column detection for predicted position
for candidate in ["Final_Position", "Predicted_Position", "Predicted_Pos"]:
    if candidate in df_pred.columns:
        col_to_use = candidate
        print(f"ℹ️  Using '{col_to_use}' column for comparison.")
        break
else:
    print(f"🚨 Available columns: {list(df_pred.columns)}")
    raise KeyError(
        "Could not find a valid position column "
        "(Final_Position, Predicted_Position, or Predicted_Pos) in the CSV."
    )

df_pred_clean = df_pred[["Driver", col_to_use]].copy()
df_pred_clean.rename(columns={col_to_use: "Predicted_Pos"}, inplace=True)

# Inner join — keeps only drivers present in both datasets
df_compare = pd.merge(df_pred_clean, df_actual, on="Driver", how="inner")

if df_compare.empty:
    print("🚨 WARNING: Merge returned an empty DataFrame. Check driver abbreviations.")
    print(f"  Prediction drivers (sample): {df_pred_clean['Driver'].unique()[:5]}")
    print(f"  Actual drivers    (sample): {df_actual['Driver'].unique()[:5]}")

df_compare["Error"] = abs(df_compare["Predicted_Pos"] - df_compare["Actual_Pos"])
df_compare = df_compare.sort_values(by="Actual_Pos").reset_index(drop=True)

# --- Core Metrics ---
total_drivers  = len(df_compare)
mae            = df_compare["Error"].mean()
exact_hits     = int((df_compare["Error"] == 0).sum())
close_hits     = int((df_compare["Error"] <= 3).sum())
exact_pct      = (exact_hits / total_drivers) * 100
close_pct      = (close_hits / total_drivers) * 100

# Top 3 accuracy
top3_actual    = set(df_compare[df_compare["Actual_Pos"]    <= 3]["Driver"])
top3_predicted = set(df_compare[df_compare["Predicted_Pos"] <= 3]["Driver"])
top3_correct   = len(top3_actual & top3_predicted)
top3_accuracy  = (top3_correct / 3) * 100 if len(top3_actual) >= 3 else 0.0

# Top 10 accuracy
top10_actual    = set(df_compare[df_compare["Actual_Pos"]    <= 10]["Driver"])
top10_predicted = set(df_compare[df_compare["Predicted_Pos"] <= 10]["Driver"])
top10_correct   = len(top10_actual & top10_predicted)
top10_accuracy  = (top10_correct / 10) * 100 if len(top10_actual) >= 10 else 0.0

# --- Error distribution buckets ---
perfect   = int((df_compare["Error"] == 0).sum())
excellent = int((df_compare["Error"] == 1).sum())
very_good = int((df_compare["Error"] == 2).sum())
good      = int((df_compare["Error"] == 3).sum())
miss      = int(((df_compare["Error"] >= 4) & (df_compare["Error"] <= 5)).sum())
big_miss  = int((df_compare["Error"] > 5).sum())

# --- Performance rating ---
if mae < 2.0:
    rating = "EXCELLENT ⭐⭐⭐⭐⭐"
elif mae < 3.0:
    rating = "VERY GOOD ⭐⭐⭐⭐"
elif mae < 4.0:
    rating = "GOOD ⭐⭐⭐"
elif mae < 5.0:
    rating = "FAIR ⭐⭐"
else:
    rating = "NEEDS IMPROVEMENT ⭐"

print("✅ Metrics calculated successfully.")


# ==============================================================================
# SECTION 5: CONSOLE OUTPUT
# ==============================================================================

def get_status_label(error: int) -> str:
    if error == 0:   return "✅ PERFECT"
    if error == 1:   return "✅ EXCELLENT"
    if error <= 2:   return "✅ VERY GOOD"
    if error <= 3:   return "✅ GOOD"
    if error <= 5:   return "❌ MISS"
    return                   "⚠️  BIG MISS"


print("\n" + "=" * 70)
print(f"📊 ACCURACY REPORT: {ANALYSIS_MODE} — {NOME_EVENTO} {ANO_ATUAL}")
print("=" * 70)

print("\n📈 OVERALL METRICS:")
print("-" * 70)
print(f"📉 MAE (Mean Absolute Error):     {mae:.2f} positions")
print(f"🎯 Exact Hits (Perfect):          {exact_hits}/{total_drivers} ({exact_pct:.1f}%)")
print(f"✅ Close Hits (±3 positions):     {close_hits}/{total_drivers} ({close_pct:.1f}%)")
print(f"🏆 Top 3 Accuracy:                {top3_correct}/3 ({top3_accuracy:.1f}%)")
print(f"🔟 Top 10 Accuracy:               {top10_correct}/10 ({top10_accuracy:.1f}%)")
print(f"📊 Overall Rating:                {rating}")
print("-" * 70)

print("\n--- DRIVER BY DRIVER COMPARISON ---")
print(f"{'DRIVER':<8} | {'PRED':<5} | {'ACTUAL':<6} | {'DIFF':<6} | STATUS")
print("-" * 60)
for _, row in df_compare.iterrows():
    diff     = int(row["Predicted_Pos"] - row["Actual_Pos"])
    diff_str = f"{diff:+d}" if diff != 0 else " 0"
    print(
        f"{row['Driver']:<8} | P{int(row['Predicted_Pos']):<4} | "
        f"P{int(row['Actual_Pos']):<5} | {diff_str:<6} | "
        f"{get_status_label(int(row['Error']))}"
    )
print("-" * 60)

print("\n📊 ERROR DISTRIBUTION:")
print("-" * 70)
print(f"Perfect  (0):    {perfect:2d} drivers ({perfect  / total_drivers * 100:5.1f}%)")
print(f"Excellent (±1):  {excellent:2d} drivers ({excellent / total_drivers * 100:5.1f}%)")
print(f"Very Good (±2):  {very_good:2d} drivers ({very_good / total_drivers * 100:5.1f}%)")
print(f"Good (±3):       {good:2d} drivers ({good     / total_drivers * 100:5.1f}%)")
print(f"Miss (4–5):      {miss:2d} drivers ({miss     / total_drivers * 100:5.1f}%)")
print(f"Big Miss (6+):   {big_miss:2d} drivers ({big_miss / total_drivers * 100:5.1f}%)")
print("-" * 70)

print("\n--- BIGGEST MISSES ---")
for _, row in df_compare.sort_values("Error", ascending=False).head(3).iterrows():
    print(
        f"❌ {row['Driver']}: Predicted P{int(row['Predicted_Pos'])}, "
        f"Finished P{int(row['Actual_Pos'])} "
        f"(Error: {int(row['Error'])} positions)"
    )

print("\n--- BEST PREDICTIONS ---")
for _, row in df_compare.sort_values("Error").head(3).iterrows():
    label = "PERFECT ✅" if row["Error"] == 0 else ("EXCELLENT" if row["Error"] <= 1 else "VERY GOOD")
    print(
        f"✅ {row['Driver']}: Predicted P{int(row['Predicted_Pos'])}, "
        f"Finished P{int(row['Actual_Pos'])} "
        f"(Error: {int(row['Error'])} — {label})"
    )

print("\n" + "=" * 70)


# ==============================================================================
# SECTION 6: SAVE RESULTS
# ==============================================================================

base_filename = f"{ANO_ATUAL}_{NOME_EVENTO.replace(' ', '_')}_{ANALYSIS_MODE}_AccuracyReport"
generated_at  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# --- 6A: Markdown Report ---
md_path = os.path.join(PATH_PREDICTIONS, f"{base_filename}.md")

driver_rows = []
for _, row in df_compare.iterrows():
    diff     = int(row["Predicted_Pos"] - row["Actual_Pos"])
    diff_str = f"{diff:+d}" if diff != 0 else "0"
    driver_rows.append(
        f"| {row['Driver']} | P{int(row['Predicted_Pos'])} | "
        f"P{int(row['Actual_Pos'])} | {diff_str} | "
        f"{get_status_label(int(row['Error']))} |"
    )

md_content = f"""# 📊 Accuracy Report: {ANALYSIS_MODE} — {NOME_EVENTO} {ANO_ATUAL}

> Generated at: `{generated_at}`

---

## 📈 Overall Metrics

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | {mae:.2f} positions |
| Exact Hits (Perfect) | {exact_hits}/{total_drivers} ({exact_pct:.1f}%) |
| Close Hits (±3 positions) | {close_hits}/{total_drivers} ({close_pct:.1f}%) |
| Top 3 Accuracy | {top3_correct}/3 ({top3_accuracy:.1f}%) |
| Top 10 Accuracy | {top10_correct}/10 ({top10_accuracy:.1f}%) |
| Overall Rating | {rating} |

---

## 🏎️ Driver by Driver Comparison

| Driver | Predicted | Actual | Diff | Status |
|---|---|---|---|---|
{chr(10).join(driver_rows)}

---

## 📊 Error Distribution

| Category | Drivers | Share |
|---|---|---|
| Perfect (0) | {perfect} | {perfect / total_drivers * 100:.1f}% |
| Excellent (±1) | {excellent} | {excellent / total_drivers * 100:.1f}% |
| Very Good (±2) | {very_good} | {very_good / total_drivers * 100:.1f}% |
| Good (±3) | {good} | {good / total_drivers * 100:.1f}% |
| Miss (4–5) | {miss} | {miss / total_drivers * 100:.1f}% |
| Big Miss (6+) | {big_miss} | {big_miss / total_drivers * 100:.1f}% |

---

## ❌ Biggest Misses

{chr(10).join(
    f"- **{row['Driver']}**: Predicted P{int(row['Predicted_Pos'])}, "
    f"Finished P{int(row['Actual_Pos'])} — Error: {int(row['Error'])} positions"
    for _, row in df_compare.sort_values("Error", ascending=False).head(3).iterrows()
)}

## ✅ Best Predictions

{chr(10).join(
    f"- **{row['Driver']}**: Predicted P{int(row['Predicted_Pos'])}, "
    f"Finished P{int(row['Actual_Pos'])} — Error: {int(row['Error'])}"
    for _, row in df_compare.sort_values("Error").head(3).iterrows()
)}
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"✅ Markdown report saved: {md_path}")


# --- 6B: JSON Export ---
json_path = os.path.join(PATH_PREDICTIONS, f"{base_filename}.json")

json_payload = {
    "metadata": {
        "event":          NOME_EVENTO,
        "year":           ANO_ATUAL,
        "mode":           ANALYSIS_MODE,
        "generated_at":   generated_at,
        "total_drivers":  total_drivers,
    },
    "metrics": {
        "mae":              round(mae, 4),
        "exact_hits":       exact_hits,
        "exact_pct":        round(exact_pct, 2),
        "close_hits":       close_hits,
        "close_pct":        round(close_pct, 2),
        "top3_correct":     top3_correct,
        "top3_accuracy":    round(top3_accuracy, 2),
        "top10_correct":    top10_correct,
        "top10_accuracy":   round(top10_accuracy, 2),
        "overall_rating":   rating,
    },
    "error_distribution": {
        "perfect":          perfect,
        "excellent":        excellent,
        "very_good":        very_good,
        "good":             good,
        "miss":             miss,
        "big_miss":         big_miss,
    },
    "drivers": [
        {
            "driver":        row["Driver"],
            "predicted_pos": int(row["Predicted_Pos"]),
            "actual_pos":    int(row["Actual_Pos"]),
            "error":         int(row["Error"]),
            "status":        get_status_label(int(row["Error"])).replace("✅ ", "").replace("❌ ", "").replace("⚠️  ", ""),
        }
        for _, row in df_compare.sort_values("Actual_Pos").iterrows()
    ],
    "biggest_misses": [
        {
            "driver":        row["Driver"],
            "predicted_pos": int(row["Predicted_Pos"]),
            "actual_pos":    int(row["Actual_Pos"]),
            "error":         int(row["Error"]),
        }
        for _, row in df_compare.sort_values("Error", ascending=False).head(3).iterrows()
    ],
    "best_predictions": [
        {
            "driver":        row["Driver"],
            "predicted_pos": int(row["Predicted_Pos"]),
            "actual_pos":    int(row["Actual_Pos"]),
            "error":         int(row["Error"]),
        }
        for _, row in df_compare.sort_values("Error").head(3).iterrows()
    ],
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2, ensure_ascii=False)
print(f"✅ JSON report saved:     {json_path}")