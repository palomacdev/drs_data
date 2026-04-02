<div align="center">

# 🏎️ DRS Data

### F1 Telemetry & Race Prediction

*A high-performance Machine Learning system for Formula 1® analysis, prediction, and race simulation.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastF1](https://img.shields.io/badge/FastF1-Telemetry-E10600?style=for-the-badge&logo=f1&logoColor=white)](https://github.com/theOehrly/Fast-F1)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-8B5CF6?style=for-the-badge)](CONTRIBUTING.md)

</div>

---

## 🧠 What is DRS Data?

**DRS Data** (Data Racing Strategies) is an end-to-end ML pipeline designed to model Formula 1 race weekends — from raw telemetry to race outcome prediction.

The system collects session data via FastF1, engineers domain-specific features, trains predictive models, simulates race scenarios, and explains performance using SHAP — all in a modular, extensible pipeline.

---

## 📊 Model Performance

<div align="center">

| Metric | Value |
|:---|:---:|
| 🏎️ Qualifying MAE | `2.18 – 2.88 positions` |
| 🏁 Race MAE | `~2.5 positions` |
| 📈 R² Score | `up to 0.65` |

</div>

> The model captures meaningful performance patterns despite the inherently stochastic nature of Formula 1 racing.

---

## ⚡ Features

| Module | Description |
|:---|:---|
| 📈 **Qualifying Prediction** | Predicts grid positions using session telemetry |
| 🏁 **Race Pace Analysis** | Models race-long pace and tire degradation |
| 🔁 **Race Simulation** | Simulates race scenarios lap by lap |
| 🔍 **SHAP Explainability** | Explains what drives each prediction |
| 📊 **Automated Reporting** | Generates JSON logs and Markdown summaries |
| ⚙️ **Modular Pipeline** | Clean flow: training → prediction → simulation |

---

## 🏗️ Project Structure

```
drs_data/
├── src/
│   ├── training/           # Data collection, feature engineering, model training
│   ├── prediction/         # Qualifying and race predictions
│   ├── simulation/         # Race simulation engine and pace analysis
│   └── analysis/           # Model accuracy reporting
│
├── data/                   # Raw & processed datasets
├── models/                 # Trained models (.pkl)
├── cache/                  # FastF1 session cache
├── outputs_predictions/    # Generated prediction outputs
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/palomacdev/drs_data
cd drs_data
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Collect & enrich data

> ⏳ First run may take a while — process data year by year for best results.

```bash
python src/training/collect_qualifying_data.py
python src/training/data_enrichment.py
```

### 4. Train the model

```bash
python src/training/train_quali_model.py
```

### 5. Run predictions

```bash
python src/prediction/predict_qualifying.py
```

---

## 📊 Model Reporting

Generate automated accuracy reports (JSON + Markdown):

```bash
python src/analysis/accuracy_report.py
```

---

## 🧪 Race Simulation

Simulate full race scenarios using trained models:

```bash
# Step 1 — run race pace analysis
python src/simulation/race_pace_overview.py

# Step 2 — run the race simulator
python src/simulation/race_simulator.py
```

---

## 🔮 Roadmap

- [ ] Weather integration (rain probability, temperature, humidity)
- [ ] Tire strategy modeling (compound, age, undercut windows)
- [ ] Real-time prediction pipeline
- [ ] REST API or Streamlit dashboard
- [ ] Improved simulation realism (safety car, pit stops)

---

## 🤝 Contributing

Contributions are welcome — from bug fixes to new features and experiments!

Check [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch conventions, guidelines, and ideas on where to start.

---

## 📄 License

Distributed under the [MIT License](LICENSE).

---

<div align="center">

*Built by someone who loves both data and the sound of a V10.*
*If you're into F1, ML, or both — this project is for you. 🏁*

**[⭐ Star this repo](https://github.com/palomacdev/drs_data) if you find it useful!**

</div>