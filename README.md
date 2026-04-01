# DRS Data — F1 Telemetry & Race Prediction

**A high-performance Machine Learning system for Formula 1® analysis, prediction, and race simulation.**

DRS Data (Data Racing Strategies) is an end-to-end ML pipeline designed to model race weekends, predict qualifying results, simulate race scenarios, and explain performance using telemetry and session data.

---

## 📊 Results

Latest model performance:

- **Qualifying MAE:** 2.18 – 2.88 positions  
- **Race MAE:** ~2.5 positions  
- **R²:** up to 0.65  

The model captures meaningful performance patterns despite the stochastic nature of Formula 1.

---

## 🧠 Features

- 📈 Qualifying prediction pipeline  
- 🏁 Race pace analysis  
- 🔁 Race simulation engine  
- 🔍 SHAP explainability  
- 📊 Automated accuracy reporting (JSON + Markdown)  
- ⚙️ Modular ML pipeline (training → prediction → simulation)

---

## 🏗️ Project Structure

``` text
drs_data/
├── src/
│ ├── training/ # data collection, feature engineering, training
│ ├── prediction/ # qualifying prediction 
│ ├── simulation/ # race simulation logic
│ ├── analisys/ # model accuracy analysis
│
├── data/ # raw & processed datasets
├── models/ # trained models (.pkl)
├── cache/ # FastF1 cache
├── outputs_predictions/ # predictions & simulation outputs
```


---

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/palomacdev/drs_data
```

```bash
cd drs_data
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Dataset

*First run may take a while. You can process data year by year.*
```bash
python src/training/collect_qualifying_data.py
```


```bash
python src/training/data_enrichment.py
```

### 4. Run Training

```bash
python src/training/train_quali_model.py
```

### 5. Run Predictions
```bash
python src/prediction/predict_qualifying.py
```

---

## 📊 Model Reporting

The project includes automated reporting:

- JSON logs for programmatic analysis  
- Markdown summaries for human-readable insights  

Generated via:
```bash
python src/analysis/accuracy_report.py
```
---

## 🧪 Simulation

Race scenarios can be simulated using:

```bash
python src/simulation/race_simulator.py
```

*To run race simulator you need to also run the race pace*
```bash
python src/simulation/race_pace_overview.py
```


---

## 🔮 Roadmap

- Improve feature engineering (weather, tire strategies, track evolution)  
- Add real-time prediction capabilities  
- Expand simulation realism  
- Deploy as API or dashboard  

---

## 🤝 Contributing

Contributions are welcome!  
Check `CONTRIBUTING.md` for guidelines.

---

## 📄 License

MIT License

---

## 💭 Final Note

This project was built as an exploration of applying Machine Learning to motorsport data.

If you're into F1, data engineering, or ML systems this is for you.