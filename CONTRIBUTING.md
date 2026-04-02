# Contributing to DRS Data 🏎️

First off — thank you for taking the time to contribute! DRS Data is an open project and every improvement, whether it's a bug fix, new feature, or documentation update, makes a real difference.

---

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Project Structure](#project-structure)
- [Ideas & Roadmap](#ideas--roadmap)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

---

## Getting Started

Before you contribute, make sure you have the project running locally:

```bash
git clone https://github.com/palomacdev/drs_data
cd drs_data
pip install -r requirements.txt
```

> **Note:** The first data collection run may take a while depending on the years you process. Start small — one season at a time.

---

## How to Contribute

1. **Fork** this repository
2. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-you-are-fixing
   ```
3. **Make your changes** — keep commits small and focused
4. **Test your changes** before submitting
5. **Open a Pull Request** with a clear description of what you changed and why

### Branch naming conventions

| Prefix | Use for |
|--------|---------|
| `feature/` | New features or enhancements |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behavior change |
| `experiment/` | Exploratory work / model experiments |

---

## Development Guidelines

- **Keep code modular** — each script should have a single, clear responsibility
- **Follow the existing project structure** — don't mix training logic into simulation scripts, etc.
- **Comment your code** — especially in ML pipelines where data transformations can be non-obvious
- **Avoid hardcoded values** — use constants or config variables for things like season years, circuit names, model paths
- **Don't commit large files** — avoid pushing `.pkl` models, large CSVs, or FastF1 cache files (`.gitignore` already covers most of these)

---

## Project Structure

Understanding the structure helps you place your contribution in the right place:

```
drs_data/
├── src/
│   ├── training/        # Data collection, feature engineering, model training
│   ├── prediction/      # Qualifying and race predictions
│   ├── simulation/      # Race simulation engine and pace analysis
│   └── analysis/        # Model accuracy reporting
│
├── outputs_predictions/ # Generated prediction outputs
├── requirements.txt
└── README.md
```

---

## Ideas & Roadmap

Looking for somewhere to start? Here are open directions for contribution:

**Model improvements**
- Incorporate weather data (rain probability, temperature) as features
- Add tire strategy modeling (compound, age, undercut windows)
- Track evolution features (rubber accumulation across a race weekend)

**Simulation enhancements**
- Safety car / VSC probability modeling
- Pit stop delta time by circuit
- Multi-driver battle simulation

**Engineering & infrastructure**
- Real-time prediction capabilities
- REST API or dashboard (Streamlit / FastAPI)
- Automated retraining pipeline after each race weekend
- Unit tests for pipeline components

**Documentation**
- Improve inline code documentation
- Add example notebooks showing predictions for specific GPs

---

## Reporting Issues

Found a bug or unexpected behavior? Please open an issue and include:

- A clear description of the problem
- Steps to reproduce it
- Relevant error messages or logs
- Your Python version and OS

---

## Code of Conduct

This project is welcoming to contributors of all experience levels. Please be respectful, constructive, and collaborative. Criticism of code is fine — criticism of people is not.

---

Questions? Open an issue or reach out directly. Happy to help you get started. 🚀