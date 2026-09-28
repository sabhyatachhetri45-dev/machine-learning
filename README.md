# Machine Learning Project

This project contains two machine learning tasks with separate scripts, datasets, outputs, and live reports.

## Setup

1. Create a virtual environment
2. Install dependencies
3. Run the script

```bash
python -m venv .venv
.venv\Scripts\python -m pip install pandas numpy matplotlib seaborn scikit-learn
.venv\Scripts\python task1\task1.py
.venv\Scripts\python task2\task2.py
```

## Files

- `task1/` - calorie prediction script, dataset, results, and graphs
- `task2/` - customer purchase script, dataset, results, and confusion matrix
- `docs/task1/` - live Task 1 report
- `docs/task2/` - live Task 2 report
- `machine.py` - separate student performance prediction script

## GitHub Pages

The project deploys automatically from the `main` branch using GitHub Actions.
After the workflow completes, the report is available at:

`https://sabhyatachhetri45-dev.github.io/machine-learning/`
