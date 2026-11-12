# Customer Churn Prediction

A reproducible machine-learning project that explores whether a subscription customer is likely to churn. It demonstrates a complete classification workflow using Python, pandas and scikit-learn.

> **Dataset note:** The included customer dataset is synthetic and intentionally small. This project is for learning and portfolio demonstration, not for production churn decisions.

## Project objectives

- Prepare numeric and categorical customer features with a scikit-learn preprocessing pipeline.
- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate classification performance using precision, recall, F1-score, ROC-AUC and a confusion matrix.
- Save the trained model and evaluation report locally.

## Repository structure

```text
.
├── data/
│   └── customers.csv
├── src/
│   └── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

The training script writes generated files to `artifacts/` when run. Model binaries are excluded from version control; the tracked evaluation report can be reviewed at [artifacts/metrics.json](artifacts/metrics.json).

## Getting started

### Requirements

- Python 3.10 or newer
- pip

### Windows — Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python src/train.py
```

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Evaluation snapshot

The committed run selected **Logistic Regression**. Its reported ROC-AUC was **1.00**, with 100% precision, recall and F1 on the five-example test split (three class-0 and two class-1 examples).

**Interpret this result cautiously:** a perfect score on such a small synthetic test set is not evidence of real-world generalization. The dataset and test sample are too small to support a reliable performance claim. A stronger next step is to evaluate on a larger, independently sourced dataset, use cross-validation, inspect possible feature leakage and compare against a realistic baseline.

## Reproduce and inspect

Run `python src/train.py` to regenerate the local model and metrics. The metrics file is tracked for transparency; generated model binaries remain local by design.

## Tools and skills demonstrated

- Python, pandas and NumPy
- scikit-learn preprocessing pipelines
- Binary classification: Logistic Regression and Random Forest
- Model evaluation: precision, recall, F1, ROC-AUC and confusion matrix
- Reproducible command-line workflows and Git

## Limitations and responsible use

- Data is synthetic and does not represent actual customer behavior.
- The reported test split is very small and should not be used to compare models conclusively.
- No deployment, live customer integration or business impact is claimed.
