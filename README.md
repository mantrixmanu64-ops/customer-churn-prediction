# Customer Churn Prediction

An end-to-end machine-learning example estimating whether a subscription customer may churn.

## Workflow
- Preprocess numeric and categorical features with scikit-learn pipelines.
- Compare logistic regression and random forest.
- Evaluate precision, recall, F1, ROC-AUC and a confusion matrix.
- Save the selected model and metrics under `artifacts/`.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/train.py
```

The sample dataset is synthetic. Run the code yourself before reporting metrics; no performance result is claimed in advance.
