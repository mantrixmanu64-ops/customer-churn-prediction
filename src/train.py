from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'customers.csv')
X, y = df.drop(columns='churn'), df['churn']
numeric = X.select_dtypes(include='number').columns.tolist()
categorical = X.select_dtypes(exclude='number').columns.tolist()
prep = ColumnTransformer([
 ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
 ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
models = {'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'), 'random_forest': RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1)}
results, best_name, best_auc, best_pipe = {}, None, -1, None
for name, model in models.items():
 pipe = Pipeline([('preprocess', prep), ('model', model)])
 pipe.fit(X_train, y_train)
 pred, prob = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
 auc = roc_auc_score(y_test, prob)
 results[name] = {'roc_auc': float(auc), 'classification_report': classification_report(y_test, pred, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(y_test, pred).tolist()}
 if auc > best_auc: best_name, best_auc, best_pipe = name, auc, pipe
out = ROOT / 'artifacts'; out.mkdir(exist_ok=True)
joblib.dump(best_pipe, out / 'churn_model.joblib')
(out / 'metrics.json').write_text(json.dumps({'best_model': best_name, 'models': results}, indent=2))
print(f'Best model: {best_name}; ROC-AUC: {best_auc:.3f}')
