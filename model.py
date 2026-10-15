from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURES = ["temperature", "vibration", "current", "pressure", "load", "age_days"]

def train_model(df, model_path="models/predictive_maintenance_model.joblib"):
    work = df.dropna(subset=["failure"]).copy()
    X = work[FEATURES]
    y = work["failure"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.22, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "features": FEATURES
    }
    p = Path(model_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics, "features": FEATURES}, p)
    return pipeline, metrics

def load_model(model_path="models/predictive_maintenance_model.joblib"):
    p = Path(model_path)
    if not p.exists():
        return None
    return joblib.load(p)

def predict_rows(model_bundle, rows):
    X = pd.DataFrame(rows)[FEATURES]
    pipe = model_bundle["pipeline"] if isinstance(model_bundle, dict) else model_bundle
    probabilities = pipe.predict_proba(X)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    return predictions, probabilities
