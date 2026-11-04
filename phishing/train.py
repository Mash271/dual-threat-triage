from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from features import extract_features   # Week-1 extractor (script dir is auto on path)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "phishing_clean.csv"
FEAT_CACHE = BASE / "data" / "phishing_features.csv"
MODEL_DIR = BASE / "models"

FEATURE_COLUMNS = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_at_signs", "digit_ratio",
    "has_ip_address", "has_https", "num_subdomains", "uses_url_shortener",
    "suspicious_keyword_count", "special_char_ratio",
]

# Promised targets from the proposal (PR table)
TARGET_ACC, TARGET_RECALL = 0.92, 0.90


def build_feature_matrix(df):
    """Apply the 14 extractors once, then cache so re-runs are instant."""
    if FEAT_CACHE.exists():
        feats = pd.read_csv(FEAT_CACHE)
        if len(feats) == len(df):          # guard: stale cache after data regen
            print("Loaded cached feature matrix.")
            return feats[FEATURE_COLUMNS]
        print("Cache stale — recomputing...")
    print("Extracting features from 30,000 URLs (one-time, ~30s)...")
    feats = pd.DataFrame(df["url"].apply(extract_features).tolist())
    feats.to_csv(FEAT_CACHE, index=False)  # NOTE: delete this file if you regen clean CSV
    return feats[FEATURE_COLUMNS]


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    m = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_phishing": precision_score(y_test, preds),   # 1 = phishing after remap
        "recall_phishing": recall_score(y_test, preds),
        "f1_phishing": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print(f"\n--- {name} ---")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted; 1=phishing):")
    print(confusion_matrix(y_test, preds))
    return m


def main():
    df = pd.read_csv(DATA)
    X = build_feature_matrix(df)
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=42))

    print("\nTraining RandomForest...")
    rf.fit(X_tr, y_tr)
    print("Training LogisticRegression baseline...")
    lr.fit(X_tr, y_tr)

    cv = cross_val_score(rf, X_tr, y_tr, cv=5, scoring="recall", n_jobs=-1)
    print(f"\nRF 5-fold CV recall on train: {cv.mean():.4f} ± {cv.std():.4f}")

    m_rf = evaluate("RandomForest (primary)", rf, X_te, y_te)
    m_lr = evaluate("LogisticRegression (baseline)", lr, X_te, y_te)

    print("\n--- Target check (RF) ---")
    print("Accuracy >= 0.92 :", "PASS" if m_rf["accuracy"] >= TARGET_ACC else "FAIL")
    print("Recall   >= 0.90 :", "PASS" if m_rf["recall_phishing"] >= TARGET_RECALL else "FAIL")

    print("\nTop-5 RF feature importances:")
    imp = sorted(zip(FEATURE_COLUMNS, rf.feature_importances_),
                 key=lambda t: -t[1])[:5]
    for name, val in imp:
        print(f"  {name}: {val:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "phishing_rf_v1.pkl")
    joblib.dump(lr, MODEL_DIR / "phishing_lr_v1.pkl")
    with open(MODEL_DIR / "metrics_v1.json", "w") as f:
        json.dump({"random_forest": m_rf, "logistic_regression": m_lr,
                   "cv_recall_mean": float(cv.mean()), "cv_recall_std": float(cv.std()),
                   "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))}, f, indent=2)
    print("\nSaved models + metrics_v1.json to phishing/models/")


if __name__ == "__main__":
    main()