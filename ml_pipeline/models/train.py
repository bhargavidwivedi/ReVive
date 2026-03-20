import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ── 1. LOAD PROCESSED DATA ────────────────────────────────────────────────────

def load_features(path: str = "data/processed_features.csv"):
    df = pd.read_csv(path)
    print(f"Loaded processed data: {df.shape[0]:,} rows × {df.shape[1]} columns")

    X = df.drop(columns=["readmitted_30d"])
    y = df["readmitted_30d"]

    print(f"Positive rate: {y.mean():.2%}")
    return X, y


# ── 2. SPLIT DATA ─────────────────────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,             # preserve class ratio in both splits
        random_state=random_state
    )
    print(f"\nTrain: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"Train positive rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ── 3. SCALE FEATURES ─────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# ── 4. DEFINE MODELS ──────────────────────────────────────────────────────────

def get_models(y_train):
    """
    All models handle class imbalance via class weights or scale_pos_weight.
    imbalance ratio ~ 88:12 in this dataset.
    """
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,     # handles imbalance
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }
    return models


# ── 5. CROSS-VALIDATE ─────────────────────────────────────────────────────────

def cross_validate_models(models, X_train, X_train_sc, y_train, cv_folds=5):
    """
    LR uses scaled data; tree models use raw data.
    Scoring: ROC-AUC (best for imbalanced datasets).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}

    print(f"\n── Cross-Validation ({cv_folds}-fold Stratified) ──────────────")
    for name, model in models.items():
        X_in = X_train_sc if name == "LogisticRegression" else X_train
        scores = cross_val_score(model, X_in, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:<22} ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# ── 6. TRAIN FINAL MODELS ─────────────────────────────────────────────────────

def train_models(models, X_train, X_train_sc, y_train):
    trained = {}
    print("\n── Training final models on full train set ───────────────")
    for name, model in models.items():
        X_in = X_train_sc if name == "LogisticRegression" else X_train
        model.fit(X_in, y_train)
        trained[name] = model
        print(f"  ✅ {name} trained")
    return trained


# ── 7. QUICK TEST EVALUATION ──────────────────────────────────────────────────

def quick_evaluate(trained_models, X_test, X_test_sc, y_test):
    print("\n── Test Set ROC-AUC ───────────────────────────────────────")
    results = {}
    for name, model in trained_models.items():
        X_in = X_test_sc if name == "LogisticRegression" else X_test
        y_prob = model.predict_proba(X_in)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[name] = auc
        print(f"  {name:<22} ROC-AUC: {auc:.4f}")
    return results


# ── 8. SAVE MODELS ────────────────────────────────────────────────────────────

def save_models(trained_models, scaler, save_dir="ml_pipeline/models/saved"):
    os.makedirs(save_dir, exist_ok=True)
    for name, model in trained_models.items():
        path = f"{save_dir}/{name}.pkl"
        joblib.dump(model, path)
        print(f"  Saved: {path}")
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    print(f"  Saved: {save_dir}/scaler.pkl")


# ── 9. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load
    X, y = load_features()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale (for LR)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Define models
    models = get_models(y_train)

    # Cross-validate
    cv_results = cross_validate_models(models, X_train, X_train_sc, y_train)

    # Train on full training set
    trained_models = train_models(models, X_train, X_train_sc, y_train)

    # Quick test evaluation
    test_results = quick_evaluate(trained_models, X_test, X_test_sc, y_test)

    # Save everything
    print("\n── Saving models ──────────────────────────────────────────")
    save_models(trained_models, scaler)

    # Summary
    print("\n── Summary ────────────────────────────────────────────────")
    best = max(test_results, key=test_results.get)
    print(f"  🏆 Best model: {best} (ROC-AUC: {test_results[best]:.4f})")
    print("\n  Full leaderboard:")
    for name, auc in sorted(test_results.items(), key=lambda x: -x[1]):
        print(f"    {name:<22} {auc:.4f}")