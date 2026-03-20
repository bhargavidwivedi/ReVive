import pandas as pd
import numpy as np
import joblib
import optuna
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv("data/processed_features.csv")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.drop(columns=["readmitted_30d"])
    y = df["readmitted_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"Positive rate: {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ── 2. OPTUNA OBJECTIVE ───────────────────────────────────────────────────────

def objective(trial, X_train, y_train):
    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 100, 500),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth"         : trial.suggest_int("max_depth", 3, 9),
        "num_leaves"        : trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples" : trial.suggest_int("min_child_samples", 10, 100),
        "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight"      : "balanced",
        "random_state"      : 42,
        "n_jobs"            : -1,
        "verbose"           : -1,
    }

    model = LGBMClassifier(**params)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ── 3. RUN TUNING ─────────────────────────────────────────────────────────────

def run_tuning(X_train, y_train, n_trials=50):
    print(f"\n── Optuna Hyperparameter Tuning ({n_trials} trials) ────────")
    print("  This will take a few minutes...\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n  Best CV ROC-AUC : {study.best_value:.4f}")
    print(f"  Best params     :")
    for k, v in study.best_params.items():
        print(f"    {k:<25} {v}")

    return study


# ── 4. TRAIN BEST MODEL ───────────────────────────────────────────────────────

def train_best_model(study, X_train, y_train, X_test, y_test):
    print("\n── Training final model with best params ────────────────")

    best_params = study.best_params
    best_params.update({
        "class_weight" : "balanced",
        "random_state" : 42,
        "n_jobs"       : -1,
        "verbose"      : -1,
    })

    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    # Evaluate
    y_prob   = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n  ✅ Tuned LightGBM Test ROC-AUC : {test_auc:.4f}")

    # Compare with baseline
    baseline = joblib.load("ml_pipeline/models/saved/LightGBM.pkl")
    base_prob = baseline.predict_proba(X_test)[:, 1]
    base_auc  = roc_auc_score(y_test, base_prob)
    improvement = (test_auc - base_auc) * 100

    print(f"  Baseline LightGBM ROC-AUC     : {base_auc:.4f}")
    print(f"  Improvement                   : +{improvement:.2f}%")

    return model, test_auc


# ── 5. SAVE TUNED MODEL ───────────────────────────────────────────────────────

def save_tuned_model(model, auc):
    path = "ml_pipeline/models/saved/LightGBM_tuned.pkl"
    joblib.dump(model, path)
    print(f"\n  Saved tuned model → {path}")
    print(f"  Final ROC-AUC     : {auc:.4f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load
    X_train, X_test, y_train, y_test = load_data()

    # Tune
    study = run_tuning(X_train, y_train, n_trials=50)

    # Train best
    best_model, best_auc = train_best_model(study, X_train, y_train, X_test, y_test)

    # Save
    save_tuned_model(best_model, best_auc)

    print("\n🏆 Tuning complete!")