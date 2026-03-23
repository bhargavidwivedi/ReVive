import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

logger   = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH      = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
NEW_MODEL_PATH  = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_retrained.pkl")
DATA_PATH       = os.path.join(BASE_DIR, "data", "processed_features.csv")
OUTCOMES_PATH   = os.path.join(BASE_DIR, "data", "outcomes.json")
METRICS_PATH    = os.path.join(BASE_DIR, "data", "model_metrics.json")


# ── 1. RECORD PATIENT OUTCOME ─────────────────────────────────────────────────

def record_outcome(patient_id: str, predicted_risk: float,
                   actually_readmitted: bool, notes: str = "") -> dict:
    """
    Records whether a patient was actually readmitted.
    This is the feedback loop that enables continuous learning.
    """
    outcomes = load_outcomes()

    outcome = {
        "patient_id"        : patient_id,
        "predicted_risk"    : predicted_risk,
        "actually_readmitted": int(actually_readmitted),
        "recorded_at"       : datetime.now().isoformat(),
        "notes"             : notes,
    }

    outcomes.append(outcome)
    save_outcomes(outcomes)

    logger.info(f"Recorded outcome for patient {patient_id}: readmitted={actually_readmitted}")
    return outcome


# ── 2. LOAD / SAVE OUTCOMES ───────────────────────────────────────────────────

def load_outcomes() -> list:
    if os.path.exists(OUTCOMES_PATH):
        with open(OUTCOMES_PATH, "r") as f:
            return json.load(f)
    return []


def save_outcomes(outcomes: list):
    os.makedirs(os.path.dirname(OUTCOMES_PATH), exist_ok=True)
    with open(OUTCOMES_PATH, "w") as f:
        json.dump(outcomes, f, indent=2)


# ── 3. DETECT MODEL DRIFT ─────────────────────────────────────────────────────

def detect_drift(min_outcomes: int = 50) -> dict:
    """
    Compares model predictions vs actual outcomes.
    Flags drift if AUC drops below threshold.
    """
    outcomes = load_outcomes()

    if len(outcomes) < min_outcomes:
        return {
            "status"       : "insufficient_data",
            "outcomes_count": len(outcomes),
            "min_required" : min_outcomes,
            "message"      : f"Need {min_outcomes - len(outcomes)} more outcomes to detect drift",
        }

    y_true = [o["actually_readmitted"] for o in outcomes]
    y_pred = [o["predicted_risk"]      for o in outcomes]

    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0.5

    drift_detected = auc < 0.60
    baseline_auc   = 0.6812

    return {
        "status"          : "drift_detected" if drift_detected else "no_drift",
        "current_auc"     : round(auc, 4),
        "baseline_auc"    : baseline_auc,
        "drift_detected"  : drift_detected,
        "outcomes_count"  : len(outcomes),
        "recommendation"  : "Retrain model immediately" if drift_detected else "Model performing well",
    }


# ── 4. RETRAIN MODEL ──────────────────────────────────────────────────────────

def retrain_model(min_new_samples: int = 100) -> dict:
    """
    Retrains the LightGBM model with the latest data.
    Saves new model only if it performs better than current.
    """
    logger.info("Starting model retraining...")

    # Load full dataset
    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.drop(columns=["readmitted_30d"])
    y = df["readmitted_30d"]

    # Split: last 20% as validation
    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # Load current model and evaluate
    current_model = joblib.load(MODEL_PATH)
    current_auc   = roc_auc_score(y_val, current_model.predict_proba(X_val)[:, 1])

    # Train new model
    new_model = LGBMClassifier(
        n_estimators  = 300,
        learning_rate = 0.05,
        max_depth     = 6,
        class_weight  = "balanced",
        random_state  = 42,
        n_jobs        = -1,
        verbose       = -1,
    )
    new_model.fit(X_train, y_train)
    new_auc = roc_auc_score(y_val, new_model.predict_proba(X_val)[:, 1])

    improved = new_auc > current_auc

    if improved:
        joblib.dump(new_model, NEW_MODEL_PATH)
        logger.info(f"New model saved. AUC improved: {current_auc:.4f} → {new_auc:.4f}")
    else:
        logger.info(f"New model not better. Current: {current_auc:.4f}, New: {new_auc:.4f}")

    result = {
        "status"         : "improved" if improved else "no_improvement",
        "current_auc"    : round(current_auc, 4),
        "new_auc"        : round(new_auc, 4),
        "improvement"    : round(new_auc - current_auc, 4),
        "model_saved"    : improved,
        "retrained_at"   : datetime.now().isoformat(),
        "training_samples": len(X_train),
    }

    save_metrics(result)
    return result


# ── 5. SAVE / LOAD METRICS ────────────────────────────────────────────────────

def save_metrics(metrics: dict):
    history = []
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            history = json.load(f)
    history.append(metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)


def load_metrics_history() -> list:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return []


# ── 6. FULL LEARNING PIPELINE ─────────────────────────────────────────────────

def run_learning_pipeline() -> dict:
    """
    Full continuous learning pipeline:
    1. Check for drift
    2. Retrain if needed
    3. Save metrics
    """
    logger.info("Running continuous learning pipeline...")

    drift_report = detect_drift()

    retrain_report = None
    if drift_report.get("drift_detected"):
        logger.info("Drift detected — triggering retraining...")
        retrain_report = retrain_model()
    else:
        logger.info("No drift detected — skipping retraining")

    return {
        "drift_report"  : drift_report,
        "retrain_report": retrain_report,
        "pipeline_ran_at": datetime.now().isoformat(),
    }


# ── 7. GET SYSTEM HEALTH ──────────────────────────────────────────────────────

def get_system_health() -> dict:
    """
    Returns overall system health including model performance and drift status.
    """
    outcomes      = load_outcomes()
    metrics_hist  = load_metrics_history()
    drift_report  = detect_drift()

    return {
        "model_version"   : "LightGBM_tuned",
        "baseline_auc"    : 0.6812,
        "outcomes_recorded": len(outcomes),
        "retraining_count": len(metrics_hist),
        "drift_status"    : drift_report.get("status", "unknown"),
        "last_retrain"    : metrics_hist[-1].get("retrained_at") if metrics_hist else "Never",
        "system_status"   : "healthy",
    }