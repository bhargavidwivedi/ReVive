import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ml_pipeline.feature_engineering.engineer import run_pipeline


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH = "ml_pipeline/models/saved/LightGBM_tuned.pkl"
THRESHOLD  = 0.369   # optimal threshold from evaluation


# ── 1. LOAD MODEL ─────────────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH):
    model = joblib.load(path)
    print(f"✅ Loaded model: {path}")
    return model


# ── 2. PREDICT ON PROCESSED DATA ─────────────────────────────────────────────

def predict_from_processed(path: str = "data/processed_features.csv",
                            threshold: float = THRESHOLD):
    """
    Run predictions on already-processed feature file.
    Returns DataFrame with risk scores and labels.
    """
    model = load_model()

    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.drop(columns=["readmitted_30d"], errors="ignore")
    feature_names = list(X.columns)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results = pd.DataFrame({
        "readmission_probability" : y_prob.round(4),
        "predicted_readmission"   : y_pred,
        "risk_level"              : pd.cut(
            y_prob,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )
    })

    if "readmitted_30d" in df.columns:
        results["actual"] = df["readmitted_30d"].values

    print(f"\n── Prediction Summary ────────────────────────────────────")
    print(f"  Total patients     : {len(results):,}")
    print(f"  High risk          : {(results['risk_level'] == 'High').sum():,} "
          f"({(results['risk_level'] == 'High').mean():.1%})")
    print(f"  Medium risk        : {(results['risk_level'] == 'Medium').sum():,} "
          f"({(results['risk_level'] == 'Medium').mean():.1%})")
    print(f"  Low risk           : {(results['risk_level'] == 'Low').sum():,} "
          f"({(results['risk_level'] == 'Low').mean():.1%})")

    return results


# ── 3. PREDICT ON RAW DATA ────────────────────────────────────────────────────

def predict_from_raw(path: str = "data/raw/readmission clinical datset.csv",
                     threshold: float = THRESHOLD):
    """
    Run full pipeline on raw data then predict.
    Use this for new incoming patient data.
    """
    print("── Running feature engineering pipeline ──────────────────")
    df = run_pipeline(path)

    return predict_from_processed(threshold=threshold)


# ── 4. PREDICT SINGLE PATIENT ─────────────────────────────────────────────────

def predict_single_patient(patient_dict: dict, threshold: float = THRESHOLD):
    """
    Predict readmission risk for a single patient.

    Example input:
    {
        "time_in_hospital"   : 5,
        "number_inpatient"   : 2,
        "number_diagnoses"   : 7,
        "num_medications"    : 12,
        "age_numeric"        : 72,
        ...
    }
    """
    model = load_model()

    # Load feature names from processed data
    df_ref = pd.read_csv("data/processed_features.csv", nrows=1)
    feature_names = [c for c in df_ref.columns if c != "readmitted_30d"]

    # Build patient row — fill missing features with 0
    patient_row = pd.DataFrame([{f: patient_dict.get(f, 0) for f in feature_names}])
    patient_row = patient_row.apply(pd.to_numeric, errors="coerce").fillna(0)

    prob  = model.predict_proba(patient_row)[0, 1]
    pred  = int(prob >= threshold)
    level = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"

    print(f"\n── Single Patient Risk Assessment ───────────────────────")
    print(f"  Readmission probability : {prob:.2%}")
    print(f"  Risk level              : {level}")
    print(f"  Predicted readmission   : {'YES ⚠️' if pred else 'NO ✅'}")

    return {"probability": prob, "risk_level": level, "predicted": pred}


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ReVive — Readmission Risk Predictor")
    print("=" * 60)

    # Batch predictions on processed data
    results = predict_from_processed()
    print("\nSample predictions (first 10):")
    print(results.head(10).to_string(index=False))

    # Save predictions
    results.to_csv("outputs/predictions.csv", index=False)
    print(f"\n✅ Predictions saved → outputs/predictions.csv")

    # Example single patient
    print("\n── Example: High-risk patient ────────────────────────────")
    predict_single_patient({
        "time_in_hospital"   : 8,
        "number_inpatient"   : 3,
        "number_diagnoses"   : 9,
        "num_medications"    : 15,
        "age_numeric"        : 75,
        "total_prior_visits" : 5,
        "complexity_score"   : 12.5,
        "is_elderly"         : 1,
        "polypharmacy"       : 1,
        "cardiac_primary"    : 1,
    })

    print("\n── Example: Low-risk patient ─────────────────────────────")
    predict_single_patient({
        "time_in_hospital"   : 2,
        "number_inpatient"   : 0,
        "number_diagnoses"   : 2,
        "num_medications"    : 3,
        "age_numeric"        : 35,
        "total_prior_visits" : 0,
        "complexity_score"   : 2.5,
        "is_elderly"         : 0,
        "polypharmacy"       : 0,
        "cardiac_primary"    : 0,
    })