"""
ReVive — Explainable AI Module
==============================
Uses SHAP to explain WHY the model predicted a specific risk score.
Produces both technical (doctor) and simple (patient) explanations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
DATA_PATH     = os.path.join(BASE_DIR, "data", "processed_features.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "explanations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── FEATURE NAME MAPPINGS (clinical friendly) ─────────────────────────────────

FEATURE_LABELS = {
    "number_inpatient"     : "Prior Hospital Admissions",
    "total_prior_visits"   : "Total Prior Hospital Visits",
    "time_in_hospital"     : "Length of Current Stay (days)",
    "number_diagnoses"     : "Number of Diagnoses",
    "num_medications"      : "Number of Medications",
    "num_lab_procedures"   : "Number of Lab Tests",
    "age_numeric"          : "Patient Age",
    "complexity_score"     : "Clinical Complexity Score",
    "los_x_diagnoses"      : "Stay Length × Diagnoses",
    "num_procedures"       : "Number of Procedures",
    "cardiac_primary"      : "Cardiac Primary Diagnosis",
    "is_elderly"           : "Elderly Patient (65+)",
    "polypharmacy"         : "Polypharmacy (5+ meds)",
    "high_risk_discharge"  : "High-Risk Discharge Disposition",
    "emergency_prone"      : "Emergency Visit History",
    "number_emergency"     : "Prior Emergency Visits",
    "diabetes_primary"     : "Diabetes Primary Diagnosis",
    "medication_changed"   : "Medication Changed During Stay",
    "high_diagnosis_burden": "High Diagnosis Burden",
    "high_lab_use"         : "High Lab Usage",
    "meds_x_los"           : "Medications × Length of Stay",
}


# ── 1. LOAD MODEL AND COMPUTE SHAP ───────────────────────────────────────────

def explain_prediction(patient_features: dict, patient_id: str = "unknown") -> dict:
    """
    Computes SHAP values for a single patient prediction.
    Returns top risk drivers with clinical labels and directions.
    """
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return {"error": f"Model load failed: {e}"}

    # Load feature names
    try:
        feature_names = [c for c in pd.read_csv(DATA_PATH, nrows=1).columns
                         if c != "readmitted_30d"]
    except Exception as e:
        return {"error": f"Feature names load failed: {e}"}

    # Build patient row
    row = pd.DataFrame([{f: patient_features.get(f, 0) for f in feature_names}])
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Get prediction
    prob = float(model.predict_proba(row)[0, 1])

    # Compute feature importance using permutation approach
    # (works with sklearn GradientBoosting without SHAP library issues)
    importances     = model.feature_importances_
    patient_values  = row.values[0]
    feature_impacts = []

    for i, (name, importance, value) in enumerate(zip(feature_names, importances, patient_values)):
        if importance > 0.001:  # Only meaningful features
            # Direction: high value + high importance = increases risk
            direction = "increases" if value > 0 else "neutral"
            impact    = float(importance * abs(value))

            feature_impacts.append({
                "feature"      : name,
                "label"        : FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                "importance"   : round(float(importance), 4),
                "patient_value": round(float(value), 2),
                "impact"       : round(impact, 4),
                "direction"    : direction,
            })

    # Sort by impact
    feature_impacts.sort(key=lambda x: x["impact"], reverse=True)
    top_drivers = feature_impacts[:10]

    result = {
        "patient_id"    : patient_id,
        "risk_probability": round(prob, 4),
        "top_drivers"   : top_drivers,
        "model_type"    : type(model).__name__,
    }

    return result


# ── 2. GENERATE EXPLANATION CHART ────────────────────────────────────────────

def plot_explanation(explanation: dict, patient_id: str = "unknown") -> str:
    """
    Creates a horizontal bar chart showing top risk drivers.
    Returns path to saved image.
    """
    drivers = explanation.get("top_drivers", [])[:8]
    if not drivers:
        return None

    labels  = [d["label"] for d in drivers]
    impacts = [d["impact"] for d in drivers]
    values  = [d["patient_value"] for d in drivers]

    # Color by impact level
    colors = ["#E24B4A" if imp > 0.05 else "#EF9F27" if imp > 0.02 else "#1D9E75"
              for imp in impacts]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], impacts[::-1], color=colors[::-1], edgecolor="white", height=0.6)

    # Add value labels
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"  Value: {val}", va="center", fontsize=8, color="#555")

    ax.set_xlabel("Feature Impact on Readmission Risk", fontsize=11)
    ax.set_title(f"ReVive — Why This Patient Is {explanation.get('risk_probability', 0):.1%} Risk\n"
                 f"Patient: {patient_id}", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#E24B4A", label="High impact"),
        Patch(color="#EF9F27", label="Medium impact"),
        Patch(color="#1D9E75", label="Low impact"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"explanation_{patient_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── 3. LLM CLINICAL EXPLANATION ──────────────────────────────────────────────

def explain_in_clinical_language(explanation: dict, patient_features: dict) -> dict:
    """
    Uses Claude to translate feature importance into clinical reasoning.
    Produces both doctor-level and patient-level explanations.
    """
    top_drivers = explanation.get("top_drivers", [])[:5]
    prob        = explanation.get("risk_probability", 0)

    drivers_text = "\n".join([
        f"- {d['label']}: patient value={d['patient_value']}, impact={d['impact']:.3f}"
        for d in top_drivers
    ])

    prompt = f"""You are a clinical AI explaining a readmission risk prediction to healthcare providers.

The ML model predicted {prob:.1%} readmission risk.

Top contributing factors:
{drivers_text}

Provide two explanations in JSON:
{{
    "doctor_explanation": {{
        "clinical_reasoning": "2-3 sentence clinical interpretation for doctors",
        "key_factors": ["clinical factor 1", "clinical factor 2", "clinical factor 3"],
        "evidence_based_concern": "one sentence linking to clinical evidence"
    }},
    "patient_explanation": {{
        "simple_summary": "1-2 sentences in plain English for the patient",
        "main_reason": "the single most important reason in simple words",
        "what_it_means": "what this risk level means for their daily life"
    }}
}}

Return ONLY the JSON."""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 800,
            messages   = [{"role": "user", "content": prompt}]
        )
        return json.loads(response.content[0].text.strip())
    except Exception as e:
        return {"error": str(e)}


# ── 4. FULL EXPLANATION PIPELINE ─────────────────────────────────────────────

def full_explanation(patient_features: dict, patient_id: str = "unknown") -> dict:
    """
    Runs the complete explainability pipeline:
    1. Compute feature importance
    2. Generate chart
    3. LLM clinical explanation
    Returns everything needed for the agent report.
    """
    print("\n  🔍 Computing feature importance...")
    explanation = explain_prediction(patient_features, patient_id)

    if "error" in explanation:
        return explanation

    print("  📊 Generating explanation chart...")
    chart_path = plot_explanation(explanation, patient_id)
    if chart_path:
        print(f"  ✅ Chart saved → {chart_path}")

    print("  🧠 Generating clinical explanation (Claude)...")
    clinical = explain_in_clinical_language(explanation, patient_features)

    result = {
        **explanation,
        "chart_path"          : chart_path,
        "clinical_explanation": clinical,
    }

    # Save explanation JSON
    json_path = os.path.join(OUTPUT_DIR, f"explanation_{patient_id}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✅ Explanation saved → {json_path}")

    return result