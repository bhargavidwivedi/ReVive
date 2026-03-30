import joblib
import pandas as pd
import numpy as np
import os
import logging

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed_features.csv")
THRESHOLD  = 0.369

try:
    model = joblib.load(MODEL_PATH)
    FEATURE_NAMES = [c for c in pd.read_csv(DATA_PATH, nrows=1).columns if c != "readmitted_30d"]
    logger.info(f"Model loaded successfully. Features: {len(FEATURE_NAMES)}")
except Exception as e:
    logger.error(f"Joblib load failed: {e} — retraining...")
    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(DATA_PATH).apply(pd.to_numeric, errors="coerce").fillna(0)
        X  = df.drop(columns=["readmitted_30d"])
        y  = df["readmitted_30d"]
        FEATURE_NAMES = list(X.columns)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH, protocol=2)
        logger.info(f"Model retrained! Features: {len(FEATURE_NAMES)}")
    except Exception as e2:
        logger.error(f"Retraining failed: {e2}")
        model         = None
        FEATURE_NAMES = []
# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    from .tasks import score_patient_on_discharge, test_celery
    CELERY_AVAILABLE = True
except Exception:
    CELERY_AVAILABLE = False

try:
    from .continuous_learning import record_outcome, detect_drift, retrain_model, get_system_health
    LEARNING_AVAILABLE = True
except Exception:
    LEARNING_AVAILABLE = False

try:
    from .care_pathway import assign_care_pathway
    PATHWAY_AVAILABLE = True
except Exception:
    PATHWAY_AVAILABLE = False

try:
    from .llm_notes import analyze_patient_notes
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

try:
    from .fhir_integration import build_features_from_fhir, search_patients
    FHIR_AVAILABLE = True
except Exception:
    FHIR_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_risk_level(prob):
    if prob >= 0.6: return "High"
    if prob >= 0.3: return "Medium"
    return "Low"


def get_recommendations(risk_level, top_features=[]):
    recs = {
        "High"  : ["Schedule follow-up within 48 hours of discharge", "Assign case manager for post-discharge support", "Review medication plan with pharmacist", "Arrange home health visit if needed"],
        "Medium": ["Schedule follow-up within 7 days of discharge", "Provide patient education on warning signs", "Confirm prescription filled before discharge"],
        "Low"   : ["Standard discharge protocol", "Provide contact number for questions"],
    }
    return recs.get(risk_level, [])


# ── 1. HEALTH ─────────────────────────────────────────────────────────────────

@api_view(["GET"])
def health(request):
    import os
    model_exists = os.path.exists(MODEL_PATH)
    data_exists  = os.path.exists(DATA_PATH)
    return Response({
        "status"      : "ok",
        "model"       : "LightGBM_tuned",
        "auc"         : 0.6812,
        "threshold"   : THRESHOLD,
        "features"    : len(FEATURE_NAMES),
        "model_loaded": model is not None,
        "model_file_exists": model_exists,
        "data_file_exists" : data_exists,
        "model_path"  : MODEL_PATH,
        "data_path"   : DATA_PATH,
    })


# ── 2. PREDICT ────────────────────────────────────────────────────────────────

@api_view(["POST"])
def predict(request):
    try:
        patient_data = request.data.get("patient_data", {})
        if not patient_data:
            return Response({"error": "patient_data is required"}, status=status.HTTP_400_BAD_REQUEST)
        if model is None:
            return Response({"error": "Model not loaded"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        row       = pd.DataFrame([{f: patient_data.get(f, 0) for f in FEATURE_NAMES}])
        row       = row.apply(pd.to_numeric, errors="coerce").fillna(0)
        prob      = float(model.predict_proba(row)[0, 1])
        predicted = int(prob >= THRESHOLD)
        risk      = get_risk_level(prob)
        recs      = get_recommendations(risk)

        return Response({
            "readmission_probability": round(prob, 4),
            "readmission_percentage" : f"{prob:.1%}",
            "predicted_readmission"  : bool(predicted),
            "risk_level"             : risk,
            "recommendations"        : recs,
        })
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── 3. BATCH PREDICT ──────────────────────────────────────────────────────────

@api_view(["POST"])
def predict_batch(request):
    try:
        patients = request.data.get("patients", [])
        if not patients:
            return Response({"error": "patients list is required"}, status=status.HTTP_400_BAD_REQUEST)
        if model is None:
            return Response({"error": "Model not loaded"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        rows  = pd.DataFrame([{f: p.get(f, 0) for f in FEATURE_NAMES} for p in patients])
        rows  = rows.apply(pd.to_numeric, errors="coerce").fillna(0)
        probs = model.predict_proba(rows)[:, 1]

        results     = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        for i, prob in enumerate(probs):
            prob  = float(prob)
            risk  = get_risk_level(prob)
            risk_counts[risk] += 1
            results.append({
                "patient_index"          : i,
                "readmission_probability": round(prob, 4),
                "readmission_percentage" : f"{prob:.1%}",
                "predicted_readmission"  : bool(prob >= THRESHOLD),
                "risk_level"             : risk,
            })

        return Response({"total_patients": len(results), "risk_summary": risk_counts, "predictions": results})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── 4. DISCHARGE SCORING ──────────────────────────────────────────────────────

@api_view(["POST"])
def trigger_discharge_scoring(request):
    if not CELERY_AVAILABLE:
        return Response({"error": "Celery not available"}, status=503)
    patient_data = request.data.get("patient_data", {})
    patient_id   = request.data.get("patient_id", "UNKNOWN")
    notify_email = request.data.get("notify_email", None)
    task = score_patient_on_discharge.delay(patient_data, patient_id, notify_email)
    return Response({"status": "queued", "task_id": task.id, "patient_id": patient_id, "message": "Patient scoring started in background"})


# ── 5. TEST CELERY ────────────────────────────────────────────────────────────

@api_view(["GET"])
def test_celery_view(request):
    if not CELERY_AVAILABLE:
        return Response({"error": "Celery not available"}, status=503)
    task = test_celery.delay()
    return Response({"status": "queued", "task_id": task.id})


# ── 6. FHIR ───────────────────────────────────────────────────────────────────

@api_view(["GET"])
def fhir_predict(request, patient_id):
    if not FHIR_AVAILABLE:
        return Response({"error": "FHIR not available"}, status=503)
    try:
        data     = build_features_from_fhir(patient_id)
        features = data["features"]
        patient  = data["patient_info"]
        row      = pd.DataFrame([{f: features.get(f, 0) for f in FEATURE_NAMES}])
        row      = row.apply(pd.to_numeric, errors="coerce").fillna(0)
        prob     = float(model.predict_proba(row)[0, 1])
        risk     = get_risk_level(prob)
        return Response({"patient": patient, "readmission_probability": round(prob, 4), "readmission_percentage": f"{prob:.1%}", "risk_level": risk, "predicted_readmission": bool(prob >= THRESHOLD), "fhir_features": features})
    except Exception as e:
        return Response({"error": str(e)}, status=400)


@api_view(["GET"])
def fhir_search(request):
    if not FHIR_AVAILABLE:
        return Response({"error": "FHIR not available"}, status=503)
    name     = request.GET.get("name", "")
    count    = int(request.GET.get("count", 5))
    patients = search_patients(name=name, count=count)
    return Response({"patients": patients, "count": len(patients)})


# ── 7. ANALYZE NOTES ─────────────────────────────────────────────────────────

@api_view(["POST"])
def analyze_notes(request):
    if not LLM_AVAILABLE:
        return Response({"error": "LLM not available"}, status=503)
    notes    = request.data.get("notes", "")
    features = request.data.get("existing_features", {})
    if not notes:
        return Response({"error": "notes field is required"}, status=400)
    result = analyze_patient_notes(notes, features)
    return Response(result)


# ── 8. CARE PATHWAY ───────────────────────────────────────────────────────────

@api_view(["POST"])
def care_pathway(request):
    if not PATHWAY_AVAILABLE:
        return Response({"error": "Care pathway not available"}, status=503)
    patient_data = request.data.get("patient_data", {})
    risk_level   = request.data.get("risk_level", "Medium")
    llm_analysis = request.data.get("llm_analysis", {})
    if not patient_data:
        return Response({"error": "patient_data is required"}, status=400)
    care_plan = assign_care_pathway(risk_level, patient_data, llm_analysis)
    return Response(care_plan)


# ── 9. OUTCOMES ───────────────────────────────────────────────────────────────

@api_view(["POST"])
def record_patient_outcome(request):
    if not LEARNING_AVAILABLE:
        return Response({"error": "Learning not available"}, status=503)
    patient_id          = request.data.get("patient_id")
    predicted_risk      = request.data.get("predicted_risk", 0.5)
    actually_readmitted = request.data.get("actually_readmitted", False)
    notes               = request.data.get("notes", "")
    result = record_outcome(patient_id, predicted_risk, actually_readmitted, notes)
    return Response(result)


@api_view(["GET"])
def drift_detection(request):
    if not LEARNING_AVAILABLE:
        return Response({"error": "Learning not available"}, status=503)
    return Response(detect_drift())


@api_view(["POST"])
def trigger_retraining(request):
    if not LEARNING_AVAILABLE:
        return Response({"error": "Learning not available"}, status=503)
    return Response(retrain_model())


@api_view(["GET"])
def system_health(request):
    if not LEARNING_AVAILABLE:
        return Response({"status": "ok", "model_loaded": model is not None, "features": len(FEATURE_NAMES)})
    return Response(get_system_health())# force redeploy 
