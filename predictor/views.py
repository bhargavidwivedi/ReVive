from .continuous_learning import record_outcome, detect_drift, retrain_model, get_system_health
from .care_pathway import assign_care_pathway
from .llm_notes import analyze_patient_notes
from .fhir_integration import build_features_from_fhir, search_patients
from .tasks import score_patient_on_discharge, test_celery
from django.shortcuts import render

# Create your views here.
import joblib
import pandas as pd
import numpy as np
import os

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# ── Load model once at startup ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed_features.csv")

_model = None
_feature_names = None

def get_model():
    global _model, _feature_names
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            _feature_names = [c for c in pd.read_csv(DATA_PATH, nrows=1).columns if c != "readmitted_30d"]
        except Exception as e:
            logger.error(f"Model load failed: {e}")
    return _model, _feature_names
THRESHOLD     = 0.369


def get_risk_level(prob):
    if prob >= 0.6:   return "High"
    if prob >= 0.3:   return "Medium"
    return "Low"


def get_recommendations(risk_level, top_features):
    recs = {
        "High": [
            "Schedule follow-up within 48 hours of discharge",
            "Assign case manager for post-discharge support",
            "Review medication plan with pharmacist",
            "Arrange home health visit if needed",
        ],
        "Medium": [
            "Schedule follow-up within 7 days of discharge",
            "Provide patient education on warning signs",
            "Confirm prescription filled before discharge",
        ],
        "Low": [
            "Standard discharge protocol",
            "Provide contact number for questions",
        ],
    }
    return recs.get(risk_level, [])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@api_view(["POST"])
def predict(request):
    """
    POST /api/predict/
    Body: { "patient_data": { "time_in_hospital": 5, "number_inpatient": 2, ... } }
    Returns: risk score, level, recommendations
    """
    try:
        patient_data = request.data.get("patient_data", {})
        if not patient_data:
            return Response(
                {"error": "patient_data is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Build feature row — fill missing with 0
        row = pd.DataFrame([{f: patient_data.get(f, 0) for f in FEATURE_NAMES}])
        row = row.apply(pd.to_numeric, errors="coerce").fillna(0)

        prob       = float(model.predict_proba(row)[0, 1])
        predicted  = int(prob >= THRESHOLD)
        risk_level = get_risk_level(prob)
        recs       = get_recommendations(risk_level, [])

        return Response({
            "readmission_probability" : round(prob, 4),
            "readmission_percentage"  : f"{prob:.1%}",
            "predicted_readmission"   : bool(predicted),
            "risk_level"              : risk_level,
            "recommendations"         : recs,
        })

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
def predict_batch(request):
    """
    POST /api/predict/batch/
    Body: { "patients": [ {...}, {...} ] }
    Returns: list of predictions with risk summary
    """
    try:
        patients = request.data.get("patients", [])
        if not patients:
            return Response(
                {"error": "patients list is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        rows = pd.DataFrame([
            {f: p.get(f, 0) for f in FEATURE_NAMES}
            for p in patients
        ])
        rows  = rows.apply(pd.to_numeric, errors="coerce").fillna(0)
        probs = model.predict_proba(rows)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            prob       = float(prob)
            risk_level = get_risk_level(prob)
            results.append({
                "patient_index"           : i,
                "readmission_probability" : round(prob, 4),
                "readmission_percentage"  : f"{prob:.1%}",
                "predicted_readmission"   : bool(prob >= THRESHOLD),
                "risk_level"              : risk_level,
            })

        # Summary stats
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        for r in results:
            risk_counts[r["risk_level"]] += 1

        return Response({
            "total_patients" : len(results),
            "risk_summary"   : risk_counts,
            "predictions"    : results,
        })

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["GET"])
def health(request):
    """GET /api/health/ — check API is live"""
    return Response({
        "status"    : "ok",
        "model"     : "LightGBM_tuned",
        "auc"       : 0.6812,
        "threshold" : THRESHOLD,
        "features"  : len(FEATURE_NAMES),
    })
@api_view(["POST"])
def trigger_discharge_scoring(request):
    """POST /api/discharge/ — trigger async scoring on patient discharge"""
    patient_data = request.data.get("patient_data", {})
    patient_id   = request.data.get("patient_id", "UNKNOWN")
    notify_email = request.data.get("notify_email", None)

    task = score_patient_on_discharge.delay(
        patient_data, patient_id, notify_email
    )
    return Response({
        "status"     : "queued",
        "task_id"    : task.id,
        "patient_id" : patient_id,
        "message"    : "Patient scoring started in background"
    })

@api_view(["GET"])
def test_celery_view(request):
    """GET /api/test-celery/ — verify Celery is working"""
    task = test_celery.delay()
    return Response({"status": "queued", "task_id": task.id})
@api_view(["GET"])
def fhir_predict(request, patient_id):
    """GET /api/fhir/<patient_id>/ — fetch from FHIR and predict"""
    try:
        data     = build_features_from_fhir(patient_id)
        features = data["features"]
        patient  = data["patient_info"]

        row    = pd.DataFrame([{f: features.get(f, 0) for f in FEATURE_NAMES}])
        row    = row.apply(pd.to_numeric, errors="coerce").fillna(0)
        prob   = float(model.predict_proba(row)[0, 1])
        risk   = get_risk_level(prob)

        return Response({
            "patient"                : patient,
            "readmission_probability": round(prob, 4),
            "readmission_percentage" : f"{prob:.1%}",
            "risk_level"             : risk,
            "predicted_readmission"  : bool(prob >= THRESHOLD),
            "fhir_features"          : features,
        })
    except Exception as e:
        return Response({"error": str(e)}, status=400)


@api_view(["GET"])
def fhir_search(request):
    """GET /api/fhir/search/?name=John — search FHIR patients"""
    name     = request.GET.get("name", "")
    count    = int(request.GET.get("count", 5))
    patients = search_patients(name=name, count=count)
    return Response({"patients": patients, "count": len(patients)})
@api_view(["POST"])
def analyze_notes(request):
    """POST /api/analyze-notes/ — LLM analysis of discharge notes"""
    notes    = request.data.get("notes", "")
    features = request.data.get("existing_features", {})

    if not notes:
        return Response({"error": "notes field is required"}, status=400)

    result = analyze_patient_notes(notes, features)
    return Response(result)
@api_view(["POST"])
def care_pathway(request):
    """POST /api/care-pathway/ — assign autonomous care pathway"""
    patient_data = request.data.get("patient_data", {})
    risk_level   = request.data.get("risk_level", "Medium")
    llm_analysis = request.data.get("llm_analysis", {})

    if not patient_data:
        return Response({"error": "patient_data is required"}, status=400)

    care_plan = assign_care_pathway(risk_level, patient_data, llm_analysis)
    return Response(care_plan)
@api_view(["POST"])
def record_patient_outcome(request):
    """POST /api/outcomes/ — record actual readmission outcome"""
    patient_id         = request.data.get("patient_id")
    predicted_risk     = request.data.get("predicted_risk", 0.5)
    actually_readmitted= request.data.get("actually_readmitted", False)
    notes              = request.data.get("notes", "")
    result = record_outcome(patient_id, predicted_risk, actually_readmitted, notes)
    return Response(result)

@api_view(["GET"])
def drift_detection(request):
    """GET /api/drift/ — check for model drift"""
    return Response(detect_drift())

@api_view(["POST"])
def trigger_retraining(request):
    """POST /api/retrain/ — trigger model retraining"""
    return Response(retrain_model())

@api_view(["GET"])
def system_health(request):
    """GET /api/system-health/ — overall system health"""
    return Response(get_system_health())