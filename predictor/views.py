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

model         = joblib.load(MODEL_PATH)
FEATURE_NAMES = [c for c in pd.read_csv(DATA_PATH, nrows=1).columns if c != "readmitted_30d"]
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