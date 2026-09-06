import joblib
import pandas as pd
import logging
import os
from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "readmission_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed_features.csv")
THRESHOLD  = 0.369

try:
    model         = joblib.load(MODEL_PATH)
    FEATURE_NAMES = [c for c in pd.read_csv(DATA_PATH, nrows=1).columns if c != "readmitted_30d"]
except Exception as e:
    logger.warning(f"Could not load model in tasks: {e}")
    model         = None
    FEATURE_NAMES = []

def get_risk_level(prob):
    if prob >= 0.6: return "High"
    if prob >= 0.3: return "Medium"
    return "Low"

@shared_task(name="predictor.tasks.score_patient_on_discharge")
def score_patient_on_discharge(patient_data: dict, patient_id: str, notify_email: str = None):
    if model is None:
        return {"error": "Model not loaded"}
    row        = pd.DataFrame([{f: patient_data.get(f, 0) for f in FEATURE_NAMES}])
    row        = row.apply(pd.to_numeric, errors="coerce").fillna(0)
    prob       = float(model.predict_proba(row)[0, 1])
    risk_level = get_risk_level(prob)
    result     = {
        "patient_id"             : patient_id,
        "readmission_probability": round(prob, 4),
        "risk_level"             : risk_level,
        "predicted_readmission"  : prob >= THRESHOLD,
    }
    if risk_level == "High" and notify_email:
        send_risk_alert.delay(patient_id, prob, risk_level, notify_email)
    return result

@shared_task(name="predictor.tasks.send_risk_alert")
def send_risk_alert(patient_id: str, prob: float, risk_level: str, recipient_email: str):
    subject = f"HIGH READMISSION RISK — Patient {patient_id}"
    message = f"""
ReVive Clinical Alert System
{'='*40}
Patient ID  : {patient_id}
Risk Level  : {risk_level.upper()}
Probability : {prob:.1%}

RECOMMENDED ACTIONS:
1. Schedule follow-up within 48 hours of discharge
2. Assign case manager for post-discharge support
3. Review medication plan with pharmacist
4. Arrange home health visit if needed

This is an automated alert from ReVive.
{'='*40}
    """
    try:
        send_mail(
            subject       = subject,
            message       = message,
            from_email    = settings.DEFAULT_FROM_EMAIL,
            recipient_list= [recipient_email],
            fail_silently = False,
        )
        return {"status": "sent", "recipient": recipient_email}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@shared_task(name="predictor.tasks.score_all_patients")
def score_all_patients():
    if model is None:
        return {"error": "Model not loaded"}
    df         = pd.read_csv(DATA_PATH).apply(pd.to_numeric, errors="coerce").fillna(0)
    X          = df.drop(columns=["readmitted_30d"], errors="ignore")
    probs      = model.predict_proba(X)[:, 1]
    summary    = {
        "total_scored": len(probs),
        "high_risk"   : int((probs >= 0.6).sum()),
        "medium_risk" : int(((probs >= 0.3) & (probs < 0.6)).sum()),
        "low_risk"    : int((probs < 0.3).sum()),
        "avg_risk"    : round(float(probs.mean()), 4),
    }
    logger.info(f"Daily batch complete: {summary}")
    return summary

@shared_task(name="predictor.tasks.test_celery")
def test_celery():
    return {"status": "ok", "message": "Celery is working!"}