import joblib
import pandas as pd
import logging
from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Load model once ───────────────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed_features.csv")

model         = joblib.load(MODEL_PATH)
FEATURE_NAMES = [c for c in pd.read_csv(DATA_PATH, nrows=1).columns if c != "readmitted_30d"]
THRESHOLD     = 0.369


def get_risk_level(prob):
    if prob >= 0.6:  return "High"
    if prob >= 0.3:  return "Medium"
    return "Low"


# ── Task 1: Score a single patient on discharge ───────────────────────────────
@shared_task(name="predictor.tasks.score_patient_on_discharge")
def score_patient_on_discharge(patient_data: dict, patient_id: str, notify_email: str = None):
    """
    Triggered automatically when a patient is discharged.
    Scores them and sends alert if high risk.
    """
    logger.info(f"Scoring patient {patient_id} on discharge...")

    row = pd.DataFrame([{f: patient_data.get(f, 0) for f in FEATURE_NAMES}])
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0)

    prob       = float(model.predict_proba(row)[0, 1])
    risk_level = get_risk_level(prob)

    result = {
        "patient_id"             : patient_id,
        "readmission_probability": round(prob, 4),
        "risk_level"             : risk_level,
        "predicted_readmission"  : prob >= THRESHOLD,
    }

    logger.info(f"Patient {patient_id}: {risk_level} risk ({prob:.2%})")

    # Send alert if high risk
    if risk_level == "High" and notify_email:
        send_risk_alert.delay(patient_id, prob, risk_level, notify_email)

    return result


# ── Task 2: Send email alert ──────────────────────────────────────────────────
@shared_task(name="predictor.tasks.send_risk_alert")
def send_risk_alert(patient_id: str, prob: float, risk_level: str, recipient_email: str):
    """
    Sends an email alert to the clinical team for high-risk patients.
    """
    subject = f"⚠️ HIGH READMISSION RISK — Patient {patient_id}"
    message = f"""
ReVive Clinical Alert System
{'='*40}

Patient ID    : {patient_id}
Risk Level    : {risk_level.upper()}
Probability   : {prob:.1%}

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
            subject      = subject,
            message      = message,
            from_email   = settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient_email],
            fail_silently= False,
        )
        logger.info(f"Alert sent to {recipient_email} for patient {patient_id}")
        return {"status": "sent", "recipient": recipient_email}
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        return {"status": "failed", "error": str(e)}


# ── Task 3: Batch score all patients daily ────────────────────────────────────
@shared_task(name="predictor.tasks.score_all_patients")
def score_all_patients():
    """
    Runs every 24 hours via Celery Beat.
    Scores all patients in the processed dataset and logs high-risk ones.
    """
    logger.info("Starting daily batch scoring...")

    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X  = df.drop(columns=["readmitted_30d"], errors="ignore")

    probs      = model.predict_proba(X)[:, 1]
    high_risk  = (probs >= 0.6).sum()
    medium_risk= ((probs >= 0.3) & (probs < 0.6)).sum()
    low_risk   = (probs < 0.3).sum()

    summary = {
        "total_scored" : len(probs),
        "high_risk"    : int(high_risk),
        "medium_risk"  : int(medium_risk),
        "low_risk"     : int(low_risk),
        "avg_risk"     : round(float(probs.mean()), 4),
    }

    logger.info(f"Daily batch complete: {summary}")
    return summary


# ── Task 4: Test task (verify Celery is working) ──────────────────────────────
@shared_task(name="predictor.tasks.test_celery")
def test_celery():
    logger.info("Celery is working correctly!")
    return {"status": "ok", "message": "Celery is working!"}