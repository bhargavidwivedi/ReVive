"""
ReVive — Autonomous Clinical Readmission Risk Agent
====================================================
Full agent loop:
1. Fetch patient data (FHIR or manual)
2. Predict readmission risk (ML)
3. Analyze clinical notes (LLM)
4. Assign care pathway (rules engine)
5. Send alert if high risk (Celery)
6. Record outcome for learning
"""

import requests
import json
import time
from datetime import datetime

API = "http://127.0.0.1:8000/api"


# ── COLORS FOR TERMINAL OUTPUT ─────────────────────────────────────────────────

class Color:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    PURPLE = "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BOLD   = "\033[1m"
    END    = "\033[0m"

def header(text):
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}  {text}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*60}{Color.END}")

def step(num, text):
    print(f"\n{Color.BOLD}{Color.BLUE}[Step {num}]{Color.END} {text}")

def success(text):
    print(f"{Color.GREEN}  ✅ {text}{Color.END}")

def warning(text):
    print(f"{Color.YELLOW}  ⚠️  {text}{Color.END}")

def error(text):
    print(f"{Color.RED}  ❌ {text}{Color.END}")

def info(text):
    print(f"{Color.WHITE}  → {text}{Color.END}")


# ── STEP 1: FETCH PATIENT DATA ────────────────────────────────────────────────

def fetch_patient_data(source="manual", patient_id=None, manual_data=None):
    """
    Fetch patient data from FHIR or use manual input.
    Returns structured patient data dict.
    """
    step(1, "Fetching Patient Data")

    if source == "fhir" and patient_id:
        try:
            r = requests.get(f"{API}/fhir/{patient_id}/", timeout=15)
            if r.status_code == 200:
                data = r.json()
                success(f"FHIR patient fetched: {data['patient']['full_name']}")
                info(f"Age: {data['patient']['age']} | Gender: {data['patient']['gender']}")
                return {
                    "patient_info" : data["patient"],
                    "features"     : data["fhir_features"],
                    "source"       : "fhir",
                }
            else:
                warning(f"FHIR fetch failed ({r.status_code}) — using manual data")
        except Exception as e:
            warning(f"FHIR unavailable: {e} — using manual data")

    # Manual patient data
    if manual_data is None:
        manual_data = {
            "age_numeric"          : 72,
            "time_in_hospital"     : 8,
            "number_inpatient"     : 3,
            "number_diagnoses"     : 9,
            "num_medications"      : 15,
            "num_lab_procedures"   : 55,
            "num_procedures"       : 2,
            "number_emergency"     : 2,
            "number_outpatient"    : 1,
            "total_prior_visits"   : 6,
            "is_elderly"           : 1,
            "polypharmacy"         : 1,
            "cardiac_primary"      : 1,
            "diabetes_primary"     : 0,
            "high_risk_discharge"  : 1,
            "medication_changed"   : 1,
            "complexity_score"     : 14.5,
            "los_x_diagnoses"      : 72,
            "meds_x_los"           : 120,
            "elderly_x_complex"    : 14.5,
            "inpatient_x_emergency": 6,
            "high_diagnosis_burden": 1,
            "high_lab_use"         : 1,
            "had_procedures"       : 1,
            "has_inpatient_history": 1,
            "emergency_prone"      : 1,
            "high_utiliser"        : 1,
        }

    success("Manual patient data loaded")
    info(f"Age: {manual_data.get('age_numeric')} | Diagnoses: {manual_data.get('number_diagnoses')} | Medications: {manual_data.get('num_medications')}")

    return {
        "patient_info": {"patient_id": patient_id or "LOCAL-001", "full_name": "Test Patient", "age": manual_data.get("age_numeric")},
        "features"    : manual_data,
        "source"      : "manual",
    }


# ── STEP 2: PREDICT READMISSION RISK ─────────────────────────────────────────

def predict_risk(features):
    """
    Call ML prediction API.
    Returns risk score and level.
    """
    step(2, "Predicting Readmission Risk (ML Model)")

    try:
        r = requests.post(f"{API}/predict/", json={"patient_data": features}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            risk = data["risk_level"]
            prob = data["readmission_percentage"]

            color = Color.RED if risk == "High" else Color.YELLOW if risk == "Medium" else Color.GREEN
            print(f"\n  {color}{Color.BOLD}Risk Level: {risk} ({prob}){Color.END}")
            success(f"Predicted readmission: {'YES ⚠️' if data['predicted_readmission'] else 'NO ✅'}")
            return data
        else:
            error(f"Prediction API failed: {r.status_code}")
            return None
    except Exception as e:
        error(f"Prediction failed: {e}")
        return None


# ── STEP 3: ANALYZE CLINICAL NOTES ───────────────────────────────────────────

def analyze_notes(discharge_notes=None):
    """
    Send discharge notes to LLM for psychosocial risk analysis.
    """
    step(3, "Analyzing Clinical Notes (Claude AI)")

    if discharge_notes is None:
        discharge_notes = """
        Patient is a 72-year-old male with history of cardiac failure and diabetes.
        Lives alone — wife passed away last year. Daughter visits occasionally but
        lives 3 hours away. Patient expressed anxiety about managing medications at home.
        Has missed follow-up appointments in the past due to transportation issues.
        Prescribed 15 medications on discharge. No home health arranged yet.
        Patient appears confused about medication schedule.
        """

    try:
        r = requests.post(
            f"{API}/analyze-notes/",
            json={"notes": discharge_notes},
            timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            analysis = data.get("llm_analysis", {})
            success("LLM analysis complete")
            info(f"Social isolation: {analysis.get('social_isolation_score', 0)}/3")
            info(f"Medication risk: {analysis.get('medication_noncompliance_risk', 0)}/2")
            info(f"Mental health flag: {'Yes' if analysis.get('mental_health_flag') else 'No'}")
            info(f"Overall psychosocial risk: {analysis.get('overall_psychosocial_risk', 0)}/10")
            info(f"Summary: {analysis.get('summary', 'N/A')}")
            return data
        else:
            warning(f"LLM analysis failed ({r.status_code}) — skipping")
            return {}
    except Exception as e:
        warning(f"LLM unavailable: {e} — skipping note analysis")
        return {}


# ── STEP 4: ASSIGN CARE PATHWAY ───────────────────────────────────────────────

def assign_pathway(features, risk_level, llm_analysis=None):
    """
    Autonomously assign care pathway based on risk + clinical flags.
    """
    step(4, "Assigning Autonomous Care Pathway")

    payload = {
        "patient_data": features,
        "risk_level"  : risk_level,
        "llm_analysis": llm_analysis.get("llm_analysis", {}) if llm_analysis else {},
    }

    try:
        r = requests.post(f"{API}/care-pathway/", json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json()
            success(f"Priority: {data['overall_priority']}")
            info(f"Pathways assigned: {len(data['assigned_pathways'])}")
            info(f"Follow-up date: {data['follow_up_date']} ({data['follow_up_in_days']} days)")
            info(f"Required specialists: {', '.join(data['required_specialists'][:4])}")
            print(f"\n  {Color.BOLD}Action Plan:{Color.END}")
            for i, action in enumerate(data["action_plan"][:5], 1):
                print(f"  {i}. {action}")
            return data
        else:
            warning(f"Care pathway failed ({r.status_code})")
            return {}
    except Exception as e:
        warning(f"Care pathway unavailable: {e}")
        return {}


# ── STEP 5: SEND ALERT ────────────────────────────────────────────────────────

def send_alert(patient_id, prob, risk_level, notify_email=None):
    """
    Trigger autonomous alert for high-risk patients.
    """
    step(5, "Triggering Autonomous Alert")

    if risk_level != "High":
        info(f"Risk level is {risk_level} — no alert needed")
        return

    payload = {
        "patient_id"  : patient_id,
        "patient_data": {},
        "notify_email": notify_email or "bhargavidwivedi56@gmail.com",
    }

    try:
        r = requests.post(f"{API}/discharge/", json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json()
            success(f"Alert queued! Task ID: {data['task_id']}")
            info("Clinical team will be notified within 30 seconds")
        else:
            warning(f"Alert failed ({r.status_code})")
    except Exception as e:
        warning(f"Alert system unavailable: {e}")


# ── STEP 6: RECORD OUTCOME ────────────────────────────────────────────────────

def record_outcome(patient_id, predicted_risk, actually_readmitted=None):
    """
    Record patient outcome for continuous learning.
    """
    step(6, "Recording Outcome for Continuous Learning")

    if actually_readmitted is None:
        info("No outcome recorded yet — will be updated after 30 days")
        return

    payload = {
        "patient_id"         : patient_id,
        "predicted_risk"     : predicted_risk,
        "actually_readmitted": actually_readmitted,
    }

    try:
        r = requests.post(f"{API}/outcomes/", json=payload, timeout=10)
        if r.status_code == 200:
            success(f"Outcome recorded: {'Readmitted' if actually_readmitted else 'Not readmitted'}")
        else:
            warning(f"Outcome recording failed ({r.status_code})")
    except Exception as e:
        warning(f"Outcome recording unavailable: {e}")


# ── FULL AGENT LOOP ───────────────────────────────────────────────────────────

def run_agent(
    patient_id      = "LOCAL-001",
    source          = "manual",
    manual_data     = None,
    discharge_notes = None,
    notify_email    = None,
    record_actual   = None,
):
    """
    Full autonomous ReVive agent loop.
    Runs all 6 steps end to end for a single patient.
    """
    start_time = time.time()

    header(f"ReVive Autonomous Agent — Patient {patient_id}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source: {source.upper()}")

    # Step 1: Fetch data
    patient = fetch_patient_data(source, patient_id, manual_data)
    if not patient:
        error("Could not fetch patient data. Aborting.")
        return

    features = patient["features"]

    # Step 2: Predict risk
    prediction = predict_risk(features)
    if not prediction:
        error("Prediction failed. Aborting.")
        return

    risk_level = prediction["risk_level"]
    prob       = prediction["readmission_probability"]

    # Step 3: Analyze notes
    llm_result = analyze_notes(discharge_notes)

    # Step 4: Assign care pathway
    pathway = assign_pathway(features, risk_level, llm_result)

    # Step 5: Send alert (only if High risk)
    send_alert(patient_id, prob, risk_level, notify_email)

    # Step 6: Record outcome
    record_outcome(patient_id, prob, record_actual)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)

    header("Agent Run Complete")
    print(f"  Patient    : {patient['patient_info'].get('full_name', patient_id)}")
    print(f"  Risk Level : {risk_level}")
    print(f"  Probability: {prediction['readmission_percentage']}")
    print(f"  Priority   : {pathway.get('overall_priority', 'N/A')}")
    print(f"  Follow-up  : {pathway.get('follow_up_date', 'N/A')}")
    print(f"  Time taken : {elapsed}s")
    print(f"\n{Color.GREEN}{Color.BOLD}  ReVive agent completed successfully! 🏥{Color.END}\n")

    return {
        "patient"   : patient,
        "prediction": prediction,
        "pathway"   : pathway,
        "llm"       : llm_result,
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Example 1: High-risk elderly cardiac patient
    run_agent(
        patient_id      = "P-HIGH-001",
        source          = "manual",
        notify_email    = "bhargavidwivedi56@gmail.com",
        discharge_notes = """
            Elderly male, 72 years old, cardiac failure history.
            Lives alone. No family support. Missed medications before.
            Anxious about going home. No follow-up arranged.
            Transportation issues noted. 15 medications prescribed.
        """,
    )