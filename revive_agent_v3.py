"""
ReVive Agent v3 — World-Class Clinical AI
==========================================
All 4 new features integrated:
1. Explainable AI — WHY the model predicted this risk
2. Clinical Knowledge Base — WHO/CDC/NIH guidelines applied
3. Patient Risk Timeline — track risk over time
4. Voice Clinical Notes — speech to structured data
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor.explainability  import full_explanation
from predictor.knowledge_base  import apply_guidelines_to_patient, get_red_flags
from predictor.risk_timeline   import record_risk_event, plot_risk_timeline, get_risk_summary
from predictor.voice_notes     import process_voice_note

API = "http://127.0.0.1:8000/api"


# ── COLORS ────────────────────────────────────────────────────────────────────
class C:
    RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"
    BLUE="\033[94m"; PURPLE="\033[95m"; CYAN="\033[96m"
    BOLD="\033[1m"; END="\033[0m"

def header(t): print(f"\n{C.BOLD}{C.CYAN}{'='*60}\n  {t}\n{'='*60}{C.END}")
def step(n,t):  print(f"\n{C.BOLD}{C.BLUE}[Step {n}]{C.END} {t}")
def ok(t):      print(f"{C.GREEN}  ✅ {t}{C.END}")
def warn(t):    print(f"{C.YELLOW}  ⚠️  {t}{C.END}")
def info(t):    print(f"  → {t}")
def think(t):   print(f"{C.PURPLE}  🧠 {t}{C.END}")
def flag(t):    print(f"{C.RED}  🚨 {t}{C.END}")


# ── FULL AGENT V3 ─────────────────────────────────────────────────────────────

def run_agent_v3(
    patient_id      : str  = "P-001",
    features        : dict = None,
    voice_note_text : str  = None,
    audio_path      : str  = None,
    notify_email    : str  = "bhargavidwivedi56@gmail.com",
):
    start = time.time()

    header(f"ReVive Agent v3 — Patient {patient_id}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Default features
    if features is None:
        features = {
            "age_numeric": 72, "time_in_hospital": 8,
            "number_inpatient": 3, "number_diagnoses": 9,
            "num_medications": 15, "num_lab_procedures": 55,
            "num_procedures": 2, "number_emergency": 2,
            "number_outpatient": 1, "total_prior_visits": 6,
            "is_elderly": 1, "polypharmacy": 1,
            "cardiac_primary": 1, "diabetes_primary": 1,
            "high_risk_discharge": 1, "medication_changed": 1,
            "complexity_score": 16.5, "los_x_diagnoses": 72,
            "meds_x_los": 120, "elderly_x_complex": 16.5,
            "inpatient_x_emergency": 6, "high_diagnosis_burden": 1,
            "high_lab_use": 1, "had_procedures": 1,
            "has_inpatient_history": 1, "emergency_prone": 1,
            "high_utiliser": 1,
        }

    # ── STEP 1: VOICE NOTE PROCESSING ────────────────────────────────────────
    step(1, "Voice Clinical Note Processing")
    voice_result = {}

    if voice_note_text or audio_path:
        voice_result = process_voice_note(
            audio_path = audio_path,
            text_input = voice_note_text,
            patient_id = patient_id,
        )
        # Merge risk signals from voice into features
        if voice_result and "risk_signals" in voice_result:
            features.update(voice_result["risk_signals"])
            ok("Voice note risk signals merged into patient features")
    else:
        info("No voice note provided — using manual features")

    # ── STEP 2: ML PREDICTION ─────────────────────────────────────────────────
    step(2, "ML Risk Prediction")
    prediction = {}
    try:
        r = requests.post(f"{API}/predict/", json={"patient_data": features}, timeout=10)
        if r.status_code == 200:
            prediction = r.json()
            risk = prediction["risk_level"]
            prob = prediction["readmission_percentage"]
            color = C.RED if risk=="High" else C.YELLOW if risk=="Medium" else C.GREEN
            print(f"\n  {color}{C.BOLD}Risk: {risk} ({prob}){C.END}")
            ok(f"Predicted readmission: {'YES ⚠️' if prediction['predicted_readmission'] else 'NO ✅'}")
    except Exception as e:
        warn(f"Prediction API failed: {e}")
        return

    risk_level = prediction.get("risk_level", "Medium")
    risk_prob  = prediction.get("readmission_probability", 0.5)

    # ── STEP 3: EXPLAINABLE AI ────────────────────────────────────────────────
    step(3, "Explainable AI — Why This Risk Score?")
    explanation = full_explanation(features, patient_id)

    if "error" not in explanation:
        print(f"\n  {C.BOLD}Top Risk Drivers:{C.END}")
        for d in explanation.get("top_drivers", [])[:5]:
            bar = "█" * int(d["impact"] * 200)
            print(f"  {d['label']:<35} {bar} ({d['patient_value']})")

        clinical = explanation.get("clinical_explanation", {})
        if clinical and "doctor_explanation" in clinical:
            print(f"\n  {C.BOLD}Doctor Explanation:{C.END}")
            print(f"  {clinical['doctor_explanation'].get('clinical_reasoning', '')}")
            print(f"\n  {C.BOLD}Patient Explanation:{C.END}")
            print(f"  {clinical['patient_explanation'].get('simple_summary', '')}")

    # ── STEP 4: CLINICAL KNOWLEDGE BASE ──────────────────────────────────────
    step(4, "Clinical Knowledge Base — Evidence-Based Guidelines")
    guidelines = apply_guidelines_to_patient(features, risk_level, patient_id)

    if "error" not in guidelines:
        ok(f"Guidelines applied: {', '.join(guidelines.get('applicable_guidelines', []))}")
        think(f"Priority intervention: {guidelines.get('priority_intervention', 'N/A')}")
        info(f"Expected readmission reduction: {guidelines.get('expected_readmission_reduction', 'N/A')}")

        print(f"\n  {C.BOLD}Evidence-Based Recommendations:{C.END}")
        for rec in guidelines.get("evidence_based_recommendations", [])[:4]:
            strength_color = C.GREEN if rec.get('evidence_strength') == 'Strong' else C.YELLOW
            print(f"  {strength_color}[{rec.get('evidence_strength', '?')}]{C.END} {rec.get('recommendation', '')} ({rec.get('source', '')})")

    # Get red flags from guidelines
    red_flags = get_red_flags(features)
    if red_flags:
        print(f"\n  {C.BOLD}{C.RED}Clinical Red Flags:{C.END}")
        for f in red_flags[:4]:
            flag(f)

    # ── STEP 5: RISK TIMELINE ─────────────────────────────────────────────────
    step(5, "Patient Risk Timeline")
    record_risk_event(
        patient_id = patient_id,
        risk_prob  = risk_prob,
        risk_level = risk_level,
        context    = {
            "diagnoses"   : features.get("number_diagnoses"),
            "medications" : features.get("num_medications"),
            "inpatient"   : features.get("number_inpatient"),
        }
    )

    chart_path = plot_risk_timeline(patient_id)
    summary    = get_risk_summary(patient_id)
    info(f"Trend: {summary.get('trend', 'unknown')}")
    info(f"Total assessments: {summary.get('total_assessments', 1)}")

    # ── STEP 6: CARE PATHWAY ──────────────────────────────────────────────────
    step(6, "Autonomous Care Pathway")
    pathway = {}
    try:
        r = requests.post(f"{API}/care-pathway/", json={
            "patient_data": features,
            "risk_level"  : risk_level,
        }, timeout=10)
        if r.status_code == 200:
            pathway = r.json()
            ok(f"Priority: {pathway.get('overall_priority')} | Follow-up: {pathway.get('follow_up_date')}")
            print(f"\n  {C.BOLD}Top Actions:{C.END}")
            for i, action in enumerate(pathway.get("action_plan", [])[:4], 1):
                print(f"  {i}. {action}")
    except Exception as e:
        warn(f"Care pathway failed: {e}")

    # ── STEP 7: ALERT ─────────────────────────────────────────────────────────
    step(7, "Smart Alert")
    if risk_level == "High":
        try:
            r = requests.post(f"{API}/discharge/", json={
                "patient_id"  : patient_id,
                "patient_data": features,
                "notify_email": notify_email,
            }, timeout=10)
            if r.status_code == 200:
                ok(f"Alert queued → {notify_email}")
        except Exception as e:
            warn(f"Alert failed: {e}")
    else:
        info(f"{risk_level} risk — no alert needed")

    # ── STEP 8: COMPREHENSIVE REPORT ─────────────────────────────────────────
    step(8, "Generating Comprehensive Clinical Report")

    report = {
        "report_id"    : f"RVV3-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "patient_id"   : patient_id,
        "generated_at" : datetime.now().isoformat(),
        "risk"         : {
            "level"      : risk_level,
            "probability": prediction.get("readmission_percentage"),
            "predicted"  : prediction.get("predicted_readmission"),
        },
        "explanation"  : {
            "top_drivers"   : explanation.get("top_drivers", [])[:5],
            "doctor_summary": explanation.get("clinical_explanation", {}).get(
                              "doctor_explanation", {}).get("clinical_reasoning", ""),
            "patient_summary": explanation.get("clinical_explanation", {}).get(
                               "patient_explanation", {}).get("simple_summary", ""),
            "chart"         : explanation.get("chart_path"),
        },
        "guidelines"   : {
            "applied"   : guidelines.get("applicable_guidelines", []),
            "priority"  : guidelines.get("priority_intervention"),
            "reduction" : guidelines.get("expected_readmission_reduction"),
            "recs"      : guidelines.get("evidence_based_recommendations", []),
        },
        "timeline"     : summary,
        "pathway"      : {
            "priority"  : pathway.get("overall_priority"),
            "follow_up" : pathway.get("follow_up_date"),
            "actions"   : pathway.get("action_plan", []),
        },
        "voice_note"   : voice_result if voice_result else None,
        "red_flags"    : red_flags,
    }

    os.makedirs("outputs/reports", exist_ok=True)
    report_path = f"outputs/reports/report_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    ok(f"Report saved → {report_path}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 2)
    header("ReVive v3 — Agent Complete")
    print(f"  Patient        : {patient_id}")
    print(f"  Risk           : {risk_level} ({prediction.get('readmission_percentage')})")
    print(f"  Red Flags      : {len(red_flags)}")
    print(f"  Timeline Trend : {summary.get('trend', 'first assessment')}")
    print(f"  Guidelines     : {len(guidelines.get('applicable_guidelines', []))} applied")
    print(f"  Report         : {report_path}")
    print(f"  Time           : {elapsed}s")
    print(f"\n{C.GREEN}{C.BOLD}  ReVive v3 — Clinical AI Complete! 🏥{C.END}\n")
    return report


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test with voice note simulation
    run_agent_v3(
        patient_id      = "P-DEMO-001",
        notify_email    = "bhargavidwivedi56@gmail.com",
        voice_note_text = """
            Patient is a 72 year old male admitted for acute decompensated heart failure
            with type 2 diabetes. He lives alone since his wife passed away six months ago.
            His daughter lives in Mumbai and cannot provide daily support.
            Patient is on 15 medications including warfarin, metformin, and furosemide.
            He expressed significant anxiety about managing medications at home.
            He has missed his last three cardiology appointments due to transportation issues.
            No home health services arranged. Follow up appointment not yet scheduled.
            Patient appears confused about his insulin dosing schedule.
            Discharge disposition is home without support services.
        """,
    )