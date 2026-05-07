"""
ReVive Agent v2 — Smarter Reasoning Layer
==========================================
Instead of blindly running all steps, the agent now:
1. Thinks about the patient context
2. Decides WHICH actions to take and WHY
3. Explains its reasoning like a clinical AI
4. Adapts its response based on risk signals
5. Uses Claude to reason about edge cases
"""

import requests
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

API    = "http://127.0.0.1:8000/api"
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── COLORS ────────────────────────────────────────────────────────────────────

class C:
    RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"
    BLUE="\033[94m"; PURPLE="\033[95m"; CYAN="\033[96m"
    BOLD="\033[1m"; END="\033[0m"

def header(t): print(f"\n{C.BOLD}{C.CYAN}{'='*60}\n  {t}\n{'='*60}{C.END}")
def step(n,t):  print(f"\n{C.BOLD}{C.BLUE}[Step {n}]{C.END} {t}")
def ok(t):      print(f"{C.GREEN}  ✅ {t}{C.END}")
def warn(t):    print(f"{C.YELLOW}  ⚠️  {t}{C.END}")
def err(t):     print(f"{C.RED}  ❌ {t}{C.END}")
def info(t):    print(f"  → {t}")
def think(t):   print(f"{C.PURPLE}  🧠 {t}{C.END}")


# ── AGENT MEMORY ──────────────────────────────────────────────────────────────

class AgentMemory:
    """Tracks everything the agent has observed about this patient."""

    def __init__(self, patient_id):
        self.patient_id  = patient_id
        self.observations = []
        self.decisions    = []
        self.risk_history = []
        self.started_at   = datetime.now().isoformat()

    def observe(self, key, value):
        self.observations.append({"key": key, "value": value, "at": datetime.now().isoformat()})

    def decide(self, action, reason, priority="normal"):
        self.decisions.append({"action": action, "reason": reason, "priority": priority})
        think(f"Decision: {action}")
        info(f"Reason: {reason}")

    def skip(self, action, reason):
        self.decisions.append({"action": action, "reason": reason, "skipped": True})
        info(f"Skipping {action}: {reason}")

    def summary(self):
        return {
            "patient_id"  : self.patient_id,
            "observations": self.observations,
            "decisions"   : self.decisions,
            "risk_history": self.risk_history,
        }


# ── STEP 1: PREDICT RISK ──────────────────────────────────────────────────────

def predict_risk(features, memory):
    step(1, "ML Risk Prediction")
    try:
        r = requests.post(f"{API}/predict/", json={"patient_data": features}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            risk = data["risk_level"]
            prob = data["readmission_probability"]
            pct  = data["readmission_percentage"]

            color = C.RED if risk=="High" else C.YELLOW if risk=="Medium" else C.GREEN
            print(f"\n  {color}{C.BOLD}Risk: {risk} ({pct}){C.END}")

            memory.observe("risk_level", risk)
            memory.observe("risk_probability", prob)
            memory.risk_history.append({"prob": prob, "risk": risk, "at": datetime.now().isoformat()})
            return data
    except Exception as e:
        err(f"Prediction failed: {e}")
    return None


# ── STEP 2: REASON ABOUT CONTEXT ─────────────────────────────────────────────

def reason_about_context(features, prediction, notes, memory):
    """
    The core reasoning step — Claude analyzes the full patient context
    and decides what actions the agent should take and why.
    This is what makes v2 smarter than v1.
    """
    step(2, "Agent Reasoning (Claude AI)")

    risk_level = prediction["risk_level"]
    prob       = prediction["readmission_probability"]

    # Build context for Claude
    context = f"""
You are ReVive, an autonomous clinical AI agent. A patient has just been discharged.

PATIENT CONTEXT:
- Age: {features.get('age_numeric', 'unknown')}
- Length of stay: {features.get('time_in_hospital', 0)} days
- Prior inpatient visits: {features.get('number_inpatient', 0)}
- Number of diagnoses: {features.get('number_diagnoses', 0)}
- Medications: {features.get('num_medications', 0)}
- Cardiac primary: {'Yes' if features.get('cardiac_primary') else 'No'}
- Diabetes primary: {'Yes' if features.get('diabetes_primary') else 'No'}
- Elderly (65+): {'Yes' if features.get('is_elderly') else 'No'}
- Polypharmacy (5+ meds): {'Yes' if features.get('polypharmacy') else 'No'}
- High-risk discharge: {'Yes' if features.get('high_risk_discharge') else 'No'}
- Emergency prone: {'Yes' if features.get('emergency_prone') else 'No'}

ML PREDICTION:
- Risk Level: {risk_level}
- Readmission Probability: {prob:.1%}

DISCHARGE NOTES:
{notes or 'No notes provided.'}

Based on this context, decide:
1. What are the TOP 3 most critical risk factors for THIS specific patient?
2. What immediate actions should be taken in the next 24 hours?
3. What actions can wait 48-72 hours?
4. Are there any red flags that need URGENT escalation?
5. What is your overall clinical reasoning summary?

Respond ONLY in this JSON format:
{{
    "critical_risk_factors": ["factor1", "factor2", "factor3"],
    "immediate_actions": ["action1", "action2"],
    "delayed_actions": ["action1", "action2"],
    "red_flags": ["flag1"] or [],
    "escalate_immediately": true or false,
    "escalation_reason": "reason if escalate_immediately is true" or null,
    "reasoning_summary": "2-3 sentence clinical reasoning summary",
    "confidence": "high/medium/low"
}}
"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1000,
            messages   = [{"role": "user", "content": context}]
        )
        raw    = response.content[0].text.strip()
        result = json.loads(raw)

        ok("Agent reasoning complete")
        print(f"\n  {C.BOLD}Critical Risk Factors:{C.END}")
        for f in result.get("critical_risk_factors", []):
            print(f"  • {f}")

        print(f"\n  {C.BOLD}Immediate Actions (24h):{C.END}")
        for a in result.get("immediate_actions", []):
            print(f"  🔴 {a}")

        print(f"\n  {C.BOLD}Delayed Actions (48-72h):{C.END}")
        for a in result.get("delayed_actions", []):
            print(f"  🟡 {a}")

        if result.get("red_flags"):
            print(f"\n  {C.RED}{C.BOLD}RED FLAGS:{C.END}")
            for f in result["red_flags"]:
                print(f"  {C.RED}🚨 {f}{C.END}")

        print(f"\n  {C.BOLD}Reasoning:{C.END}")
        print(f"  {result.get('reasoning_summary', '')}")
        print(f"  Confidence: {result.get('confidence', 'unknown').upper()}")

        # Store decisions in memory
        for action in result.get("immediate_actions", []):
            memory.decide(action, "Identified as immediate priority by agent reasoning", "urgent")
        for action in result.get("delayed_actions", []):
            memory.decide(action, "Identified as delayed priority by agent reasoning", "normal")

        memory.observe("escalate_immediately", result.get("escalate_immediately", False))
        memory.observe("red_flags", result.get("red_flags", []))

        return result

    except json.JSONDecodeError:
        warn("Could not parse reasoning — using defaults")
        return {}
    except Exception as e:
        warn(f"Reasoning unavailable: {e}")
        return {}


# ── STEP 3: SMART CARE PATHWAY ────────────────────────────────────────────────

def smart_care_pathway(features, prediction, reasoning, memory):
    """
    Assigns care pathway but now uses reasoning to prioritize actions.
    """
    step(3, "Smart Care Pathway Assignment")

    risk_level = prediction["risk_level"]

    # Agent decides whether to assign pathway
    if risk_level == "Low" and not reasoning.get("escalate_immediately"):
        memory.skip("full_care_pathway", "Low risk patient — standard discharge protocol sufficient")
        ok("Standard discharge protocol applied")
        return {"overall_priority": "LOW", "action_plan": ["Standard discharge", "GP follow-up in 14 days"]}

    try:
        r = requests.post(f"{API}/care-pathway/", json={
            "patient_data": features,
            "risk_level"  : risk_level,
            "llm_analysis": {},
        }, timeout=10)

        if r.status_code == 200:
            data = r.json()
            ok(f"Priority: {data['overall_priority']}")
            info(f"Pathways: {len(data['assigned_pathways'])}")
            info(f"Follow-up: {data['follow_up_date']} ({data['follow_up_in_days']} days)")

            # Agent prioritizes actions based on reasoning
            immediate = reasoning.get("immediate_actions", [])
            if immediate:
                print(f"\n  {C.BOLD}Agent-Prioritized Actions:{C.END}")
                for i, action in enumerate(immediate[:3], 1):
                    print(f"  {i}. 🔴 {action}")

            memory.observe("pathway_priority", data["overall_priority"])
            return data
    except Exception as e:
        warn(f"Care pathway failed: {e}")
    return {}


# ── STEP 4: SMART ALERT DECISION ─────────────────────────────────────────────

def smart_alert(patient_id, prediction, reasoning, memory, notify_email):
    """
    Agent decides whether to alert, who to alert, and how urgently.
    """
    step(4, "Smart Alert Decision")

    risk_level = prediction["risk_level"]
    escalate   = reasoning.get("escalate_immediately", False)

    # Decision tree
    if risk_level == "Low" and not escalate:
        memory.skip("alert", "Low risk — no alert needed")
        ok("No alert needed for low risk patient")
        return

    if escalate:
        memory.decide("URGENT_ESCALATION", "Agent identified red flags requiring immediate escalation", "critical")
        print(f"\n  {C.RED}{C.BOLD}⚠️  URGENT ESCALATION TRIGGERED{C.END}")
        reason = reasoning.get("escalation_reason", "Red flags identified")
        info(f"Reason: {reason}")

    if risk_level in ["High", "Medium"] or escalate:
        try:
            r = requests.post(f"{API}/discharge/", json={
                "patient_id"  : patient_id,
                "patient_data": {},
                "notify_email": notify_email,
            }, timeout=10)

            if r.status_code == 200:
                data = r.json()
                ok(f"Alert queued! Task: {data['task_id']}")
                memory.observe("alert_sent", True)
                memory.observe("alert_task_id", data["task_id"])
            else:
                warn(f"Alert failed: {r.status_code}")
        except Exception as e:
            warn(f"Alert system unavailable: {e}")


# ── STEP 5: AGENT REPORT ─────────────────────────────────────────────────────

def generate_report(patient_id, prediction, reasoning, pathway, memory):
    """
    Generate a structured clinical report from the agent run.
    """
    step(5, "Generating Clinical Report")

    report = {
        "report_id"        : f"RVV-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "patient_id"       : patient_id,
        "generated_at"     : datetime.now().isoformat(),
        "risk_assessment"  : {
            "level"      : prediction.get("risk_level"),
            "probability": prediction.get("readmission_percentage"),
            "predicted"  : prediction.get("predicted_readmission"),
        },
        "agent_reasoning"  : {
            "critical_factors": reasoning.get("critical_risk_factors", []),
            "red_flags"       : reasoning.get("red_flags", []),
            "escalate"        : reasoning.get("escalate_immediately", False),
            "confidence"      : reasoning.get("confidence", "unknown"),
            "summary"         : reasoning.get("reasoning_summary", ""),
        },
        "care_plan"        : {
            "priority"        : pathway.get("overall_priority"),
            "follow_up_date"  : pathway.get("follow_up_date"),
            "immediate_actions": reasoning.get("immediate_actions", []),
            "delayed_actions"  : reasoning.get("delayed_actions", []),
        },
        "agent_decisions"  : memory.decisions,
        "total_observations": len(memory.observations),
    }

    # Save report
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/report_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    ok(f"Report saved → {path}")
    return report


# ── FULL AGENT v2 LOOP ────────────────────────────────────────────────────────

def run_agent_v2(
    patient_id      = "P-001",
    features        = None,
    discharge_notes = None,
    notify_email    = "bhargavidwivedi56@gmail.com",
):
    start = time.time()
    memory = AgentMemory(patient_id)

    header(f"ReVive Agent v2 — Patient {patient_id}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Default high-risk patient
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

    if discharge_notes is None:
        discharge_notes = """
        72-year-old male with cardiac failure and Type 2 diabetes.
        Lives alone — wife passed 6 months ago. No family nearby.
        Very anxious about managing 15 medications at home.
        Has history of missing appointments. No transport available.
        Confusion noted about medication schedule during education session.
        Home health not yet arranged. Follow-up not scheduled.
        Patient tearful during discharge counseling.
        """

    # Step 1: Predict
    prediction = predict_risk(features, memory)
    if not prediction:
        err("Cannot proceed without risk prediction.")
        return

    # Step 2: Reason
    reasoning = reason_about_context(features, prediction, discharge_notes, memory)

    # Step 3: Care pathway
    pathway = smart_care_pathway(features, prediction, reasoning, memory)

    # Step 4: Alert
    smart_alert(patient_id, prediction, reasoning, memory, notify_email)

    # Step 5: Report
    report = generate_report(patient_id, prediction, reasoning, pathway, memory)

    # Summary
    elapsed = round(time.time() - start, 2)
    header("Agent v2 Complete")
    print(f"  Patient     : {patient_id}")
    print(f"  Risk        : {prediction['risk_level']} ({prediction['readmission_percentage']})")
    print(f"  Decisions   : {len(memory.decisions)}")
    print(f"  Escalate    : {'YES 🚨' if reasoning.get('escalate_immediately') else 'No'}")
    print(f"  Report      : outputs/report_{patient_id}_*.json")
    print(f"  Time        : {elapsed}s")
    print(f"\n{C.GREEN}{C.BOLD}  ReVive v2 completed! 🏥{C.END}\n")

    return report


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent_v2(patient_id="P-HIGH-001")