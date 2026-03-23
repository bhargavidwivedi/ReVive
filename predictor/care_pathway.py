import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ── CARE PATHWAY DEFINITIONS ──────────────────────────────────────────────────

PATHWAYS = {
    "cardiac_high_risk": {
        "name"       : "Cardiac High-Risk Pathway",
        "priority"   : "URGENT",
        "color"      : "red",
        "actions"    : [
            "Schedule cardiology follow-up within 48 hours",
            "Arrange cardiac rehabilitation assessment",
            "Daily remote monitoring for first 2 weeks",
            "Medication reconciliation with cardiologist",
            "Patient education on warning signs",
        ],
        "specialists": ["Cardiologist", "Cardiac Nurse", "Pharmacist"],
        "follow_up_days": 2,
    },
    "diabetes_high_risk": {
        "name"       : "Diabetes Management Pathway",
        "priority"   : "URGENT",
        "color"      : "red",
        "actions"    : [
            "Schedule endocrinology review within 48 hours",
            "HbA1c and glucose monitoring plan",
            "Diabetes nurse educator referral",
            "Medication adherence assessment",
            "Dietary counselling referral",
        ],
        "specialists": ["Endocrinologist", "Diabetes Nurse", "Dietitian"],
        "follow_up_days": 2,
    },
    "elderly_high_risk": {
        "name"       : "Elderly High-Risk Pathway",
        "priority"   : "URGENT",
        "color"      : "red",
        "actions"    : [
            "Arrange home health visit within 24 hours",
            "Social work assessment for home support",
            "Falls risk assessment",
            "Medication review for polypharmacy",
            "Cognitive assessment if indicated",
            "Family/carer education session",
        ],
        "specialists": ["Geriatrician", "Social Worker", "Occupational Therapist"],
        "follow_up_days": 1,
    },
    "mental_health_risk": {
        "name"       : "Mental Health Support Pathway",
        "priority"   : "HIGH",
        "color"      : "orange",
        "actions"    : [
            "Psychiatry or psychology consult within 72 hours",
            "Mental health crisis plan provided",
            "Community mental health team referral",
            "Medication compliance support",
            "Support network assessment",
        ],
        "specialists": ["Psychiatrist", "Psychologist", "Mental Health Nurse"],
        "follow_up_days": 3,
    },
    "social_isolation_risk": {
        "name"       : "Social Support Pathway",
        "priority"   : "HIGH",
        "color"      : "orange",
        "actions"    : [
            "Social worker referral within 48 hours",
            "Community support services assessment",
            "Transport assistance arrangement",
            "Meals on wheels referral if needed",
            "Volunteer visitor program referral",
        ],
        "specialists": ["Social Worker", "Community Health Nurse"],
        "follow_up_days": 3,
    },
    "medication_risk": {
        "name"       : "Medication Management Pathway",
        "priority"   : "HIGH",
        "color"      : "orange",
        "actions"    : [
            "Pharmacist medication review before discharge",
            "Blister pack dispensing arrangement",
            "Medication reminder system setup",
            "Patient education on each medication",
            "GP follow-up within 5 days",
        ],
        "specialists": ["Pharmacist", "GP", "Medication Nurse"],
        "follow_up_days": 5,
    },
    "standard_medium_risk": {
        "name"       : "Standard Medium-Risk Pathway",
        "priority"   : "MEDIUM",
        "color"      : "yellow",
        "actions"    : [
            "GP follow-up within 7 days",
            "Discharge summary sent to GP",
            "Patient education materials provided",
            "Contact number for questions provided",
            "Outpatient review if needed",
        ],
        "specialists": ["GP"],
        "follow_up_days": 7,
    },
    "standard_low_risk": {
        "name"       : "Standard Discharge Protocol",
        "priority"   : "LOW",
        "color"      : "green",
        "actions"    : [
            "Standard discharge instructions provided",
            "GP follow-up within 14 days",
            "Emergency contact number provided",
        ],
        "specialists": ["GP"],
        "follow_up_days": 14,
    },
}


# ── PATHWAY ASSIGNMENT ENGINE ─────────────────────────────────────────────────

def assign_care_pathway(
    risk_level      : str,
    features        : dict,
    llm_analysis    : dict = None,
) -> dict:
    """
    Assigns one or more care pathways based on:
    - ML risk level (High/Medium/Low)
    - Clinical features (cardiac, diabetes, elderly)
    - LLM psychosocial analysis (mental health, isolation, medication)

    Returns a comprehensive care plan.
    """
    assigned_pathways = []
    llm = llm_analysis or {}

    # ── High risk routing ─────────────────────────────────────────────────────
    if risk_level == "High":

        # Cardiac
        if features.get("cardiac_primary", 0):
            assigned_pathways.append("cardiac_high_risk")

        # Diabetes
        if features.get("diabetes_primary", 0):
            assigned_pathways.append("diabetes_high_risk")

        # Elderly
        if features.get("is_elderly", 0) or features.get("age_numeric", 0) >= 65:
            assigned_pathways.append("elderly_high_risk")

        # Mental health from LLM
        if llm.get("mental_health_flag", 0):
            assigned_pathways.append("mental_health_risk")

        # Social isolation from LLM
        if llm.get("social_isolation_score", 0) >= 2:
            assigned_pathways.append("social_isolation_risk")

        # Medication non-compliance from LLM
        if llm.get("medication_noncompliance_risk", 0) >= 1:
            assigned_pathways.append("medication_risk")

        # If no specific pathway matched, use general high risk
        if not assigned_pathways:
            assigned_pathways.append("elderly_high_risk")

    # ── Medium risk routing ───────────────────────────────────────────────────
    elif risk_level == "Medium":
        assigned_pathways.append("standard_medium_risk")

        # Add specific pathways if flags present
        if llm.get("mental_health_flag", 0):
            assigned_pathways.append("mental_health_risk")
        if llm.get("medication_noncompliance_risk", 0) >= 2:
            assigned_pathways.append("medication_risk")

    # ── Low risk routing ──────────────────────────────────────────────────────
    else:
        assigned_pathways.append("standard_low_risk")

    # ── Build care plan ───────────────────────────────────────────────────────
    care_plan = build_care_plan(assigned_pathways, features)
    logger.info(f"Assigned pathways: {assigned_pathways}")
    return care_plan


# ── BUILD CARE PLAN ───────────────────────────────────────────────────────────

def build_care_plan(pathway_ids: list, features: dict) -> dict:
    """
    Builds a comprehensive care plan from assigned pathways.
    """
    today       = datetime.now()
    all_actions = []
    specialists = set()
    earliest_fu = 999
    priority    = "LOW"

    priority_order = {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

    pathway_details = []
    for pid in pathway_ids:
        p = PATHWAYS.get(pid, {})
        if not p:
            continue

        pathway_details.append({
            "id"      : pid,
            "name"    : p["name"],
            "priority": p["priority"],
            "color"   : p["color"],
            "actions" : p["actions"],
        })

        all_actions.extend(p["actions"])
        specialists.update(p.get("specialists", []))

        fu_days    = p.get("follow_up_days", 14)
        earliest_fu = min(earliest_fu, fu_days)

        if priority_order.get(p["priority"], 0) > priority_order.get(priority, 0):
            priority = p["priority"]

    follow_up_date = today + timedelta(days=earliest_fu)

    # Remove duplicate actions
    unique_actions = list(dict.fromkeys(all_actions))

    return {
        "assigned_pathways"  : pathway_details,
        "overall_priority"   : priority,
        "total_actions"      : len(unique_actions),
        "action_plan"        : unique_actions[:10],
        "required_specialists": list(specialists),
        "follow_up_date"     : follow_up_date.strftime("%Y-%m-%d"),
        "follow_up_in_days"  : earliest_fu,
        "generated_at"       : today.strftime("%Y-%m-%d %H:%M:%S"),
    }