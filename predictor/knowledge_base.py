"""
ReVive — Clinical Knowledge Base
=================================
WHO, CDC, NIH, AHA guidelines for readmission prevention.
Claude retrieves and applies relevant guidelines to each patient.
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── CURATED CLINICAL GUIDELINES ───────────────────────────────────────────────

GUIDELINES = {
    "cardiac": {
        "source"   : "AHA/ACC 2022",
        "condition": "Heart Failure",
        "key_points": [
            "Heart failure patients should have follow-up within 7-14 days of discharge (AHA Class I)",
            "Daily weight monitoring — contact provider if weight increases >2kg in 2 days",
            "Sodium restriction <2g/day reduces fluid retention and readmission risk",
            "ACE inhibitors/ARBs and beta-blockers reduce 30-day readmission by 25%",
            "Cardiac rehabilitation reduces readmission by 30% in HF patients",
            "BNP/NT-proBNP monitoring post-discharge identifies deterioration early",
        ],
        "red_flags": [
            "Shortness of breath at rest",
            "Weight gain >2kg in 2 days",
            "Swollen ankles/legs worsening",
            "Chest pain or palpitations",
        ],
        "readmission_risk_reduction": "Structured discharge + 7-day follow-up reduces readmission by 20-30%",
    },

    "diabetes": {
        "source"   : "ADA Standards 2024 / WHO Diabetes Guidelines",
        "condition": "Diabetes Mellitus",
        "key_points": [
            "HbA1c target <7% for most adults (ADA 2024)",
            "Self-monitoring blood glucose 2-4x/day post-discharge",
            "Structured diabetes education reduces readmission by 20% (NIH evidence)",
            "Hypoglycemia is leading cause of diabetes-related readmission",
            "Metformin should be resumed 48h post-discharge if renal function stable",
            "Carbohydrate counting education reduces glucose variability",
        ],
        "red_flags": [
            "Blood glucose <4 mmol/L (hypoglycemia)",
            "Blood glucose >15 mmol/L (hyperglycemia)",
            "Confusion or shakiness",
            "Wound not healing",
        ],
        "readmission_risk_reduction": "Diabetes-specific discharge planning reduces 30-day readmission by 15%",
    },

    "medication": {
        "source"   : "WHO Medication Safety 2023 / ISMP Guidelines",
        "condition": "Polypharmacy / Medication Safety",
        "key_points": [
            "Medication reconciliation at discharge reduces ADEs by 70% (WHO 2023)",
            "Blister packs improve adherence by 40% in elderly polypharmacy patients",
            "Pharmacist-led medication review reduces readmission by 17%",
            "High-risk medications (warfarin, insulin, digoxin) require priority counseling",
            "Simplified dosing schedules improve compliance in patients with 5+ medications",
            "Follow-up phone call within 72h of discharge reduces medication errors",
        ],
        "red_flags": [
            "Patient cannot name their medications",
            "Confusion about dosing schedule",
            "5 or more medications prescribed",
            "New medications added at discharge",
        ],
        "readmission_risk_reduction": "Pharmacist medication review at discharge reduces readmission by 17-30%",
    },

    "social": {
        "source"   : "CDC Social Determinants of Health / WHO SDOH Framework",
        "condition": "Social Determinants of Health",
        "key_points": [
            "Social isolation increases readmission risk by 2x (CDC 2023)",
            "Transportation barriers responsible for 25% of missed follow-up appointments",
            "Patients living alone have 34% higher 30-day readmission rate",
            "Food insecurity associated with 27% higher readmission in diabetic patients",
            "Social work intervention reduces readmission by 25% in high-risk patients",
            "Community health worker programs reduce readmission by 20% (NIH evidence)",
        ],
        "red_flags": [
            "Patient lives alone with no support",
            "No transport to follow-up appointments",
            "History of missed appointments",
            "Financial stress affecting medication purchase",
        ],
        "readmission_risk_reduction": "Social work intervention + community support reduces readmission by 20-25%",
    },

    "general_readmission": {
        "source"   : "CMS Hospital Readmissions Reduction Program / AHRQ 2023",
        "condition": "General Readmission Prevention",
        "key_points": [
            "Teach-back method improves patient understanding by 40% (AHRQ 2023)",
            "Follow-up within 7 days reduces readmission by 20% (CMS data)",
            "Structured discharge checklist reduces readmission by 15%",
            "Transitional care programs reduce 30-day readmission by 20-30%",
            "Post-discharge phone calls within 72h reduce readmission by 12%",
            "Electronic health records flagging reduces time to readmission by 25%",
        ],
        "red_flags": [
            "No follow-up appointment arranged",
            "Patient does not understand discharge instructions",
            "Multiple prior admissions in 12 months",
            "Discharge against medical advice",
        ],
        "readmission_risk_reduction": "Comprehensive discharge planning reduces all-cause 30-day readmission by 25%",
    },

    "elderly": {
        "source"   : "AGS Beers Criteria 2023 / WHO Ageing Guidelines",
        "condition": "Geriatric Care",
        "key_points": [
            "Falls risk assessment mandatory for all patients 65+ (AGS 2023)",
            "Beers Criteria medications (anticholinergics, benzodiazepines) increase readmission risk",
            "Delirium post-discharge affects 15% of elderly patients and doubles readmission risk",
            "Cognitive assessment (MMSE/MoCA) identifies patients needing extra support",
            "Occupational therapy assessment reduces readmission by 20% in elderly",
            "Home safety evaluation reduces falls and emergency readmissions by 25%",
        ],
        "red_flags": [
            "Confusion or memory problems",
            "Falls history in past 12 months",
            "Multiple high-risk medications (Beers list)",
            "Functional decline noted during admission",
        ],
        "readmission_risk_reduction": "Comprehensive geriatric assessment reduces 30-day readmission by 22%",
    },
}


# ── 1. RETRIEVE RELEVANT GUIDELINES ──────────────────────────────────────────

def get_relevant_guidelines(patient_features: dict) -> dict:
    """
    Retrieves relevant clinical guidelines based on patient profile.
    """
    relevant = {}

    # Always include general readmission guidelines
    relevant["general_readmission"] = GUIDELINES["general_readmission"]

    # Cardiac
    if patient_features.get("cardiac_primary"):
        relevant["cardiac"] = GUIDELINES["cardiac"]

    # Diabetes
    if patient_features.get("diabetes_primary"):
        relevant["diabetes"] = GUIDELINES["diabetes"]

    # Polypharmacy / medication
    if patient_features.get("polypharmacy") or patient_features.get("num_medications", 0) >= 5:
        relevant["medication"] = GUIDELINES["medication"]

    # Social isolation
    if patient_features.get("social_isolation_score", 0) >= 2:
        relevant["social"] = GUIDELINES["social"]

    # Elderly
    if patient_features.get("is_elderly") or patient_features.get("age_numeric", 0) >= 65:
        relevant["elderly"] = GUIDELINES["elderly"]

    return relevant


# ── 2. APPLY GUIDELINES WITH CLAUDE ──────────────────────────────────────────

def apply_guidelines_to_patient(patient_features: dict, risk_level: str, patient_id: str = "unknown") -> dict:
    """
    Uses Claude to apply relevant clinical guidelines to this specific patient.
    Returns evidence-based recommendations tailored to the patient.
    """
    guidelines = get_relevant_guidelines(patient_features)

    # Build guidelines text
    guidelines_text = ""
    for key, g in guidelines.items():
        guidelines_text += f"\n### {g['condition']} ({g['source']})\n"
        for point in g["key_points"][:3]:
            guidelines_text += f"- {point}\n"
        guidelines_text += f"Evidence: {g['readmission_risk_reduction']}\n"

    patient_summary = f"""
Patient Profile:
- Age: {patient_features.get('age_numeric', 'unknown')}
- Risk Level: {risk_level}
- Cardiac diagnosis: {'Yes' if patient_features.get('cardiac_primary') else 'No'}
- Diabetes: {'Yes' if patient_features.get('diabetes_primary') else 'No'}
- Medications: {patient_features.get('num_medications', 0)}
- Prior admissions: {patient_features.get('number_inpatient', 0)}
- Elderly: {'Yes' if patient_features.get('is_elderly') else 'No'}
- High-risk discharge: {'Yes' if patient_features.get('high_risk_discharge') else 'No'}
"""

    prompt = f"""You are a clinical knowledge expert applying evidence-based guidelines.

{patient_summary}

Relevant Clinical Guidelines:
{guidelines_text}

Based on these guidelines, provide specific recommendations for THIS patient in JSON:
{{
    "applicable_guidelines": ["guideline source 1", "guideline source 2"],
    "evidence_based_recommendations": [
        {{"recommendation": "specific action", "source": "guideline source", "evidence_strength": "Strong/Moderate/Weak"}},
        {{"recommendation": "specific action", "source": "guideline source", "evidence_strength": "Strong/Moderate/Weak"}},
        {{"recommendation": "specific action", "source": "guideline source", "evidence_strength": "Strong/Moderate/Weak"}}
    ],
    "expected_readmission_reduction": "percentage range if recommendations followed",
    "priority_intervention": "single most impactful intervention for this patient",
    "clinical_note": "one sentence summary for clinical documentation"
}}

Return ONLY the JSON."""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1000,
            messages   = [{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text.strip())
        result["raw_guidelines"] = {k: v["key_points"][:2] for k, v in guidelines.items()}
        return result
    except Exception as e:
        # Return raw guidelines if Claude fails
        return {
            "applicable_guidelines": list(guidelines.keys()),
            "raw_guidelines"       : {k: v["key_points"][:3] for k, v in guidelines.items()},
            "error"                : str(e),
        }


# ── 3. GET RED FLAGS FOR PATIENT ──────────────────────────────────────────────

def get_red_flags(patient_features: dict) -> list:
    """
    Returns all applicable red flags from guidelines for this patient.
    """
    guidelines = get_relevant_guidelines(patient_features)
    all_flags  = []
    for g in guidelines.values():
        all_flags.extend(g.get("red_flags", []))
    return list(set(all_flags))