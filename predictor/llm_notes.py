import os
import json
import logging
from dotenv import load_dotenv
import anthropic

load_dotenv()
logger     = logging.getLogger(__name__)
client     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL      = "claude-sonnet-4-20250514"


# ── 1. EXTRACT RISK FEATURES FROM CLINICAL NOTES ─────────────────────────────

def analyze_clinical_notes(notes: str) -> dict:
    """
    Sends discharge notes to Claude and extracts structured risk features.
    Returns a dict of risk signals that can be merged with ML features.
    """
    prompt = f"""You are a clinical AI assistant analyzing hospital discharge notes to identify readmission risk factors.

Analyze the following discharge notes and extract risk signals. Return ONLY a valid JSON object with these exact fields:

{{
    "social_isolation_score": 0-3 (0=none, 1=mild, 2=moderate, 3=severe),
    "medication_noncompliance_risk": 0-2 (0=low, 1=medium, 2=high),
    "mental_health_flag": 0 or 1,
    "caregiver_support": 0-2 (0=none, 1=partial, 2=full),
    "follow_up_arranged": 0 or 1,
    "substance_abuse_flag": 0 or 1,
    "fall_risk": 0 or 1,
    "cognitive_impairment": 0 or 1,
    "financial_stress": 0 or 1,
    "transport_barrier": 0 or 1,
    "overall_psychosocial_risk": 0-10,
    "key_risk_factors": ["list", "of", "main", "risks"],
    "recommended_interventions": ["list", "of", "interventions"],
    "summary": "one sentence summary of readmission risk"
}}

Discharge Notes:
{notes}

Return ONLY the JSON object. No explanation, no markdown, no extra text."""

    response = client.messages.create(
        model      = MODEL,
        max_tokens = 1000,
        messages   = [{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    try:
        result = json.loads(raw)
        logger.info(f"LLM analysis complete. Risk score: {result.get('overall_psychosocial_risk')}")
        return result
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response: {raw}")
        return get_default_analysis()


# ── 2. MERGE LLM FEATURES WITH ML FEATURES ───────────────────────────────────

def merge_llm_features(ml_features: dict, llm_features: dict) -> dict:
    """
    Merges structured ML features with LLM-extracted psychosocial features.
    """
    merged = ml_features.copy()

    merged["social_isolation_score"]      = llm_features.get("social_isolation_score", 0)
    merged["medication_noncompliance"]     = llm_features.get("medication_noncompliance_risk", 0)
    merged["mental_health_flag"]           = llm_features.get("mental_health_flag", 0)
    merged["caregiver_support"]            = llm_features.get("caregiver_support", 2)
    merged["follow_up_arranged"]           = llm_features.get("follow_up_arranged", 0)
    merged["psychosocial_risk_score"]      = llm_features.get("overall_psychosocial_risk", 0)

    # Boost complexity score based on psychosocial risk
    merged["complexity_score"] = round(
        merged.get("complexity_score", 0) +
        llm_features.get("overall_psychosocial_risk", 0) * 0.5, 2
    )

    return merged


# ── 3. FULL ANALYSIS PIPELINE ─────────────────────────────────────────────────

def analyze_patient_notes(notes: str, existing_features: dict = None) -> dict:
    """
    Full pipeline: analyze notes + merge with existing ML features.
    """
    llm_analysis = analyze_clinical_notes(notes)

    if existing_features:
        merged = merge_llm_features(existing_features, llm_analysis)
    else:
        merged = {}

    return {
        "llm_analysis"   : llm_analysis,
        "merged_features": merged,
    }


# ── 4. DEFAULT FALLBACK ───────────────────────────────────────────────────────

def get_default_analysis() -> dict:
    return {
        "social_isolation_score"      : 0,
        "medication_noncompliance_risk": 0,
        "mental_health_flag"           : 0,
        "caregiver_support"            : 2,
        "follow_up_arranged"           : 0,
        "substance_abuse_flag"         : 0,
        "fall_risk"                    : 0,
        "cognitive_impairment"         : 0,
        "financial_stress"             : 0,
        "transport_barrier"            : 0,
        "overall_psychosocial_risk"    : 0,
        "key_risk_factors"             : [],
        "recommended_interventions"    : [],
        "summary"                      : "Unable to analyze notes",
    }