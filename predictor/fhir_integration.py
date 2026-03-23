import requests
import logging

logger = logging.getLogger(__name__)

# ── HAPI FHIR Public Sandbox ──────────────────────────────────────────────────
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"


# ── 1. FETCH PATIENT ──────────────────────────────────────────────────────────

def fetch_patient(patient_id: str) -> dict:
    url  = f"{FHIR_BASE_URL}/Patient/{patient_id}"
    resp = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)

    if resp.status_code != 200:
        raise ValueError(f"Patient {patient_id} not found. Status: {resp.status_code}")

    data      = resp.json()
    name      = data.get("name", [{}])[0]
    full_name = " ".join(name.get("given", []) + [name.get("family", "")])
    gender    = data.get("gender", "unknown")
    dob       = data.get("birthDate", "")

    age = 0
    if dob:
        from datetime import date
        birth = date.fromisoformat(dob)
        age   = (date.today() - birth).days // 365

    return {
        "patient_id": patient_id,
        "full_name" : full_name.strip(),
        "gender"    : gender,
        "birth_date": dob,
        "age"       : age,
    }


# ── 2. FETCH CONDITIONS ───────────────────────────────────────────────────────

def fetch_conditions(patient_id: str) -> list:
    url  = f"{FHIR_BASE_URL}/Condition?patient={patient_id}&_count=50"
    resp = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)
    if resp.status_code != 200:
        return []

    conditions = []
    for entry in resp.json().get("entry", []):
        resource = entry.get("resource", {})
        code     = resource.get("code", {})
        codings  = code.get("coding", [{}])
        conditions.append({
            "text": code.get("text", ""),
            "code": codings[0].get("code", "") if codings else ""
        })
    return conditions


# ── 3. FETCH MEDICATIONS ──────────────────────────────────────────────────────

def fetch_medications(patient_id: str) -> list:
    url  = f"{FHIR_BASE_URL}/MedicationRequest?patient={patient_id}&_count=50"
    resp = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)
    if resp.status_code != 200:
        return []

    meds = []
    for entry in resp.json().get("entry", []):
        med  = entry.get("resource", {}).get("medicationCodeableConcept", {})
        meds.append(med.get("text", "Unknown"))
    return meds


# ── 4. FETCH ENCOUNTERS ───────────────────────────────────────────────────────

def fetch_encounters(patient_id: str) -> dict:
    url  = f"{FHIR_BASE_URL}/Encounter?patient={patient_id}&_count=100"
    resp = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)
    if resp.status_code != 200:
        return {"inpatient": 0, "outpatient": 0, "emergency": 0, "total": 0}

    counts = {"inpatient": 0, "outpatient": 0, "emergency": 0, "total": 0}
    entries = resp.json().get("entry", [])
    counts["total"] = len(entries)

    for entry in entries:
        c = entry.get("resource", {}).get("class", {}).get("code", "").lower()
        if "inp" in c or "acute" in c: counts["inpatient"]  += 1
        elif "out" in c:               counts["outpatient"] += 1
        elif "eme" in c:               counts["emergency"]  += 1
    return counts


# ── 5. FETCH LAB RESULTS ──────────────────────────────────────────────────────

def fetch_lab_results(patient_id: str) -> int:
    url  = f"{FHIR_BASE_URL}/Observation?patient={patient_id}&category=laboratory&_count=100"
    resp = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)
    if resp.status_code != 200:
        return 0
    data = resp.json()
    return data.get("total", len(data.get("entry", [])))


# ── 6. BUILD ML FEATURES FROM FHIR ───────────────────────────────────────────

def build_features_from_fhir(patient_id: str, length_of_stay: int = 5) -> dict:
    patient    = fetch_patient(patient_id)
    conditions = fetch_conditions(patient_id)
    meds       = fetch_medications(patient_id)
    encounters = fetch_encounters(patient_id)
    lab_count  = fetch_lab_results(patient_id)

    age          = patient["age"]
    n_conditions = len(conditions)
    n_meds       = len(meds)
    n_inpatient  = encounters["inpatient"]
    n_emergency  = encounters["emergency"]
    total_visits = encounters["total"]

    cardiac_codes  = ["I21","I22","I50","I48","410","428"]
    diabetes_codes = ["E11","E12","E13","250"]

    cardiac_primary  = int(any(any(c in (cond.get("code","") or "") for c in cardiac_codes)  for cond in conditions))
    diabetes_primary = int(any(any(c in (cond.get("code","") or "") for c in diabetes_codes) for cond in conditions))

    features = {
        "age_numeric"          : age,
        "time_in_hospital"     : length_of_stay,
        "number_inpatient"     : n_inpatient,
        "number_outpatient"    : encounters["outpatient"],
        "number_emergency"     : n_emergency,
        "number_diagnoses"     : n_conditions,
        "num_medications"      : n_meds,
        "num_lab_procedures"   : lab_count,
        "num_procedures"       : 0,
        "total_prior_visits"   : total_visits,
        "is_elderly"           : int(age >= 65),
        "polypharmacy"         : int(n_meds >= 5),
        "cardiac_primary"      : cardiac_primary,
        "diabetes_primary"     : diabetes_primary,
        "high_risk_discharge"  : 0,
        "medication_changed"   : 0,
        "complexity_score"     : round(n_conditions + n_meds * 0.5 + lab_count * 0.2, 2),
        "los_x_diagnoses"      : length_of_stay * n_conditions,
        "meds_x_los"           : n_meds * length_of_stay,
        "elderly_x_complex"    : int(age >= 65) * (n_conditions + n_meds * 0.5),
        "inpatient_x_emergency": n_inpatient * n_emergency,
        "high_diagnosis_burden": int(n_conditions >= 5),
        "high_lab_use"         : int(lab_count > 40),
        "had_procedures"       : 0,
        "has_inpatient_history": int(n_inpatient >= 1),
        "emergency_prone"      : int(n_emergency >= 2),
        "high_utiliser"        : int(total_visits >= 3),
    }

    return {
        "patient_info": patient,
        "conditions"  : conditions,
        "medications" : meds,
        "encounters"  : encounters,
        "features"    : features,
    }


# ── 7. SEARCH PATIENTS ────────────────────────────────────────────────────────

def search_patients(name: str = "", count: int = 5) -> list:
    url  = f"{FHIR_BASE_URL}/Patient?_count={count}"
    if name:
        url += f"&name={name}"

    resp = requests.get(url, headers={"Accept": "application/fhir+json"}, timeout=10)
    if resp.status_code != 200:
        return []

    patients = []
    for entry in resp.json().get("entry", []):
        r         = entry.get("resource", {})
        name_data = r.get("name", [{}])[0]
        full      = " ".join(name_data.get("given", []) + [name_data.get("family", "")])
        patients.append({
            "id"    : r.get("id", ""),
            "name"  : full.strip(),
            "gender": r.get("gender", "")
        })
    return patients