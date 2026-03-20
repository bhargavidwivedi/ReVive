import pandas as pd
import numpy as np


# ── 1. LOAD ──────────────────────────────────────────────────────────────────

def load_data(path: str = "data/raw/readmission clinical datset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 2. TARGET ENCODING ───────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    readmitted values: 'NO', '>30', '<30'
    We predict 30-day readmission → only '<30' is a positive case.
    """
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] != "NO").astype(int)
    print(f"Target distribution:\n{df['readmitted_30d'].value_counts()}")
    return df


# ── 3. CLEAN RAW COLUMNS ─────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicates (keep first encounter per patient)
    df = df.drop_duplicates(subset="encounter_id")

    # Replace '?' with NaN (common in this dataset)
    df.replace("?", np.nan, inplace=True)

    # Weight has ~97% missing — drop it
    df.drop(columns=["weight", "ndc_code", "payer_code"], inplace=True)

    # Drop the original target (we have readmitted_30d now)
    df.drop(columns=["readmitted"], inplace=True)

    return df


# ── 4. AGE FEATURE ───────────────────────────────────────────────────────────

def process_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age comes as a bracket string e.g. '[70-80)' → convert to numeric midpoint.
    """
    df = df.copy()
    age_map = {
        "[0-10)"  : 5,  "[10-20)": 15, "[20-30)": 25,
        "[30-40)" : 35, "[40-50)": 45, "[50-60)": 55,
        "[60-70)" : 65, "[70-80)": 75, "[80-90)": 85,
        "[90-100)": 95
    }
    df["age_numeric"] = df["age"].map(age_map)
    df["is_elderly"]  = (df["age_numeric"] >= 65).astype(int)
    df["age_group"]   = pd.cut(
        df["age_numeric"],
        bins=[0, 30, 50, 65, 80, 100],
        labels=["<30", "30-50", "50-65", "65-80", "80+"]
    )
    df.drop(columns=["age"], inplace=True)
    return df


# ── 5. PRIOR UTILISATION FEATURES ────────────────────────────────────────────

def utilisation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total prior visits across all channels
    df["total_prior_visits"] = (
        df["number_outpatient"] +
        df["number_inpatient"] +
        df["number_emergency"]
    )

    # High utiliser flag (top quartile)
    threshold = df["total_prior_visits"].quantile(0.75)
    df["high_utiliser"] = (df["total_prior_visits"] >= threshold).astype(int)

    # Emergency-heavy patient flag
    df["emergency_prone"] = (df["number_emergency"] >= 2).astype(int)

    # Inpatient history flag
    df["has_inpatient_history"] = (df["number_inpatient"] >= 1).astype(int)

    return df


# ── 6. CLINICAL COMPLEXITY FEATURES ──────────────────────────────────────────

def clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Polypharmacy flag (≥5 medications is standard clinical threshold)
    df["polypharmacy"] = (df["num_medications"] >= 5).astype(int)

    # High diagnosis burden
    df["high_diagnosis_burden"] = (df["number_diagnoses"] >= 5).astype(int)

    # Lab intensity (many labs = complex case)
    df["high_lab_use"] = (
        df["num_lab_procedures"] > df["num_lab_procedures"].median()
    ).astype(int)

    # Procedure count risk
    df["had_procedures"] = (df["num_procedures"] > 0).astype(int)

    # Comorbidity proxy score
    df["complexity_score"] = (
        df["number_diagnoses"] +
        df["num_medications"] * 0.5 +
        df["num_lab_procedures"] * 0.2
    ).round(2)

    return df


# ── 7. INTERACTION FEATURES ───────────────────────────────────────────────────

def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Long stay + many diagnoses = high risk
    df["los_x_diagnoses"]   = df["time_in_hospital"] * df["number_diagnoses"]

    # Elderly + high complexity
    df["elderly_x_complex"] = df["is_elderly"] * df["complexity_score"]

    # Prior inpatient + emergency combo
    df["inpatient_x_emergency"] = (
        df["number_inpatient"] * df["number_emergency"]
    )

    # High medication + long stay
    df["meds_x_los"] = df["num_medications"] * df["time_in_hospital"]

    return df


# ── 8. GLUCOSE & HbA1c FEATURES ──────────────────────────────────────────────

def glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    max_glu_serum: None / Norm / >200 / >300
    A1Cresult    : None / Norm / >7   / >8
    """
    df = df.copy()

    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    a1c_map = {"None": 0, "Norm": 1, ">7"  : 2, ">8"  : 3}

    df["glu_serum_num"] = df["max_glu_serum"].map(glu_map).fillna(0)
    df["a1c_num"]       = df["A1Cresult"].map(a1c_map).fillna(0)

    # Combined metabolic risk
    df["metabolic_risk"] = df["glu_serum_num"] + df["a1c_num"]
    df["uncontrolled_diabetes"] = (df["metabolic_risk"] >= 4).astype(int)

    df.drop(columns=["max_glu_serum", "A1Cresult"], inplace=True)
    return df


# ── 9. DISCHARGE DISPOSITION RISK ────────────────────────────────────────────

def discharge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    discharge_disposition_id:
      1  = Home (low risk)
      2-5 = Transfers (medium)
      6  = Home with health service (high)
      11 = Expired
      13,14 = Hospice (exclude from prediction)
    """
    df = df.copy()

    high_risk_ids = [2, 3, 4, 5, 6, 8, 9, 15, 16, 17]
    df["high_risk_discharge"] = (
        df["discharge_disposition_id"].isin(high_risk_ids)
    ).astype(int)

    # Remove deceased / hospice patients (not relevant for readmission)
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]

    return df


# ── 10. DIAGNOSIS CODE FEATURES ──────────────────────────────────────────────

def diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map primary_diagnosis_code ICD-9 ranges to disease categories.
    """
    df = df.copy()

    def categorise_diag(code):
        if pd.isna(code):
            return "Unknown"
        code = str(code)
        if code.startswith("V") or code.startswith("E"):
            return "External"
        try:
            c = float(code)
            if 390 <= c <= 459 or c == 785: return "Circulatory"
            if 460 <= c <= 519 or c == 786: return "Respiratory"
            if 520 <= c <= 579 or c == 787: return "Digestive"
            if 250 <= c <= 250.99:          return "Diabetes"
            if 800 <= c <= 999:             return "Injury"
            if 710 <= c <= 739:             return "Musculoskeletal"
            if 580 <= c <= 629 or c == 788: return "Genitourinary"
            if 140 <= c <= 239:             return "Neoplasms"
            return "Other"
        except ValueError:
            return "Unknown"

    df["primary_diag_category"] = df["primary_diagnosis_code"].apply(categorise_diag)

    # Diabetes as primary diagnosis flag
    df["diabetes_primary"] = (df["primary_diag_category"] == "Diabetes").astype(int)

    # Circulatory / cardiac risk
    df["cardiac_primary"] = (df["primary_diag_category"] == "Circulatory").astype(int)

    df.drop(columns=["primary_diagnosis_code", "other_diagnosis_codes"], inplace=True)

    return df


# ── 11. ENCODE CATEGORICALS ───────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary encode gender
    df["gender_male"] = (df["gender"] == "Male").astype(int)
    df.drop(columns=["gender"], inplace=True)

    # Medication change flag (already binary-ish)
    df["medication_changed"] = (df["change"] == "Ch").astype(int)
    df.drop(columns=["change"], inplace=True)

    # One-hot encode remaining categoricals
    cat_cols = ["race", "medical_specialty", "primary_diag_category", "age_group"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


# ── 12. DROP ID COLUMNS ───────────────────────────────────────────────────────

def drop_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    id_cols = ["encounter_id", "patient_nbr",
               "admission_type_id", "admission_source_id",
               "discharge_disposition_id"]
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)
    return df


# ── 13. FULL PIPELINE ─────────────────────────────────────────────────────────

def run_pipeline(path: str = "data/raw/readmission clinical datset.csv") -> pd.DataFrame:
    print("── Loading data ──────────────────────────────")
    df = load_data(path)

    print("── Encoding target ───────────────────────────")
    df = encode_target(df)

    print("── Cleaning ──────────────────────────────────")
    df = clean_data(df)

    print("── Age features ──────────────────────────────")
    df = process_age(df)

    print("── Utilisation features ──────────────────────")
    df = utilisation_features(df)

    print("── Clinical features ─────────────────────────")
    df = clinical_features(df)

    print("── Glucose features ──────────────────────────")
    df = glucose_features(df)

    print("── Discharge features ────────────────────────")
    df = discharge_features(df)

    print("── Diagnosis features ────────────────────────")
    df = diagnosis_features(df)

    print("── Interaction features ──────────────────────")
    df = interaction_features(df)

    print("── Encoding categoricals ─────────────────────")
    df = encode_categoricals(df)

    print("── Dropping IDs ──────────────────────────────")
    df = drop_ids(df)

    print(f"\n✅ Pipeline complete: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Positive rate (30d readmission): {df['readmitted_30d'].mean():.2%}")

    # Save processed data locally
    df.to_csv("data/processed_features.csv", index=False)
    print("   Saved → data/processed_features.csv")

    return df


if __name__ == "__main__":
    df = run_pipeline()
    print("\nFinal columns:")
    print(df.columns.tolist())