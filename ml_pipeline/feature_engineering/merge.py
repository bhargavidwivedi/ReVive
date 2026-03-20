import pandas as pd
import numpy as np
import os

RAW  = "data/raw/archive (8)"
OUT  = "data"


# ── 1. LOAD ALL FILES ─────────────────────────────────────────────────────────

def load_all():
    appointments = pd.read_csv(f"{RAW}/appointments.csv")
    billing      = pd.read_csv(f"{RAW}/billing.csv")
    doctors      = pd.read_csv(f"{RAW}/doctors.csv")
    patients     = pd.read_csv(f"{RAW}/patients.csv")
    treatments   = pd.read_csv(f"{RAW}/treatments.csv")

    print("Loaded:")
    for name, df in [("appointments", appointments), ("billing", billing),
                     ("doctors", doctors), ("patients", patients),
                     ("treatments", treatments)]:
        print(f"  {name:<15} {df.shape[0]:>4} rows × {df.shape[1]} cols")

    return appointments, billing, doctors, patients, treatments


# ── 2. MERGE PIPELINE ─────────────────────────────────────────────────────────

def merge_datasets(appointments, billing, doctors, patients, treatments):

    # Step 1: appointments + doctors → get doctor info per appointment
    df = appointments.merge(
        doctors[["doctor_id", "specialization", "years_experience", "hospital_branch"]],
        on="doctor_id", how="left"
    )
    print(f"\nAfter appointments + doctors : {df.shape}")

    # Step 2: + patients → get patient demographics
    df = df.merge(
        patients[["patient_id", "gender", "date_of_birth",
                  "insurance_provider", "registration_date"]],
        on="patient_id", how="left"
    )
    print(f"After + patients             : {df.shape}")

    # Step 3: + billing → get financial info per appointment
    # billing links via patient_id (some patients have multiple bills)
    billing_agg = billing.groupby("patient_id").agg(
        total_billed       = ("amount", "sum"),
        avg_bill           = ("amount", "mean"),
        num_bills          = ("amount", "count"),
        unpaid_bills       = ("payment_status", lambda x: (x == "Pending").sum()),
        insurance_used     = ("payment_method", lambda x: (x == "Insurance").sum()),
    ).reset_index()

    df = df.merge(billing_agg, on="patient_id", how="left")
    print(f"After + billing              : {df.shape}")

    # Step 4: + treatments → get treatment info
    treatments_agg = treatments.groupby("appointment_id").agg(
        num_treatments       = ("treatment_id", "count"),
        total_treatment_cost = ("cost", "sum"),
    ).reset_index()

    df = df.merge(treatments_agg, on="appointment_id", how="left")
    print(f"After + treatments           : {df.shape}")

    return df


# ── 3. FEATURE ENGINEERING ON MERGED DATA ────────────────────────────────────

def engineer_merged_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Date features ---
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
    df["date_of_birth"]    = pd.to_datetime(df["date_of_birth"],    errors="coerce")

    df["appt_month"]       = df["appointment_date"].dt.month
    df["appt_day_of_week"] = df["appointment_date"].dt.dayofweek
    df["appt_is_weekend"]  = df["appt_day_of_week"].isin([5, 6]).astype(int)

    # Patient age from DOB
    ref_date = pd.Timestamp("2024-01-01")
    df["patient_age"] = ((ref_date - df["date_of_birth"]).dt.days / 365.25).round(1)
    df["is_elderly"]  = (df["patient_age"] >= 65).astype(int)

    # --- Appointment risk features ---
    df["is_noshow"]        = (df["status"] == "No-show").astype(int)
    df["is_cancelled"]     = (df["status"] == "Cancelled").astype(int)
    df["appointment_risk"] = df["is_noshow"] + df["is_cancelled"]

    # --- Doctor experience risk ---
    df["junior_doctor"]    = (df["years_experience"] < 5).astype(int)
    df["senior_doctor"]    = (df["years_experience"] >= 10).astype(int)

    # --- Billing risk features ---
    df["has_unpaid_bills"]    = (df["unpaid_bills"] > 0).astype(int)
    df["high_bill_amount"]    = (df["total_billed"] > df["total_billed"].median()).astype(int)
    df["insurance_dependent"] = (df["insurance_used"] == df["num_bills"]).astype(int)
    df["bill_per_treatment"]  = (
        df["total_billed"] / (df["num_treatments"] + 1)
    ).round(2)

    # --- Gender encode ---
    df["gender_male"] = (df["gender"] == "Male").astype(int)

    # --- Specialization risk (cardiology/internal medicine = higher risk) ---
    high_risk_specs = ["Cardiology", "Internal Medicine", "Nephrology", "Pulmonology"]
    df["high_risk_specialty"] = df["specialization"].isin(high_risk_specs).astype(int)

    # --- Hospital branch encode ---
    df = pd.get_dummies(df, columns=["specialization", "hospital_branch",
                                      "insurance_provider"],
                        drop_first=True)

    return df


# ── 4. CLEAN & SAVE ───────────────────────────────────────────────────────────

def clean_and_save(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns not useful for ML
    drop_cols = [
        "appointment_id", "doctor_id", "appointment_date", "appointment_time",
        "reason_for_visit", "status", "date_of_birth", "registration_date",
        "gender"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill missing numerics
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Save
    os.makedirs(OUT, exist_ok=True)
    out_path = f"{OUT}/merged_hospital_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved merged dataset → {out_path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── 5. SUMMARY ────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n── Feature Summary ───────────────────────────────────────")
    print(f"  Total patients    : {df['patient_id'].nunique()}")
    print(f"  Total features    : {df.shape[1] - 1}")
    print(f"  No-show rate      : {df['is_noshow'].mean():.2%}")
    print(f"  Unpaid bills rate : {df['has_unpaid_bills'].mean():.2%}")
    print(f"  Elderly patients  : {df['is_elderly'].mean():.2%}")
    print(f"\n  Columns:")
    for col in df.columns:
        print(f"    {col}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load
    appointments, billing, doctors, patients, treatments = load_all()

    # Merge
    merged = merge_datasets(appointments, billing, doctors, patients, treatments)

    # Engineer features
    print("\n── Engineering features ──────────────────────────────────")
    merged = engineer_merged_features(merged)

    # Clean & save
    merged = clean_and_save(merged)

    # Summary
    print_summary(merged)