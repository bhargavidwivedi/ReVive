import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/api"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReVive — Readmission Risk",
    page_icon="🏥",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏥 ReVive — Hospital Readmission Risk Predictor")
st.markdown("Predict 30-day readmission risk using AI. Enter patient details below.")
st.divider()

# ── Sidebar — Patient Input ───────────────────────────────────────────────────
st.sidebar.markdown("## 👤 Patient Details")

age         = st.sidebar.slider("Age",                    0,   100, 65)
los         = st.sidebar.slider("Length of stay (days)",  1,    30,  5)
n_inpatient = st.sidebar.slider("Prior inpatient visits", 0,    10,  2)
n_diagnoses = st.sidebar.slider("Number of diagnoses",    1,    20,  7)
n_meds      = st.sidebar.slider("Number of medications",  0,    40, 12)
n_labs      = st.sidebar.slider("Number of lab tests",    0,   100, 40)
n_procs     = st.sidebar.slider("Number of procedures",   0,    10,  2)
n_outpat    = st.sidebar.slider("Prior outpatient visits",0,    10,  1)
n_emergency = st.sidebar.slider("Prior emergency visits", 0,    10,  1)

st.sidebar.divider()
st.sidebar.markdown("## 🏥 Clinical Flags")
is_elderly        = st.sidebar.checkbox("Elderly (65+)",           value=age >= 65)
polypharmacy      = st.sidebar.checkbox("Polypharmacy (5+ meds)",  value=n_meds >= 5)
cardiac_primary   = st.sidebar.checkbox("Cardiac primary diagnosis")
diabetes_primary  = st.sidebar.checkbox("Diabetes primary diagnosis")
high_risk_dc      = st.sidebar.checkbox("High-risk discharge disposition")
medication_changed= st.sidebar.checkbox("Medication changed during stay")

# ── Build patient dict ────────────────────────────────────────────────────────
patient = {
    "age_numeric"          : age,
    "time_in_hospital"     : los,
    "number_inpatient"     : n_inpatient,
    "number_diagnoses"     : n_diagnoses,
    "num_medications"      : n_meds,
    "num_lab_procedures"   : n_labs,
    "num_procedures"       : n_procs,
    "number_outpatient"    : n_outpat,
    "number_emergency"     : n_emergency,
    "total_prior_visits"   : n_inpatient + n_outpat + n_emergency,
    "is_elderly"           : int(is_elderly),
    "polypharmacy"         : int(polypharmacy),
    "cardiac_primary"      : int(cardiac_primary),
    "diabetes_primary"     : int(diabetes_primary),
    "high_risk_discharge"  : int(high_risk_dc),
    "medication_changed"   : int(medication_changed),
    "complexity_score"     : round(n_diagnoses + n_meds * 0.5 + n_labs * 0.2, 2),
    "los_x_diagnoses"      : los * n_diagnoses,
    "meds_x_los"           : n_meds * los,
    "elderly_x_complex"    : int(is_elderly) * (n_diagnoses + n_meds * 0.5),
    "inpatient_x_emergency": n_inpatient * n_emergency,
    "high_diagnosis_burden": int(n_diagnoses >= 5),
    "high_lab_use"         : int(n_labs > 40),
    "had_procedures"       : int(n_procs > 0),
    "has_inpatient_history": int(n_inpatient >= 1),
    "emergency_prone"      : int(n_emergency >= 2),
    "high_utiliser"        : int((n_inpatient + n_outpat + n_emergency) >= 3),
}

# ── Predict button ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_btn = st.button("🔍 Predict Readmission Risk", use_container_width=True)

st.divider()

if predict_btn:
    try:
        resp = requests.post(
            f"{API_URL}/predict/",
            json={"patient_data": patient},
            timeout=10
        )
        result = resp.json()

        prob       = result["readmission_probability"]
        risk_level = result["risk_level"]
        recs       = result["recommendations"]
        pct        = result["readmission_percentage"]

        # ── Risk colour ───────────────────────────────────────────────────────
        colour = {"High": "#E24B4A", "Medium": "#EF9F27", "Low": "#1D9E75"}[risk_level]
        emoji  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk_level]

        # ── Top metrics ───────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Readmission Probability", pct)
        m2.metric("Risk Level", f"{emoji} {risk_level}")
        m3.metric("Predicted Readmission", "YES ⚠️" if result["predicted_readmission"] else "NO ✅")

        st.divider()

        # ── Gauge chart ───────────────────────────────────────────────────────
        c1, c2 = st.columns([1, 1])

        with c1:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = round(prob * 100, 1),
                title = {"text": "Readmission Risk %"},
                gauge = {
                    "axis" : {"range": [0, 100]},
                    "bar"  : {"color": colour},
                    "steps": [
                        {"range": [0,  30], "color": "#E1F5EE"},
                        {"range": [30, 60], "color": "#FAEEDA"},
                        {"range": [60, 100],"color": "#FCEBEB"},
                    ],
                    "threshold": {
                        "line" : {"color": colour, "width": 4},
                        "thickness": 0.75,
                        "value": round(prob * 100, 1)
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # ── Recommendations ───────────────────────────────────────────────────
        with c2:
            st.markdown(f"### {emoji} Clinical Recommendations")
            for rec in recs:
                st.markdown(f"- {rec}")

            st.markdown("### 📊 Key Risk Factors")
            risk_factors = {
                "Prior inpatient visits" : n_inpatient,
                "Number of diagnoses"    : n_diagnoses,
                "Medications"            : n_meds,
                "Length of stay"         : los,
                "Emergency visits"       : n_emergency,
            }
            for factor, value in risk_factors.items():
                st.markdown(f"- **{factor}:** {value}")

        st.divider()

        # ── Risk summary bar ──────────────────────────────────────────────────
        st.markdown("### 🎯 Risk Breakdown")
        fig2 = go.Figure(go.Bar(
            x    = list(risk_factors.values()),
            y    = list(risk_factors.keys()),
            orientation = "h",
            marker_color = colour,
        ))
        fig2.update_layout(height=250, margin=dict(t=20, b=0),
                           xaxis_title="Value", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"API Error: {e}. Make sure the Django server is running!")

else:
    st.info("👈 Fill in patient details in the sidebar and click **Predict Readmission Risk**")

    # ── API status check ──────────────────────────────────────────────────────
    st.markdown("### 🔌 API Status")
    try:
        h = requests.get(f"{API_URL}/health/", timeout=3)
        info = h.json()
        st.success(f"✅ API is live | Model: {info['model']} | AUC: {info['auc']}")
    except:
        st.error("❌ API is offline — start Django server first!")