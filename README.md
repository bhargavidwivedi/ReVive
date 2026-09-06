# ReVive — Intelligent Readmission Risk Prediction System

ReVive predicts a patient's risk of **hospital readmission within 30 days** from
structured clinical features, and exposes that prediction through a Django REST
API with a Streamlit dashboard on top.

> **Model transparency note:** The served model is a scikit-learn
> **`GradientBoostingClassifier`**. Some legacy filenames and comments in the
> repository history referred to "LightGBM"; the actual artifact used for
> serving is scikit-learn gradient boosting, saved as
> `ml_pipeline/models/saved/readmission_model.pkl`. This README documents the
> real, verified behaviour of the code as it stands.

---

## Overview

Unplanned 30-day readmissions are costly and often preventable. ReVive scores a
patient at (or before) discharge and returns a probability, a risk band, and
simple follow-up recommendations, so a care team can prioritise who needs extra
attention.

## Problem Statement

Given a patient's encounter data (diagnoses, procedures, medications, prior
utilisation, demographics, etc.), estimate the probability that the patient will
be readmitted within 30 days, and turn that probability into an actionable risk
level.

## Solution

A trained gradient-boosting classifier serves predictions through a Django REST
Framework API. A Streamlit dashboard and a Django-served HTML page provide simple
front-ends that call the API. The machine-learning pipeline (feature
engineering, training, tuning, evaluation) lives under `ml_pipeline/` and is
reproducible from the processed dataset.

## Key Features

- REST API for single and batch readmission-risk prediction.
- Trained scikit-learn `GradientBoostingClassifier` (121 input features).
- Risk banding (Low / Medium / High) and follow-up recommendations.
- Streamlit dashboard that calls the API.
- Reproducible training/evaluation pipeline with a model leaderboard.
- Optional/experimental modules (FHIR, LLM note analysis, SHAP, drift/retrain) —
  see [Experimental / optional components](#experimental--optional-components).

## Tech Stack

- **Language:** Python 3.11+ (verified on 3.12)
- **ML:** scikit-learn 1.7.2, LightGBM, XGBoost, NumPy, pandas, joblib
- **API:** Django 5, Django REST Framework
- **Dashboard:** Streamlit, Plotly, requests
- **Serving:** Gunicorn, WhiteNoise

## Project Architecture

```
Streamlit dashboard (dashboard.py)  ─┐
Django HTML dashboard (templates/)  ─┤──HTTP──▶  Django REST API (predictor/)
                                     │              │
                                     │              ▼
                                     │     readmission_model.pkl
                                     │   (GradientBoostingClassifier)
                                     │              │
                                     │              ▼
                                     └───────  prediction + risk band
ml_pipeline/  ── feature engineering → training → tuning → evaluation
```

## Project Structure

```
ReVive/
├── core/                     # Django project (settings, urls, wsgi)
├── predictor/                # API app: views, urls, models, apps
│   ├── views.py              #   /api/predict, /api/health, ...
│   ├── apps.py               #   loads the model at startup (no silent retrain)
│   └── (experimental modules: fhir_integration, llm_notes, explainability, ...)
├── ml_pipeline/
│   ├── feature_engineering/  # feature construction
│   ├── models/               # train.py, tune.py, predict.py
│   │   └── saved/            # committed model binaries (normal Git files)
│   └── evaluation/           # metrics.py
├── templates/dashboard.html  # Django-served dashboard
├── dashboard.py              # Streamlit dashboard
├── data/processed_features.csv
├── requirements.txt          # core (pinned, verified)
├── requirements-optional.txt # experimental extras
└── .env.example
```

## Dataset

- **Source:** UCI *Diabetes 130-US hospitals (1999–2008)* readmission dataset.
- **Processed file:** `data/processed_features.csv` — **99,343 rows × 121
  feature columns** plus the target `readmitted_30d`.
- **Target balance:** ~47% positive in the processed file. Note this is a
  **balanced** version of the problem; the natural 30-day readmission rate is
  much lower, so metrics here reflect the balanced dataset (see
  [Known limitations](#known-limitations)).

## Data Preprocessing

Feature engineering lives in `ml_pipeline/feature_engineering/`. The processed
matrix contains 121 numeric features (utilisation counts, diagnosis categories,
medication indicators, demographic encodings, lab summaries, etc.). Missing
features supplied to the API at inference time are defaulted to 0.

## Machine Learning Model

- **Served model:** scikit-learn **`GradientBoostingClassifier`**
  (`ml_pipeline/models/saved/readmission_model.pkl`, ~256 KB).
- **Input:** 121 features (the columns of `processed_features.csv` excluding
  `readmitted_30d`).
- **Decision threshold:** `0.369` (tuned to favour recall — catching
  readmissions — over precision).
- Baseline models (Logistic Regression, Random Forest, XGBoost, LightGBM) are
  also trained by `ml_pipeline/models/train.py` for comparison.

## Model Evaluation

All numbers below were **reproduced locally** from this repository on the
seed-42 stratified hold-out split (19,869 rows). They are not estimated or
invented.

**Served model — `GradientBoostingClassifier` @ threshold 0.369:**

| Metric | Value |
|---|---|
| ROC-AUC | **0.6755** |
| Precision | 0.528 |
| Recall | 0.861 |
| F1 | 0.655 |

Confusion matrix (hold-out): TN 3304 · FP 7202 · FN 1304 · TP 8059 — i.e. the
model catches ~86% of true readmissions at the cost of lower precision, a
deliberate clinical trade-off.

**Model comparison (hold-out ROC-AUC, reproduced via `train.py`):**

| Model | ROC-AUC |
|---|---|
| XGBoost | 0.681 |
| LightGBM | 0.680 |
| Random Forest | 0.670 |
| Logistic Regression | 0.667 |

An AUC around 0.68 is modest but realistic for this dataset and task.

## Application / Dashboard

- **Streamlit** (`dashboard.py`): a form-based UI that posts to the API and
  displays the risk result.
- **Django HTML** (`templates/dashboard.html`): a server-rendered dashboard at
  `/` that also calls the API.

Both are thin clients over the REST API. (No screenshots are included; add real
ones after running locally if desired.)

## API

Base path: `/api/`

| Method | Endpoint | Purpose |
|---|---|---|
| GET  | `/api/health/` | Service + model status |
| POST | `/api/predict/` | Single-patient prediction |
| POST | `/api/predict/batch/` | Batch prediction |

Additional endpoints (FHIR, analyze-notes, care-pathway, drift, retrain,
system-health, stats, logs) belong to the experimental layer and are **not
verified** — see below.

### Example — single prediction

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"patient_data": {"time_in_hospital": 5, "num_medications": 15,
        "number_diagnoses": 9, "num_lab_procedures": 40, "age_numeric": 70}}'
```

Any of the 121 features may be provided; omitted features default to 0.

**Verified response (200):**

```json
{
  "readmission_probability": 0.3328,
  "readmission_percentage": "33.3%",
  "predicted_readmission": false,
  "risk_level": "Medium",
  "recommendations": ["Schedule follow-up within 7 days of discharge", "..."]
}
```

## Installation

Requires **Python 3.11+** and the pinned dependencies (note **scikit-learn
1.7.2** — the saved model requires it to deserialize).

```bash
git clone <your-repo-url> ReVive
cd ReVive

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional experimental features:
# pip install -r requirements-optional.txt

cp .env.example .env      # then fill in values as needed
```

> **Model files are committed as normal Git files (not Git LFS)**, so a plain
> `git clone` gives you the real, usable binaries — no `git lfs pull` required.

## How to Run

### Django API

```bash
python manage.py migrate
python manage.py runserver
# API now at http://127.0.0.1:8000/api/
```

### Streamlit dashboard

Run the API (above) first, then in a second terminal:

```bash
streamlit run dashboard.py
```

## Model-loading verification

Confirm the model loaded successfully:

```bash
# Option A — health endpoint (server running)
curl http://127.0.0.1:8000/api/health/
# expect: "model_loaded": true, "model": "GradientBoostingClassifier", "features": 121

# Option B — direct load check
python -c "import joblib; m=joblib.load('ml_pipeline/models/saved/readmission_model.pkl'); print(type(m).__name__)"
# expect: GradientBoostingClassifier
```

If loading fails with `No module named '_loss'`, your scikit-learn version is
wrong — install `scikit-learn==1.7.2`.

## Model files

Committed as normal Git objects under `ml_pipeline/models/saved/`:

| File | Size | Type |
|---|---|---|
| `readmission_model.pkl` (**served**) | 256 KB | GradientBoostingClassifier |
| `LightGBM.pkl` | 983 KB | LGBMClassifier |
| `XGBoost.pkl` | 757 KB | XGBClassifier |
| `RandomForest.pkl` | 9.8 MB | RandomForestClassifier |
| `LogisticRegression.pkl` | 1.8 KB | LogisticRegression |

`scaler.pkl` is **not** committed: it is only used by evaluation and the
Logistic Regression baseline, not by the serving path, and is regenerated by
`ml_pipeline/models/train.py`.

## Known limitations

- **Balanced dataset:** metrics reflect the ~47%-positive processed file, not the
  lower natural readmission base rate.
- **Dashboard feature coverage:** the interactive dashboard collects a subset of
  the 121 features; unspecified features are sent as 0, so dashboard predictions
  are approximate compared with a fully-populated feature vector.
- **scikit-learn version:** the served model requires scikit-learn 1.7.2 to
  deserialize. All four baseline `.pkl` files load successfully under this
  pinned version; they are reference artifacts and can also be regenerated with
  `python ml_pipeline/models/train.py`.
- **No verified deployment:** there is no verified live/hosted deployment; run
  locally as documented.

## Experimental / optional components

The following modules exist in the codebase but are **not part of the verified
core** and were **not validated end-to-end**. They are imported lazily, so the
core API runs without them. Each requires extra dependencies
(`requirements-optional.txt`) and, in some cases, external services or API keys:

- **LLM clinical-note analysis** (`predictor/llm_notes.py`) — requires
  `ANTHROPIC_API_KEY`.
- **SHAP explainability** (`predictor/explainability.py`).
- **FHIR integration** (`predictor/fhir_integration.py`) — requires a FHIR server.
- **Voice-note transcription** (`predictor/voice_notes.py`) — requires Whisper.
- **Care-pathway assignment, risk timeline, knowledge base.**
- **Drift detection / retraining and Celery background scoring**
  (`predictor/continuous_learning.py`, `tasks.py`) — require Redis + Celery.

These are presented as experimental work, not as production-ready features.

## Author

**Bhargavi Dwivedi** — Integrated M.Tech (AI/ML), VIT Bhopal.
