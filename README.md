<!-- ════════════════════════════════════════════════════════════ -->
<!--              REVIVE · HOSPITAL READMISSION PREDICTOR         -->
<!-- ════════════════════════════════════════════════════════════ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=C9A84C&height=240&section=header&text=ReVive&fontSize=72&fontColor=000000&fontAlignY=40&desc=Hospital%20Readmission%20Prediction%20System&descSize=18&descAlignY=64&descColor=000000&animation=fadeIn" width="100%"/>

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Georgia&size=17&pause=1200&color=C9A84C&background=00000000&center=true&vCenter=true&width=640&lines=Predicting+30-day+hospital+readmission+risk;Clinical+Decision+Support+powered+by+XGBoost;Django+REST+API+%C2%B7+Streamlit+Frontend;Turning+patient+data+into+life-saving+decisions" />

<br/><br/>

![Status](https://img.shields.io/badge/Status-In%20Progress-C9A84C?style=flat-square&labelColor=000000)
![Python](https://img.shields.io/badge/Python-3.10+-000000?style=flat-square&logo=python&logoColor=C9A84C)
![Django](https://img.shields.io/badge/Django-REST%20API-C9A84C?style=flat-square&logo=django&logoColor=000000)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-000000?style=flat-square&logo=streamlit&logoColor=C9A84C)
![ML](https://img.shields.io/badge/Model-XGBoost-C9A84C?style=flat-square&labelColor=000000)

</div>

---

## 🔴 Problem Statement

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20PROBLEM%20STATEMENT&fontSize=13&fontColor=C9A84C&fontAlign=6&fontAlignY=65" width="100%"/>

<br/>

Hospital readmissions within 30 days of discharge are one of the most critical and costly challenges in modern healthcare.

- 💸 **Cost** — Over **$26 billion** spent annually on preventable readmissions
- ⚠️ **Signal** — Readmissions indicate gaps in post-discharge care or premature discharge
- 🏥 **Impact** — Disproportionately affects patients with diabetes, heart failure & COPD
- 💡 **Solution** — ReVive gives clinicians a real-time risk score per patient, enabling proactive care planning before discharge

---

## ✨ Features

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20FEATURES&fontSize=13&fontColor=C9A84C&fontAlign=5&fontAlignY=65" width="100%"/>

<br/>

- 🔍 **Risk Prediction** — Predicts 30-day readmission probability from clinical & behavioral patient data
- 📊 **Clinician Dashboard** — Interactive Streamlit UI for real-time patient input & risk score display
- ⚙️ **REST API Backend** — Django-powered `/predict` endpoint for scalable real-time inference
- 🚨 **High-Risk Flagging** — Automatically flags patients above risk threshold for early clinical intervention
- 🧹 **Robust Preprocessing** — Handles missing values, encodes categorical variables, scales features automatically
- 📈 **Model Benchmarking** — Logistic Regression, Random Forest & XGBoost compared on AUC-ROC and F1-score

---

## 🛠️ Tech Stack

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20TECH%20STACK&fontSize=13&fontColor=C9A84C&fontAlign=5&fontAlignY=65" width="100%"/>

<br/>

| Layer | Technology |
|:------|:-----------|
| **Language** | ![Python](https://img.shields.io/badge/Python%203.10+-C9A84C?style=flat-square&logo=python&logoColor=000000) |
| **ML Models** | ![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=flat-square&logoColor=C9A84C) ![RandomForest](https://img.shields.io/badge/Random%20Forest-C9A84C?style=flat-square&logoColor=000000) ![LogReg](https://img.shields.io/badge/Logistic%20Regression-000000?style=flat-square&logoColor=C9A84C) |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-C9A84C?style=flat-square&logo=pandas&logoColor=000000) ![NumPy](https://img.shields.io/badge/NumPy-000000?style=flat-square&logo=numpy&logoColor=C9A84C) ![Sklearn](https://img.shields.io/badge/Scikit--learn-C9A84C?style=flat-square&logo=scikit-learn&logoColor=000000) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=flat-square&logoColor=C9A84C) ![Plotly](https://img.shields.io/badge/Plotly-C9A84C?style=flat-square&logo=plotly&logoColor=000000) |
| **Backend** | ![Django](https://img.shields.io/badge/Django%20REST-000000?style=flat-square&logo=django&logoColor=C9A84C) |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-C9A84C?style=flat-square&logo=streamlit&logoColor=000000) |
| **Tools** | ![Git](https://img.shields.io/badge/Git-000000?style=flat-square&logo=git&logoColor=C9A84C) ![Jupyter](https://img.shields.io/badge/Jupyter-C9A84C?style=flat-square&logo=jupyter&logoColor=000000) ![Colab](https://img.shields.io/badge/Colab-000000?style=flat-square&logo=googlecolab&logoColor=C9A84C) |

---

## 🏗️ Project Architecture

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20PROJECT%20ARCHITECTURE&fontSize=13&fontColor=C9A84C&fontAlign=6&fontAlignY=65" width="100%"/>

<br/>

**Folder Structure:**

```
ReVive/
│
├── data/
│   ├── raw/                    ← Raw dataset files
│   └── processed/              ← Cleaned & preprocessed data
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  ← Feature Engineering & Cleaning
│   └── 03_model_training.ipynb ← Model Training & Evaluation
│
├── ml/
│   ├── preprocess.py           ← Data preprocessing pipeline
│   ├── train.py                ← Model training script
│   ├── evaluate.py             ← Evaluation metrics
│   └── model.pkl               ← Saved trained model
│
├── backend/                    ← Django REST API
│   ├── manage.py
│   └── predictor/
│       ├── views.py            ← /predict endpoint logic
│       └── serializers.py
│
├── frontend/                   ← Streamlit UI
│   └── app.py                  ← Clinician dashboard
│
├── requirements.txt
└── README.md
```

**Data Flow:**

```
Clinician Input (Streamlit)
        ↓
  HTTP POST → /predict
        ↓
  Preprocessing Pipeline
        ↓
  XGBoost Model Inference
        ↓
  Risk Score + Flag Returned
        ↓
  Result Displayed on Dashboard
```

---

## 🚀 Setup Instructions

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20SETUP%20INSTRUCTIONS&fontSize=13&fontColor=C9A84C&fontAlign=6&fontAlignY=65" width="100%"/>

<br/>

**`STEP 01` — Clone the repository**

```bash
git clone https://github.com/bhargavidwivedi/ReVive.git
cd ReVive
```

**`STEP 02` — Create & activate virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**`STEP 03` — Install dependencies**

```bash
pip install -r requirements.txt
```

**`STEP 04` — Run the Django backend**

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

> API live at → `http://127.0.0.1:8000/`

**`STEP 05` — Run the Streamlit frontend** *(open a new terminal)*

```bash
cd frontend
streamlit run app.py
```

> Dashboard live at → `http://localhost:8501/`

---

## 📡 API Reference

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20API%20REFERENCE&fontSize=13&fontColor=C9A84C&fontAlign=5&fontAlignY=65" width="100%"/>

<br/>

**Endpoint:** `POST http://127.0.0.1:8000/predict/`

**Request Body:**

```json
{
  "age": 67,
  "gender": "Female",
  "diagnosis": "Heart Failure",
  "num_prior_admissions": 3,
  "length_of_stay": 7,
  "num_medications": 12,
  "discharge_type": "Home",
  "comorbidity_score": 4
}
```

**Response:**

```json
{
  "readmission_probability": 0.78,
  "risk_level": "High",
  "flag": true,
  "message": "Patient is at HIGH risk. Early intervention recommended."
}
```

**Response Fields:**

| Field | Type | Description |
|:------|:----:|:------------|
| `readmission_probability` | `float` | Score between 0 and 1 |
| `risk_level` | `string` | `Low` · `Medium` · `High` |
| `flag` | `boolean` | `true` if probability > 0.5 |
| `message` | `string` | Clinical recommendation note |

---

## 📸 Screenshots

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20SCREENSHOTS&fontSize=13&fontColor=C9A84C&fontAlign=5&fontAlignY=65" width="100%"/>

<br/>

> Screenshots will be added upon project completion.

| View | Status |
|:-----|:------:|
| Clinician Input Form | ![Soon](https://img.shields.io/badge/Coming%20Soon-C9A84C?style=flat-square&labelColor=000000) |
| Risk Score Dashboard | ![Soon](https://img.shields.io/badge/Coming%20Soon-000000?style=flat-square&labelColor=C9A84C) |
| Model Performance Metrics | ![Soon](https://img.shields.io/badge/Coming%20Soon-C9A84C?style=flat-square&labelColor=000000) |
| High-Risk Patient Alert | ![Soon](https://img.shields.io/badge/Coming%20Soon-000000?style=flat-square&labelColor=C9A84C) |

---

## 🔮 Future Improvements

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20FUTURE%20IMPROVEMENTS&fontSize=13&fontColor=C9A84C&fontAlign=6&fontAlignY=65" width="100%"/>

<br/>

- [ ] 🔐 Clinician authentication & role-based access control
- [ ] 🐳 Dockerize the full application for easy deployment
- [ ] ☁️ Deploy on AWS with a live demo link
- [ ] 🧠 SHAP / LIME explainability for model transparency
- [ ] 📉 Deep learning models — LSTM for temporal patient data
- [ ] 🏥 Integrate with Electronic Health Record (EHR) system API
- [ ] ⚛️ React.js frontend to replace Streamlit

---

## 👩‍💻 Author

<img src="https://capsule-render.vercel.app/api?type=rect&color=000000&height=18&text=%E2%9C%A6%20%20AUTHOR&fontSize=13&fontColor=C9A84C&fontAlign=5&fontAlignY=65" width="100%"/>

<br/>

<div align="center">

**Bhargavi Dwivedi**

*Integrated M.Tech · Artificial Intelligence · VIT Bhopal · CGPA 8.73*

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-C9A84C?style=for-the-badge&logo=linkedin&logoColor=000000)](https://www.linkedin.com/in/bhargavi-dwivedi-093620291/)
[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=C9A84C)](https://github.com/bhargavidwivedi)
[![Gmail](https://img.shields.io/badge/Gmail-C9A84C?style=for-the-badge&logo=gmail&logoColor=000000)](mailto:bhargavidwivedi56@gmail.com)

</div>

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=C9A84C&height=140&section=footer&text=%22Turning%20patient%20data%20into%20life-saving%20decisions%22&fontSize=15&fontColor=000000&fontAlignY=65&animation=fadeIn" width="100%"/>
