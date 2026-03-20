\# ReVive — Hospital Readmission Prediction



A machine learning pipeline to predict 30-day hospital readmissions

using patient demographics, clinical history, and treatment data.



\## Project Structure

\- data/          → Raw and processed datasets (not tracked by Git)

\- ml\_pipeline/   → Feature engineering, model training, evaluation

\- models/        → Saved trained models

\- outputs/       → Charts, SHAP plots, reports



\## Setup

pip install -r requirements.txt



\## Status

🚧 In Progress

\# 🏥 ReVive — Hospital Readmission Prediction



A full end-to-end machine learning pipeline to predict hospital readmission risk using patient demographics, clinical history, treatment data, and behavioural signals.



\---



\## 📊 Final Model Performance



| Model | ROC-AUC | Recall | Precision | F1 |

|---|---|---|---|---|

| Logistic Regression | 0.6672 | - | - | - |

| Random Forest | 0.6702 | - | - | - |

| LightGBM (baseline) | 0.6803 | - | - | - |

| \*\*LightGBM (tuned) ✅\*\* | \*\*0.6812\*\* | \*\*88%\*\* | \*\*53%\*\* | \*\*0.66\*\* |



> Model trained on 99,343 patient encounters. Predicts any hospital readmission (30-day and beyond).



\---



\## 🔍 Top 10 Readmission Risk Drivers (SHAP)



1\. \*\*total\_prior\_visits\*\* — Frequent hospital users always come back

2\. \*\*number\_inpatient\*\* — Prior inpatient stays are the strongest signal

3\. \*\*number\_diagnoses\*\* — More conditions = higher complexity

4\. \*\*num\_procedures\*\* — More procedures = sicker patient

5\. \*\*num\_medications\*\* — Polypharmacy increases readmission risk

6\. \*\*age\_numeric\*\* — Older patients readmit more frequently

7\. \*\*num\_lab\_procedures\*\* — Lab-intensive cases = complex patients

8\. \*\*los\_x\_diagnoses\*\* — Long stay × many diagnoses interaction

9\. \*\*complexity\_score\*\* — Overall clinical burden score

10\. \*\*cardiac\_primary\*\* — Heart conditions are a strong readmission driver



\---



\## 📁 Project Structure



```

ReVive/

├── data/

│   └── raw/                          # Raw CSVs (not tracked by Git)

│       ├── readmission clinical datset.csv

│       ├── archive (8)/              # appointments, billing, doctors, patients, treatments

│       └── behavioral dataset/       # demographic data

│

├── ml\_pipeline/

│   ├── feature\_engineering/

│   │   ├── engineer.py               # 122-feature clinical pipeline

│   │   └── merge.py                  # Multi-source CSV merge pipeline

│   ├── models/

│   │   ├── train.py                  # Train 4 models with CV

│   │   ├── tune.py                   # Optuna hyperparameter tuning

│   │   └── predict.py                # Batch + single patient prediction

│   └── evaluation/

│       └── metrics.py                # ROC, PR, SHAP, confusion matrix

│

├── outputs/                          # Charts and predictions (local only)

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## ⚙️ Setup



```bash

git clone https://github.com/bhargavidwivedi/ReVive.git

cd ReVive

pip install -r requirements.txt

```



\---



\## 🚀 How to Run



\### 1. Feature Engineering

```bash

python ml\_pipeline/feature\_engineering/engineer.py

```



\### 2. Merge Additional Data Sources

```bash

python ml\_pipeline/feature\_engineering/merge.py

```



\### 3. Train Models

```bash

python ml\_pipeline/models/train.py

```



\### 4. Evaluate + SHAP Analysis

```bash

python ml\_pipeline/evaluation/metrics.py

```



\### 5. Hyperparameter Tuning

```bash

python ml\_pipeline/models/tune.py

```



\### 6. Predict

```bash

python ml\_pipeline/models/predict.py

```



\---



\## 🧪 Single Patient Risk Scoring



```python

from ml\_pipeline.models.predict import predict\_single\_patient



predict\_single\_patient({

&#x20;   "time\_in\_hospital"   : 8,

&#x20;   "number\_inpatient"   : 3,

&#x20;   "number\_diagnoses"   : 9,

&#x20;   "num\_medications"    : 15,

&#x20;   "age\_numeric"        : 75,

&#x20;   "total\_prior\_visits" : 5,

&#x20;   "is\_elderly"         : 1,

&#x20;   "cardiac\_primary"    : 1,

})

\# Output: Readmission probability: 76.01% | Risk: High ⚠️

```



\---



\## 📦 Dependencies



```

pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, shap,

matplotlib, seaborn, joblib, django, djangorestframework

```



\---



\## 📈 Risk Stratification



| Risk Level | Probability | Count (test set) | Action |

|---|---|---|---|

| 🔴 High | ≥ 60% | 26,059 (26.2%) | Immediate intervention |

| 🟡 Medium | 30–60% | 62,645 (63.1%) | Monitor closely |

| 🟢 Low | < 30% | 10,639 (10.7%) | Standard discharge |



\---



\## 👩‍💻 Author



\*\*Bhargavi Dwivedi\*\*

Hospital Readmission Prediction — ReVive Project

