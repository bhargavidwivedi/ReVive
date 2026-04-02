import os
import joblib
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed_features.csv")

def train_and_save():
    print("Training model on Railway...")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH).apply(pd.to_numeric, errors="coerce").fillna(0)
    X  = df.drop(columns=["readmitted_30d"])
    y  = df["readmitted_30d"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, protocol=2)
    print("Model trained and saved!")

# Always delete old incompatible pickle and retrain fresh
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print("Removed old model, retraining fresh...")

train_and_save()
print("Startup complete!")