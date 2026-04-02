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

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig").apply(pd.to_numeric, errors="coerce").fillna(0)
    print("Columns:", df.columns.tolist()[:5])
    print("Shape:", df.shape)

    # Find target column flexibly
    target_col = None
    for col in df.columns:
        if "readmit" in col.lower():
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"No readmission column found! Columns: {df.columns.tolist()}")

    print(f"Using target column: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, protocol=2)
    print("Model trained and saved!")

# Always retrain fresh
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

train_and_save()
print("Startup complete!")