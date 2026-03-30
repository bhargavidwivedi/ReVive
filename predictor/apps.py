from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    def ready(self):
        import os
        import joblib
        import pandas as pd

        BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "LightGBM_tuned.pkl")
        DATA_PATH  = os.path.join(BASE_DIR, "data", "processed_features.csv")

        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded in apps.py: {type(model)}")
        except Exception as e:
            print(f"❌ Model load failed in apps.py: {e}")
            try:
                from lightgbm import LGBMClassifier
                from sklearn.model_selection import train_test_split
                df = pd.read_csv(DATA_PATH).apply(pd.to_numeric, errors="coerce").fillna(0)
                X  = df.drop(columns=["readmitted_30d"])
                y  = df["readmitted_30d"]
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                model = LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                joblib.dump(model, MODEL_PATH, protocol=2)
                print("✅ Model retrained and saved!")
            except Exception as e2:
                print(f"❌ Retraining failed: {e2}")