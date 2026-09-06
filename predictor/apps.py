from django.apps import AppConfig
import os
import logging

logger = logging.getLogger(__name__)


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    def ready(self):
        """Load the trained model at startup.

        IMPORTANT: This intentionally does NOT retrain a model on failure.
        The application must never silently replace the intended trained
        model with a different, newly-trained one. If loading fails we log a
        clear, actionable error and let the API report the model as not
        loaded (see /api/health/ and /api/predict/).
        """
        import joblib

        BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "models", "saved", "readmission_model.pkl")

        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully: %s from %s", type(model).__name__, MODEL_PATH)
        except Exception as e:
            logger.error(
                "Failed to load model at %s: %s. "
                "The app will start, but /api/predict/ will return an error until this is fixed. "
                "Checklist: (1) the real model binary is present (it is committed as a normal Git "
                "file, not Git LFS); (2) scikit-learn==1.7.2 is installed — the saved "
                "GradientBoostingClassifier requires it to deserialize. See README > Model loading.",
                MODEL_PATH, e,
            )
