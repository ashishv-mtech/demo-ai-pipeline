import joblib
import pandas as pd
from config import MODELS_DIR, FASTAPI_MODEL_NAME, FASTAPI_PROCESSOR
from .schema import MLModelInput, MLModelOutput


class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self, model_name=None):
        """Load model from models directory into memory."""
        if self.model is not None:
            return self.model

        model_path = MODELS_DIR / FASTAPI_MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model file '{FASTAPI_MODEL_NAME}' not found in '{MODELS_DIR}'.")

        print(f"Loading model into memory: {model_path}")
        self.model = joblib.load(model_path)
        return self.model

    def predict(self, input_data: MLModelInput) -> MLModelOutput:
        """Run ML model inference on input data."""
        if self.model is None:
            self.load_model()

        df_input = pd.DataFrame([input_data.model_dump()])

        try:
            pred_class = int(self.model.predict(df_input)[0])
            probas = self.model.predict_proba(df_input)[0]
            probability = float(probas[1]) if len(probas) > 1 else float(pred_class)
        except Exception as e:
            preprocessor_path = MODELS_DIR / FASTAPI_PROCESSOR
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                X_proc = preprocessor.transform(df_input)
                pred_class = int(self.model.predict(X_proc)[0])
                probas = self.model.predict_proba(X_proc)[0]
                probability = float(probas[1]) if len(probas) > 1 else float(pred_class)
            else:
                raise RuntimeError(f"Prediction failed: {str(e)}")

        prob_percent = round(probability * 100, 2)

        if probability < 0.35:
            risk_level = "Low Risk"
        elif probability < 0.70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return MLModelOutput(
            has_heart_disease=pred_class,
            heart_disease_probability_percent=prob_percent,
            probability=round(probability, 4),
            risk_level=risk_level,
            model_used=FASTAPI_MODEL_NAME
        )


# Global instance
model_service = ModelService()


