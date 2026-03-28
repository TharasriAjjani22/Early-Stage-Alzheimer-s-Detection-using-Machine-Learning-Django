import numpy as np
import joblib
from django.conf import settings


_model = None
_features = None


def load_model():
    global _model, _features
    if _model is None:
        _model = joblib.load(settings.ML_MODEL_PATH)
        _features = joblib.load(settings.FEATURE_NAMES_PATH)

        # Debug logs (optional)
        print(f"Model type: {_model.__class__.__name__}")

    return _model, _features


def predict(form_data: dict) -> dict:
    model, features = load_model()

    # Create feature vector (must match training order)
    feature_vec = np.array([[
        float(form_data.get('patient_age', 70)),
        float(form_data.get('mmse_score', 28)),
        float(form_data.get('cdr_score', 0)),
        float(form_data.get('education_years', 12)),
        float(form_data.get('socioeconomic_status', 3)),
        float(form_data.get('etiv', 1480)),
        float(form_data.get('nwbv', 0.82)),
        float(form_data.get('asf', 1.18)),
        float(form_data.get('memory_score', 85)),
        float(form_data.get('attention_score', 85)),
        float(form_data.get('language_score', 85)),
        float(form_data.get('visuospatial_score', 85)),
        1 if form_data.get('gender') == 'M' else 0,
        1 if form_data.get('family_history') else 0,
        int(form_data.get('apoe4_alleles', 0)),
    ]])

    # Predict probabilities
    probabilities = model.predict_proba(feature_vec)[0]
    predicted_class = int(np.argmax(probabilities))

    label_map = {0: 'healthy', 1: 'mci', 2: 'alzheimer'}
    prediction = label_map.get(predicted_class, "unknown")

    # Risk score calculation
    risk_score = (probabilities[1] * 50 + probabilities[2] * 100) * 100
    risk_score = min(100, risk_score)

    return {
        'prediction': prediction,
        'confidence_healthy': round(float(probabilities[0]) * 100, 1),
        'confidence_mci': round(float(probabilities[1]) * 100, 1),
        'confidence_alzheimer': round(float(probabilities[2]) * 100, 1),
        'risk_score': round(float(risk_score), 1),
    }


def get_feature_importances():
    model, features = load_model()

    # Handle Pipeline models safely
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get('clf', model)
    else:
        clf = model

    print("Model type:", type(clf))  # debug

    # Case 1: Tree-based models (RandomForest, GradientBoosting)
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_

    # Case 2: Linear models (LogisticRegression)
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    # Case 3: Fallback
    else:
        importances = np.zeros(len(features))

    return sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )