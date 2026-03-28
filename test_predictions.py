import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from alzheimer_app.ml_service import predict

# Test 1: Healthy person
print("=== Test 1: Healthy Patient ===")
healthy = {
    'patient_age': 65, 'mmse_score': 29.5, 'cdr_score': 0,
    'education_years': 15, 'socioeconomic_status': 3,
    'etiv': 1500, 'nwbv': 0.85, 'asf': 1.15,
    'memory_score': 95, 'attention_score': 94,
    'language_score': 92, 'visuospatial_score': 90,
    'gender': 'M', 'family_history': False, 'apoe4_alleles': 0
}
result = predict(healthy)
print(f"Prediction: {result['prediction']}")
print(f"Healthy: {result['confidence_healthy']}% | MCI: {result['confidence_mci']}% | Alzheimer: {result['confidence_alzheimer']}%")
print(f"Risk Score: {result['risk_score']}\n")

# Test 2: MCI patient
print("=== Test 2: MCI Patient ===")
mci = {
    'patient_age': 75, 'mmse_score': 24.5, 'cdr_score': 0.5,
    'education_years': 12, 'socioeconomic_status': 2,
    'etiv': 1400, 'nwbv': 0.77, 'asf': 1.25,
    'memory_score': 65, 'attention_score': 60,
    'language_score': 70, 'visuospatial_score': 60,
    'gender': 'F', 'family_history': True, 'apoe4_alleles': 1
}
result = predict(mci)
print(f"Prediction: {result['prediction']}")
print(f"Healthy: {result['confidence_healthy']}% | MCI: {result['confidence_mci']}% | Alzheimer: {result['confidence_alzheimer']}%")
print(f"Risk Score: {result['risk_score']}\n")

# Test 3: Alzheimer's patient
print("=== Test 3: Alzheimer's Patient ===")
alzheimers = {
    'patient_age': 82, 'mmse_score': 18.0, 'cdr_score': 2.0,
    'education_years': 10, 'socioeconomic_status': 2,
    'etiv': 1350, 'nwbv': 0.71, 'asf': 1.35,
    'memory_score': 35, 'attention_score': 40,
    'language_score': 45, 'visuospatial_score': 30,
    'gender': 'M', 'family_history': True, 'apoe4_alleles': 2
}
result = predict(alzheimers)
print(f"Prediction: {result['prediction']}")
print(f"Healthy: {result['confidence_healthy']}% | MCI: {result['confidence_mci']}% | Alzheimer: {result['confidence_alzheimer']}%")
print(f"Risk Score: {result['risk_score']}")
