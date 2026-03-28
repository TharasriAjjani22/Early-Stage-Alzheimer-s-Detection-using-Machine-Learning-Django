import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from alzheimer_app.ml_service import predict
import json

print("=" * 80)
print("ALZHEIMER'S DETECTION MODEL — COMPREHENSIVE TEST")
print("=" * 80)

# Test Case 1: HEALTHY PATIENT
print("\n✅ TEST 1: HEALTHY PATIENT")
print("-" * 80)
healthy_data = {
    'patient_name': 'John Smith',
    'patient_age': 65,
    'gender': 'M',
    'mmse_score': 29.0,
    'cdr_score': 0,
    'education_years': 15,
    'socioeconomic_status': 3,
    'etiv': 1500,
    'nwbv': 0.85,
    'asf': 1.15,
    'memory_score': 95,
    'attention_score': 94,
    'language_score': 92,
    'visuospatial_score': 90,
    'family_history': False,
    'apoe4_alleles': 0
}

print("Input Data:")
print(f"  Name: {healthy_data['patient_name']}, Age: {healthy_data['patient_age']}")
print(f"  MMSE: {healthy_data['mmse_score']}, CDR: {healthy_data['cdr_score']}")
print(f"  Memory: {healthy_data['memory_score']}, Attention: {healthy_data['attention_score']}")
print(f"  Language: {healthy_data['language_score']}, Visuospatial: {healthy_data['visuospatial_score']}")
print(f"  nWBV: {healthy_data['nwbv']}, Family History: {healthy_data['family_history']}")

result1 = predict(healthy_data)
print("\n✓ PREDICTION RESULT:")
print(f"  Diagnosis: {result1['prediction'].upper()}")
print(f"  Confidence Healthy: {result1['confidence_healthy']}%")
print(f"  Confidence MCI: {result1['confidence_mci']}%")
print(f"  Confidence Alzheimer: {result1['confidence_alzheimer']}%")
print(f"  Risk Score: {result1['risk_score']}%")
print(f"  ✓ EXPECTED: healthy (100% confidence, 0% risk)")

# Test Case 2: MCI PATIENT
print("\n\n⚠️  TEST 2: MCI PATIENT (Early Warning)")
print("-" * 80)
mci_data = {
    'patient_name': 'Sarah Johnson',
    'patient_age': 75,
    'gender': 'F',
    'mmse_score': 24.5,
    'cdr_score': 0.5,
    'education_years': 12,
    'socioeconomic_status': 2,
    'etiv': 1400,
    'nwbv': 0.77,
    'asf': 1.25,
    'memory_score': 65,
    'attention_score': 60,
    'language_score': 70,
    'visuospatial_score': 60,
    'family_history': True,
    'apoe4_alleles': 1
}

print("Input Data:")
print(f"  Name: {mci_data['patient_name']}, Age: {mci_data['patient_age']}")
print(f"  MMSE: {mci_data['mmse_score']}, CDR: {mci_data['cdr_score']}")
print(f"  Memory: {mci_data['memory_score']}, Attention: {mci_data['attention_score']}")
print(f"  Language: {mci_data['language_score']}, Visuospatial: {mci_data['visuospatial_score']}")
print(f"  nWBV: {mci_data['nwbv']}, Family History: {mci_data['family_history']}")

result2 = predict(mci_data)
print("\n✓ PREDICTION RESULT:")
print(f"  Diagnosis: {result2['prediction'].upper()}")
print(f"  Confidence Healthy: {result2['confidence_healthy']}%")
print(f"  Confidence MCI: {result2['confidence_mci']}%")
print(f"  Confidence Alzheimer: {result2['confidence_alzheimer']}%")
print(f"  Risk Score: {result2['risk_score']}%")
print(f"  ✓ EXPECTED: mci (high MCI confidence, elevated risk)")

# Test Case 3: ALZHEIMER'S PATIENT
print("\n\n🔴 TEST 3: ALZHEIMER'S PATIENT (High Risk)")
print("-" * 80)
alzheimers_data = {
    'patient_name': 'Robert Davis',
    'patient_age': 82,
    'gender': 'M',
    'mmse_score': 18.0,
    'cdr_score': 2.0,
    'education_years': 10,
    'socioeconomic_status': 2,
    'etiv': 1350,
    'nwbv': 0.71,
    'asf': 1.35,
    'memory_score': 35,
    'attention_score': 40,
    'language_score': 45,
    'visuospatial_score': 30,
    'family_history': True,
    'apoe4_alleles': 2
}

print("Input Data:")
print(f"  Name: {alzheimers_data['patient_name']}, Age: {alzheimers_data['patient_age']}")
print(f"  MMSE: {alzheimers_data['mmse_score']}, CDR: {alzheimers_data['cdr_score']}")
print(f"  Memory: {alzheimers_data['memory_score']}, Attention: {alzheimers_data['attention_score']}")
print(f"  Language: {alzheimers_data['language_score']}, Visuospatial: {alzheimers_data['visuospatial_score']}")
print(f"  nWBV: {alzheimers_data['nwbv']}, Family History: {alzheimers_data['family_history']}")

result3 = predict(alzheimers_data)
print("\n✓ PREDICTION RESULT:")
print(f"  Diagnosis: {result3['prediction'].upper()}")
print(f"  Confidence Healthy: {result3['confidence_healthy']}%")
print(f"  Confidence MCI: {result3['confidence_mci']}%")
print(f"  Confidence Alzheimer: {result3['confidence_alzheimer']}%")
print(f"  Risk Score: {result3['risk_score']}%")
print(f"  ✓ EXPECTED: alzheimer (high Alzheimer confidence, 100% risk)")

# Summary
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

test_results = [
    ("Healthy Patient", result1['prediction'], 'healthy'),
    ("MCI Patient", result2['prediction'], 'mci'),
    ("Alzheimer's Patient", result3['prediction'], 'alzheimer'),
]

all_passed = True
for test_name, actual, expected in test_results:
    status = "✓ PASS" if actual == expected else "✗ FAIL"
    print(f"{status}  {test_name}: got {actual}, expected {expected}")
    if actual != expected:
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("🎉 ALL TESTS PASSED! Model is working correctly.")
else:
    print("⚠️  SOME TESTS FAILED. Please review the results.")
print("=" * 80)
