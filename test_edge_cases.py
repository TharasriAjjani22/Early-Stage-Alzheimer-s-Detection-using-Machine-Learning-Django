import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from alzheimer_app.ml_service import predict

print("=" * 80)
print("EDGE CASES & BOUNDARY VALUES TEST")
print("=" * 80)

# Test 1: Minimum Age (50)
print("\n🔵 TEST 1: Young Patient (Age 50)")
print("-" * 80)
young = {
    'patient_age': 50, 'mmse_score': 30, 'cdr_score': 0,
    'education_years': 20, 'socioeconomic_status': 5,
    'etiv': 1600, 'nwbv': 0.90, 'asf': 1.0,
    'memory_score': 100, 'attention_score': 100, 'language_score': 100, 'visuospatial_score': 100,
    'gender': 'M', 'family_history': False, 'apoe4_alleles': 0
}
result = predict(young)
print(f"Age: {young['patient_age']} | MMSE: {young['mmse_score']} | nWBV: {young['nwbv']}")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 2: Maximum Age (100)
print("🔵 TEST 2: Elderly Patient (Age 100)")
print("-" * 80)
elderly = {
    'patient_age': 100, 'mmse_score': 15, 'cdr_score': 2.5,
    'education_years': 6, 'socioeconomic_status': 1,
    'etiv': 1200, 'nwbv': 0.65, 'asf': 1.5,
    'memory_score': 20, 'attention_score': 25, 'language_score': 30, 'visuospatial_score': 20,
    'gender': 'F', 'family_history': True, 'apoe4_alleles': 2
}
result = predict(elderly)
print(f"Age: {elderly['patient_age']} | MMSE: {elderly['mmse_score']} | nWBV: {elderly['nwbv']}")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 3: Perfect MMSE (30)
print("🔵 TEST 3: Perfect MMSE Score (30)")
print("-" * 80)
perfect_mmse = {
    'patient_age': 78, 'mmse_score': 30.0, 'cdr_score': 0,
    'education_years': 18, 'socioeconomic_status': 4,
    'etiv': 1550, 'nwbv': 0.88, 'asf': 1.08,
    'memory_score': 98, 'attention_score': 99, 'language_score': 97, 'visuospatial_score': 96,
    'gender': 'F', 'family_history': False, 'apoe4_alleles': 0
}
result = predict(perfect_mmse)
print(f"Age: {perfect_mmse['patient_age']} | MMSE: {perfect_mmse['mmse_score']} (PERFECT)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 4: Minimum MMSE (8)
print("🔵 TEST 4: Severe Cognitive Decline (MMSE 8)")
print("-" * 80)
severe = {
    'patient_age': 85, 'mmse_score': 8.0, 'cdr_score': 3.0,
    'education_years': 9, 'socioeconomic_status': 2,
    'etiv': 1250, 'nwbv': 0.62, 'asf': 1.48,
    'memory_score': 15, 'attention_score': 18, 'language_score': 20, 'visuospatial_score': 15,
    'gender': 'M', 'family_history': True, 'apoe4_alleles': 2
}
result = predict(severe)
print(f"Age: {severe['patient_age']} | MMSE: {severe['mmse_score']} (SEVERE)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 5: High Brain Volume (nWBV 0.95)
print("🔵 TEST 5: Very High Brain Volume (nWBV 0.95)")
print("-" * 80)
high_volume = {
    'patient_age': 72, 'mmse_score': 28, 'cdr_score': 0,
    'education_years': 16, 'socioeconomic_status': 4,
    'etiv': 1800, 'nwbv': 0.95, 'asf': 0.95,
    'memory_score': 92, 'attention_score': 90, 'language_score': 91, 'visuospatial_score': 89,
    'gender': 'M', 'family_history': False, 'apoe4_alleles': 0
}
result = predict(high_volume)
print(f"Age: {high_volume['patient_age']} | nWBV: {high_volume['nwbv']} (HIGH)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 6: Low Brain Volume (nWBV 0.62)
print("🔵 TEST 6: Very Low Brain Volume (nWBV 0.62)")
print("-" * 80)
low_volume = {
    'patient_age': 79, 'mmse_score': 19, 'cdr_score': 2.0,
    'education_years': 11, 'socioeconomic_status': 2,
    'etiv': 1100, 'nwbv': 0.62, 'asf': 1.58,
    'memory_score': 38, 'attention_score': 42, 'language_score': 50, 'visuospatial_score': 35,
    'gender': 'F', 'family_history': True, 'apoe4_alleles': 2
}
result = predict(low_volume)
print(f"Age: {low_volume['patient_age']} | nWBV: {low_volume['nwbv']} (LOW)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 7: No APOE4 Risk
print("🔵 TEST 7: No APOE4 Genetic Risk (0 alleles)")
print("-" * 80)
no_apoe = {
    'patient_age': 76, 'mmse_score': 20, 'cdr_score': 1.0,
    'education_years': 12, 'socioeconomic_status': 3,
    'etiv': 1420, 'nwbv': 0.75, 'asf': 1.22,
    'memory_score': 55, 'attention_score': 50, 'language_score': 60, 'visuospatial_score': 50,
    'gender': 'M', 'family_history': False, 'apoe4_alleles': 0
}
result = predict(no_apoe)
print(f"Age: {no_apoe['patient_age']} | APOE4: {no_apoe['apoe4_alleles']} (NO RISK)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 8: Maximum APOE4 Risk (2 alleles)
print("🔵 TEST 8: Maximum APOE4 Genetic Risk (2 alleles)")
print("-" * 80)
max_apoe = {
    'patient_age': 73, 'mmse_score': 20, 'cdr_score': 1.0,
    'education_years': 12, 'socioeconomic_status': 3,
    'etiv': 1420, 'nwbv': 0.75, 'asf': 1.22,
    'memory_score': 55, 'attention_score': 50, 'language_score': 60, 'visuospatial_score': 50,
    'gender': 'M', 'family_history': True, 'apoe4_alleles': 2
}
result = predict(max_apoe)
print(f"Age: {max_apoe['patient_age']} | APOE4: {max_apoe['apoe4_alleles']} (MAX RISK)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 9: Low Education
print("🔵 TEST 9: Very Low Education (4 years)")
print("-" * 80)
low_edu = {
    'patient_age': 77, 'mmse_score': 22, 'cdr_score': 1.5,
    'education_years': 4, 'socioeconomic_status': 1,
    'etiv': 1350, 'nwbv': 0.72, 'asf': 1.30,
    'memory_score': 48, 'attention_score': 52, 'language_score': 55, 'visuospatial_score': 45,
    'gender': 'F', 'family_history': True, 'apoe4_alleles': 1
}
result = predict(low_edu)
print(f"Age: {low_edu['patient_age']} | Education: {low_edu['education_years']} years (LOW)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 10: High Education
print("🔵 TEST 10: Very High Education (25 years)")
print("-" * 80)
high_edu = {
    'patient_age': 74, 'mmse_score': 24, 'cdr_score': 0.5,
    'education_years': 25, 'socioeconomic_status': 5,
    'etiv': 1600, 'nwbv': 0.84, 'asf': 1.05,
    'memory_score': 78, 'attention_score': 81, 'language_score': 83, 'visuospatial_score': 79,
    'gender': 'M', 'family_history': False, 'apoe4_alleles': 1
}
result = predict(high_edu)
print(f"Age: {high_edu['patient_age']} | Education: {high_edu['education_years']} years (HIGH)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 11: Borderline MCI (exactly CDR 0.5)
print("🔵 TEST 11: Borderline MCI (CDR exactly 0.5)")
print("-" * 80)
borderline = {
    'patient_age': 71, 'mmse_score': 26.0, 'cdr_score': 0.5,
    'education_years': 13, 'socioeconomic_status': 3,
    'etiv': 1480, 'nwbv': 0.80, 'asf': 1.18,
    'memory_score': 75, 'attention_score': 72, 'language_score': 76, 'visuospatial_score': 71,
    'gender': 'F', 'family_history': False, 'apoe4_alleles': 1
}
result = predict(borderline)
print(f"Age: {borderline['patient_age']} | CDR: {borderline['cdr_score']} (BORDERLINE)")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

# Test 12: All Cognitive Scores = 0 (Worst case)
print("🔵 TEST 12: WORST CASE - All scores at minimum")
print("-" * 80)
worst = {
    'patient_age': 95, 'mmse_score': 8, 'cdr_score': 3,
    'education_years': 4, 'socioeconomic_status': 1,
    'etiv': 1100, 'nwbv': 0.60, 'asf': 1.67,
    'memory_score': 15, 'attention_score': 15, 'language_score': 16, 'visuospatial_score': 15,
    'gender': 'M', 'family_history': True, 'apoe4_alleles': 2
}
result = predict(worst)
print(f"Age: {worst['patient_age']} | MMSE: {worst['mmse_score']} | nWBV: {worst['nwbv']}")
print(f"Result: {result['prediction'].upper()} | Risk: {result['risk_score']}%\n")

print("=" * 80)
print("✓ ALL EDGE CASE TESTS COMPLETED!")
print("=" * 80)
