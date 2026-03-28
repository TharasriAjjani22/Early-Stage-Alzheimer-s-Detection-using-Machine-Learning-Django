"""
run_pipeline.py — Master Pipeline Runner
==========================================
Run the complete dataset + training pipeline in one command:

    python run_pipeline.py

Steps:
  1. Generate OASIS-style dataset  (dataset/generate_dataset.py)
  2. EDA + preprocessing           (dataset/eda_preprocess.py)
  3. Train + evaluate all models   (dataset/train_evaluate.py)
  4. Verify model loads correctly  (utils/alzheimer_model.pkl)
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Generate Dataset",          ["python", os.path.join(BASE, "dataset", "generate_dataset.py")]),
    ("EDA & Preprocessing",       ["python", os.path.join(BASE, "dataset", "eda_preprocess.py")]),
    ("Train & Evaluate Models",   ["python", os.path.join(BASE, "dataset", "train_evaluate.py")]),
]

def run_step(title, cmd):
    print(f"\n{'='*55}")
    print(f"  STEP: {title}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n❌ Step '{title}' failed. Aborting.")
        sys.exit(1)
    print(f"\n✅ {title} — done.")

def verify_model():
    print(f"\n{'='*55}")
    print("  STEP: Verify Model")
    print(f"{'='*55}")
    import joblib, numpy as np
    model_path    = os.path.join(BASE, "utils", "alzheimer_model.pkl")
    features_path = os.path.join(BASE, "utils", "feature_names.pkl")
    model    = joblib.load(model_path)
    features = joblib.load(features_path)
    print(f"Model loaded: {type(model.named_steps['clf']).__name__}")
    print(f"Features ({len(features)}): {features}")

    # Quick prediction test
    test_input = np.array([[72, 14, 3, 29, 0, 1490, 0.84, 1.14,
                            88, 87, 86, 87, 0, 0, 0]])
    pred  = model.predict(test_input)
    proba = model.predict_proba(test_input)
    label = ['Nondemented','MCI','Demented'][pred[0]]
    print(f"\nTest prediction: {label}  |  Probabilities: {dict(zip(['Nondem','MCI','Dem'], proba[0].round(3)))}")
    print("\n✅ Model verified — ready for Django!")


if __name__ == '__main__':
    print("\n🧠 NeuroDetect — Full Pipeline Runner")
    for title, cmd in STEPS:
        run_step(title, cmd)
    verify_model()
    print("\n" + "="*55)
    print("  ALL STEPS COMPLETE")
    print("  Next: python manage.py runserver")
    print("="*55 + "\n")
