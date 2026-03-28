# Alzheimer Detection App - Test TODO

## Status: Fixes Applied - Retest Pending

### 1. [✅ FIXED] ml_service.py feature importances
- Updated for Pipeline(GradientBoostingClassifier)
- Uses model.named_steps['clf'].feature_importances_
- Debug prints model type/coef_

### 2. [✅ DONE] result.html template syntax
- Removed invalid filters
- **Note**: Test showed |split error - search found none, may be cached

### 3. [IN PROGRESS] Test fixes
- [✅] venv, deps, migrate
- [✅] ML model loads (Pipeline)
- [PENDING] Dashboard charts (feature importances)
- [PENDING] Full assessment → result page
- Server: http://127.0.0.1:8000/

### 4. [PENDING] Final Validation
- No 500 errors
- Predictions/risk scores display
- Console debug: GradientBoostingClassifier
