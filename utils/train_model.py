"""
Alzheimer's Disease Detection ML Model Training
Uses clinical/cognitive features based on OASIS dataset structure
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# Seed for reproducibility
np.random.seed(42)

def load_local_dataset():
    df = pd.read_csv('../dataset/raw/oasis_alzheimer.csv')
    
    # Rename columns to match model
    df = df.rename(columns={
        'Age': 'age',
        'Gender': 'gender',
        'MMSE': 'mmse',
        'EDUC': 'educ',
        'SES': 'ses',
        'eTIV': 'etiv',
        'nWBV': 'nwbv',
        'ASF': 'asf',
        'Memory_Score': 'memory_score',
        'Attention_Score': 'attention_score',
        'Language_Score': 'language_score',
        'Visuospatial_Score': 'visuospatial_score',
        'Family_History': 'family_history',
        'APOE4_Alleles': 'apoe4',
        'CDR': 'cdr'
    })
    
    # Encode gender (M=1, F=0)
    df['gender'] = (df['gender'] == 'M').astype(int)
    
    # Map Group to label
    group_map = {'Nondemented': 0, 'MCI': 1, 'Demented': 2}
    df['label'] = df['Group'].map(group_map)
    df = df.dropna(subset=['label'])
    
    print(f"Loaded dataset: {len(df)} records")
    print("Group distribution:", df['Group'].value_counts().to_dict())
    return df


def train_and_save():
    print("Loading local dataset...")
    df = load_local_dataset()

    feature_cols = [
        'age', 'mmse', 'cdr', 'educ', 'ses', 'etiv', 'nwbv', 'asf',
        'memory_score', 'attention_score', 'language_score',
        'visuospatial_score', 'gender', 'family_history', 'apoe4'
    ]

    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient Boosting pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred,
          target_names=['Healthy', 'MCI', 'Alzheimer\'s']))

    # Save model and feature names
    out_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(out_dir, 'alzheimer_model.pkl'))
    joblib.dump(feature_cols, os.path.join(out_dir, 'feature_names.pkl'))
    print("\nModel saved!")

    return model, feature_cols


if __name__ == '__main__':
    train_and_save()
