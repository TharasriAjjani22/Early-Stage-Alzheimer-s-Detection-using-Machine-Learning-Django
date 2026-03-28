"""
Step 3 — Model Training, Comparison & Evaluation
==================================================
Loads processed CSV → trains 6 ML algorithms → compares performance →
saves best model + detailed evaluation report + plots.

Run:
    python dataset/train_evaluate.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib, os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve, accuracy_score,
                                     f1_score, precision_score, recall_score)

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROCESSED_PATH  = os.path.join(os.path.dirname(__file__), "processed", "oasis_processed.csv")
MODEL_OUT       = os.path.join(os.path.dirname(__file__), "..", "utils", "alzheimer_model.pkl")
FEATURES_OUT    = os.path.join(os.path.dirname(__file__), "..", "utils", "feature_names.pkl")
PLOTS_DIR       = os.path.join(os.path.dirname(__file__), "plots")
REPORT_PATH     = os.path.join(os.path.dirname(__file__), "reports", "training_report.txt")

FEATURE_COLS = ['Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF',
                'Memory_Score','Attention_Score','Language_Score',
                'Visuospatial_Score','Gender_enc','Family_History','APOE4_Alleles']
TARGET_NAMES = ['Nondemented', 'MCI', 'Demented']

plt.rcParams.update({
    'figure.facecolor':'#0d1420','axes.facecolor':'#121a28',
    'axes.edgecolor':'#1e2d45','axes.labelcolor':'#7a90b0',
    'xtick.color':'#4a6080','ytick.color':'#4a6080',
    'text.color':'#e8edf5','grid.color':'#1e2d45','grid.alpha':0.5,
    'font.family':'sans-serif','axes.titlesize':11,'axes.labelsize':9,
})


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    X  = df[FEATURE_COLS].values
    y  = df['Label'].values
    print(f"Loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    return X, y


def build_models():
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   KNeighborsClassifier(n_neighbors=7))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))
        ]),
        'AdaBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, random_state=42))
        ]),
    }
    if HAS_XGB:
        models['XGBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',   XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                    use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ])
    return models


def cross_validate_all(models, X, y, cv=5):
    print(f"\n── {cv}-Fold Cross-Validation ──")
    results = {}
    for name, model in models.items():
        t0 = time.time()
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=42),
                                 scoring='accuracy', n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {
            'cv_mean': scores.mean(),
            'cv_std':  scores.std(),
            'cv_time': elapsed,
            'scores':  scores,
        }
        print(f"  {name:<25}  {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%  ({elapsed:.1f}s)")
    return results


def train_and_evaluate(models, X_train, X_test, y_train, y_test, report_lines):
    def log(s=''):
        print(s)
        report_lines.append(s)

    log("\n" + "="*60)
    log("  MODEL EVALUATION ON HELD-OUT TEST SET  (80/20 split)")
    log("="*60)

    eval_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted')

        # OvR AUC
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc = roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr')

        eval_results[name] = {
            'model': model, 'acc': acc, 'f1': f1,
            'prec': prec, 'rec': rec, 'auc': auc,
            'y_pred': y_pred, 'y_prob': y_prob,
            'cm': confusion_matrix(y_test, y_pred),
        }

        log(f"\n── {name} ──")
        log(f"  Accuracy  : {acc*100:.2f}%")
        log(f"  F1 (weighted): {f1:.4f}")
        log(f"  AUC (OvR) : {auc:.4f}")
        log(classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=3))

    return eval_results


# ── Plot: Model Comparison Bar Chart ──────────────────────────────────────
def plot_model_comparison(eval_results, cv_results):
    names   = list(eval_results.keys())
    acc     = [eval_results[n]['acc']  for n in names]
    f1      = [eval_results[n]['f1']   for n in names]
    auc     = [eval_results[n]['auc']  for n in names]
    cv_mean = [cv_results[n]['cv_mean'] for n in names]
    cv_std  = [cv_results[n]['cv_std']  for n in names]

    x = np.arange(len(names))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1420')

    # Grouped bars
    ax = axes[0]
    bars1 = ax.bar(x - w, acc,    w, label='Accuracy',  color='#3b82f6', alpha=0.85)
    bars2 = ax.bar(x,     f1,     w, label='F1',         color='#10b981', alpha=0.85)
    bars3 = ax.bar(x + w, auc,    w, label='AUC (OvR)',  color='#8b5cf6', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=28, ha='right', fontsize=8)
    ax.set_ylim(0.5, 1.04); ax.set_title('Model Performance Comparison')
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis='y', alpha=0.4)
    ax.set_ylabel('Score')

    # CV scores
    ax2 = axes[1]
    ax2.barh(names, cv_mean, xerr=cv_std, color='#f59e0b', alpha=0.8, height=0.55,
             error_kw=dict(ecolor='#7a90b0', capsize=4))
    ax2.set_xlim(0.5, 1.04)
    ax2.set_title(f'5-Fold Cross-Validation Accuracy')
    ax2.set_xlabel('CV Accuracy')
    ax2.grid(axis='x', alpha=0.4)
    for i, (m, s) in enumerate(zip(cv_mean, cv_std)):
        ax2.text(m + s + 0.003, i, f'{m*100:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_model_comparison.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_model_comparison.png")


# ── Plot: Confusion Matrices ───────────────────────────────────────────────
def plot_confusion_matrices(eval_results):
    n = len(eval_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*3.8))
    fig.patch.set_facecolor('#0d1420')
    axes = axes.flatten()

    cmap = sns.diverging_palette(230, 10, as_cmap=True)

    for i, (name, res) in enumerate(eval_results.items()):
        cm = res['cm']
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=['Norm','MCI','AD'], yticklabels=['Norm','MCI','AD'],
                    ax=axes[i], linewidths=0.5, linecolor='#0d1420',
                    cbar=False, annot_kws={'size':9})
        axes[i].set_title(f'{name}\n(Acc: {res["acc"]*100:.1f}%)', fontsize=9)
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Confusion Matrices — All Models (% of true class)', y=1.01,
                 color='#e8edf5', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_confusion_matrices.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 08_confusion_matrices.png")


# ── Plot: ROC Curves (best model) ─────────────────────────────────────────
def plot_roc_curves(eval_results, y_test, best_name):
    res = eval_results[best_name]
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    y_prob = res['y_prob']

    colors = ['#10b981', '#f59e0b', '#ef4444']
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0d1420')

    for i, (cls, color) in enumerate(zip(TARGET_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{cls} (AUC={auc_i:.3f})')

    ax.plot([0,1],[0,1], '--', color='#4a6080', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {best_name}')
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_roc_curves.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 09_roc_curves.png")


# ── Plot: Feature Importance ───────────────────────────────────────────────
def plot_feature_importance(best_model, best_name):
    clf = best_model.named_steps['clf']
    if not hasattr(clf, 'feature_importances_'):
        print("  Skipping feature importance (not available for this model)")
        return

    importances = clf.feature_importances_
    feat_labels = [f.replace('_', ' ') for f in FEATURE_COLS]
    sorted_idx  = np.argsort(importances)[::-1]

    colors_list = ['#3b82f6','#60a5fa','#93c5fd','#bfdbfe',
                   '#10b981','#34d399','#6ee7b7','#a7f3d0',
                   '#f59e0b','#fbbf24','#fcd34d','#fde68a',
                   '#ef4444','#f87171','#fca5a5']

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0d1420')

    bars = ax.barh(
        [feat_labels[i] for i in sorted_idx],
        [importances[i] for i in sorted_idx],
        color=[colors_list[i % len(colors_list)] for i in range(len(sorted_idx))],
        edgecolor='none', height=0.65
    )
    ax.invert_yaxis()
    ax.set_title(f'Feature Importances — {best_name}')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.4)

    for bar, val in zip(bars, [importances[i] for i in sorted_idx]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, color='#7a90b0')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_feature_importance.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 10_feature_importance.png")


def main():
    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION PIPELINE")
    print("="*55)

    report_lines = []
    def log(s=''):
        print(s)
        report_lines.append(s)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    models = build_models()

    log("\n── Cross-Validation Results ──")
    cv_results = cross_validate_all(models, X, y, cv=5)
    for name, r in cv_results.items():
        log(f"  {name:<25} CV Acc: {r['cv_mean']*100:.2f}% ± {r['cv_std']*100:.2f}%")

    eval_results = train_and_evaluate(models, X_train, X_test, y_train, y_test, report_lines)

    # Pick best by F1
    best_name  = max(eval_results, key=lambda n: eval_results[n]['f1'])
    best_model = eval_results[best_name]['model']

    log(f"\n{'='*55}")
    log(f"  BEST MODEL: {best_name}")
    log(f"  Accuracy : {eval_results[best_name]['acc']*100:.2f}%")
    log(f"  F1 Score : {eval_results[best_name]['f1']:.4f}")
    log(f"  AUC (OvR): {eval_results[best_name]['auc']:.4f}")
    log(f"{'='*55}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(FEATURE_COLS, FEATURES_OUT)
    log(f"\nModel saved  → {MODEL_OUT}")
    log(f"Features saved → {FEATURES_OUT}")

    # Save report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))
    log(f"Report saved → {REPORT_PATH}")

    print("\n── Generating Plots ──")
    plot_model_comparison(eval_results, cv_results)
    plot_confusion_matrices(eval_results)
    plot_roc_curves(eval_results, y_test, best_name)
    plot_feature_importance(best_model, best_name)

    print(f"\n✅ Done! Best model ({best_name}) saved and ready for Django.")


if __name__ == '__main__':
    main()
