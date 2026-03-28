"""
Step 2 — Exploratory Data Analysis & Preprocessing
====================================================
Loads raw CSV → analyses distributions, missing values, correlations,
class balance → outputs cleaned processed CSV + EDA report.

Run:
    python dataset/eda_preprocess.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

RAW_PATH       = os.path.join(os.path.dirname(__file__), "raw",       "oasis_alzheimer.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "processed", "oasis_processed.csv")
PLOTS_DIR      = os.path.join(os.path.dirname(__file__), "plots")
REPORT_PATH    = os.path.join(os.path.dirname(__file__), "reports",   "eda_report.txt")

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

PALETTE = {'Nondemented': '#10b981', 'MCI': '#f59e0b', 'Demented': '#ef4444'}
FEATURE_COLS = ['Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF',
                'Memory_Score','Attention_Score','Language_Score',
                'Visuospatial_Score','Family_History','APOE4_Alleles']

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#0d1420','axes.facecolor':'#121a28',
    'axes.edgecolor':'#1e2d45','axes.labelcolor':'#7a90b0',
    'xtick.color':'#4a6080','ytick.color':'#4a6080',
    'text.color':'#e8edf5','grid.color':'#1e2d45','grid.alpha':0.6,
    'font.family':'sans-serif','axes.titlesize':11,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,
})


def load_data():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def eda_report(df, report_lines):
    def log(s=''):
        print(s)
        report_lines.append(s)

    log("=" * 60)
    log("  ALZHEIMER'S DETECTION — EDA REPORT")
    log("=" * 60)
    log(f"\nDataset shape : {df.shape}")
    log(f"Subjects      : {df['Subject_ID'].nunique()}")
    log(f"Visits/subject: {df.groupby('Subject_ID').size().mean():.1f} avg")

    log("\n── Class Distribution ──")
    vc = df['Group'].value_counts()
    for g, c in vc.items():
        log(f"  {g:<15} {c:>5}  ({c/len(df)*100:.1f}%)")

    log("\n── Numeric Summary ──")
    log(df[FEATURE_COLS].describe().round(2).to_string())

    log("\n── Missing Values ──")
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if len(mv) == 0:
        log("  None found.")
    else:
        log(mv.to_string())

    log("\n── Gender Distribution ──")
    log(df.groupby(['Group','Gender']).size().unstack(fill_value=0).to_string())

    log("\n── Key Feature Means by Group ──")
    key = ['MMSE','CDR','nWBV','Memory_Score','Age']
    log(df.groupby('Group')[key].mean().round(2).to_string())


# ── Plot 1: Class Distribution ─────────────────────────────────────────────
def plot_class_dist(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('#0d1420')

    # Bar chart
    vc = df['Group'].value_counts()
    colors = [PALETTE[g] for g in vc.index]
    bars = axes[0].bar(vc.index, vc.values, color=colors, edgecolor='none', width=0.6)
    axes[0].set_title('Class Distribution (All Visits)')
    axes[0].set_ylabel('Record Count')
    for bar, val in zip(bars, vc.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                     str(val), ha='center', color='#e8edf5', fontsize=9)
    axes[0].set_ylim(0, vc.max() * 1.18)
    axes[0].grid(axis='y', alpha=0.4)

    # Subject-level pie
    subject_groups = df.drop_duplicates('Subject_ID')['Group'].value_counts()
    wedge_colors = [PALETTE[g] for g in subject_groups.index]
    axes[1].pie(subject_groups.values, labels=subject_groups.index, colors=wedge_colors,
                autopct='%1.0f%%', startangle=90,
                textprops={'color':'#e8edf5','fontsize':9},
                wedgeprops={'edgecolor':'#0d1420','linewidth':1.5})
    axes[1].set_title('Subject-Level Class Split')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_class_distribution.png")


# ── Plot 2: Feature distributions per group ────────────────────────────────
def plot_feature_distributions(df):
    features = ['Age','MMSE','CDR','nWBV','Memory_Score','Attention_Score',
                'Language_Score','Visuospatial_Score']
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.patch.set_facecolor('#0d1420')
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        for group, color in PALETTE.items():
            subset = df[df['Group'] == group][feat].dropna()
            ax.hist(subset, bins=22, alpha=0.55, color=color, edgecolor='none',
                    label=group, density=True)
        ax.set_title(feat)
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)

    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.7) for c in PALETTE.values()]
    fig.legend(handles, PALETTE.keys(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.2,
               labelcolor='#e8edf5', fontsize=9)
    fig.suptitle('Feature Distributions by Diagnostic Group', y=1.01, color='#e8edf5', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_feature_distributions.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_feature_distributions.png")


# ── Plot 3: Correlation heatmap ────────────────────────────────────────────
def plot_correlation(df):
    num = df[FEATURE_COLS + ['Group']].copy()
    num['Group_enc'] = num['Group'].map({'Nondemented':0,'MCI':1,'Demented':2})
    corr = num.drop(columns='Group').corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#0d1420')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', linewidths=0.4, linecolor='#0d1420',
                annot_kws={'size':7}, ax=ax,
                cbar_kws={'shrink':0.7})
    ax.set_title('Feature Correlation Matrix', pad=14, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_correlation_heatmap.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_correlation_heatmap.png")


# ── Plot 4: Box plots — key clinical markers ───────────────────────────────
def plot_boxplots(df):
    features = ['MMSE','nWBV','Memory_Score','CDR']
    labels   = ['MMSE Score','Norm. Brain Volume','Memory Score','CDR Rating']
    order = ['Nondemented','MCI','Demented']
    colors = [PALETTE[g] for g in order]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1420')

    for ax, feat, label in zip(axes, features, labels):
        data = [df[df['Group']==g][feat].dropna().values for g in order]
        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        medianprops=dict(color='white', linewidth=1.5),
                        whiskerprops=dict(color='#4a6080'),
                        capprops=dict(color='#4a6080'),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(['Normal','MCI','AD'], fontsize=8)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle('Clinical Markers by Diagnostic Group', y=1.02, color='#e8edf5', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_boxplots.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_boxplots.png")


# ── Plot 5: MMSE vs nWBV scatter ───────────────────────────────────────────
def plot_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1420')

    for group, color in PALETTE.items():
        sub = df[df['Group'] == group]
        axes[0].scatter(sub['MMSE'], sub['nWBV'], c=color, alpha=0.35, s=18, label=group)
        axes[1].scatter(sub['Age'],  sub['Memory_Score'], c=color, alpha=0.35, s=18)

    axes[0].set_xlabel('MMSE Score')
    axes[0].set_ylabel('Normalized Brain Volume (nWBV)')
    axes[0].set_title('MMSE vs Brain Volume')
    axes[0].legend(fontsize=8, framealpha=0.2)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Memory Score')
    axes[1].set_title('Age vs Memory Score')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_scatter_plots.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_scatter_plots.png")


# ── Plot 6: Age distribution + longitudinal progression ───────────────────
def plot_longitudinal(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1420')

    # Age distribution
    for group, color in PALETTE.items():
        axes[0].hist(df[df['Group']==group]['Age'], bins=25, alpha=0.55,
                     color=color, edgecolor='none', label=group, density=True)
    axes[0].set_title('Age Distribution by Group')
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8, framealpha=0.2)
    axes[0].grid(alpha=0.3)

    # MMSE over visits (mean per visit per group)
    visit_means = df.groupby(['Group','Visit'])['MMSE'].mean().reset_index()
    for group, color in PALETTE.items():
        sub = visit_means[visit_means['Group']==group]
        axes[1].plot(sub['Visit'], sub['MMSE'], color=color, marker='o',
                     linewidth=2, markersize=6, label=group)
    axes[1].set_title('Mean MMSE Score Over Visits')
    axes[1].set_xlabel('Visit Number')
    axes[1].set_ylabel('Mean MMSE')
    axes[1].legend(fontsize=8, framealpha=0.2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_longitudinal.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_longitudinal.png")


# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df):
    """
    1. Encode categorical columns
    2. Use LATEST visit per subject (most informative for diagnosis)
    3. Handle missing values (mean imputation)
    4. Export cleaned CSV
    """
    # Use the latest visit per subject
    df_latest = df.sort_values('Visit').groupby('Subject_ID').last().reset_index()

    # Encode gender
    df_latest['Gender_enc'] = (df_latest['Gender'] == 'M').astype(int)

    # Encode target
    label_map = {'Nondemented': 0, 'MCI': 1, 'Demented': 2}
    df_latest['Label'] = df_latest['Group'].map(label_map)

    # Feature columns for ML
    ml_features = ['Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF',
                   'Memory_Score','Attention_Score','Language_Score',
                   'Visuospatial_Score','Gender_enc','Family_History','APOE4_Alleles']

    # Mean-impute any nulls
    for col in ml_features:
        if df_latest[col].isnull().any():
            df_latest[col].fillna(df_latest[col].mean(), inplace=True)

    # Keep only relevant columns
    keep = ['Subject_ID','Group','Label'] + ml_features
    df_clean = df_latest[keep].copy()
    df_clean.to_csv(PROCESSED_PATH, index=False)

    print(f"\nProcessed dataset saved → {PROCESSED_PATH}")
    print(f"Shape: {df_clean.shape}")
    print(f"Label distribution:\n{df_clean['Label'].value_counts().to_string()}")
    return df_clean


def main():
    print("\n" + "="*55)
    print("  EDA & PREPROCESSING PIPELINE")
    print("="*55 + "\n")

    df = load_data()
    report_lines = []

    print("\n[1/6] Running EDA report...")
    eda_report(df, report_lines)

    print("\n[2/6] Plotting class distribution...")
    plot_class_dist(df)

    print("\n[3/6] Plotting feature distributions...")
    plot_feature_distributions(df)

    print("\n[4/6] Plotting correlation heatmap...")
    plot_correlation(df)

    print("\n[5/6] Plotting box plots...")
    plot_boxplots(df)

    print("\n[6/6] Scatter & longitudinal plots...")
    plot_scatter(df)
    plot_longitudinal(df)

    # Save text report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nEDA report saved → {REPORT_PATH}")

    print("\n── Preprocessing ──")
    df_clean = preprocess(df)
    return df_clean


if __name__ == '__main__':
    main()
