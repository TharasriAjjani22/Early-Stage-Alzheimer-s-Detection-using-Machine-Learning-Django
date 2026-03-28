"""
Step 1 — Dataset Generator
===========================
Generates a realistic OASIS (Open Access Series of Imaging Studies) -style
longitudinal Alzheimer's dataset as a CSV file.

Columns match the real OASIS-2 dataset schema:
  Subject ID, Visit, MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR,
  eTIV, nWBV, ASF

We add cognitive domain scores + genetic risk to enrich the feature set.

Run:
    python dataset/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(2024)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw", "oasis_alzheimer.csv")


def generate_subject(subject_id, group):
    """
    Generate longitudinal records (2–5 visits) for a single subject.
    group: 0=Nondemented, 1=MCI, 2=Demented
    """
    records = []

    # Base demographics — fixed per subject
    age_base = {0: np.random.normal(69, 7), 1: np.random.normal(74, 7), 2: np.random.normal(77, 7)}[group]
    age_base = float(np.clip(age_base, 55, 94))
    gender  = np.random.choice(['M', 'F'])
    hand    = 'R'  # OASIS dataset is mostly right-handed
    educ    = float(np.clip(np.random.normal([14.5, 13.2, 12.1][group], 3), 4, 23))
    ses     = int(np.clip(np.random.choice([1,2,3,4,5], p=[[.1,.2,.4,.2,.1],[.15,.25,.35,.15,.1],[.2,.3,.3,.15,.05]][group]), 1, 5))
    etiv    = float(np.clip(np.random.normal([1490, 1455, 1420][group], 130), 1050, 1900))
    asf     = round(float(np.clip(np.random.normal([1.16, 1.22, 1.29][group], 0.14), 0.88, 1.72)), 3)
    family_history = int(np.random.choice([0,1], p=[[0.82,0.18],[0.62,0.38],[0.47,0.53]][group]))
    apoe4          = int(np.random.choice([0,1,2], p=[[0.62,0.29,0.09],[0.46,0.39,0.15],[0.30,0.44,0.26]][group]))

    n_visits = np.random.randint(2, 6)
    mr_delay = 0

    # Baseline scores
    mmse_base  = float(np.clip(np.random.normal([28.6, 25.4, 20.3][group], [1.2, 2.0, 4.2][group]), 8, 30))
    cdr_base   = float(np.random.choice([0, 0.5, 1.0, 2.0, 3.0],
                        p=[[0.92,0.06,0.01,0.01,0.00],
                           [0.05,0.72,0.18,0.04,0.01],
                           [0.01,0.10,0.44,0.35,0.10]][group]))
    nwbv_base  = float(np.clip(np.random.normal([0.835, 0.782, 0.733][group], 0.025), 0.62, 0.94))
    mem_base   = float(np.clip(np.random.normal([87, 67, 49][group], [7, 11, 13][group]), 15, 100))
    att_base   = float(np.clip(np.random.normal([88, 71, 53][group], [7, 11, 13][group]), 15, 100))
    lang_base  = float(np.clip(np.random.normal([88, 73, 54][group], [7, 10, 12][group]), 15, 100))
    vis_base   = float(np.clip(np.random.normal([87, 70, 51][group], [8, 11, 13][group]), 15, 100))

    # Progression rates per visit (group-dependent)
    mmse_decline = [0.05, 0.35, 0.90][group]
    nwbv_decline = [0.001, 0.004, 0.008][group]

    for visit in range(1, n_visits + 1):
        noise = lambda s: np.random.normal(0, s)

        mmse  = round(float(np.clip(mmse_base  - mmse_decline * (visit-1) + noise(0.5),  8, 30)), 1)
        nwbv  = round(float(np.clip(nwbv_base  - nwbv_decline * (visit-1) + noise(0.004), 0.62, 0.94)), 3)
        mem   = round(float(np.clip(mem_base   - [0.2,1.2,2.5][group] * (visit-1) + noise(2), 15, 100)), 1)
        att   = round(float(np.clip(att_base   - [0.2,1.1,2.3][group] * (visit-1) + noise(2), 15, 100)), 1)
        lang  = round(float(np.clip(lang_base  - [0.2,1.0,2.2][group] * (visit-1) + noise(2), 15, 100)), 1)
        vis   = round(float(np.clip(vis_base   - [0.2,1.1,2.4][group] * (visit-1) + noise(2), 15, 100)), 1)

        # CDR sometimes progresses
        cdr = cdr_base
        if group == 1 and visit > 2 and np.random.random() < 0.15:
            cdr = min(3.0, cdr + 0.5)
        if group == 2 and visit > 1 and np.random.random() < 0.2:
            cdr = min(3.0, cdr + 0.5)

        label_map = {0: 'Nondemented', 1: 'MCI', 2: 'Demented'}

        records.append({
            'Subject_ID':     f'OAS2_{subject_id:04d}',
            'Visit':          visit,
            'MR_Delay':       mr_delay,
            'Gender':         gender,
            'Hand':           hand,
            'Age':            round(age_base + (visit - 1) * 0.5, 1),
            'EDUC':           educ,
            'SES':            ses,
            'MMSE':           mmse,
            'CDR':            round(cdr, 1),
            'eTIV':           round(etiv, 1),
            'nWBV':           nwbv,
            'ASF':            asf,
            'Memory_Score':   mem,
            'Attention_Score':att,
            'Language_Score': lang,
            'Visuospatial_Score': vis,
            'Family_History': family_history,
            'APOE4_Alleles':  apoe4,
            'Group':          label_map[group],
        })

        mr_delay += np.random.randint(180, 730)  # ~6–24 months between visits

    return records


def generate_full_dataset(n_subjects=600):
    print(f"Generating {n_subjects} subjects across 3 groups...")
    all_records = []

    # Balanced groups: 50% Nondemented, 25% MCI, 25% Demented
    n_nd  = n_subjects // 2
    n_mci = n_subjects // 4
    n_dem = n_subjects - n_nd - n_mci

    for sid in range(1, n_nd + 1):
        all_records.extend(generate_subject(sid, 0))
    for sid in range(n_nd + 1, n_nd + n_mci + 1):
        all_records.extend(generate_subject(sid, 1))
    for sid in range(n_nd + n_mci + 1, n_subjects + 1):
        all_records.extend(generate_subject(sid, 2))

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDataset saved → {OUTPUT_PATH}")
    print(f"Total records : {len(df)}")
    print(f"Total subjects: {df['Subject_ID'].nunique()}")
    print("\nClass distribution:")
    print(df['Group'].value_counts().to_string())
    print("\nSample:")
    print(df.head(3).to_string())
    return df


if __name__ == '__main__':
    generate_full_dataset(600)
