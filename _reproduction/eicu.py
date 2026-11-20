"""Reproduce the eICU arm.

Their ML_eICU.ml_eICU() reads 338 pre-balanced CSVs from `data_processed_eicu/`.
That folder is not in the repository — only the aggregated cohort CSV
`shared_dataset_eicu.csv` ships. So we rebuild the balanced sets from the cohort
CSV using their own splitting rule (sequential control partition, cases
repeated in every set), then train with their *fixed* hyperparameters.

This is a faithful reproduction of the *method*; it is not byte-identical to
their prepared folder, and the report says so. Their split proportions are
applied per role (60/15/10/15 as in common.py of their pipeline), stratified by
label, exactly as balanced_datasets_eICU + the main pipeline do.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from _reproduction.common import (PREDICTIONS, REPO, RESULTS, SEED, evaluate,
                                  save_json, summarise)

# Their modal hyperparameters, hard-coded in ML_eICU.ml_eICU()
FIXED_PARAMS = dict(learning_rate=0.1, n_estimators=250, subsample=0.9,
                    max_depth=6, max_features=1)

PROPORTIONS = {'train': 0.60, 'val': 0.15, 'test': 0.15}   # remaining 0.10 = policy


def build_cohort(max_splits: int | None = None):
    """Split the eICU cohort into DISJOINT roles, then balance within each.

    Their pipeline splits first (60/15/10/15) and balances inside each role. An
    earlier version of this script sampled each role independently from the same
    pool, which put the same patients in train and test and produced a spurious
    AUC of 0.9995. The split below assigns every patient to exactly one role.
    """
    df = pd.read_csv(DATA_ROOT / 'shared_dataset_eicu.csv')
    df = df.drop(columns=[c for c in ('index',) if c in df.columns])
    print(f"  eICU cohort: {len(df):,} rows, {int(df['label'].sum())} sepsis "
          f"({100*df['label'].mean():.2f}%)")

    # Disjoint, stratified role assignment (their PROPORTIONS in common.py)
    roles_of = ['train', 'val', 'policy', 'test']
    fracs = [0.60, 0.15, 0.10, 0.15]
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    role = np.empty(len(df), dtype=object)
    for lab in (0, 1):
        idx = df.index[df.label == lab].to_numpy()
        cuts = (np.cumsum(fracs) * len(idx)).astype(int)
        bounds = np.split(idx, cuts[:-1])
        for name, part in zip(roles_of, bounds):
            role[part] = name
    df['role'] = role

    counts = df.groupby(['role', 'label']).size().unstack(fill_value=0)
    print(counts.to_string())

    # Verify disjointness explicitly - this is the guard that would have caught
    # the earlier bug.
    assert df.groupby('label').size().sum() == len(df)
    assert set(df['role'].unique()) <= set(roles_of)

    sets = {}
    for r in ('train', 'val', 'test'):
        sub = df[df.role == r]
        cases = sub[sub.label == 1]
        ctrl = sub[sub.label == 0].sample(frac=1.0, random_state=SEED)
        size = len(cases)
        chunks = [ctrl.iloc[i:i + size] for i in range(0, len(ctrl), size)]
        chunks = [c for c in chunks if len(c) == size]      # their rule: discard remainder
        if max_splits:
            chunks = chunks[:max_splits]
        sets[r] = [(pd.concat([c, cases]).reset_index(drop=True), size) for c in chunks]
        print(f"  {r:5s}: {len(chunks)} balanced sets of {2*size} rows")
    return sets, df


def run(max_splits: int | None):
    sets, _ = build_cohort(max_splits)
    feat_cols = [c for c in sets['train'][0][0].columns
                 if c not in ('label', 'role')]

    rows, probs = [], {}
    t0 = time.time()
    n = len(sets['test'])
    for i in range(n):
        tr = sets['train'][i % len(sets['train'])][0]
        te = sets['test'][i][0]
        model = GradientBoostingClassifier(random_state=SEED, **FIXED_PARAMS)
        model.fit(tr[feat_cols], tr['label'])
        p = model.predict_proba(te[feat_cols])[:, 1]
        m = evaluate(p, te['label'])
        m['split'] = i + 1
        rows.append(m)
        probs[i + 1] = p
        if (i + 1) % 20 == 0:
            print(f"  eICU split {i+1}/{n}  AUC={m['auc']:.4f}  ({time.time()-t0:.0f}s)")

    pd.DataFrame(probs).to_csv(PREDICTIONS / 'GBM_eICU_prob.csv', index=False)
    pd.DataFrame(rows).to_csv(RESULTS / 'gbm_eicu_per_split.csv', index=False)

    summary = summarise(rows)
    summary.update({'arm': 'GBM eICU', 'n_splits': n,
                    'params': FIXED_PARAMS,
                    'seconds': round(time.time() - t0, 1)})
    save_json(summary, 'gbm_eicu_summary.json')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-splits', type=int, default=None)
    a = ap.parse_args()
    print("=== eICU GBM reproduction (fixed hyperparameters, 338 balanced sets) ===")
    s = run(a.max_splits)
    print(f"\n  AUC {s['test_auc']:.4f} | SEN@90 {s['test_sen_90']:.4f} "
          f"| SPE@90 {s['test_spec_90']:.4f} | PRE {s['test_prec_90']:.4f}")
    print(f"  @maxYouden  SEN {s['test_sen_y']:.4f} SPE {s['test_spec_y']:.4f} "
          f"PRE {s['test_prec_y']:.4f}")
