"""Experiment B — artefact-conditional performance.

Question: when a near-constant "hourly difference" feature fires, what does it
mean clinically, and how much of the model's true-positive yield arrives through
that channel?

The mechanism under audit is their preprocessing. `data_processing2.py:152-154`
applies `ffill -> bfill -> median-fill` to hourly values before differencing. A lab
drawn once and forward-filled produces a constant series, so every consecutive
difference is exactly 0. A lab drawn twice, at adjacent hours, produces exactly
one nonzero difference. The `*_diff_*` features therefore encode *whether the lab
was measured in consecutive hours*, not merely what changed.

This matters because a model can treat "this feature is nonzero" as an event and
learn a rule that is really about the observation process.

Quantities reported per artefact feature:

  coverage            fraction of patients where the feature is nonzero
  conditional PPV     sepsis prevalence among those patients (vs base rate)
  recall contribution fraction of all true positives that the firing subgroup
                      contains
  AUC as a binary     discrimination of the bare indicator, with no model

Then a leave-the-artefacts-out ablation: retrain the full 40-feature model with
the artefact columns removed and measure the AUC delta.

Usage: py -m _audit.experiments.exp_b_artefact
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from _audit import common as C

# Features that are exactly zero for >=90% of patients in the balanced test set.
ARTEFACT_FEATURES = ['225624_diff_0', '225624_diff_1', '225624_diff_2',
                     '220621_diff_0', '220621_diff_1', '220621_diff_2',
                     '220645_diff_1', '220545_diff_2']


def model():
    return GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                      learning_rate=0.1, subsample=0.9,
                                      random_state=C.SEED)


def cv_auc(X, y, groups=None, n_splits=5):
    it = (GroupKFold(n_splits).split(X, y, groups) if groups is not None
          else StratifiedKFold(n_splits, shuffle=True,
                               random_state=C.SEED).split(X, y))
    oof = np.full(len(y), np.nan)
    for tr, te in it:
        m = model(); m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def feature_profile(df: pd.DataFrame, y: np.ndarray, feats):
    """Per-feature firing statistics in a given cohort."""
    base = float(y.mean())
    rows = []
    for f in feats:
        if f not in df.columns:
            continue
        v = df[f].to_numpy(float)
        fires = v != 0.0
        n_fire = int(fires.sum())
        pos_fire = int(((fires) & (y == 1)).sum())
        rows.append({
            'feature': f,
            'itemid': C.itemid_of(f),
            'label': C.LAB_NAMES.get(C.itemid_of(f), f),
            'pct_nonzero': round(100 * n_fire / len(y), 2),
            'n_fire': n_fire,
            'sepsis_if_fires': pos_fire,
            'ppv_if_fires': round(pos_fire / n_fire, 4) if n_fire else None,
            'base_rate': round(base, 4),
            'lift': round((pos_fire / n_fire) / base, 2) if n_fire else None,
            'recall_contribution': round(pos_fire / max(int(y.sum()), 1), 4),
            'auc_as_binary': round(float(roc_auc_score(y, fires.astype(int))), 4),
        })
    return pd.DataFrame(rows)


def cluster_analysis(df, y, name):
    """The union of all artefact indicators, as a single channel."""
    present = [f for f in ARTEFACT_FEATURES if f in df.columns]
    M = df[present].to_numpy(float) != 0.0
    any_fire = M.any(axis=1)
    n_fire = int(any_fire.sum())
    pos = int((any_fire & (y == 1)).sum())
    n_pos = int(y.sum())
    print(f"    [{name}] any artefact fires: {n_fire:,}/{len(y):,} "
          f"({100*n_fire/len(y):.1f}%)  sepsis among them {pos}/{n_fire} "
          f"-> PPV {100*pos/max(n_fire,1):.1f}%  "
          f"captures {100*pos/max(n_pos,1):.1f}% of all sepsis")
    return {
        'cohort': name,
        'n_patients': int(len(y)),
        'n_sepsis': n_pos,
        'base_rate': round(float(y.mean()), 4),
        'n_fire': n_fire,
        'pct_fire': round(100 * n_fire / len(y), 2),
        'sepsis_if_fires': pos,
        'ppv_if_fires': round(pos / n_fire, 4) if n_fire else None,
        'captures_fraction_of_sepsis': round(pos / max(n_pos, 1), 4),
        'auc_union_indicator': round(float(roc_auc_score(y, any_fire.astype(int))), 4),
    }


def ablation(df, y, groups, sel, name):
    """Full 40-feature model vs the same model with artefact columns removed."""
    present = [f for f in sel if f in df.columns]
    without = [c for c in present if c not in ARTEFACT_FEATURES]
    print(f"    [{name}] full {len(present)} feats vs {len(without)} without artefacts")
    a_full = cv_auc(df[present].to_numpy(float), y, groups)
    a_wo = cv_auc(df[without].to_numpy(float), y, groups)
    print(f"      AUC {a_full:.4f} -> {a_wo:.4f}   delta {a_wo-a_full:+.4f}")
    return {'cohort': name, 'auc_full': round(a_full, 4),
            'auc_without_artefacts': round(a_wo, 4),
            'delta': round(a_wo - a_full, 4),
            'n_features_full': len(present),
            'n_features_without': len(without)}


def ablation_per_split(sel, name='balanced_per_split'):
    """Balanced-cohort ablation using the paper's own train/test pairs.

    Pooling the 40 balanced sets and cross-validating leaks the 218 repeated
    sepsis patients across folds (observed AUC 0.998). This uses train_k/test_k
    per split instead, then averages.
    """
    tr_df = C.read_data('data_processed', 'train_selected_feat40.csv')
    va_df = C.read_data('data_processed', 'val_selected_feat40.csv')
    tr_off, te_off = 0, len(tr_df) + len(va_df)
    full = C.load_full()

    present = [f for f in sel if f in full.columns]
    without = [c for c in present if c not in ARTEFACT_FEATURES]
    a_full, a_wo = [], []
    for i in range(1, C.N_SPLITS + 1):
        dtr = C.read_data('data_processed', f'train_{i}.csv')
        dte = C.read_data('data_processed', f'test_{i}.csv')
        itr, ite = dtr['index'].to_numpy(), dte['index'].to_numpy()
        ytr, yte = dtr['label'].to_numpy(), dte['label'].to_numpy()
        for cols, sink in ((present, a_full), (without, a_wo)):
            m = model()
            m.fit(full.iloc[tr_off:tr_off+len(tr_df)].iloc[itr][cols].to_numpy(float), ytr)
            p = m.predict_proba(
                full.iloc[te_off:].iloc[ite][cols].to_numpy(float))[:, 1]
            sink.append(roc_auc_score(yte, p))
    print(f"    [{name}] full {len(present)} vs {len(without)} feats: "
          f"{np.mean(a_full):.4f} -> {np.mean(a_wo):.4f} "
          f"({np.mean(a_wo)-np.mean(a_full):+.4f})")
    return {'cohort': name, 'auc_full': round(float(np.mean(a_full)), 4),
            'auc_without_artefacts': round(float(np.mean(a_wo)), 4),
            'delta': round(float(np.mean(a_wo) - np.mean(a_full)), 4),
            'n_features_full': len(present), 'n_features_without': len(without)}


def load_balanced_with_availability():
    tr = C.read_data('data_processed', 'train_selected_feat40.csv')
    va = C.read_data('data_processed', 'val_selected_feat40.csv')
    offset = len(tr) + len(va)
    full = C.load_full()
    frames = []
    for i in range(1, C.N_SPLITS + 1):
        d = C.read_data('data_processed', f'test_{i}.csv')
        idx = d['index'].to_numpy()
        b = full.iloc[offset + idx].reset_index(drop=True)
        b['split'] = i
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT B — artefact-conditional performance")
    print("=" * 74)
    sel = [str(s) for s in pickle.load(
        open(C.DATA_ROOT / 'models' / 'GBM_feat_selection_feat50(40).pkl', 'rb'))]

    print("\n[1/3] per-feature profile, FULL cohort (2.41%)")
    full = C.load_full()
    y_full = full['label'].to_numpy()
    prof_full = feature_profile(full, y_full, ARTEFACT_FEATURES)
    print(prof_full.to_string(index=False))

    print("\n[2/3] union-indicator behaviour")
    cl_full = cluster_analysis(full, y_full, 'full')

    bal = load_balanced_with_availability()
    y_bal, g_bal = bal['label'].to_numpy(), bal['split'].to_numpy()
    cl_bal = cluster_analysis(bal, y_bal, 'balanced')

    print("\n[3/3] leave-the-artefacts-out ablation")
    abl_full = ablation(full, y_full, None, sel, 'full')
    abl_bal = ablation_per_split(sel)

    C.save(prof_full, 'exp_b_feature_profile_full')
    C.save({'union_full': cl_full, 'union_balanced': cl_bal,
            'ablation_full': abl_full, 'ablation_balanced': abl_bal,
            'artefact_features': ARTEFACT_FEATURES}, 'exp_b_summary')

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  full cohort: artefact cluster reaches {cl_full['pct_fire']:.1f}% of "
          f"patients, PPV {100*cl_full['ppv_if_fires']:.1f}% vs base "
          f"{100*cl_full['base_rate']:.2f}%, captures "
          f"{100*cl_full['captures_fraction_of_sepsis']:.1f}% of all sepsis")
    print(f"  ablation: {abl_full['auc_full']:.4f} -> "
          f"{abl_full['auc_without_artefacts']:.4f} "
          f"({abl_full['delta']:+.4f})")
