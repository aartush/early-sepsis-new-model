"""Experiment A — the measurement-availability null model (`M_only`).

Question: how much of a sepsis model's discriminative signal is recoverable from
*whether and when measurements were taken*, with no physiological value seen?

Four nested views, identical folds, identical labels:

  V     physiology only — the 40 selected features' values (their model's view)
  M     measurement only — availability features, no values
  M+S   measurement + workflow/static
  A     all — values + availability

Headline quantity, the **measurement-explained fraction**:

      MEF = (AUC_M - 0.5) / (AUC_A - 0.5)

the proportion of the full model's above-chance signal that survives when every
physiological *value* is deleted and only the measurement process is kept.

Three cohorts, each answering a different question:

  FULL     45,148 patients, 2.41% prevalence. The honest population.
  PAPER    9,030 patients — `test_selected_feat40`, i.e. the paper's own test
           split before 1:1 balancing. This is the cohort the paper's numbers
           would have applied to had it not balanced.
  BAL      17,440 rows — the 40 balanced test sets stacked. Comparable to the
           published table but patient-repeated, so it is reported last and its
           optimism is flagged.

Cohort mapping is verified, not assumed: `train/val/test_selected_feat40`
concatenate to exactly 45,148 rows and match `shared_dataset_mimiciv.csv`
row-for-row, and `test_k.csv`'s `index` column indexes into
`test_selected_feat40` (checked in `verify_cohort_mapping`).

Guard: GroupKFold by balanced-set id in the BAL cohort, so a patient can never be
in both a training and a scoring fold. Plain KFold inflates AUC here; only the
grouped version is reported.

Usage:
    py -m _audit.experiments.exp_a_availability
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from _audit import common as C

N_EST, DEPTH = 120, 4


def model():
    return GradientBoostingClassifier(n_estimators=N_EST, max_depth=DEPTH,
                                      learning_rate=0.1, subsample=0.9,
                                      random_state=C.SEED)


def cv_auc(X, y, groups=None, n_splits=5):
    it = (GroupKFold(n_splits=n_splits).split(X, y, groups) if groups is not None
          else StratifiedKFold(n_splits, shuffle=True,
                               random_state=C.SEED).split(X, y))
    oof = np.full(len(y), np.nan)
    for tr, te in it:
        m = model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def selected_features():
    return [str(s) for s in pickle.load(
        open(C.DATA_ROOT / 'models' / 'GBM_feat_selection_feat50(40).pkl', 'rb'))]


def verify_cohort_mapping():
    """Assert the joins this experiment depends on. Fails loudly if violated."""
    tr = C.read_data('data_processed', 'train_selected_feat40.csv')
    va = C.read_data('data_processed', 'val_selected_feat40.csv')
    te = C.read_data('data_processed', 'test_selected_feat40.csv')
    full = C.load_full()
    feat = [c for c in tr.columns if c != 'label']
    joined = pd.concat([tr, va, te], ignore_index=True)
    assert len(joined) == len(full), (len(joined), len(full))
    assert np.allclose(joined[feat].to_numpy(float), full[feat].to_numpy(float),
                       equal_nan=True), "split files are not row-aligned with cohort"
    t1 = C.read_data('data_processed', 'test_1.csv')
    idx = t1['index'].to_numpy()
    assert np.allclose(te.iloc[idx][feat].to_numpy(float),
                       t1[feat].to_numpy(float)), "index does not map into test split"
    return tr, va, te, full, feat


def build_views(df, sel, has_values=True):
    """Return (view_name -> column list, augmented frame).

    Availability columns are *derived* names (e.g. '220621_n_change') that do not
    exist in the cohort frame, so they must be concatenated on before any column
    filter. Filtering them against `df.columns` silently reduces the M view to the
    one real column (`hosp_to_icu`) — that bug produced an M+S view of "1 feats".
    """
    avail = C.availability_features(df)
    avail = avail.loc[:, ~avail.columns.duplicated()]
    # hosp_to_icu already exists in the cohort frame; carry only new columns so
    # the augmented frame has no duplicate names.
    carry = [c for c in avail.columns if c not in df.columns]
    aug = pd.concat([df.reset_index(drop=True),
                     avail[carry].reset_index(drop=True)], axis=1)
    meas = [c for c in avail.columns if c != 'hosp_to_icu']
    views = {}
    if has_values:
        views['V_physiology'] = [c for c in sel if c in df.columns]
        views['M_measurement'] = meas
        views['M+S_meas_static'] = list(avail.columns)
        views['A_all'] = list(dict.fromkeys(views['V_physiology'] + meas))
    else:
        views['M_measurement'] = meas
        views['M+S_meas_static'] = list(avail.columns)
    return views, aug


def evaluate_views(df, views, y, groups=None):
    """`df` here is the augmented frame, so every view name resolves."""
    out = {}
    for name, cols in views.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            raise ValueError(f"view {name} resolved to zero columns")
        auc = cv_auc(df[cols].to_numpy(float), y, groups)
        out[name] = {'auc': round(auc, 4), 'n_features': len(cols)}
        print(f"    {name:18s} {len(cols):4d} feats  AUC={auc:.4f}")
    if 'A_all' in out:
        out['MEF_measurement_only'] = round(
            (out['M_measurement']['auc'] - 0.5) / (out['A_all']['auc'] - 0.5), 4)
        out['MEF_meas_plus_static'] = round(
            (out['M+S_meas_static']['auc'] - 0.5) / (out['A_all']['auc'] - 0.5), 4)
    return out


def run_full(sel):
    print("\n[1/3] FULL cohort (2.41% prevalence, honest)")
    df = C.load_full()
    y = df['label'].to_numpy()
    print(f"    n={len(df):,}  sepsis={int(y.sum())}  prevalence={100*y.mean():.2f}%")
    views, aug = build_views(df, sel, has_values=True)
    r = evaluate_views(aug, views, y)
    r.update({'cohort': 'full', 'n': int(len(df))})
    return r


def run_paper_split(sel):
    print("\n[2/3] PAPER test split (9,030 patients, unbalanced — pre-balancing)")
    tr, va, te, full, feat = verify_cohort_mapping()
    offset = len(tr) + len(va)
    sub = full.iloc[offset:offset + len(te)].reset_index(drop=True)
    y = sub['label'].to_numpy()
    print(f"    n={len(sub):,}  sepsis={int(y.sum())}  prevalence={100*y.mean():.2f}%")
    views, aug = build_views(sub, sel, has_values=True)
    r = evaluate_views(aug, views, y)
    r.update({'cohort': 'paper_test_split', 'n': int(len(sub)),
              'prevalence': round(float(y.mean()), 4)})
    return r


def run_balanced(sel):
    """Balanced cohort, evaluated the way the paper actually did it.

    Important: the same 218 sepsis patients appear in ALL 40 balanced sets, so
    pooling them and cross-validating leaks patients across folds — an earlier
    draft of this function produced AUC 0.998 for exactly that reason. The only
    valid construction is the paper's own: train on train_k, test on test_k, and
    average over the 40 patient-disjoint pairs.
    """
    print("\n[3/3] BALANCED cohort, per-split train/test (paper protocol)")
    tr_df = C.read_data('data_processed', 'train_selected_feat40.csv')
    va_df = C.read_data('data_processed', 'val_selected_feat40.csv')
    te_df = C.read_data('data_processed', 'test_selected_feat40.csv')
    tr_off, te_off = 0, len(tr_df) + len(va_df)

    full = C.load_full()
    tr_aug = pd.concat([full.iloc[tr_off:tr_off + len(tr_df)].reset_index(drop=True),
                        C.availability_features(full.iloc[tr_off:tr_off + len(tr_df)]
                                                .reset_index(drop=True))], axis=1)
    te_aug = pd.concat([full.iloc[te_off:].reset_index(drop=True),
                        C.availability_features(full.iloc[te_off:]
                                                .reset_index(drop=True))], axis=1)
    tr_aug = tr_aug.loc[:, ~tr_aug.columns.duplicated()]
    te_aug = te_aug.loc[:, ~te_aug.columns.duplicated()]

    views, _ = build_views(full, sel, has_values=True)

    scores = {k: [] for k in views}
    for i in range(1, C.N_SPLITS + 1):
        dtr = C.read_data('data_processed', f'train_{i}.csv')
        dte = C.read_data('data_processed', f'test_{i}.csv')
        itr, ite = dtr['index'].to_numpy(), dte['index'].to_numpy()
        ytr = dtr['label'].to_numpy(); yte = dte['label'].to_numpy()
        for name, cols in views.items():
            cols = [c for c in cols if c in tr_aug.columns]
            m = model()
            m.fit(tr_aug.iloc[itr][cols].to_numpy(float), ytr)
            p = m.predict_proba(te_aug.iloc[ite][cols].to_numpy(float))[:, 1]
            scores[name].append(roc_auc_score(yte, p))

    out = {}
    for name, v in scores.items():
        out[name] = {'auc': round(float(np.mean(v)), 4),
                     'auc_sd': round(float(np.std(v)), 4),
                     'n_features': len([c for c in views[name] if c in tr_aug.columns])}
        print(f"    {name:18s} {out[name]['n_features']:4d} feats  "
              f"AUC={out[name]['auc']:.4f} (sd {out[name]['auc_sd']:.4f}, 40 splits)")
    out['MEF_measurement_only'] = round(
        (out['M_measurement']['auc'] - 0.5) / (out['A_all']['auc'] - 0.5), 4)
    out['MEF_meas_plus_static'] = round(
        (out['M+S_meas_static']['auc'] - 0.5) / (out['A_all']['auc'] - 0.5), 4)
    out.update({'cohort': 'balanced_per_split', 'n': int(len(te_df)) * C.N_SPLITS})
    return out


def feature_ranking():
    print("\n[+] single-feature availability AUC ranking (full cohort)")
    df = C.load_full()
    y = df['label'].to_numpy()
    avail = C.availability_features(df)
    rows = []
    for c in avail.columns:
        v = avail[c].to_numpy(float)
        if not np.isfinite(v).all() or v.std() == 0:
            continue
        auc = float(roc_auc_score(y, v))
        iid = C.itemid_of(c)
        rows.append({'feature': c, 'itemid': iid,
                     'label': C.LAB_NAMES.get(iid, c),
                     'kind': c.rsplit('_', 1)[1] if iid else 'static',
                     'auc': round(auc, 4),
                     'equiv_distance_from_chance': round(abs(auc - 0.5), 4)})
    return pd.DataFrame(rows).sort_values('auc', ascending=False)


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT A — measurement-availability null model")
    print("=" * 74)
    sel = selected_features()
    res = {'full': run_full(sel),
           'paper_test_split': run_paper_split(sel),
           'balanced_per_split': run_balanced(sel)}
    rank = feature_ranking()
    C.save(rank, 'exp_a_feature_ranking')
    res['top_availability_features'] = rank.head(12).to_dict('records')
    C.save(res, 'exp_a_summary')

    print("\n" + "=" * 74)
    print("MEASUREMENT-EXPLAINED FRACTION  (MEF = (AUC_M-0.5)/(AUC_A-0.5))")
    print("=" * 74)
    print(f"{'cohort':18s} {'V':>7s} {'M':>7s} {'M+S':>7s} {'A':>7s} {'MEF(M)':>9s} {'MEF(M+S)':>10s}")
    for k in ("full", "paper_test_split", "balanced_per_split"):
        r = res[k]
        print(f"{k:18s} {r['V_physiology']['auc']:7.4f} "
              f"{r['M_measurement']['auc']:7.4f} "
              f"{r['M+S_meas_static']['auc']:7.4f} "
              f"{r['A_all']['auc']:7.4f} "
              f"{100*r['MEF_measurement_only']:8.1f}% "
              f"{100*r['MEF_meas_plus_static']:9.1f}%")
