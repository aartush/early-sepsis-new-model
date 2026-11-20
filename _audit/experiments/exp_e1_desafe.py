"""Experiment E1 — additive decomposition and per-patient surveillance share.

DeSAFE's logit is exactly f_P + f_M + f_S, so the decomposition is not a post-hoc
attribution: it is the model. This script reports

  * AUC of each channel alone and of the additive combination,
  * the distribution of the surveillance share sigma per patient,
  * how sigma behaves for cases vs controls, and across stay durations,
  * what happens to risk when the surveillance channel is neutralised
    (the counterfactual "as if monitoring were uninformative" prediction).

The point is not a better AUC. It is that sigma is a quantity the baseline cannot
produce at all.

Usage: py -m _audit.experiments.exp_e1_desafe
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _audit import common as C
from _audit import desafe as D


def cv_eval(df, y, n_splits=5, cma_copies=0):
    P, M, I, S = D.build_all(df)
    dec = _stay_decile(df)
    pool = None
    if cma_copies:
        pool, dec = D.MaskDonorPool.fit(df, y)

    oof = np.zeros(len(y))
    comp_rows = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True,
                                  random_state=C.SEED).split(P, y):
        model = D.DeSAFE()
        if cma_copies and pool is not None:
            Pa, Ma, Ia, ya = D.cma_augment(
                df.iloc[tr].reset_index(drop=True), y[tr], dec[tr], pool,
                copies=cma_copies)
            model.fit(Pa, Ma, Ia, S.iloc[tr].reset_index(drop=True), ya)
        else:
            model.fit(P.iloc[tr], M.iloc[tr], I.iloc[tr], S.iloc[tr], y[tr])
        oof[te] = model.predict_proba(P.iloc[te], M.iloc[te], I.iloc[te], S.iloc[te])
        comp = model.components(P.iloc[te], M.iloc[te], I.iloc[te], S.iloc[te])
        comp_rows.append((comp, te))
    return oof, comp_rows, P, M, S


def _stay_decile(df):
    for iid in ['220045', '220052']:
        B = D.hourly(df, iid)
        if B is not None:
            d = (~np.isclose(np.diff(B, axis=1), 0, atol=1e-9)).sum(axis=1)
            return _quant(df, d)
    return np.zeros(len(df), dtype=int)


def _quant(df, v):
    import pandas as pd
    return pd.qcut(pd.Series(v), 10, labels=False, duplicates='drop').to_numpy()


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT E1 — DeSAFE additive decomposition")
    print("=" * 74)
    df = C.load_full()
    y = df['label'].to_numpy()
    print(f"  cohort n={len(df):,}  sepsis={int(y.sum())}  "
          f"prevalence={100*y.mean():.2f}%")

    oof, comp_rows, P, M, S = cv_eval(df, y, cma_copies=0)
    auc = roc_auc_score(y, oof)
    print(f"\n  DeSAFE (no CMA) out-of-fold AUC = {auc:.4f}")

    # per-channel AUC by fitting the same architecture on one channel at a time
    print("\n  single-channel AUCs (same folds, own expert only):")
    ch_auc = {}
    for name, X in (('P physiology', P), ('M surveillance', M)):
        from sklearn.ensemble import GradientBoostingClassifier
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(X, y):
            m = GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                           learning_rate=0.1, subsample=0.9,
                                           random_state=C.SEED)
            m.fit(X.iloc[tr], y[tr])
            o[te] = m.predict_proba(X.iloc[te])[:, 1]
        ch_auc[name] = roc_auc_score(y, o)
        print(f"    {name:16s} AUC={ch_auc[name]:.4f}")

    # surveillance share per patient
    sigma = np.full(len(y), np.nan)
    for comp, te in comp_rows:
        den = np.abs(comp['P']) + np.abs(comp['M']) + np.abs(comp['S']) + 1e-9
        sigma[te] = np.abs(comp['M']) / den
    sigma = sigma[np.isfinite(sigma)]
    q = np.percentile(sigma, [10, 25, 50, 75, 90])
    print(f"\n  surveillance share sigma = |f_M| / (|f_P|+|f_M|+|f_S|)")
    print(f"    median {np.median(sigma):.3f}   IQR [{q[1]:.3f}, {q[3]:.3f}]   "
          f"p10 {q[0]:.3f}  p90 {q[4]:.3f}")
    yv = y[:len(sigma)]
    print(f"    cases    mean sigma = {sigma[yv == 1].mean():.3f}")
    print(f"    controls mean sigma = {sigma[yv == 0].mean():.3f}")

    # neutralise the surveillance channel
    print("\n  counterfactual: risk with surveillance channel neutralised")
    phys = np.full(len(y), np.nan)
    for comp, te in comp_rows:
        phys[te] = comp['P'] + comp['S']
    ok = np.isfinite(phys)
    if ok.sum() and len(np.unique(y[ok])) == 2:
        print(f"    AUC of physiology-only risk = {roc_auc_score(y[ok], phys[ok]):.4f}")

    C.save({'auc_full': round(float(auc), 4),
            'auc_P': round(float(ch_auc['P physiology']), 4),
            'auc_M': round(float(ch_auc['M surveillance']), 4),
            'sigma_median': round(float(np.median(sigma)), 4),
            'sigma_iqr': [round(float(x), 4) for x in (q[1], q[3])],
            'sigma_cases': round(float(sigma[yv == 1].mean()), 4),
            'sigma_controls': round(float(sigma[yv == 0].mean()), 4),
            'n': int(len(y))}, 'exp_e1_desafe')

    import pandas as pd
    pd.DataFrame({'sigma': sigma, 'label': yv}).to_csv(
        C.RESULTS / 'exp_e1_sigma.csv', index=False)
    print(f"\n  wrote results/exp_e1_desafe.json and exp_e1_sigma.csv")
