"""Experiment E2 — does counterfactual measurement augmentation help?

The scientific claim DeSAFE makes is deliberately not "higher AUC". CMA removes
information, so on a same-distribution test set AUC should fall. Three claims are
tested instead:

  C1  calibration is at least as good, because the surveillance channel is now
      explicit rather than folded into value features
  C2  net benefit at a clinically plausible threshold is not worse
  C3  the model exposes a per-patient surveillance share

The baseline is an `EntangledBaseline` fitted on the reference pipeline's own
feature style: value summaries computed over the full 24-hour grid, which is what
makes range and diff features track the mask.

Usage: py -m _audit.experiments.exp_e2_desafe
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _audit import common as C
from _audit import desafe as D


def entangled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reference-style value features: computed over the whole grid.

    Deliberately reproduces the artefact — `spread` and `nchange` are functions of
    the mask, exactly as `*_range` and `*_diff_*` are in the reference pipeline.
    """
    out = {}
    for iid in D.LAB_ITEMIDS + D.VITAL_ITEMIDS:
        B = D.hourly(df, iid)
        if B is None:
            continue
        out[f'E_{iid}_mean'] = np.nanmean(B, axis=1)
        out[f'E_{iid}_range'] = np.nanmax(B, axis=1) - np.nanmin(B, axis=1)
        out[f'E_{iid}_nchange'] = (~np.isclose(np.diff(B, axis=1), 0, atol=1e-12)).sum(axis=1)
    if 'hosp_to_icu' in df.columns:
        out['E_hosp_to_icu'] = df['hosp_to_icu'].to_numpy(float)
    return pd.DataFrame(out, index=df.index)


def net_benefit(y, p, t):
    pred = p >= t
    n = len(y)
    tp = ((pred) & (y == 1)).sum() / n
    fp = ((pred) & (y == 0)).sum() / n
    return float(tp - fp * (t / (1 - t)))


def run(use_cma=True, copies=2):
    df = C.load_full()
    y = df['label'].to_numpy()
    P, M, I, S = D.build_all(df)
    E = entangled_features(df)
    dec = _decile(df)
    pool, _ = D.MaskDonorPool.fit(df, y) if use_cma else (None, None)

    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True,
                                                    random_state=C.SEED).split(P, y)):
        # --- baseline
        from _audit import desafe as DD
        base = DD.EntangledBaseline()
        base.fit(E.iloc[tr], y[tr])
        pb = base.predict_proba(E.iloc[te])

        # --- DeSAFE
        m = DD.DeSAFE()
        if use_cma:
            Pa, Ma, Ia, ya = DD.cma_augment(
                df.iloc[tr].reset_index(drop=True), y[tr], dec[tr], pool,
                copies=copies)
            m.fit(Pa, Ma, Ia, S.iloc[tr].reset_index(drop=True), ya)
        else:
            m.fit(P.iloc[tr], M.iloc[tr], I.iloc[tr], S.iloc[tr], y[tr])
        pd_ = m.predict_proba(P.iloc[te], M.iloc[te], I.iloc[te], S.iloc[te])

        rows.append({
            'fold': fold,
            'base_auc': roc_auc_score(y[te], pb),
            'desafe_auc': roc_auc_score(y[te], pd_),
            'base_brier': brier_score_loss(y[te], pb),
            'desafe_brier': brier_score_loss(y[te], pd_),
            'base_nb05': net_benefit(y[te], pb, 0.05),
            'desafe_nb05': net_benefit(y[te], pd_, 0.05),
            'base_nb10': net_benefit(y[te], pb, 0.10),
            'desafe_nb10': net_benefit(y[te], pd_, 0.10),
        })
        print(f"  fold {fold}: base AUC {rows[-1]['base_auc']:.4f} "
              f"-> desafe {rows[-1]['desafe_auc']:.4f}")
    return pd.DataFrame(rows)


def _decile(df):
    for iid in ['220045', '220052']:
        B = D.hourly(df, iid)
        if B is not None:
            d = (~np.isclose(np.diff(B, axis=1), 0, atol=1e-9)).sum(axis=1)
            return pd.qcut(pd.Series(d), 10, labels=False, duplicates='drop').to_numpy()
    return np.zeros(len(df), dtype=int)


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT E2 — CMA on/off, versus entangled baseline")
    print("=" * 74)
    out = {}
    for tag, cma in (('no_cma', False), ('cma', True)):
        print(f"\n[{tag}]")
        d = run(use_cma=cma)
        d.to_csv(C.RESULTS / f'exp_e2_{tag}.csv', index=False)
        s = d.mean(numeric_only=True).round(4).to_dict()
        out[tag] = s
        print(f"  mean: base AUC {s['base_auc']:.4f} | desafe AUC {s['desafe_auc']:.4f} "
              f"({s['desafe_auc']-s['base_auc']:+.4f})")
        print(f"        base Brier {s['base_brier']:.4f} | desafe {s['desafe_brier']:.4f}")
        print(f"        base NB@0.10 {s['base_nb10']:+.4f} | desafe {s['desafe_nb10']:+.4f}")
    C.save(out, 'exp_e2_summary')
