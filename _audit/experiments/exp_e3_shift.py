"""Experiment E3 — robustness under a shift in monitoring practice.

This is the experiment that justifies DeSAFE's existence. The claim is not that it
predicts better on the distribution it was trained on; it is that it degrades less
when the *monitoring policy* changes, because CMA removed the coupling between the
observation mask and the outcome.

Shift operators (applied to the test set only, never to training)

  sparsify(k)   randomly delete a fraction k of observed lab hours, refilling with
                the previous value — exactly what a hospital that draws labs less
                often would produce
  densify(k)    insert interpolated observations between existing ones — a unit
                that measures more aggressively
  policy-flip   replace a patient's mask with a donor mask drawn from the opposite
                stay-duration decile, simulating transfer to a unit with a
                different monitoring culture

Three models are compared at each shift level:

  entangled     reference-style model on mask-contaminated value features
  desafe        additive dual-channel model, trained on clean data
  desafe+cma    additive dual-channel model, trained with counterfactual
                measurement augmentation

Reported: AUC at each shift level, and the **degradation slope** (AUC lost per unit
of shift). A more robust model has a flatter slope. Because the entangled model can
read the mask directly, it should be the most fragile.

Usage: py -m _audit.experiments.exp_e3_shift
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _audit import common as C
from _audit import desafe as D
from _audit.experiments.exp_e2_desafe import entangled_features


# ---------------------------------------------------------------------------
# shift operators
# ---------------------------------------------------------------------------
def sparsify(df, frac, rng):
    """Delete a fraction of observed lab hours by forward-filling across them."""
    out = df.copy()
    for iid in D.LAB_ITEMIDS:
        B = D.hourly(df, iid)
        if B is None:
            continue
        cols = [c for c in df.columns
                if C.feature_family(c) == 'hour' and C.itemid_of(c) == iid]
        cols = sorted(cols, key=lambda x: int(x.rsplit('_', 1)[1]))
        M = D.recover_mask(B, 'lab')
        B2 = B.copy()
        for j in range(len(B2)):
            obs = np.flatnonzero(M[j])
            if len(obs) <= 1:
                continue
            drop = rng.random(len(obs)) < frac
            keep = obs[~drop]
            if len(keep) == 0:
                keep = obs[:1]
            # forward-fill kept values across the grid
            filled = np.empty(B2.shape[1])
            cur = B2[j][keep[0]]
            ki = 0
            for t in range(B2.shape[1]):
                if ki < len(keep) and t == keep[ki]:
                    cur = B2[j][t]; ki += 1
                filled[t] = cur
            B2[j] = filled
        for t, c in enumerate(cols):
            out[c] = B2[:, t]
    return out


def policy_flip(df, rng):
    """Replace each patient's lab mask with one from the opposite duration decile."""
    out = df.copy()
    y = df['label'].to_numpy()
    dur = None
    for iid in ['220045', '220052']:
        B = D.hourly(df, iid)
        if B is not None:
            d = (~np.isclose(np.diff(B, axis=1), 0, atol=1e-9)).sum(axis=1)
            dur = d if dur is None else np.maximum(dur, d)
    dec = pd.qcut(pd.Series(dur), 10, labels=False, duplicates='drop').to_numpy()
    # only as many deciles as actually exist; mirroring around the observed max
    # avoids the earlier out-of-range bug (9 - dec when dec max was 6)
    n_dec = int(np.nanmax(dec)) + 1
    flip = (n_dec - 1) - dec

    for iid in D.LAB_ITEMIDS:
        B = D.hourly(df, iid)
        if B is None:
            continue
        cols = [c for c in df.columns
                if C.feature_family(c) == 'hour' and C.itemid_of(c) == iid]
        cols = sorted(cols, key=lambda x: int(x.rsplit('_', 1)[1]))
        M = D.recover_mask(B, 'lab')
        B2 = B.copy()
        for j in range(len(B2)):
            src = np.flatnonzero((dec == flip[j]) & (y == y[j]))
            if len(src) == 0:
                continue
            donor = M[rng.integers(len(src), size=1)[0]] if False else \
                M[src[rng.integers(len(src))]]
            obs = B2[j][M[j]]
            if obs.size == 0:
                continue
            B2[j][donor] = rng.choice(obs, size=int(donor.sum()), replace=True)
        for t, c in enumerate(cols):
            out[c] = B2[:, t]
    return out


def _decile(df):
    for iid in ['220045', '220052']:
        B = D.hourly(df, iid)
        if B is not None:
            d = (~np.isclose(np.diff(B, axis=1), 0, atol=1e-9)).sum(axis=1)
            return pd.qcut(pd.Series(d), 10, labels=False, duplicates='drop').to_numpy()
    return np.zeros(len(df), dtype=int)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    df = C.load_full()
    y = df['label'].to_numpy()
    dec = _decile(df)
    pool, _ = D.MaskDonorPool.fit(df, y)
    rng = np.random.default_rng(C.SEED)

    P, M, I, S = D.build_all(df)
    E = entangled_features(df)

    fold = next(iter(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(P, y)))
    tr, te = fold

    # --- train the three models once
    print("  training models ...")
    base = D.EntangledBaseline(); base.fit(E.iloc[tr], y[tr])

    d_nocma = D.DeSAFE(); d_nocma.fit(P.iloc[tr], M.iloc[tr], I.iloc[tr], S.iloc[tr], y[tr])

    Pa, Ma, Ia, ya = D.cma_augment(
        df.iloc[tr].reset_index(drop=True), y[tr], dec[tr], pool, copies=2)
    d_cma = D.DeSAFE(); d_cma.fit(Pa, Ma, Ia, S.iloc[tr].reset_index(drop=True), ya)

    te_df = df.iloc[te].reset_index(drop=True)
    y_te = y[te]

    levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    rows = []
    for frac in levels:
        if frac == 0:
            shifted = te_df
        else:
            shifted = sparsify(te_df, frac, np.random.default_rng(C.SEED + int(frac * 100)))
        Ps, Ms, Is, Ss = D.build_all(shifted)
        Es = entangled_features(shifted)
        rows.append({
            'shift': frac,
            'entangled': roc_auc_score(y_te, base.predict_proba(Es)),
            'desafe': roc_auc_score(y_te, d_nocma.predict_proba(Ps, Ms, Is, Ss)),
            'desafe_cma': roc_auc_score(y_te, d_cma.predict_proba(Ps, Ms, Is, Ss)),
        })
        print(f"  sparsify {frac:.1f}: entangled {rows[-1]['entangled']:.4f} | "
              f"desafe {rows[-1]['desafe']:.4f} | desafe+cma {rows[-1]['desafe_cma']:.4f}")

    # policy flip
    flipped = policy_flip(te_df, np.random.default_rng(C.SEED + 7))
    Pf, Mf, If, Sf = D.build_all(flipped)
    Ef = entangled_features(flipped)
    flip_row = {
        'shift': 'policy_flip',
        'entangled': roc_auc_score(y_te, base.predict_proba(Ef)),
        'desafe': roc_auc_score(y_te, d_nocma.predict_proba(Pf, Mf, If, Sf)),
        'desafe_cma': roc_auc_score(y_te, d_cma.predict_proba(Pf, Mf, If, Sf)),
    }
    print(f"  policy flip: entangled {flip_row['entangled']:.4f} | "
          f"desafe {flip_row['desafe']:.4f} | desafe+cma {flip_row['desafe_cma']:.4f}")

    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / 'exp_e3_shift_curve.csv', index=False)

    # degradation slope over the sparsify range
    slopes = {}
    x = d['shift'].to_numpy()
    for col in ('entangled', 'desafe', 'desafe_cma'):
        yv = d[col].to_numpy()
        A = np.vstack([x, np.ones_like(x)]).T
        slope = np.linalg.lstsq(A, yv, rcond=None)[0][0]
        slopes[col] = round(float(slope), 4)
    print("\n  degradation slope (AUC lost per unit sparsification):")
    for k, v in slopes.items():
        print(f"    {k:12s} {v:+.4f}")

    C.save({'curve': rows, 'policy_flip': flip_row,
            'slopes': slopes,
            'shift_operator': 'random deletion of observed lab hours, forward-filled',
            'folds': 'single held-out fold (fold 0 of 5)'}, 'exp_e3_summary')


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT E3 — robustness to monitoring-practice shift")
    print("=" * 74)
    main()
