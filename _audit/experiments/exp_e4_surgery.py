"""Experiment E4 — the mask-surgery test.

The one real vulnerability in DeSAFE is the interaction channel `I`. I argued it
carries genuine conditional information (a lab that is *trended and moving* is more
alarming than either alone), not smuggled surveillance. But `I` is conditioned on
having >=2 observations, so a reviewer can reasonably claim it partially
reintroduces mask dependence, and the reported robustness advantage is therefore
illusory.

That claim is testable, and the test is worth having independently of DeSAFE.

The surgery
-----------
At inference time only, replace each patient's observation mask with a donor mask
drawn from a *different* patient (stratified on label only). Values are untouched:
the patient still has their own measured levels, they are merely placed at hours
that someone else was measured. Then re-score every model.

What each outcome means
  * A model that collapses  -> it was reading the mask, not the physiology.
  * A model that holds      -> its predictions were invariant to observation
                               pattern, which is the property DeSAFE claims.

Three models are compared at five surgery intensities (fraction of patients
re-masked, 0.0 to 1.0):

  entangled        reference-style, mask-contaminated value features
  desafe_physio    DeSAFE with channels P + S only (surveillance removed by
                   construction) — the honest lower bound
  desafe_cma       full DeSAFE (P + M + I + S) trained with CMA

Prediction: the entangled model's AUC should fall furthest, because `range` and
`nchange` are functions of the mask. `desafe_physio` should be nearly flat by
construction. The interesting question is whether `desafe_cma` sits closer to the
flat line than to the entangled collapse — that is the quantitative answer to
"is `I` smuggled surveillance or not".

Reported per model: AUC at each surgery level, and a **mask-sensitivity index**
(MSI) = AUC(0%) - AUC(100%). Lower is better.

Usage: py -m _audit.experiments.exp_e4_surgery
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _audit import common as C
from _audit import desafe as D
from _audit.experiments.exp_e2_desafe import entangled_features
from _audit.experiments.exp_e3_shift import _decile

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def surgery(df: pd.DataFrame, y: np.ndarray, frac: float, rng) -> pd.DataFrame:
    """Re-mask a fraction of patients, leaving their values intact.

    For each selected patient and each lab, a donor mask is taken from a patient
    with the same label. The patient's own observed values are resampled into the
    donor's positions, so the value distribution is preserved and only the
    *placement* changes.
    """
    if frac == 0:
        return df.copy()
    out = df.copy()
    n = len(df)
    chosen = rng.random(n) < frac

    for iid in D.LAB_ITEMIDS:
        B = D.hourly(df, iid)
        if B is None:
            continue
        cols = [c for c in df.columns
                if C.feature_family(c) == 'hour' and C.itemid_of(c) == iid]
        cols = sorted(cols, key=lambda x: int(x.rsplit('_', 1)[1]))
        M = D.recover_mask(B, 'lab')
        B2 = B.copy()
        for j in np.flatnonzero(chosen):
            donors = np.flatnonzero(y == y[j])
            if len(donors) == 0:
                continue
            d = donors[rng.integers(len(donors))]
            donor_mask = M[d]
            obs = B2[j][M[j]]
            if obs.size == 0:
                continue
            B2[j][donor_mask] = rng.choice(obs, size=int(donor_mask.sum()),
                                           replace=True)
        for t, c in enumerate(cols):
            out[c] = B2[:, t]
    return out


def main():
    df = C.load_full()
    y = df['label'].to_numpy()
    dec = _decile(df)
    pool, _ = D.MaskDonorPool.fit(df, y)

    P, M, I, S = D.build_all(df)
    E = entangled_features(df)

    folds = list(StratifiedKFold(5, shuffle=True,
                                 random_state=C.SEED).split(P, y))
    rows = []

    for fi, (tr, te) in enumerate(folds):
        print(f"\n[fold {fi}] training ...")
        base = D.EntangledBaseline(); base.fit(E.iloc[tr], y[tr])

        # P+S only: surveillance removed by construction
        d_phys = D.DeSAFE()
        zero_I = I.mul(0)
        zero_M = M.mul(0)
        d_phys.fit(P.iloc[tr], zero_M.iloc[tr], zero_I.iloc[tr],
                   S.iloc[tr], y[tr])

        Pa, Ma, Ia, ya = D.cma_augment(
            df.iloc[tr].reset_index(drop=True), y[tr], dec[tr], pool, copies=2)
        d_cma = D.DeSAFE(); d_cma.fit(Pa, Ma, Ia,
                                      S.iloc[tr].reset_index(drop=True), ya)

        te_df = df.iloc[te].reset_index(drop=True)
        y_te = y[te]

        for frac in LEVELS:
            surg = surgery(te_df, y_te, frac,
                           np.random.default_rng(C.SEED + int(frac * 100)))
            Ps, Ms, Is, Ss = D.build_all(surg)
            Es = entangled_features(surg)
            rows.append({
                'fold': fi, 'mask_surgery_frac': frac,
                'entangled': roc_auc_score(y_te, base.predict_proba(Es)),
                'desafe_physio': roc_auc_score(
                    y_te, d_phys.predict_proba(Ps, Ms.mul(0), Is.mul(0), Ss)),
                'desafe_cma': roc_auc_score(
                    y_te, d_cma.predict_proba(Ps, Ms, Is, Ss)),
            })
        print(f"  fold {fi} done: " + " ".join(
            f"{r['mask_surgery_frac']:.2f}={r['desafe_cma']:.3f}"
            for r in rows if r['fold'] == fi))

    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / 'exp_e4_per_fold.csv', index=False)

    curve = []
    for frac in LEVELS:
        sub = d[d.mask_surgery_frac == frac]
        curve.append({
            'surgery': frac,
            'entangled': round(float(sub['entangled'].mean()), 4),
            'desafe_physio': round(float(sub['desafe_physio'].mean()), 4),
            'desafe_cma': round(float(sub['desafe_cma'].mean()), 4),
            'entangled_sd': round(float(sub['entangled'].std()), 4),
            'desafe_cma_sd': round(float(sub['desafe_cma'].std()), 4),
        })

    msi = {}
    for k in ('entangled', 'desafe_physio', 'desafe_cma'):
        msi[k] = round(curve[0][k] - curve[-1][k], 4)

    print("\n" + "=" * 74)
    print("MASK SURGERY — AUC as the observation pattern is replaced")
    print("=" * 74)
    print(f"{'surgery':>8s} {'entangled':>12s} {'desafe_P+S':>12s} {'desafe_cma':>12s}")
    for c in curve:
        print(f"{c['surgery']:8.2f} {c['entangled']:12.4f} "
              f"{c['desafe_physio']:12.4f} {c['desafe_cma']:12.4f}")

    print("\nMASK-SENSITIVITY INDEX  (AUC drop from 0% to 100% surgery; lower = better)")
    for k, v in sorted(msi.items(), key=lambda x: x[1]):
        print(f"  {k:16s} {v:+.4f}")

    verdict = {}
    if msi['desafe_cma'] < msi['entangled'] * 0.6:
        verdict['conclusion'] = (
            'DeSAFE is substantially less mask-dependent than the entangled '
            'baseline; the interaction channel does not behave as smuggled '
            'surveillance.')
    elif msi['desafe_cma'] < msi['entangled']:
        verdict['conclusion'] = (
            'DeSAFE is less mask-dependent than the entangled baseline, but the '
            'margin is modest. The interaction channel retains partial mask '
            'sensitivity and the paper must say so.')
    else:
        verdict['conclusion'] = (
            'DeSAFE is no less mask-dependent than the entangled baseline. The '
            'interaction-channel defence does NOT hold and the robustness claim in '
            'E3 must be withdrawn.')
    print(f"\n  VERDICT: {verdict['conclusion']}")

    C.save({'curve': curve, 'msi': msi, 'verdict': verdict,
            'n_folds': len(folds),
            'definition': 'MSI = AUC(0% surgery) - AUC(100% surgery)'},
           'exp_e4_surgery')


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT E4 — mask surgery (is the interaction channel smuggled?)")
    print("=" * 74)
    main()
