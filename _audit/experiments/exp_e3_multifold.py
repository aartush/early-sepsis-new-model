"""Experiment E3 (multi-fold, reproducible) — robustness to monitoring-practice shift.

Replaces the single-fold version. Two problems with the original:

1. It evaluated on ONE held-out fold. Two runs of identical code returned 0.8886
   and 0.8825 for the same quantity — a 0.006 swing that is larger than several of
   the effects being claimed. Reporting one fold would have been the same error we
   criticise in the study under audit.
2. `HistGradientBoosting` uses threaded reductions, so exact bit-reproducibility
   is not available. Averaging over folds is the honest answer.

This version trains and evaluates on all 5 folds and reports mean ± sd, with the
per-fold values written out so the variance is visible.

Shift operators (test set only, never training):
  sparsify(k)   delete a fraction k of observed lab hours, forward-filling
  policy_flip   replace a patient's mask with one from an opposite-duration donor

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
from _audit.experiments.exp_e3_shift import sparsify, policy_flip, _decile

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]


def main():
    df = C.load_full()
    y = df['label'].to_numpy()
    dec = _decile(df)
    pool, _ = D.MaskDonorPool.fit(df, y)

    P, M, I, S = D.build_all(df)
    E = entangled_features(df)

    folds = list(StratifiedKFold(5, shuffle=True,
                                 random_state=C.SEED).split(P, y))
    per_fold = []
    for fi, (tr, te) in enumerate(folds):
        print(f"\n[fold {fi}] training ...")
        base = D.EntangledBaseline(); base.fit(E.iloc[tr], y[tr])
        d_nocma = D.DeSAFE(); d_nocma.fit(P.iloc[tr], M.iloc[tr], I.iloc[tr],
                                          S.iloc[tr], y[tr])
        Pa, Ma, Ia, ya = D.cma_augment(
            df.iloc[tr].reset_index(drop=True), y[tr], dec[tr], pool, copies=2)
        d_cma = D.DeSAFE(); d_cma.fit(Pa, Ma, Ia,
                                      S.iloc[tr].reset_index(drop=True), ya)

        te_df = df.iloc[te].reset_index(drop=True)
        y_te = y[te]

        for frac in LEVELS:
            shifted = te_df if frac == 0 else sparsify(
                te_df, frac, np.random.default_rng(C.SEED + int(frac * 100)))
            Ps, Ms, Is, Ss = D.build_all(shifted)
            Es = entangled_features(shifted)
            per_fold.append({
                'fold': fi, 'operator': 'sparsify', 'level': frac,
                'entangled': roc_auc_score(y_te, base.predict_proba(Es)),
                'desafe': roc_auc_score(y_te, d_nocma.predict_proba(Ps, Ms, Is, Ss)),
                'desafe_cma': roc_auc_score(y_te, d_cma.predict_proba(Ps, Ms, Is, Ss)),
            })

        flipped = policy_flip(te_df, np.random.default_rng(C.SEED + 7))
        Pf, Mf, If, Sf = D.build_all(flipped)
        Ef = entangled_features(flipped)
        per_fold.append({
            'fold': fi, 'operator': 'policy_flip', 'level': np.nan,
            'entangled': roc_auc_score(y_te, base.predict_proba(Ef)),
            'desafe': roc_auc_score(y_te, d_nocma.predict_proba(Pf, Mf, If, Sf)),
            'desafe_cma': roc_auc_score(y_te, d_cma.predict_proba(Pf, Mf, If, Sf)),
        })
        print(f"  fold {fi}: " + " ".join(
            f"{r['operator'][:5]}{r['level'] if r['level']==r['level'] else ''}"
            f"={r['desafe_cma']:.3f}"
            for r in per_fold if r['fold'] == fi))

    d = pd.DataFrame(per_fold)
    d.to_csv(C.RESULTS / 'exp_e3_per_fold.csv', index=False)

    curve = []
    for frac in LEVELS:
        sub = d[(d.operator == 'sparsify') & (d.level == frac)]
        curve.append({
            'shift': frac,
            'entangled': round(float(sub['entangled'].mean()), 4),
            'desafe': round(float(sub['desafe'].mean()), 4),
            'desafe_cma': round(float(sub['desafe_cma'].mean()), 4),
            'entangled_sd': round(float(sub['entangled'].std()), 4),
            'desafe_cma_sd': round(float(sub['desafe_cma'].std()), 4),
            'n_folds': int(len(sub)),
        })
        print(f"  sparsify {frac:.1f}: entangled {curve[-1]['entangled']:.4f} "
              f"(±{curve[-1]['entangled_sd']:.4f}) | "
              f"desafe_cma {curve[-1]['desafe_cma']:.4f} "
              f"(±{curve[-1]['desafe_cma_sd']:.4f})")

    flip_sub = d[d.operator == 'policy_flip']
    flip_row = {
        'entangled': round(float(flip_sub['entangled'].mean()), 4),
        'desafe': round(float(flip_sub['desafe'].mean()), 4),
        'desafe_cma': round(float(flip_sub['desafe_cma'].mean()), 4),
        'entangled_sd': round(float(flip_sub['entangled'].std()), 4),
        'desafe_cma_sd': round(float(flip_sub['desafe_cma'].std()), 4),
    }
    print(f"  policy flip: entangled {flip_row['entangled']:.4f} | "
          f"desafe_cma {flip_row['desafe_cma']:.4f}")

    slopes = {}
    x = np.array([c['shift'] for c in curve])
    for col in ('entangled', 'desafe', 'desafe_cma'):
        yv = np.array([c[col] for c in curve])
        A = np.vstack([x, np.ones_like(x)]).T
        slopes[col] = round(float(np.linalg.lstsq(A, yv, rcond=None)[0][0]), 4)

    # paired win/loss over folds at each level — the honest significance check
    diffs = {}
    for frac in LEVELS:
        sub = d[(d.operator == 'sparsify') & (d.level == frac)]
        delta = (sub['desafe_cma'] - sub['entangled']).to_numpy()
        diffs[str(frac)] = {'mean_delta': round(float(delta.mean()), 4),
                            'folds_won': int((delta > 0).sum()),
                            'n_folds': int(len(delta))}
        print(f"  delta vs baseline @{frac:.1f}: "
              f"{delta.mean():+.4f}  ({int((delta>0).sum())}/{len(delta)} folds won)")

    C.save({'curve': curve, 'policy_flip': flip_row, 'slopes': slopes,
            'paired_deltas': diffs,
            'n_folds': len(folds),
            'note': 'mean over 5 folds; per-fold values in exp_e3_per_fold.csv'},
           'exp_e3_summary')


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT E3 — robustness to monitoring shift (5-fold, reproducible)")
    print("=" * 74)
    main()
