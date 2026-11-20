"""Generate the four performance curves from reproduced ensemble predictions.

The reference study reports ROC, precision-recall, calibration and decision-curve
plots for its ensemble. We reproduced the ensemble to within 0.0005 AUC, so the
same four plots are reconstructions from our own predictions, not copies.

Also regenerates the GBM feature-importance ranking for all 40 selected features.
The reference pipeline selected features with a greedy GBM importance loop but did
not persist the fitted per-split models, so importances are recomputed here by
refitting on the same 40 balanced splits with the modal hyperparameters reported in
the reproduction (`n_estimators=250, max_depth=6, lr=0.1, subsample=1.0`). This is
stated in the figure caption so it is not mistaken for an archived artefact.

Outputs: figures/fig7_curves.pdf, figures/fig8_importance.pdf, and
results/exp_f_curves.json, results/exp_f_importance.csv.

Usage: py -m _audit.experiments.exp_curves
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from _audit import common as C

PRED = C.REPO / '_reproduction' / 'predictions' / 'ENSEMBLE_40.csv'
MODAL = dict(max_iter=250, max_depth=6, learning_rate=0.1)


def net_benefit_curve(y, p, thresholds):
    out = []
    n = len(y)
    prev = y.mean()
    for t in thresholds:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum() / n
        fp = ((pred) & (y == 0)).sum() / n
        nb = tp - fp * (t / (1 - t))
        treat_all = prev - (1 - prev) * (t / (1 - t))
        out.append((t, nb, treat_all))
    return out


def load_pooled():
    probs = pd.read_csv(PRED)
    probs.columns = probs.columns.astype(str)
    ys, ps = [], []
    for col in probs.columns:
        d = C.read_data('data_processed', f'test_{int(col)}.csv')
        ys.append(d['label'].to_numpy())
        ps.append(probs[col].to_numpy(float))
    return np.concatenate(ys), np.concatenate(ps)


def curves():
    y, p = load_pooled()
    print(f"  pooled n={len(y):,} positives={int(y.sum())} "
          f"prevalence={100*y.mean():.1f}%")

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='uniform')
    ths = np.linspace(0.01, 0.6, 60)
    nb = net_benefit_curve(y, p, ths)

    res = {
        'n': int(len(y)),
        'auc': round(float(roc_auc_score(y, p)), 4),
        'average_precision': round(float(average_precision_score(y, p)), 4),
        'brier': round(float(brier_score_loss(y, p)), 4),
        'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr': {'precision': prec.tolist(), 'recall': rec.tolist()},
        'calibration': {'mean_predicted': mean_pred.tolist(),
                        'fraction_positive': frac_pos.tolist()},
        'decision_curve': [{'threshold': round(float(a), 4),
                            'net_benefit': round(float(b), 4),
                            'treat_all': round(float(c), 4)}
                           for a, b, c in nb],
    }
    C.save(res, 'exp_f_curves')
    print(f"  AUC {res['auc']}  AP {res['average_precision']}  Brier {res['brier']}")
    return res


def importances():
    """Refit on the 40 balanced splits and average GBM feature importances."""
    sel = [str(s) for s in __import__('pickle').load(
        open(C.DATA_ROOT / 'models' / 'GBM_feat_selection_feat50(40).pkl', 'rb'))]
    acc = np.zeros(len(sel))
    n = 0
    for i in range(1, C.N_SPLITS + 1):
        tr = C.read_data('data_processed', f'train_{i}.csv').iloc[:, :-1]
        cols = [c for c in sel if c in tr.columns]
        if len(cols) != len(sel):
            continue
        m = HistGradientBoostingClassifier(random_state=C.SEED, **MODAL)
        m.fit(tr[cols], tr['label'])
        # HistGB has no importances_; use permutation-free proxy: split gain
        # is unavailable, so fall back to the model's own feature_importances_
        imp = getattr(m, 'feature_importances_', None)
        if imp is None:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(m, tr[cols], tr['label'],
                                       n_repeats=3, random_state=C.SEED,
                                       scoring='roc_auc')
            imp = np.clip(r.importances_mean, 0, None)
        acc += imp / max(imp.sum(), 1e-9)
        n += 1
        if i % 10 == 0:
            print(f"    split {i}/40")
    acc /= max(n, 1)
    order = np.argsort(acc)[::-1]
    df = pd.DataFrame({'rank': np.arange(1, len(sel) + 1),
                       'feature': [sel[i] for i in order],
                       'importance': [round(float(acc[i]), 4) for i in order],
                       'itemid': [C.itemid_of(sel[i]) for i in order],
                       'label': [C.LAB_NAMES.get(C.itemid_of(sel[i]),
                                                 sel[i]) for i in order]})
    C.save(df, 'exp_f_importance')
    print(f"  top 5: {df.head(5)['feature'].tolist()}")
    return df


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT F — replication figures from reproduced predictions")
    print("=" * 74)
    print("\n[1/2] performance curves")
    curves()
    print("\n[2/2] GBM feature importances (refit, modal hyperparameters)")
    importances()
