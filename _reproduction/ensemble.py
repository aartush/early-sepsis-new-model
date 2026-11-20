"""Ensemble (soft voting) and full operating-point metrics, per their ensemble.py.

Averages GBM and H-LSTM probabilities per balanced test set, then reports the
SEN>=0.90 and max-Youden operating points, plus the curve summaries the paper
reports (mean AUC, mean AP, Brier, net benefit).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve, auc)

from _reproduction.common import (N_SPLITS, PREDICTIONS, RESULTS, evaluate,
                                  read, save_json, summarise)


def net_benefit(y, p, thresholds):
    y = np.asarray(y); p = np.asarray(p)
    n = len(y); prev = y.mean()
    out = {}
    for t in thresholds:
        pred = p >= t
        tp = np.sum((pred == 1) & (y == 1)) / n
        fp = np.sum((pred == 1) & (y == 0)) / n
        out[float(t)] = float(tp - fp * (t / (1 - t)))
    return out


def run(ml_tag: str, dl_tag: str, name: str):
    ml = pd.read_csv(PREDICTIONS / f'{ml_tag}.csv')
    dl = pd.read_csv(PREDICTIONS / f'{dl_tag}.csv')
    ml.columns = ml.columns.astype(str)
    dl.columns = dl.columns.astype(str)

    # Their run_ensemble assumes both files carry all 40 split columns. When an
    # arm is still in progress we must not silently average mismatched splits,
    # so intersect and report which splits were used.
    shared = [c for c in ml.columns if c in dl.columns]
    if not shared:
        raise SystemExit(f"no shared split columns between {ml_tag} and {dl_tag}")
    ml = ml[shared]
    dl = dl[shared]
    if ml.shape != dl.shape:
        raise SystemExit(f"row mismatch: {ml.shape} vs {dl.shape}")

    avg = pd.DataFrame(np.nanmean([ml.values.astype(float), dl.values.astype(float)],
                                  axis=0), columns=shared, index=ml.index)
    avg.to_csv(PREDICTIONS / f'{name}.csv', index=False)
    print(f"  ensembled {len(shared)} splits: {shared[0]}..{shared[-1]}")

    rows, rocs, prs, briers, nbs = [], [], [], [], []
    for i in shared:
        idx = int(i)
        df = read('data_processed', f'test_{idx}.csv').iloc[:, :-1]
        y = df.iloc[:, -1].to_numpy()
        p = avg[i].to_numpy()

        m = evaluate(p, y)
        m['split'] = idx
        rows.append(m)

        fpr, tpr, _ = roc_curve(y, p)
        rocs.append(auc(fpr, tpr))
        prec, rec, _ = precision_recall_curve(y, p)
        prs.append(average_precision_score(y, p))
        briers.append(brier_score_loss(y, p))
        nbs.append(net_benefit(y, p, [0.05, 0.1, 0.2, 0.3, 0.45]))

    pd.DataFrame(rows).to_csv(RESULTS / f'{name}_per_split.csv', index=False)

    summary = summarise(rows)
    summary.update({
        'ensemble': name,
        'mean_roc_auc': round(float(np.mean(rocs)), 4),
        'mean_average_precision': round(float(np.mean(prs)), 4),
        'mean_brier': round(float(np.mean(briers)), 4),
        'net_benefit_mean': {str(t): round(float(np.mean([d[t] for d in nbs])), 4)
                             for t in (0.05, 0.1, 0.2, 0.3, 0.45)},
        'n_splits': len(rows),
    })
    save_json(summary, f'{name}_summary.json')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ml', default='GBM_balanced_prob')
    ap.add_argument('--dl', default='LSTM_1_balanced_prob')
    ap.add_argument('--name', default='GBM-LSTM_balanced_prob')
    a = ap.parse_args()
    s = run(a.ml, a.dl, a.name)
    print(f"\n=== ensemble {a.name} ===")
    print(f"  AUC {s['test_auc']:.4f} | AP {s['mean_average_precision']:.4f} | "
          f"Brier {s['mean_brier']:.4f}")
    print(f"  @SEN>=0.90  SEN {s['test_sen_90']:.2f} SPE {s['test_spec_90']:.2f} "
          f"PRE {s['test_prec_90']:.2f} NPV {s['test_npv_90']:.2f}")
    print(f"  @maxYouden  SEN {s['test_sen_y']:.2f} SPE {s['test_spec_y']:.2f} "
          f"PRE {s['test_prec_y']:.2f} NPV {s['test_npv_y']:.2f}")
    print(f"  net benefit {s['net_benefit_mean']}")
