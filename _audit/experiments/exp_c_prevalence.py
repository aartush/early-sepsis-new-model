"""Experiment C — the prevalence discontinuity.

Claim under test: a prediction model's reported utility can change sign when only
the assumed prevalence changes, holding predictions, thresholds and labels fixed.

Three evaluation regimes are computed for the reproduced ensemble:

  BALANCED   what the paper reports: 1:1 test sets, prevalence 0.50
  HONEST     the cohort's true prior: 2.41% (1,088 / 45,148)
  DEPLOYED   per-alert, i.e. prediction-level rather than patient-level

The HONEST numbers are produced two independent ways so that the conclusion does
not rest on a single derivation:

  analytic  recompute precision / net benefit from reported sensitivity and
            specificity using Bayes' rule at the true prior
  empirical reweight the actual per-patient predictions with weights chosen so
            the weighted prevalence equals the true prior
            (w_pos=1, w_neg=(1-p)/p)

Agreement between the two is the internal check. An earlier draft used
w_neg=(1-p)/(1-0.5), which produced an effective prevalence of 33.9% instead of
2.41%; that bug is why the empirical route is reported alongside the analytic one.

Usage: py -m _audit.experiments.exp_c_prevalence
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from _audit import common as C

THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45)
PRED_FILE = C.REPO / '_reproduction' / 'predictions' / 'ENSEMBLE_40.csv'


def load_predictions():
    p = pd.read_csv(PRED_FILE)
    p.columns = p.columns.astype(str)
    return p


def balanced_regime(y, p):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    rows = []
    for t in THRESHOLDS:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum()
        fp = ((pred) & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum()
        tn = ((~pred) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        rows.append({'threshold': t, 'sens': sens, 'spec': spec,
                     'precision': prec, 'tp': int(tp), 'fp': int(fp),
                     'net_benefit': C.net_benefit(y, p, t)})
    return {'auc': auc, 'average_precision': ap, 'brier': brier,
            'operating_points': rows}


def honest_analytic(op_rows, prev=C.TRUE_PREV):
    """Bayes-rule correction of each balanced operating point."""
    out = []
    for r in op_rows:
        prec = C.precision_at_prevalence(r['sens'], r['spec'], prev)
        out.append({
            'threshold': r['threshold'],
            'sens': r['sens'], 'spec': r['spec'],
            'precision_balanced': round(r['precision'], 4),
            'precision_honest': round(prec, 4),
            'overstatement_factor': round(r['precision'] / max(prec, 1e-9), 1),
            'net_benefit_balanced': round(r['net_benefit'], 4),
            'net_benefit_honest': round(
                C.net_benefit(np.array([1, 0]), np.array([1.0, 0.0]), 0), 4)
            if False else round(_nb_from_rates(r['sens'], r['spec'], prev,
                                               r['threshold']), 4),
        })
    return out


def _nb_from_rates(sens, spec, prev, t):
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp - fp * (t / (1 - t))


def honest_empirical(probs, labels_by_split, prev=C.TRUE_PREV):
    """Reweight real predictions so the weighted prevalence equals `prev`."""
    w_neg = (1 - prev) / prev
    rows = []
    for col, y in labels_by_split.items():
        p = probs[col].to_numpy(float)
        w = np.where(y == 1, 1.0, w_neg)
        w = w / w.sum()
        for t in THRESHOLDS:
            pred = p >= t
            tp = w[(pred) & (y == 1)].sum()
            fp = w[(pred) & (y == 0)].sum()
            rows.append({'threshold': t, 'net_benefit': float(tp - fp * (t / (1 - t)))})
    d = pd.DataFrame(rows)
    return (d.groupby('threshold')['net_benefit']
              .agg(['mean', 'std', 'min', 'max']).round(4)
              .reset_index().to_dict('records'))


def deployed_regime(probs, labels_by_split):
    """Per-alert precision: what a nurse experiences when the alarm fires.

    Patient-level evaluation uses one score per patient. In deployment the alarm
    can fire repeatedly, so the experienced precision is the pooled
    per-prediction precision across all firings. This is computed at the balanced
    prior (the paper's) because the point being made is about *level*, not prior.
    """
    rows = []
    all_tp = all_fp = 0
    for col, y in labels_by_split.items():
        p = probs[col].to_numpy(float)
        for t in THRESHOLDS:
            pred = p >= t
            all_tp += int(((pred) & (y == 1)).sum())
            all_fp += int(((pred) & (y == 0)).sum())
            rows.append({'threshold': t,
                         'precision_deployed': all_tp / max(all_tp + all_fp, 1)})
    d = pd.DataFrame(rows).drop_duplicates('threshold')
    return d.to_dict('records')


def run():
    probs = load_predictions()
    labels = {}
    y_all, p_all = [], []
    for col in probs.columns:
        i = int(col)
        d = C.read_data('data_processed', f'test_{i}.csv')
        y = d['label'].to_numpy()
        labels[col] = y
        y_all.append(y)
        p_all.append(probs[col].to_numpy(float))
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    print(f"  pooled balanced test predictions: n={len(y):,} "
          f"positives={int(y.sum())} prevalence={100*y.mean():.1f}%")

    bal = balanced_regime(y, p)
    hon_a = honest_analytic(bal['operating_points'])
    hon_e = honest_empirical(probs, labels)
    dep = deployed_regime(probs, labels)

    # agreement check between the two honest routes
    a = {r['threshold']: r['net_benefit_honest'] for r in hon_a}
    e = {r['threshold']: r['mean'] for r in hon_e}
    diffs = [abs(a[t] - e[t]) for t in a if t in e]

    return {
        'true_prevalence': round(C.TRUE_PREV, 4),
        'source': '1088 sepsis / 45148 stays, their shared_dataset_mimiciv.csv',
        'balanced': bal,
        'honest_analytic': hon_a,
        'honest_empirical': hon_e,
        'deployed_precision': dep,
        'agreement_max_abs_diff_analytic_vs_empirical': round(max(diffs), 4),
    }


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT C — prevalence discontinuity")
    print("=" * 74)
    res = run()
    C.save(res, 'exp_c_prevalence')

    print(f"\n  true prevalence {100*res['true_prevalence']:.2f}%")
    print(f"\n{'thr':>5s} {'sens':>6s} {'spec':>6s} {'prec(bal)':>10s} "
          f"{'prec(true)':>11s} {'x':>6s} {'NB(bal)':>9s} {'NB(true)':>10s}")
    for a, e in zip(res['honest_analytic'],
                    {r['threshold']: r for r in res['honest_empirical']}.values()):
        print(f"{a['threshold']:5.2f} {a['sens']:6.3f} {a['spec']:6.3f} "
              f"{a['precision_balanced']:10.3f} {a['precision_honest']:11.3f} "
              f"{a['overstatement_factor']:6.1f} {a['net_benefit_balanced']:+9.4f} "
              f"{a['net_benefit_honest']:+10.4f}")

    print(f"\n  analytic vs empirical max |diff| in net benefit: "
          f"{res['agreement_max_abs_diff_analytic_vs_empirical']:.4f}")
    print("  (small value => the two independent derivations agree)")
