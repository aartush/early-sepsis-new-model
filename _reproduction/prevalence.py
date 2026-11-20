"""The prevalence correction that the reproduction unlocks.

We have exactly reproduced the paper's ensemble and its operating points. That
means we can now take the *same predictions at the same thresholds* and change
only the prior, which is the one thing the paper never does.

At the cohort's true prevalence (2.41%), reported precision, PR-AUC, Brier and
net benefit all shift. This script quantifies by how much, using:
  (a) their own reported sensitivity/specificity, and
  (b) our reproduced predictions, reweighted to the true prior.

Usage: py -m _reproduction.prevalence
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _reproduction.common import PREDICTIONS, REPO, RESULTS, read, save_json

TRUE_PREV = 1088 / 45148          # 2.41%, from their own cohort file
BALANCED_PREV = 0.5


def nb(sens, spec, prev, threshold):
    """Net benefit at a given prevalence."""
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp - fp * (threshold / (1 - threshold))


def precision_at(sens, spec, prev):
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp / (tp + fp) if (tp + fp) else 0.0


def analytic():
    """Recompute their reported operating points at true prevalence."""
    ops = {
        'SEN>=0.90': (0.91, 0.59, 0.69),
        'max Youden': (0.79, 0.81, 0.81),
        'after argumentation': (0.81, 0.81, 0.82),
    }
    rows = []
    for name, (sens, spec, reported_prec) in ops.items():
        p = precision_at(sens, spec, TRUE_PREV)
        rows.append({
            'operating_point': name,
            'sens': sens, 'spec': spec,
            'precision_balanced_reported': reported_prec,
            'precision_true_prevalence': round(p, 4),
            'overstatement_factor': round(reported_prec / max(p, 1e-9), 1),
            'ppv_if_50_cases_per_day': round(p, 3),
            'net_benefit_t05': round(nb(sens, spec, TRUE_PREV, 0.05), 4),
            'net_benefit_t10': round(nb(sens, spec, TRUE_PREV, 0.10), 4),
            'net_benefit_t20': round(nb(sens, spec, TRUE_PREV, 0.20), 4),
            'net_benefit_t45': round(nb(sens, spec, TRUE_PREV, 0.45), 4),
        })
    return rows


def empirical():
    """Reweight reproduced ensemble predictions from 50% to 2.41% prevalence.

    To move a balanced sample to prevalence p we hold positives at weight 1 and
    up-weight negatives so that n_pos / (n_pos + w_neg*n_neg) = p, i.e.
    w_neg = (1-p)/p. (An earlier version divided by (1-0.5), which weighted
    negatives *down* and produced an effective prevalence of 33.9% - a sign error.
    The analytic and empirical columns now agree, which is the check.)
    """
    probs = pd.read_csv(PREDICTIONS / 'ENSEMBLE_40.csv')
    w_neg = (1 - TRUE_PREV) / TRUE_PREV
    out = []
    for col in probs.columns:
        idx = int(col)
        df = read('data_processed', f'test_{idx}.csv').iloc[:, :-1]
        y = df['label'].to_numpy()
        p = probs[col].to_numpy()
        w = np.where(y == 1, 1.0, w_neg)
        w = w / w.sum()
        for t in (0.05, 0.10, 0.20, 0.30, 0.45):
            pred = p >= t
            tp = w[(pred == 1) & (y == 1)].sum()
            fp = w[(pred == 1) & (y == 0)].sum()
            out.append({'split': idx, 'threshold': t,
                        'net_benefit': float(tp - fp * (t / (1 - t)))})
    df = pd.DataFrame(out)
    return (df.groupby('threshold')['net_benefit']
              .agg(['mean', 'std', 'min', 'max']).round(4)
              .reset_index().to_dict('records'))


if __name__ == '__main__':
    report = {
        'true_prevalence': round(TRUE_PREV, 4),
        'source': '1088 sepsis / 45148 stays in their shared_dataset_mimiciv.csv',
        'analytic': analytic(),
        'empirical_net_benefit_from_reproduced_predictions': empirical(),
    }
    save_json(report, 'prevalence_correction.json')

    print(f"\nTrue cohort prevalence: {TRUE_PREV:.4f} ({100*TRUE_PREV:.2f}%)")
    print(f"\n{'operating point':22s} {'SEN':>5s} {'SPE':>5s} {'PRE(bal)':>9s} "
          f"{'PRE(true)':>10s} {'factor':>7s}")
    for r in report['analytic']:
        print(f"{r['operating_point']:22s} {r['sens']:5.2f} {r['spec']:5.2f} "
              f"{r['precision_balanced_reported']:9.2f} "
              f"{r['precision_true_prevalence']:10.3f} "
              f"{r['overstatement_factor']:6.1f}x")

    print("\nNet benefit at true prevalence (their reported operating points):")
    print(f"{'operating point':22s} {'@0.05':>8s} {'@0.10':>8s} {'@0.20':>8s} {'@0.45':>8s}")
    for r in report['analytic']:
        print(f"{r['operating_point']:22s} {r['net_benefit_t05']:+8.4f} "
              f"{r['net_benefit_t10']:+8.4f} {r['net_benefit_t20']:+8.4f} "
              f"{r['net_benefit_t45']:+8.4f}")

    print("\nNet benefit from reproduced predictions, reweighted to 2.41%:")
    print(f"{'threshold':>10s} {'mean':>9s} {'sd':>8s} {'min':>8s} {'max':>8s}")
    for r in report['empirical_net_benefit_from_reproduced_predictions']:
        print(f"{r['threshold']:10.2f} {r['mean']:+9.4f} {r['std']:8.4f} "
              f"{r['min']:+8.4f} {r['max']:+8.4f}")
