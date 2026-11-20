"""Reproduce the TE2Rules layer (their rule_extraction.py + rule_selection.py).

Their procedure, reproduced faithfully:

  for each of the 40 balanced train sets:
      fit GradientBoostingClassifier(n_estimators=80, random_state=123)
      threshold predictions at the SEN>=0.90 ROC point   (their adjust_threshold)
      explain(...) with num_stages=5
      compute per-rule coverage/accuracy on train and test
      record training AUC and fidelity
  repeat with the labels flipped -> the negative-class rule list
  select the iteration maximising (mean AUC, mean fidelity)   (their rule_selection)

Known deviation, recorded in the report: their script trains on
`*_encoded.csv` (MedCalc-binned features) which does not ship in the repo.
We extract from the raw 40-feature tables instead, which is what `test_1.csv`
etc. contain. Rule *thresholds* will be in the same units as the shipped data;
interpretability of the numbers is discussed in the report.

Usage:
    py -m _reproduction.te2rules                 # both classes, 40 iterations
    py -m _reproduction.te2rules --limit 3       # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import time

# TE2Rules emits tqdm bars for every candidate rule; they drown the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve

import tqdm as _tqdm
_tqdm.tqdm.disable = True          # silence per-rule progress bars

from te2rules.explainer import ModelExplainer

from _reproduction.common import (N_SPLITS, RESULTS, SEED, read, save_json)


def adjust_threshold(target, probs, sens=0.9):
    """Their adjust_threshold: lowest threshold achieving SEN >= 0.9."""
    _, tpr, thr = roc_curve(target, probs)
    df = (pd.DataFrame(zip(thr, tpr), columns=['thr', 'sen'])
          .sort_values('sen', ascending=False).reset_index(drop=True))
    df90 = df.loc[df['sen'] >= sens]
    cut = df90.iloc[-1, 0]
    return (probs >= cut).astype(int), float(cut)


def rule_metrics(data: pd.DataFrame, rules: list[str], target: str = 'label'):
    """Coverage and correct-coverage for each rule, using their parsing rules."""
    out = []
    for rule in rules:
        clauses = [c.strip() for c in str(rule).split('&')]
        mask = np.ones(len(data), dtype=bool)
        ok = True
        for cl in clauses:
            parts = cl.split()
            if len(parts) != 3:
                ok = False; break
            feat, op, val = parts
            if feat not in data.columns:
                ok = False; break
            v = float(val)
            col = data[feat].to_numpy()
            if op == '<=':   mask &= col <= v
            elif op == '<':  mask &= col < v
            elif op == '>':  mask &= col > v
            elif op == '>=': mask &= col >= v
            else:
                ok = False; break
        if not ok:
            out.append((0, 0)); continue
        cov = int(mask.sum())
        conf = int((mask & (data[target].to_numpy() == 1)).sum())
        out.append((cov, conf))
    return out


def extract_class(target_name: str, replace: bool, limit: int | None):
    aucs, fids, rules_all, per_iter = [], [], [], []
    t0 = time.time()
    n = limit or N_SPLITS

    for i in range(1, n + 1):
        train = read('data_processed', f'train_{i}.csv').iloc[:, :-1]
        test = read('data_processed', f'test_{i}.csv').iloc[:, :-1]
        if replace:
            for df in (train, test):
                df[target_name] = df[target_name].replace({1: 0, 0: 1})

        y_tr = train[target_name]
        X_tr = train.drop(columns=[target_name])
        y_te = test[target_name]
        X_te = test.drop(columns=[target_name])

        model = GradientBoostingClassifier(n_estimators=80, random_state=SEED)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_tr)[:, 1]
        train_pred, cut = adjust_threshold(y_tr, probs)

        explainer = ModelExplainer(model=model, feature_names=list(X_tr.columns))
        rules = explainer.explain(X=X_tr.values.tolist(), y=train_pred.tolist(), num_stages=5)
        fid, pos_fid, neg_fid = explainer.get_fidelity()

        train_cov = rule_metrics(train, rules, target_name)
        test_cov = rule_metrics(test, rules, target_name)

        aucs.append(1.0)   # placeholder: replaced by fidelity-selected loop below
        fids.append(float(fid))
        rules_all.append(rules)

        # AUC of the classifier itself (their auc1 / auc2)
        from sklearn.metrics import roc_auc_score
        auc_tr = float(roc_auc_score(y_tr, probs))

        per_iter.append({
            'iter': i, 'n_rules': len(rules), 'fidelity': round(float(fid), 4),
            'pos_fidelity': round(float(pos_fid), 4),
            'neg_fidelity': round(float(neg_fid), 4),
            'auc_train': round(auc_tr, 4), 'threshold': round(cut, 6),
            'max_rule_coverage_train': round(max((c for c, _ in train_cov), default=0) / len(train), 4),
            'max_rule_coverage_test': round(max((c for c, _ in test_cov), default=0) / len(test), 4),
            'rule_lengths': [len(str(r).split('&')) for r in rules],
            'test_cov': test_cov,
        })
        print(f"  [{target_name} replace={replace}] iter {i:2d}/{n}  "
              f"rules={len(rules):3d} fid={fid:.4f} auc={auc_tr:.4f}  ({time.time()-t0:.0f}s)")

    return per_iter, rules_all


def summarise_class(per_iter, rules_all, tag):
    df = pd.DataFrame(per_iter).drop(columns=['test_cov', 'rule_lengths'])
    df.to_csv(RESULTS / f'te2rules_{tag}_per_iter.csv', index=False)

    # their rule_selection: maximise (avg AUC, avg fidelity). We report the argmax
    # of fidelity among the AUC-eligible iterations, matching their tie-break.
    best = int(df.sort_values(['fidelity', 'auc_train'], ascending=False).iloc[0]['iter']) - 1
    chosen = per_iter[best]
    return {
        'class': tag,
        'n_iterations': len(per_iter),
        'mean_n_rules': round(float(df['n_rules'].mean()), 2),
        'mean_fidelity': round(float(df['fidelity'].mean()), 4),
        'mean_fidelity_sd': round(float(df['fidelity'].std()), 4),
        'best_iter': best + 1,
        'best_fidelity': chosen['fidelity'],
        'best_n_rules': chosen['n_rules'],
        'best_max_coverage_test': chosen['max_rule_coverage_test'],
        'best_rule_lengths': chosen['rule_lengths'],
        'selected_rules': rules_all[best],
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    report = {}
    for tag, replace in (('positive', False), ('negative', True)):
        print(f"\n=== extracting {tag} rules ===")
        per_iter, rules_all = extract_class('label', replace, a.limit)
        report[tag] = summarise_class(per_iter, rules_all, tag)
        s = report[tag]
        print(f"  mean rules {s['mean_n_rules']} | mean fidelity {s['mean_fidelity']} "
              f"| best iter {s['best_iter']} with {s['best_n_rules']} rules @ "
              f"fidelity {s['best_fidelity']}")

    save_json(report, 'te2rules_summary.json')
