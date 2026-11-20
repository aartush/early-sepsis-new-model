"""Experiment D — vacuity scoring of extracted rules.

A rule can look multi-clause while behaving as a single clause. If one conjunct is
true for almost every patient, it contributes no discrimination; removing it does
not change who the rule selects. Reporting such a rule as "4 features" overstates
its interpretability and hides the fact that its discriminative content is
narrower than advertised.

For each rule extracted by TE2Rules (the paper's own extractor, on their own 40
balanced sets) this computes:

  vacuity(clause)   fraction of patients for whom the clause alone is satisfied
  support(rule)     fraction of patients the full rule selects
  marginal(clause)  support(rule) / support(rule minus that clause)
                    = 1.0 when the clause is doing no work
  effective length  number of clauses with vacuity below the threshold
  AUC as a rule     discrimination of the bare rule indicator

Rules are evaluated on the balanced cohort (their extraction population) and the
full cohort (the honest population), because a clause that is vacuous in one need
not be vacuous in the other.

Usage: py -m _audit.experiments.exp_d_vacuity
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _audit import common as C

VACUITY_HIGH = 0.95        # clause true for >=95% of patients = effectively free
RULE_SOURCE = C.REPO / '_reproduction' / 'results' / 'te2rules_summary.json'

CLAUSE_RE = re.compile(r'^\s*([A-Za-z0-9_]+)\s*(<=|>=|<|>)\s*([-\d.eE+]+)\s*$')


def parse_rule(rule: str):
    clauses = []
    for part in str(rule).split('&'):
        m = CLAUSE_RE.match(part)
        if not m:
            return None
        clauses.append((m.group(1), m.group(2), float(m.group(3))))
    return clauses


def clause_mask(df: pd.DataFrame, clause):
    f, op, v = clause
    if f not in df.columns:
        return None
    col = df[f].to_numpy(float)
    if op == '<=': return col <= v
    if op == '<':  return col < v
    if op == '>=': return col >= v
    if op == '>':  return col > v
    return None


def rule_mask(df: pd.DataFrame, clauses):
    m = np.ones(len(df), dtype=bool)
    for cl in clauses:
        cm = clause_mask(df, cl)
        if cm is None:
            return None
        m &= cm
    return m


def summarise_rules(df, y, rules, cohort, top=15):
    n = len(df)
    base = float(y.mean())
    rows = []
    for i, r in enumerate(rules):
        clauses = parse_rule(r)
        if not clauses:
            continue
        full = rule_mask(df, clauses)
        if full is None:
            continue
        support = int(full.sum())
        pos = int((full & (y == 1)).sum())

        clause_info = []
        for j, cl in enumerate(clauses):
            cm = clause_mask(df, cl)
            vac = float(cm.mean()) if cm is not None else np.nan
            rest = [c for k, c in enumerate(clauses) if k != j]
            rm = rule_mask(df, rest) if rest else np.ones(n, dtype=bool)
            rs = int(rm.sum()) if rm is not None else 0
            marginal = (support / rs) if rs else np.nan
            clause_info.append({'clause': f'{cl[0]} {cl[1]} {cl[2]}',
                                'vacuity': vac, 'support_without': rs,
                                'marginal': marginal})

        vacuous = [c for c in clause_info if c['vacuity'] >= VACUITY_HIGH]
        effective = [c for c in clause_info if c['vacuity'] < VACUITY_HIGH]
        rows.append({
            'rule_id': i,
            'rule': str(r),
            'n_clauses': len(clauses),
            'n_vacuous_clauses': len(vacuous),
            'effective_length': len(effective),
            'max_clause_vacuity': round(max(c['vacuity'] for c in clause_info), 4),
            'support_frac': round(support / n, 4),
            'conditional_ppv': round(pos / support, 4) if support else None,
            'lift_vs_base': round((pos / support) / base, 2) if support else None,
            'auc_as_rule': round(float(roc_auc_score(y, full.astype(int))), 4),
            'vacuous_clauses': '; '.join(c['clause'] for c in vacuous),
        })
    d = pd.DataFrame(rows).sort_values(['n_clauses', 'support_frac'],
                                       ascending=[False, False])
    return d


def cohort_stats(d: pd.DataFrame):
    if d.empty:
        return {}
    return {
        'n_rules': int(len(d)),
        'mean_clauses': round(float(d['n_clauses'].mean()), 2),
        'mean_effective_length': round(float(d['effective_length'].mean()), 2),
        'rules_with_vacuous_clause': int((d['n_vacuous_clauses'] > 0).sum()),
        'pct_rules_with_vacuous_clause': round(
            100 * (d['n_vacuous_clauses'] > 0).mean(), 1),
        'total_clauses': int(d['n_clauses'].sum()),
        'total_vacuous_clauses': int(d['n_vacuous_clauses'].sum()),
        'pct_clauses_vacuous': round(
            100 * d['n_vacuous_clauses'].sum() / max(d['n_clauses'].sum(), 1), 1),
        'shrinkage_in_mean_length': round(
            float(d['n_clauses'].mean() - d['effective_length'].mean()), 2),
    }


def load_balanced_full():
    tr = C.read_data('data_processed', 'train_selected_feat40.csv')
    va = C.read_data('data_processed', 'val_selected_feat40.csv')
    offset = len(tr) + len(va)
    full = C.load_full()
    frames = []
    for i in range(1, C.N_SPLITS + 1):
        d = C.read_data('data_processed', f'test_{i}.csv')
        idx = d['index'].to_numpy()
        b = full.iloc[offset + idx].reset_index(drop=True)
        b['split'] = i
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT D — vacuity scoring of extracted rules")
    print("=" * 74)
    src = json.loads(RULE_SOURCE.read_text())
    pos_rules = src['positive']['selected_rules']
    neg_rules = src['negative']['selected_rules']
    print(f"  positive rules: {len(pos_rules)}   negative rules: {len(neg_rules)}")
    print(f"  vacuity threshold: clause true for >= {100*VACUITY_HIGH:.0f}% of patients")

    print("\n[1/2] balanced cohort (their extraction population)")
    bal = load_balanced_full()
    y_bal = bal['label'].to_numpy()
    d_pos_bal = summarise_rules(bal, y_bal, pos_rules, 'balanced')
    d_neg_bal = summarise_rules(bal, y_bal, neg_rules, 'balanced')
    C.save(d_pos_bal, 'exp_d_rules_positive_balanced')
    C.save(d_neg_bal, 'exp_d_rules_negative_balanced')

    print("\n[2/2] full cohort (honest 2.41% population)")
    full = C.load_full()
    y_full = full['label'].to_numpy()
    d_pos_full = summarise_rules(full, y_full, pos_rules, 'full')
    d_neg_full = summarise_rules(full, y_full, neg_rules, 'full')
    C.save(d_pos_full, 'exp_d_rules_positive_full')
    C.save(d_neg_full, 'exp_d_rules_negative_full')

    summary = {
        'vacuity_threshold': VACUITY_HIGH,
        'positive_balanced': cohort_stats(d_pos_bal),
        'negative_balanced': cohort_stats(d_neg_bal),
        'positive_full': cohort_stats(d_pos_full),
        'negative_full': cohort_stats(d_neg_full),
    }
    C.save(summary, 'exp_d_summary')

    print("\n" + "=" * 74)
    print("VACUITY SUMMARY")
    print("=" * 74)
    print(f"{'cohort/class':22s} {'rules':>6s} {'mean len':>9s} {'eff len':>8s} "
          f"{'w/ vacuous':>11s} {'vacuous clauses':>16s}")
    for k in ('positive_balanced', 'negative_balanced', 'positive_full', 'negative_full'):
        s = summary[k]
        if not s:
            continue
        print(f"{k:22s} {s['n_rules']:6d} {s['mean_clauses']:9.2f} "
              f"{s['mean_effective_length']:8.2f} "
              f"{s['pct_rules_with_vacuous_clause']:10.1f}% "
              f"{s['total_vacuous_clauses']:5d}/{s['total_clauses']:<10d}")

    print("\n  Most vacuous rules in the balanced cohort:")
    top = d_pos_bal.nlargest(5, 'n_vacuous_clauses')[
        ['rule_id', 'n_clauses', 'effective_length', 'max_clause_vacuity',
         'support_frac', 'conditional_ppv']]
    print(top.to_string(index=False))
