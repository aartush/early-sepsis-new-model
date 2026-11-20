"""Reproduce the GBM arm (their ML_balanced.run_ml_balanced).

Exact hyperparameter search as theirs: RandomizedSearchCV over n_iter=150,
StratifiedKFold(5), scoring='roc_auc', random_state=123, n_jobs=-1.

Usage:
    py -m _reproduction.gbm            # full 40 splits
    py -m _reproduction.gbm --limit 3  # smoke test
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from _reproduction.common import (N_SPLITS, SEED, PREDICTIONS, RESULTS,
                                  evaluate, read, save_json, summarise)

PARAM_GRID = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2],
    'n_estimators': [80, 100, 150, 200, 250, 300],
    'subsample': [0.8, 0.9, 1],
    'max_depth': [3, 4, 5, 6],
    'max_features': [0.8, 0.9, 1],
}


def lit_drop_index(df: pd.DataFrame) -> pd.DataFrame:
    """Their `.iloc[:, :-1]` drops the trailing `index` column."""
    return df.iloc[:, :-1]


def run(limit: int | None = None) -> dict:
    rows, params, probs_all = [], [], {}
    splits = range(1, (limit or N_SPLITS) + 1)
    t0 = time.time()

    for i in splits:
        df_train = lit_drop_index(read('data_processed', f'train_{i}.csv'))
        X_tr, y_tr = df_train.iloc[:, :-1], df_train.iloc[:, -1]
        df_test = lit_drop_index(read('data_processed', f'test_{i}.csv'))
        X_te, y_te = df_test.iloc[:, :-1], df_test.iloc[:, -1]

        search = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=SEED), PARAM_GRID,
            cv=StratifiedKFold(5), scoring='roc_auc',
            random_state=SEED, n_iter=150, n_jobs=-1)
        search.fit(X_tr, y_tr)

        prob = search.best_estimator_.predict_proba(X_te)[:, 1]
        m = evaluate(prob, y_te)
        m['split'] = i
        m['train_cv_auc'] = float(search.cv_results_['mean_test_score'][search.best_index_])
        m['train_cv_auc_sd'] = float(search.cv_results_['std_test_score'][search.best_index_])
        m['best_params'] = search.best_params_
        rows.append(m)
        params.append(search.best_params_)
        probs_all[i] = prob

        print(f"  split {i:2d}/{len(splits)}  test AUC={m['auc']:.4f}  "
              f"cv AUC={m['train_cv_auc']:.4f}  ({time.time()-t0:.0f}s)")

    pd.DataFrame(probs_all).to_csv(PREDICTIONS / 'GBM_balanced_prob.csv', index=False)
    pd.DataFrame(rows).drop(columns=['best_params']).to_csv(
        RESULTS / 'gbm_per_split.csv', index=False)

    summary = summarise(rows, 'test_')
    summary['train_cv_auc_mean'] = round(float(np.mean([r['train_cv_auc'] for r in rows])), 4)
    summary['n_splits'] = len(rows)
    summary['seconds'] = round(time.time() - t0, 1)

    # Their reported modal hyperparameters (paper Table II narrative)
    allp = pd.DataFrame(params)
    summary['mode_params'] = {
        'n_estimators': int(allp['n_estimators'].mode()[0]),
        'max_depth': int(allp['max_depth'].mode()[0]),
        'learning_rate_median': float(allp['learning_rate'].median()),
        'subsample_median': float(allp['subsample'].median()),
        'max_features_median': float(allp['max_features'].median()),
    }
    save_json(summary, 'gbm_summary.json')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    print("=== GBM reproduction (40 balanced splits, RandomizedSearchCV n_iter=150) ===")
    s = run(a.limit)
    print("\n--- mean over splits ---")
    for k in ('test_auc', 'test_sen_90', 'test_spec_90', 'test_prec_90', 'test_npv_90',
              'test_sen_y', 'test_spec_y', 'test_prec_y', 'test_npv_y'):
        if k in s:
            print(f"  {k:16s} {s[k]:.4f}  (sd {s.get(k + '_sd', 0):.4f})")
    print(f"  cv AUC          {s['train_cv_auc_mean']:.4f}")
    print(f"  modal params    {s['mode_params']}")
