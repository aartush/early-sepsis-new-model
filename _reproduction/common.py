"""Shared infrastructure for reproducing Stylianides et al., IEEE JBHI 2026.

The reference implementation (github.com/cstylianides96/sepsis-prediction) uses
absolute POSIX paths like '/data_processed/train_1.csv'. Rather than editing
their scripts in place (which would make it impossible to tell their code from
mine), this module resolves every path against the cloned repo and reproduces
their evaluation logic verbatim.

Their `evaluate()` is copied exactly from model_evaluation.py so that any
difference in numbers is a modelling difference, not a metric difference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

import os
REPO = Path(__file__).resolve().parents[1]
# Reference-study artefacts are not redistributed (PhysioNet DUA); see
# data/DATASETS.md. Set SEPSIS_REFERENCE_ROOT or place the checkout at
# ./reference-repo.
DATA_ROOT = Path(os.environ.get('SEPSIS_REFERENCE_ROOT',
                                REPO / 'reference-repo'))
DATA = DATA_ROOT / "data_processed"
OUT = Path(__file__).resolve().parents[1] / "_reproduction"
RESULTS = OUT / "results"
PREDICTIONS = OUT / "predictions"
MODELS = OUT / "models"
PLOTS = OUT / "plots"

for d in (RESULTS, PREDICTIONS, MODELS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

N_SPLITS = 40
SEED = 123   # their random_state everywhere


def path(*parts) -> Path:
    """Resolve a path inside the reference study's checkout (DATA_ROOT)."""
    return DATA_ROOT.joinpath(*parts)


def read(*parts) -> pd.DataFrame:
    return pd.read_csv(path(*parts))


# ---------------------------------------------------------------------------
# Verbatim copy of their evaluation logic (model_evaluation.py)
# ---------------------------------------------------------------------------
def _youden_index(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1


def evaluate(prob, actual, acc=True):
    """Exact port of their evaluate(): SEN>=0.90 point and max-Youden point."""
    from sklearn.metrics import roc_curve

    auc = np.round(roc_auc_score(actual, prob), 5)
    fpr, tpr, thresholds = roc_curve(actual, prob)
    sen_fpr_thres = (pd.DataFrame(zip(thresholds, tpr, fpr),
                                  columns=['thresholds', 'sen', 'fpr'])
                     .sort_values(by='sen', ascending=False).reset_index(drop=True))

    sen_fpr_thres_90 = sen_fpr_thres.loc[sen_fpr_thres['sen'] >= 0.9]
    threshold_90 = sen_fpr_thres_90.iloc[-1, 0]

    pred = (prob >= threshold_90)
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    sen_90 = np.round(recall_score(actual, pred), 5)
    spec_90 = np.round(tn / (tn + fp), 5)
    prec_90 = np.round(precision_score(actual, pred, zero_division=0), 5)
    npv_90 = np.round(tn / (tn + fn), 5) if (tn + fn) else 0.0
    acc_90 = np.round(accuracy_score(actual, pred), 5)

    youden = [_youden_index(actual, prob, t) for t in sen_fpr_thres['thresholds']]
    best = int(np.argmax(youden))
    threshold_y = sen_fpr_thres.iloc[best, 0]

    pred = (prob >= threshold_y)
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    sen_y = np.round(recall_score(actual, pred), 5)
    spec_y = np.round(tn / (tn + fp), 5)
    prec_y = np.round(precision_score(actual, pred, zero_division=0), 5)
    npv_y = np.round(tn / (tn + fn), 5) if (tn + fn) else 0.0
    acc_y = np.round(accuracy_score(actual, pred), 5)

    return dict(auc=auc,
                sen_90=sen_90, spec_90=spec_90, prec_90=prec_90, npv_90=npv_90, acc_90=acc_90,
                thres_90=float(threshold_90),
                sen_y=sen_y, spec_y=spec_y, prec_y=prec_y, npv_y=npv_y, acc_y=acc_y,
                thres_y=float(threshold_y))


def summarise(rows: list[dict], prefix: str = 'test_') -> dict:
    """Mean and sd across the 40 balanced test sets, as the paper reports.

    `rows` are the dicts returned by evaluate(), whose metric keys are unprefixed
    (auc, sen_90, ...). The returned dict is prefixed so callers can tell model
    metrics from other fields.
    """
    df = pd.DataFrame(rows)
    metrics = ('auc', 'sen_90', 'spec_90', 'prec_90', 'npv_90',
               'sen_y', 'spec_y', 'prec_y', 'npv_y', 'acc_90', 'acc_y')
    cols = [m for m in metrics if m in df.columns]
    out = {f'{prefix}{c}': round(float(df[c].mean()), 4) for c in cols}
    out |= {f'{prefix}{c}_sd': round(float(df[c].std()), 4) for c in cols}
    return out


def save_json(obj, name):
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    print(f"  wrote {p.relative_to(OUT)}")
    return p


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))
