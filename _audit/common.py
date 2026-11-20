"""Shared infrastructure for the measurement-artefact audit.

Design note on why this lives beside `_reproduction/` rather than inside it:
the reproduction harness answers "does their paper reproduce?" (it does). This
package answers "what does the reproduced number mean?" The two have different
provenance requirements — reproduction must be faithful to their code, the audit
must be free to choose its own estimators — so they are kept separate and the
audit imports the reproduction's *predictions* as inputs, never its internals.

Three cohort views are used throughout:

  FULL    the 45,148-patient cohort with 2,211 columns, imbalanced (2.41%
          sepsis). This is the honest population.
  BAL     the 40 balanced 1:1 test sets the paper reports on (218 pos / 218 neg
          each). This is the paper's population.
  Both are needed because several claims are about the *difference* between them.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# The reference study's released artefacts (data_processed/, models/,
# shared_dataset_mimiciv.csv) are NOT redistributed in this repository: they are
# covered by the PhysioNet credentialed data use agreement. Point
# SEPSIS_REFERENCE_ROOT at a local checkout of
# https://github.com/cstylianides96/sepsis-prediction, or place that checkout at
# ./reference-repo. See data/DATASETS.md.
REFERENCE_ROOT = Path(os.environ.get('SEPSIS_REFERENCE_ROOT',
                                     REPO / 'reference-repo'))
# Alias kept so data lookups read naturally as `C.DATA_ROOT / 'data_processed' / ...`
DATA_ROOT = REFERENCE_ROOT

OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
for d in (RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 123
N_SPLITS = 40
TRUE_PREV = 1088 / 45148          # 0.0241, from their own cohort file

# The 31 itemids that carry 24-hour series in their cohort file.
SERIES_ITEMIDS = ['220045', '220050', '220051', '220052', '220210', '220545',
                  '220546', '220615', '220621', '220632', '220635', '223900',
                  '223901', '225612', '225624', '225641', '225677', '227444',
                  '227468', '220051', '220052']

LAB_NAMES = {
    '220045': 'Heart rate', '220050': 'Arterial BP systolic',
    '220051': 'Arterial BP diastolic', '220052': 'Arterial BP mean',
    '220210': 'Respiratory rate', '220545': 'Hematocrit',
    '220546': 'White blood cell count', '220615': 'Creatinine',
    '220621': 'Glucose (serum)', '220632': 'Magnesium',
    '220635': 'Phosphate', '223900': 'GCS verbal response',
    '223901': 'GCS motor response', '225612': 'BUN',
    '225624': 'BUN (alternate)', '225641': 'Differential lymphs',
    '225677': 'Platelet count', '227444': 'PTT', '227468': 'Fibrinogen',
}

# Features that are administrative or workflow, not physiological.
NON_PHYSIOLOGICAL = {'hosp_to_icu'}


def itemid_of(col: str) -> str | None:
    m = re.match(r'^(\d+)_', col)
    return m.group(1) if m else None


def feature_family(col: str) -> str:
    """Classify a column into hour / stat / diff / other."""
    m = re.match(r'^(\d+)_(.+)$', col)
    if not m:
        return 'other'
    suf = m.group(2)
    if suf.isdigit():
        return 'hour'
    if suf in ('mean', 'min', 'max', 'range'):
        return 'stat'
    if suf.startswith('diff_'):
        return 'diff'
    return 'other'


# ---------------------------------------------------------------------------
# Cohort loaders
# ---------------------------------------------------------------------------
_MISSING_HINT = (
    "\n\nThe reference study's released artefacts are not redistributed in this "
    "repository (PhysioNet credentialed data use agreement).\n"
    "Set SEPSIS_REFERENCE_ROOT to a local checkout of\n"
    "  https://github.com/cstylianides96/sepsis-prediction\n"
    "or place that checkout at ./reference-repo. See data/DATASETS.md."
)


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"missing reference artefact: {p}{_MISSING_HINT}")
    return p


def read_data(*parts, **kw):
    """Read a reference-study artefact, with a helpful error when it is absent."""
    return pd.read_csv(_require(DATA_ROOT.joinpath(*parts)), **kw)


def load_full(years=None) -> pd.DataFrame:
    """The full 45,148-patient cohort."""
    return pd.read_csv(_require(DATA_ROOT / 'shared_dataset_mimiciv.csv'),
                       low_memory=False)


def load_balanced(splits=range(1, N_SPLITS + 1)):
    """The paper's 40 balanced test sets, stacked with a split marker."""
    frames = []
    for i in splits:
        d = pd.read_csv(_require(DATA_ROOT / 'data_processed' / f'test_{i}.csv'))
        d = d.drop(columns=[c for c in ('index',) if c in d.columns])
        d['split'] = i
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Measurement-availability features  (the M-only view)
# ---------------------------------------------------------------------------
def hourly_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """itemid -> ordered list of its 24 hourly column names."""
    out = defaultdict(list)
    for c in df.columns:
        if feature_family(c) == 'hour':
            iid = itemid_of(c)
            if iid is not None:          # guard: 'hour' family must have an itemid
                out[iid].append(c)
    return {k: sorted(v, key=lambda x: int(x.rsplit('_', 1)[1]))
            for k, v in out.items()}


def diff_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    out = defaultdict(list)
    for c in df.columns:
        if feature_family(c) == 'diff':
            iid = itemid_of(c)
            if iid is not None:
                out[iid].append(c)
    return {k: sorted(v, key=lambda x: int(x.rsplit('_', 1)[1]))
            for k, v in out.items()}


def availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a table that encodes ONLY the measurement process.

    Every column here is a function of *whether and when* something was
    observed, never of the observed value. Concretely, for each itemid:

      <id>_n_distinct   number of distinct hourly values        (imputation-aware)
      <id>_n_change     number of nonzero consecutive diffs     (the signal their
                                                                 pipeline leaks)
      <id>_first_hour   first hour with a nonzero diff, else -1
      <id>_span         last-first hour of variation, else 0
      <id>_n_missing    hours identical to the previous hour    (imputed stretch)

    plus static workflow columns.

    Why `n_distinct` and not `n_notna`: because `ffill -> bfill -> median`
    (their data_processing2.py:152-154) has already removed NaN. A lab that was
    drawn once and forward-filled for 23 hours produces a constant series, which
    is observable as n_distinct == 1. This is the honest proxy for "was this
    measurement repeated?" given their published pipeline.
    """
    hourly = hourly_columns(df)
    diffs = diff_columns(df)
    cols_out: dict[str, np.ndarray] = {}

    for iid, cols in hourly.items():
        block = df[cols].to_numpy(dtype=np.float64)
        cols_out[f'{iid}_n_distinct'] = (np.nanmax(block, axis=1)
                                         != np.nanmin(block, axis=1)).astype(np.int8)
        if iid in diffs:
            dblock = df[diffs[iid]].to_numpy(dtype=np.float64)
            nonzero = (dblock != 0.0)
            cols_out[f'{iid}_n_change'] = nonzero.sum(axis=1).astype(np.int16)
            first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), -1)
            cols_out[f'{iid}_first_hour'] = first.astype(np.int16)
            rev = nonzero[:, ::-1]
            last = np.where(nonzero.any(axis=1),
                            len(cols) - 1 - rev.argmax(axis=1), -1)
            cols_out[f'{iid}_span'] = np.where(nonzero.any(axis=1),
                                               last - first + 1, 0).astype(np.int16)
            cols_out[f'{iid}_n_static_hours'] = (len(cols) - 1
                                                 - nonzero.sum(axis=1)).astype(np.int16)

    if 'hosp_to_icu' in df.columns:
        cols_out['hosp_to_icu'] = df['hosp_to_icu'].to_numpy(dtype=np.float64)

    # single concat avoids pandas' fragmentation warning from repeated inserts
    return pd.DataFrame(cols_out, index=df.index)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def net_benefit(y, p, threshold):
    y = np.asarray(y); p = np.asarray(p)
    n = len(y)
    pred = p >= threshold
    tp = ((pred == 1) & (y == 1)).sum() / n
    fp = ((pred == 1) & (y == 0)).sum() / n
    return float(tp - fp * (threshold / (1 - threshold)))


def precision_at_prevalence(sens, spec, prev):
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def sensitivity_threshold(y, p, target=0.90):
    from sklearn.metrics import roc_curve
    _, tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero(tpr >= target)
    return float(thr[ok[0]]) if len(ok) else 1.0


def oof_predictions(df, y, feature_cols, groups, model_factory, n_folds=None):
    """Out-of-fold predictions grouped by patient-level split id.

    `groups` must identify independent units (here: the balanced-set id), so no
    patient can appear in both a training and a scoring fold. This is the guard
    that makes the M-only comparison honest.
    """
    from sklearn.model_selection import GroupKFold
    uniq = np.unique(groups)
    gkf = GroupKFold(n_splits=n_folds or min(5, len(uniq)))
    pred = np.full(len(df), np.nan)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def save(obj, name):
    p = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        p = p.with_suffix('.csv')
        obj.to_csv(p, index=False)
    else:
        p = p.with_suffix('.json')
        p.write_text(json.dumps(obj, indent=2, default=_default), encoding='utf-8')
    print(f"  wrote {p.relative_to(OUT)}")
    return p


def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
