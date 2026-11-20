"""Experiment A verification — is the measurement signal real, or an artefact?

The headline result is that availability-only features reach AUC ~0.86 against a
full-model ~0.88. That is such a large number that it deserves adversarial
testing before it goes in a paper. Three checks:

  A4  permutation test
      Shuffle the labels and rebuild M. If M's AUC collapses to ~0.5 the signal
      came from label structure, not from a leak in the pipeline.

  A5  incremental information
      Does M add anything to V? A model on (V ∪ M) beating V means part of the
      measurement signal is complementary to physiology, not merely a proxy for
      it. Equality means M is redundant given V.

  A6  mechanism decomposition
      Which availability families carry the signal? If `n_change`/`span`
      (repetition) dominate over `n_distinct` (presence), the mechanism is
      "repeated measurement tracks illness severity", which is a defensible
      clinical statement rather than a computational trick.

Usage: py -m _audit.experiments.exp_a_verify
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _audit import common as C


def model():
    return GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                      learning_rate=0.1, subsample=0.9,
                                      random_state=C.SEED)


def cv_auc(X, y, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(n_splits, shuffle=True,
                                  random_state=C.SEED).split(X, y):
        m = model(); m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    df = C.load_full()
    y = df['label'].to_numpy()
    sel = [str(s) for s in pickle.load(
        open(C.DATA_ROOT / 'models' / 'GBM_feat_selection_feat50(40).pkl', 'rb'))]
    avail = C.availability_features(df)
    meas = [c for c in avail.columns if c != 'hosp_to_icu']
    V = [c for c in sel if c in df.columns]

    Xv = df[V].to_numpy(float)
    Xm = avail[meas].to_numpy(float)
    Xa = np.hstack([Xv, Xm])

    out = {}

    print("[A4] permutation test — labels shuffled")
    auc_real = cv_auc(Xm, y)
    rng = np.random.default_rng(C.SEED)
    perm = []
    for k in range(3):
        yp = rng.permutation(y)
        perm.append(cv_auc(Xm, yp))
        print(f"      shuffle {k+1}: AUC={perm[-1]:.4f}")
    out['permutation'] = {'real_auc': round(auc_real, 4),
                          'permuted_aucs': [round(p, 4) for p in perm],
                          'permuted_mean': round(float(np.mean(perm)), 4)}
    print(f"      real M AUC {auc_real:.4f} vs permuted {np.mean(perm):.4f}")

    print("\n[A5] incremental information over physiology")
    a_v = cv_auc(Xv, y)
    a_vm = cv_auc(Xa, y)
    out['incremental'] = {'V': round(a_v, 4), 'V_plus_M': round(a_vm, 4),
                          'delta': round(a_vm - a_v, 4)}
    print(f"      V={a_v:.4f}  V+M={a_vm:.4f}  delta={a_vm-a_v:+.4f}")

    print("\n[A6] mechanism decomposition by availability family")
    fams = {'presence (n_distinct)': [c for c in meas if c.endswith('_n_distinct')],
            'repetition (n_change)': [c for c in meas if c.endswith('_n_change')],
            'timing (first_hour/span)': [c for c in meas
                                         if c.endswith('_first_hour') or c.endswith('_span')],
            'imputed-run (n_static_hours)': [c for c in meas
                                             if c.endswith('_n_static_hours')]}
    fam_out = {}
    for name, cols in fams.items():
        if not cols:
            continue
        a = cv_auc(avail[cols].to_numpy(float), y)
        fam_out[name] = {'auc': round(a, 4), 'n_features': len(cols)}
        print(f"      {name:32s} {len(cols):3d} feats  AUC={a:.4f}")
    out['family_decomposition'] = fam_out

    # single strongest feature, for the record
    singles = []
    for c in meas:
        v = avail[c].to_numpy(float)
        if v.std() == 0:
            continue
        singles.append((c, float(roc_auc_score(y, v))))
    singles.sort(key=lambda t: -t[1])
    out['top_single_features'] = [{'feature': c, 'auc': round(a, 4)}
                                  for c, a in singles[:10]]
    print("\n      top single availability features:")
    for c, a in singles[:10]:
        print(f"        {c:28s} {a:.4f}")

    C.save(out, 'exp_a_verification')
    return out


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT A VERIFICATION — adversarial checks")
    print("=" * 74)
    main()
