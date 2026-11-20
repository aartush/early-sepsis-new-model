"""Experiment A capacity control — is the M signal an artefact of feature count?

Experiment A compares V (40 features) against M (155 features). A larger feature
set can win on capacity alone, so the MEF figure is only defensible if M still
recovers most of the signal when its capacity is matched to V.

Three controls:

  A7  top-K M features   rank M's single features by univariate AUC, keep the
                         same K as V (40), and re-evaluate. If AUC stays high,
                         the signal is in the features, not in their count.

  A8  equal-capacity V   restrict V to its 40 features (it already is) and compare
                         against M restricted to 40, and against M with a single
                         feature per itemid family so the comparison is not
                         dominated by having five derived columns per lab.

  A9  latency control    drop all timing features (first_hour, span) and keep only
                         counts. Timing could encode "how far into the stay we
                         are", which is a legitimate confounder worth isolating.

Usage: py -m _audit.experiments.exp_a_capacity
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

    a_v = cv_auc(df[V].to_numpy(float), y)
    print(f"[baseline] V ({len(V)} feats)            AUC={a_v:.4f}")

    # A7: rank M by univariate AUC, take top-K
    scored = []
    for c in meas:
        v = avail[c].to_numpy(float)
        if v.std() == 0:
            continue
        scored.append((c, float(roc_auc_score(y, v))))
    scored.sort(key=lambda t: -t[1])
    out = {'V': {'auc': round(a_v, 4), 'n_features': len(V)}}

    print("\n[A7] M restricted to K features (K matched to V)")
    for K in (5, 10, 20, 40, 60):
        top = [c for c, _ in scored[:K]]
        a = cv_auc(avail[top].to_numpy(float), y)
        out[f'M_top{K}'] = {'auc': round(a, 4), 'n_features': K,
                            'mef': round((a - 0.5) / (a_v - 0.5), 4)}
        print(f"      K={K:3d}  AUC={a:.4f}   MEF-vs-V={100*(a-0.5)/(a_v-0.5):.1f}%")

    # A8: one representative column per itemid (no per-family multiplicity)
    print("\n[A8] one column per itemid family (removes derived-column multiplicity)")
    per_family = {}
    for c in meas:
        iid = C.itemid_of(c)
        fam = c.rsplit('_', 1)[1]
        per_family.setdefault((iid, fam), []).append(c)
    one_each = []
    for key, cols in per_family.items():
        best = max(cols, key=lambda c: dict((s[0], s[1]) for s in scored).get(c, 0))
        one_each.append(best)
    a = cv_auc(avail[one_each].to_numpy(float), y)
    out['M_one_per_family'] = {'auc': round(a, 4), 'n_features': len(one_each)}
    print(f"      {len(one_each)} feats  AUC={a:.4f}")

    # A9: latency control — remove timing
    print("\n[A9] latency control (timing features removed)")
    no_timing = [c for c in meas
                 if not (c.endswith('_first_hour') or c.endswith('_span'))]
    a = cv_auc(avail[no_timing].to_numpy(float), y)
    out['M_no_timing'] = {'auc': round(a, 4), 'n_features': len(no_timing)}
    print(f"      {len(no_timing)} feats  AUC={a:.4f}")
    only_timing = [c for c in meas if c not in no_timing]
    a2 = cv_auc(avail[only_timing].to_numpy(float), y)
    out['M_timing_only'] = {'auc': round(a2, 4), 'n_features': len(only_timing)}
    print(f"      timing only ({len(only_timing)} feats)  AUC={a2:.4f}")

    C.save(out, 'exp_a_capacity_control')
    print("\n  key result: M at 40 features vs V at 40 features")
    print(f"    V  AUC={out['V']['auc']:.4f}")
    print(f"    M  AUC={out['M_top40']['auc']:.4f}")


if __name__ == '__main__':
    print("=" * 74)
    print("EXPERIMENT A CAPACITY CONTROL")
    print("=" * 74)
    main()
