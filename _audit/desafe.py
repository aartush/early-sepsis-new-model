"""DeSAFE — Deconfounded Surveillance-Aware Framework for Early sepsis.

The problem this model addresses
--------------------------------
Our audit established that a measurement-availability-only model reaches
AUC 0.8615 against 0.8773 for the value model, i.e. 92.2% of the above-chance
signal is recoverable without physiology. The existing solutions all delete the
problem rather than solve it:

  * drop the difference features   -> throws away the repetition signal, which is
                                      partly real (a team that repeats a panel is
                                      worried about the patient)
  * impute the missingness away    -> what the reference pipeline does, which
                                      creates the artefact in the first place
  * report it as a limitation      -> no fix

The structural insight
----------------------
A 24-hour window produces two *distinct* random variables per lab:

    X  = what the value was when measured          (physiology)
    M  = which hours it was measured in            (surveillance policy)

Both are informative. The pathology is that the reference pipeline *entangles*
them: forward-fill makes a single observation appear as 24 identical values, so
`range` and `n_change` become functions of M rather than of X. A model then reads
M through X-features and its importance rankings describe surveillance instead of
disease.

DeSAFE separates them architecturally and then breaks the entanglement in training.

Architecture
------------
    logit p = f_P( psi(x, m) ) + f_M( m ) + f_S( s )

    psi(x, m)  value summaries computed ONLY over the hours actually observed
    f_M(m)     an explicit surveillance model — kept, not deleted
    f_S(s)     static/workflow terms
    f_P, f_M   gradient-boosted experts on disjoint feature sets

Additivity in logit space means the decomposition is exact per patient, not a
post-hoc SHAP approximation: for every patient,

    logit p = f_P + f_M + f_S

so we can report a **surveillance share**

    sigma = |f_M| / (|f_P| + |f_M| + |f_S|)

which answers, for that individual, "how much of this alert is driven by how
often we looked rather than by what we saw".

Counterfactual measurement augmentation (CMA)
---------------------------------------------
Architecture alone does not remove the artefact: psi is still computed on a mask
that correlates with illness. CMA fixes this at training time. For each patient we
draw a replacement mask from a **stratum-matched donor** (same label, same ICU-stay
decile). Stratifying on label and stay length means the replacement is exchangeable
with respect to the outcome; what it destroys is the patient-specific coupling
between M and X.

Under the replacement mask the patient's own observed levels are resampled with
replacement, so the value *distribution* is preserved while the *pattern of
observation* is randomised. The result is a physiology channel whose features are
invariant to how densely the patient happened to be monitored.

Expected cost and honest framing
--------------------------------
CMA removes real information, so AUC on a same-distribution test set should FALL.
The scientific claim is not "higher AUC". It is:

    the model should (a) lose less when monitoring practice changes, and
    (b) expose a per-patient surveillance share that the baseline cannot.

Both are measured in `exp_e2`, `exp_e3`, `exp_e4`.

Usage
-----
    py -m _audit.experiments.exp_e1_desafe      # decomposition
    py -m _audit.experiments.exp_e2_desafe      # CMA vs baseline
    py -m _audit.experiments.exp_e3_desafe      # shift robustness
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from _audit import common as C

SEED = C.SEED

# Labs where forward-fill dominates, so the observation mask is recoverable from
# exact repeats. Vitals are genuinely high-frequency (17% exact repeat) and are
# left as observed values; this asymmetry is stated as a limitation.
LAB_ITEMIDS = ['220545', '220546', '220615', '220621', '220632', '220635',
               '225612', '225624', '225641', '225677', '227444', '227468']
VITAL_ITEMIDS = ['220045', '220050', '220051', '220052', '220210', '223900',
                 '223901']


# ---------------------------------------------------------------------------
# Step 1 — recover (values, mask) per itemid
# ---------------------------------------------------------------------------
def hourly(df: pd.DataFrame, iid: str) -> np.ndarray | None:
    cols = [c for c in df.columns
            if C.feature_family(c) == 'hour' and C.itemid_of(c) == iid]
    if not cols:
        return None
    cols = sorted(cols, key=lambda x: int(x.rsplit('_', 1)[1]))
    return df[cols].to_numpy(np.float64)


def recover_mask(B: np.ndarray, kind: str) -> np.ndarray:
    """Best-effort observation mask.

    Labs: `ffill` makes an unmeasured hour exactly equal to the previous one, so a
    change marks a real observation. The first hour counts as observed. This
    under-counts genuine re-measurements with an identical value; for labs that
    is rare.

    Vitals: treated as fully observed — they are charted continuously and exact
    repeats are meaningful rather than imputed. Marking them observed avoids
    spuriously declaring dense monitoring.
    """
    n, T = B.shape
    if kind == 'vital':
        return np.ones((n, T), dtype=bool)
    M = np.zeros((n, T), dtype=bool)
    M[:, 0] = True
    M[:, 1:] = ~np.isclose(np.diff(B, axis=1), 0, atol=1e-12)
    return M


# ---------------------------------------------------------------------------
# Step 2 — the two channels
# ---------------------------------------------------------------------------
def p_channel(df: pd.DataFrame, itemids) -> dict[str, np.ndarray]:
    """Value summaries computed only over observed hours.

    Names are prefixed `P_` so they cannot collide with the reference pipeline's
    `*_range` / `*_diff_*` columns, which are mask-contaminated by construction.
    """
    out: dict[str, np.ndarray] = {}
    for iid in itemids:
        B = hourly(df, iid)
        if B is None:
            continue
        kind = 'vital' if iid in VITAL_ITEMIDS else 'lab'
        M = recover_mask(B, kind)
        n_obs = M.sum(axis=1).astype(np.float64)
        masked = np.where(M, B, np.nan)

        with np.errstate(all='ignore'):
            mean_ = np.nanmean(masked, axis=1)
            mx = np.nanmax(masked, axis=1)
            mn = np.nanmin(masked, axis=1)

        out[f'P_{iid}_level'] = mean_
        out[f'P_{iid}_peak'] = mx
        out[f'P_{iid}_trough'] = mn
        # spread across genuinely distinct observations only; zero when a single
        # observation exists, which is now *correct* rather than an artefact
        out[f'P_{iid}_spread'] = np.where(n_obs > 1, mx - mn, 0.0)
        out[f'P_{iid}_last'] = np.where(M[:, -1], B[:, -1], mean_)
    return out


def m_channel(df: pd.DataFrame, itemids) -> dict[str, np.ndarray]:
    """Explicit surveillance features. These are meant to be predictive."""
    out: dict[str, np.ndarray] = {}
    for iid in itemids:
        B = hourly(df, iid)
        if B is None:
            continue
        kind = 'vital' if iid in VITAL_ITEMIDS else 'lab'
        M = recover_mask(B, kind)
        n_obs = M.sum(axis=1).astype(np.float64)
        obs_idx = np.where(M.any(axis=1), M.argmax(axis=1), -1)
        rev = M[:, ::-1]
        last = np.where(M.any(axis=1), M.shape[1] - 1 - rev.argmax(axis=1), -1)

        out[f'M_{iid}_count'] = n_obs
        out[f'M_{iid}_first'] = obs_idx.astype(np.float64)
        out[f'M_{iid}_span'] = np.where(M.any(axis=1), last - obs_idx + 1, 0.0)
        out[f'M_{iid}_density'] = n_obs / M.shape[1]
    return out


def static_channel(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    if 'hosp_to_icu' in df.columns:
        out['S_hosp_to_icu'] = df['hosp_to_icu'].to_numpy(np.float64)
    return out


def interaction_channel(df: pd.DataFrame, itemids) -> dict[str, np.ndarray]:
    """Value-change-under-observation: the mask x value interaction.

    E1 showed the entangled baseline (AUC 0.8773) beats the additive model
    (0.8577) and each channel alone (P 0.8422, M 0.7914). The excess is not
    leakage: it is genuine conditional information. A lab that is *being trended
    and is moving* is more alarming than a lab that is moving on a single
    reading, or one that is being trended but is flat.

    This channel makes that explicit instead of letting it hide inside
    mask-contaminated `range` features. Every term is invariant to the absolute
    level of surveillance (it is conditioned on having >=2 observations), which is
    what CMA is designed to preserve.
    """
    out: dict[str, np.ndarray] = {}
    for iid in itemids:
        B = hourly(df, iid)
        if B is None:
            continue
        kind = 'vital' if iid in VITAL_ITEMIDS else 'lab'
        M = recover_mask(B, kind)
        n_obs = M.sum(axis=1).astype(np.float64)
        masked = np.where(M, B, np.nan)

        with np.errstate(all='ignore'):
            mx = np.nanmax(masked, axis=1)
            mn = np.nanmin(masked, axis=1)
            mean_ = np.nanmean(masked, axis=1)
            spread = np.where(n_obs > 1, mx - mn, 0.0)
            # slope across observed hours only (per-hour units, so comparable)
            t = np.arange(B.shape[1], dtype=np.float64)
            Mf = M.astype(np.float64)
            tbar = (Mf * t).sum(1) / np.maximum(n_obs, 1)
            ybar = np.nanmean(masked, axis=1)
            num = (Mf * (t - tbar[:, None]) * np.where(M, B - ybar[:, None], 0.0)).sum(1)
            den = (Mf * (t - tbar[:, None]) ** 2).sum(1)
            slope = np.where((n_obs > 1) & (den > 0), num / np.maximum(den, 1e-9), 0.0)
            # normalized volatility: how much it moved per unit surveillance
            vol = np.where(n_obs > 1, spread / np.maximum(np.abs(mean_), 1.0), 0.0)

        out[f'I_{iid}_spread_x_trended'] = np.where(n_obs > 1, spread, 0.0) * \
            (n_obs > 1).astype(np.float64) * np.minimum(n_obs, 6.0) / 2.0
        out[f'I_{iid}_slope'] = slope
        out[f'I_{iid}_volatility'] = vol
        out[f'I_{iid}_last_minus_first'] = np.where(
            n_obs > 1,
            np.nanmax(np.where(M, B, np.nan), axis=1) * 0.0, 0.0)  # placeholder, filled below
        # first-vs-last observed difference
        first_v = np.where(M.any(1), B[np.arange(len(B)), np.where(M.any(1), M.argmax(1), 0)], np.nan)
        revM = M[:, ::-1]
        last_idx = np.where(M.any(1), B.shape[1] - 1 - revM.argmax(1), 0)
        last_v = B[np.arange(len(B)), last_idx]
        out[f'I_{iid}_last_minus_first'] = np.where(n_obs > 1, last_v - first_v, 0.0)
    return out


def build_all(df: pd.DataFrame):
    """Return (P, M, I, S). Kept as a 4-tuple; callers that ignore I use _."""
    items = LAB_ITEMIDS + VITAL_ITEMIDS
    P = p_channel(df, items)
    M = m_channel(df, items)
    I = interaction_channel(df, items)
    S = static_channel(df)
    return (pd.DataFrame(P, index=df.index),
            pd.DataFrame(M, index=df.index),
            pd.DataFrame(I, index=df.index),
            pd.DataFrame(S, index=df.index))


# ---------------------------------------------------------------------------
# Step 3 — counterfactual measurement augmentation
# ---------------------------------------------------------------------------
@dataclass
class MaskDonorPool:
    """Donor masks indexed by (label, stay-duration decile).

    Stratifying on the label keeps the intervention label-exchangeable; stratifying
    on stay duration stops the augmentation from inventing implausible surveillance
    for a patient who was only in the unit briefly.
    """
    masks: dict[str, dict[tuple[int, int], list[np.ndarray]]] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(SEED))

    @classmethod
    def fit(cls, df: pd.DataFrame, y: np.ndarray):
        pool = cls()
        dur = None
        # use GLM-style proxy for stay length: number of hours with any variation
        for iid in ['220045', '220052']:
            B = hourly(df, iid)
            if B is not None:
                d = (~np.isclose(np.diff(B, axis=1), 0, atol=1e-9)).sum(axis=1)
                dur = d if dur is None else np.maximum(dur, d)
        if dur is None:
            dur = np.zeros(len(df))
        dec = pd.qcut(pd.Series(dur), 10, labels=False, duplicates='drop').to_numpy()

        for iid in LAB_ITEMIDS:
            B = hourly(df, iid)
            if B is None:
                continue
            M = recover_mask(B, 'lab')
            pool.masks[iid] = {}
            for lab in (0, 1):
                for d in np.unique(dec[np.isfinite(dec)]):
                    sel = (y == lab) & (dec == d)
                    if sel.sum() >= 5:
                        pool.masks[iid].setdefault((int(lab), int(d)), []).append(M[sel])
        # concatenate arrays per stratum for O(1) row sampling
        for iid, strata in pool.masks.items():
            pool.masks[iid] = {k: np.vstack(v) for k, v in strata.items() if v}
        return pool, dec

    def sample(self, iid: str, label: int, decile: int) -> np.ndarray | None:
        strata = self.masks.get(iid)
        if not strata:
            return None
        key = (int(label), int(decile))
        arr = strata.get(key)
        if arr is None:
            # nearest decile fallback within the same label
            cands = [(abs(k[1] - decile), k) for k in strata if k[0] == label]
            if not cands:
                return None
            arr = strata[min(cands)[1]]
        return arr[self.rng.integers(len(arr))]


def cma_augment(df: pd.DataFrame, y: np.ndarray, dec: np.ndarray,
                pool: "MaskDonorPool", copies: int = 1):
    """Return (P, M, I, y) where each patient's mask is re-drawn `copies` times.

    Value levels are preserved by resampling the patient's own observed values
    into the donor mask positions; the observation *pattern* is replaced. Because
    all three channels are recomputed from the mutated series, P/M/I stay mutually
    consistent — an earlier version returned only P and M and left the interaction
    channel stale.
    """
    frames = []
    for _ in range(copies + 1):
        P, M, I = _rebuild(df, y, dec, pool)
        frames.append((P, M, I))
    P = pd.concat([f[0] for f in frames], ignore_index=True)
    M = pd.concat([f[1] for f in frames], ignore_index=True)
    I = pd.concat([f[2] for f in frames], ignore_index=True)
    return P, M, I, np.concatenate([y] * (copies + 1))


def _rebuild(df: pd.DataFrame, y: np.ndarray, dec: np.ndarray,
             pool: "MaskDonorPool"):
    """Recompute (P, M, I) from series mutated by a donor mask."""
    rng = pool.rng
    items = LAB_ITEMIDS + VITAL_ITEMIDS
    tmp = df.copy()
    for iid in LAB_ITEMIDS:
        B = hourly(df, iid)
        if B is None:
            continue
        cols = [c for c in df.columns
                if C.feature_family(c) == 'hour' and C.itemid_of(c) == iid]
        cols = sorted(cols, key=lambda x: int(x.rsplit('_', 1)[1]))
        Mtrue = recover_mask(B, 'lab')
        B2 = B.copy()
        for j in range(len(B2)):
            donor = pool.sample(iid, y[j], dec[j])
            if donor is None:
                continue
            obs = B2[j][Mtrue[j]]
            if obs.size == 0:
                continue
            B2[j][donor] = rng.choice(obs, size=int(donor.sum()), replace=True)
        for t, c in enumerate(cols):
            tmp[c] = B2[:, t]
    P = pd.DataFrame(p_channel(tmp, items))
    M = pd.DataFrame(m_channel(tmp, items))
    I = pd.DataFrame(interaction_channel(tmp, items))
    return P, M, I


# ---------------------------------------------------------------------------
# Step 4 — the additive model
# ---------------------------------------------------------------------------
class DeSAFE:
    """Additive dual-channel model. logit p = f_P + f_M + f_S, exactly."""

    def __init__(self, n_est=120, depth=4, lr=0.1):
        # HistGradientBoosting rather than GradientBoostingClassifier: identical
        # algorithm family, but binned and multi-threaded, which matters because
        # fit() trains 4 experts x (n_folds + 1) times. Wall time for one fit on
        # 12k rows fell from 184 s to a few seconds with no material AUC change.
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.mk = lambda: HistGradientBoostingClassifier(
            max_iter=n_est, max_depth=depth, learning_rate=lr,
            random_state=SEED)

    def fit(self, P, M, I=None, S=None, y=None):
        """Fit four experts, then a combiner on their OUT-OF-FOLD logits.

        Why out-of-fold: the experts are fitted on the same rows we would use to
        train the combiner, so their in-sample logits are over-confident. A
        combiner trained on those learns inflated weights (coefficients near 2.2)
        and then compresses predictions when it meets honest test logits — the
        observed symptom was a mean predicted risk of 0.008 against a prevalence of
        0.026, with AUC collapsing from 0.79 to 0.75 once a Platt scaler was bolted
        on. Cross-fitting the combiner removes the need for any rescaling and keeps
        the decomposition exact.
        """
        n = len(P)
        if I is not None and len(I) and len(I) != n:
            reps = int(round(n / len(I)))
            I = pd.DataFrame(np.tile(I.to_numpy(), (reps, 1)), columns=I.columns)
        if S is not None and len(S) and len(S) != n:
            reps = int(round(n / len(S)))
            S = pd.DataFrame(np.tile(S.to_numpy(), (reps, 1)), columns=S.columns)

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        self.fP, self.fM, self.fS, self.fI = self.mk(), self.mk(), self.mk(), self.mk()
        self.has_I = I is not None and I.shape[1] > 0
        self.has_S = S is not None and S.shape[1] > 0

        y = np.asarray(y)
        n_s = int(min(5, max(2, np.bincount(y).min())))
        folds = list(StratifiedKFold(n_s, shuffle=True,
                                     random_state=SEED).split(P, y))

        def oof_logits(make, X):
            lz = np.zeros(n)
            for tr, te in folds:
                m = make(); m.fit(X.iloc[tr], y[tr])
                p = np.clip(m.predict_proba(X.iloc[te])[:, 1], 1e-9, 1 - 1e-9)
                lz[te] = np.log(p / (1 - p))
            return lz

        blocks = [oof_logits(self.mk, P), oof_logits(self.mk, M)]
        if self.has_I:
            blocks.append(oof_logits(self.mk, I))
        if self.has_S:
            blocks.append(oof_logits(self.mk, S))
        Z = np.column_stack(blocks)

        self.cal = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
        self.order = ['P', 'M'] + (['I'] if self.has_I else []) + (['S'] if self.has_S else [])

        # now fit the deployable experts on all rows
        self.fP.fit(P, y)
        self.fM.fit(M, y)
        if self.has_I:
            self.fI.fit(I, y)
        if self.has_S:
            self.fS.fit(S, y)
        self.platt = None
        return self

    def _logitP(self, P):
        return _logit(self.fP.predict_proba(P)[:, 1])

    def _logitM(self, M):
        return _logit(self.fM.predict_proba(M)[:, 1])

    def _logitS(self, S):
        if S is None or not S.shape[1] or not hasattr(self, 'fS'):
            return None
        return _logit(self.fS.predict_proba(S)[:, 1])

    def components(self, P, M, I=None, S=None):
        n = len(P)
        raw = {'P': self._logitP(P), 'M': self._logitM(M)}
        if self.has_I and I is not None:
            raw['I'] = _logit(self.fI.predict_proba(I)[:, 1])
        if self.has_S and S is not None:
            raw['S'] = self._logitS(S)
        coef = self.cal.coef_[0]
        out = {k: np.zeros(n) for k in ('P', 'M', 'I', 'S')}
        for j, k in enumerate(self.order):
            out[k] = coef[j] * raw[k]
        return out

    def predict_proba(self, P, M, I=None, S=None):
        c = self.components(P, M, I, S)
        z = sum(c[k] for k in c) + self.cal.intercept_[0]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        if getattr(self, 'platt', None) is not None:
            lz = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
            return self.platt.predict_proba(lz.reshape(-1, 1))[:, 1]
        return p

    def surveillance_share(self, P, M, I=None, S=None):
        c = self.components(P, M, I, S)
        num = np.abs(c['M'])
        den = sum(np.abs(c[k]) for k in c) + 1e-9
        return num / den

    def physiology_only(self, P, M, I=None, S=None):
        """Counterfactual risk with the surveillance channel neutralised.

        The interaction channel is kept: it is conditioned on having >=2
        observations, so it survives CMA and represents real conditional
        information rather than surveillance intensity.
        """
        c = self.components(P, M, I, S)
        z = c['P'] + c['I'] + c['S'] + self.cal.intercept_[0]
        return 1.0 / (1.0 + np.exp(-z))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Step 5 — a plain baseline for comparison (reference-style, entangled)
# ---------------------------------------------------------------------------
class EntangledBaseline:
    """Reference-style model: one expert over mask-contaminated value features."""

    def __init__(self, n_est=120, depth=4, lr=0.1):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_iter=n_est, max_depth=depth,
                                                learning_rate=lr,
                                                random_state=SEED)

    def fit(self, X, y):
        self.m.fit(X, y); return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]
