# Evidence manifest

Every quantitative claim in `paper/main.tex` maps to a generated file below.
Numbers must be regenerated from these files, never hand-edited.

Generated: 2026-11-17.

---

## §III-B — Reproduction of the target study (Table I)

| Claim | Value | Source | Script |
|---|---|---|---|
| GBM AUC | 0.8704 | `_reproduction/results/gbm_summary.json` → `test_auc` | `_reproduction/gbm.py` |
| H-LSTM AUC | 0.781 | `_reproduction/results/dl_LSTM_1_summary.json` | `_reproduction/dl.py` |
| Ensemble AUC | 0.8646 | `_reproduction/results/ENSEMBLE_40_summary.json` → `test_auc` | `_reproduction/ensemble.py` |
| Average precision | 0.8622 | same file → `mean_average_precision` | same |
| Brier | 0.1513 | same file → `mean_brier` | same |
| Net benefit @ 0.1 | 0.4501 | same file → `net_benefit_mean["0.1"]` | same |
| Positive rules | 39.95 mean / 38 best | `_reproduction/results/te2rules_summary.json` | `_reproduction/te2rules.py` |
| Fidelity | 0.945 ± 0.005 | same → `positive.mean_fidelity` | same |

**Guard:** the paper states nine of nine GBM metrics land within ±0.01. Verified in
`_reproduction/REPRODUCTION.md` Table A. The eICU arm is excluded and reported only
as a caveat: `_reproduction/results/gbm_eicu_summary.json` gives 0.8063.

---

## §V — Experiment A, measurement-availability null model (Table II, Fig. 1, Fig. 2)

| Claim | Value | Source | Script |
|---|---|---|---|
| Full cohort: V / M / M+S / A | 0.8773 / 0.8615 / 0.8641 / 0.8923 | `_audit/results/exp_a_summary.json` → `full` | `_audit/experiments/exp_a_availability.py` |
| MEF full | 92.2% | same → `full.MEF_measurement_only` | same |
| Paper test split: V / M / A | 0.8522 / 0.8441 / 0.8642 | same → `paper_test_split` | same |
| MEF paper test | 94.5% | same | same |
| Balanced per-split: V / M / A | 0.8691 / 0.8498 / 0.8889 | same → `balanced_per_split` | same |
| MEF balanced | 90.0% | same | same |
| Cohort size / prevalence | 45,148 / 2.41% | same → `full.n` | same |

**Guards:**
- Cohort mapping is asserted, not assumed: `verify_cohort_mapping()` in the script
  checks that `train/val/test_selected_feat40` concatenate to 45,148 rows and match
  the cohort row-for-row, and that `test_k.csv`'s `index` maps into the test split.
- The balanced cohort is evaluated per-split (train_k/test_k, averaged over 40), not
  by pooling. Pooling leaks the 218 repeated sepsis patients across folds and
  produced AUC 0.998 in an earlier draft.

---

## §V-C — Verification (Fig. 3)

| Claim | Value | Source | Script |
|---|---|---|---|
| Permutation: real vs shuffled | 0.8615 vs 0.5018 / 0.4922 / 0.4968 | `_audit/results/exp_a_verification.json` → `permutation` | `_audit/experiments/exp_a_verify.py` |
| Incremental V → V+M | 0.8773 → 0.8923 (+0.0151) | same → `incremental` | same |
| Family: presence / repetition / imputed-run / timing | 0.7629 / 0.8111 / 0.8110 / 0.8465 | same → `family_decomposition` | same |
| Top single feature (glucose span) | 0.6769 | same → `top_single_features[0]` | same |
| M at 5 / 10 / 20 / 40 / 60 features | 0.7219 / 0.7507 / 0.7715 / 0.8310 / 0.8464 | `_audit/results/exp_a_capacity_control.json` | `_audit/experiments/exp_a_capacity.py` |
| M at 40 features, MEF vs V | 87.7% | same → `M_top40` | same |
| One column per itemid (31 feats) | 0.8222, MEF 85.4% | same → `M_one_per_itemid` | same |
| Timing removed / timing only | 0.8124 / 0.8465 | same → `M_no_timing`, `M_timing_only` | same |

**Guard the paper protects:** the abstract quotes 92.2% but §V-C and §VI report the
capacity-matched 85–88% figure. The two are not interchangeable and the paper says so.

---

## §VI — Experiment B, artefact-conditional (Fig. 4)

| Claim | Value | Source | Script |
|---|---|---|---|
| BUN diff_0 fires / precision / lift | 321 patients, 18.1%, 7.5× | `_audit/results/exp_b_feature_profile_full.csv` | `_audit/experiments/exp_b_artefact.py` |
| Glucose diff_0 fires / precision / lift | 361, 19.9%, 8.3× | same | same |
| Union fires | 2,539 / 45,148 = 5.62% | `_audit/results/exp_b_summary.json` → `union_full.pct_fire` | same |
| Union precision | 9.85% (vs 2.41% base) | same → `ppv_if_fires` | same |
| Share of all sepsis captured | 23.0% | same → `captures_fraction_of_sepsis` | same |
| Ablation full cohort | 0.8773 → 0.8662 (−0.011) | same → `ablation_full` | same |
| Ablation balanced | 0.8691 → 0.8664 (−0.003) | same → `ablation_balanced` | same |

---

## §VII — Experiment C, prevalence discontinuity (Table III, Fig. 5)

| Claim | Value | Source | Script |
|---|---|---|---|
| True prevalence | 2.41% (1,088 / 45,148) | `_audit/results/exp_c_prevalence.json` → `true_prevalence` | `_audit/experiments/exp_c_prevalence.py` |
| Precision @ 0.10, balanced → honest | 0.576 → 0.033 | same → `honest_analytic` entry with `threshold == 0.1` | same |
| Overstatement factor | 17.7× | same → `overstatement_factor` | same |
| Net benefit @ 0.10, balanced → honest | +0.4501 → −0.0546 | same → `net_benefit_balanced`, `net_benefit_honest` | same |
| Net benefit @ 0.05 honest | −0.0203 | same | same |
| Analytic vs empirical agreement | max abs diff 0.0000 | same → `agreement_max_abs_diff_analytic_vs_empirical` | same |

**Guard the paper protects (§VII-C):** the paper explicitly states that negative net
benefit at threshold 0.1 does *not* mean the model is uninformative — at 0.05 the
analytic net benefit is marginally positive. The claim is scoped to the stated
operating region.

---

## §VIII — Experiment D, rule vacuity (Fig. 6)

| Claim | Value | Source | Script |
|---|---|---|---|
| Positive rules | 38 | `_audit/results/exp_d_summary.json` → `positive_balanced.n_rules` | `_audit/experiments/exp_d_vacuity.py` |
| Rules with a vacuous clause (positive) | 10 / 38 = 26.3% | same → `pct_rules_with_vacuous_clause` | same |
| Vacuous clauses (positive) | 11 / 173 = 6.4% | same → `total_vacuous_clauses` / `total_clauses` | same |
| Negative: rules with vacuous clause | 21 / 46 = 45.7% | same → `negative_full` | same |
| Negative: vacuous clauses | 26 / 246 = 10.6% | same | same |
| Worked example rule | 5 clauses, 2 vacuous | `_audit/results/exp_d_rules_positive_balanced.csv` → `rule_id == 1` | same |

**Vacuity threshold:** 0.95, i.e. a clause true for ≥95% of patients counts as
non-discriminative. Stated in the paper and in `exp_d_vacuity.py:VACUITY_HIGH`.

---

## Figures

| Figure | File | Generated by |
|---|---|---|
| Fig. 1 (MEF bars) | `_audit/figures/fig2_measurement_explained.pdf` | `_audit/make_figures.py::fig_mef` |
| Fig. 2 (verification) | `_audit/figures/fig3_verification.pdf` | `::fig_verification` |
| Fig. 3 (artefact) | `_audit/figures/fig4_artefact.pdf` | `::fig_artefact` |
| Fig. 4 (prevalence) | `_audit/figures/fig1_prevalence.pdf` | `::fig_prevalence` |
| Fig. 5 (vacuity) | `_audit/figures/fig5_vacuity.pdf` | `::fig_vacuity` |

Figure numbering in the manuscript follows placement order, which differs from
filename order; the mapping above is the authority.

---

## Bugs found and fixed during construction

Recorded so the harness can be trusted rather than assumed:

1. **Availability columns filtered away.** `build_views` filtered derived columns
   against `df.columns`, reducing the M view from 155 columns to the one real
   shared column. Symptom: "M+S 1 feats". Fixed by concatenating the derived frame
   before filtering and asserting each view resolves to >0 columns.

2. **`itemid_of` returned `None` for `hosp_to_icu`,** silently creating an
   `hourly_columns` entry keyed `None` and dropping 150 real columns from M.

3. **Balanced-cohort leakage in the audit.** Pooling the 40 balanced sets and
   cross-validating put the 218 repeated sepsis patients in both training and
   scoring folds, giving AUC 0.998. Replaced with the paper's own train_k/test_k
   protocol. This is the same class of error the paper itself avoids, and catching
   it was necessary before criticising anyone else's evaluation.

4. **Prevalence reweighting sign error.** `w_neg = (1-p)/(1-0.5)` weights negatives
   *down*, yielding an effective prevalence of 33.9% rather than 2.41%. Correct
   weight is `(1-p)/p`. The analytic route is reported alongside the empirical one
   specifically so this class of error cannot pass unnoticed.

5. **Capacity confound in Experiment A.** M has 155 features against V's 40, so M
   could win on dimensionality alone. The capacity control was added after the
   first results looked implausibly strong; it attenuates the headline from 92.2%
   to 85–88% and both numbers are reported.

---

## What is not claimed

- We do not claim the target study contains a coding error. It reproduces.
- We do not claim the model is useless. Net benefit is marginally positive at low
  thresholds.
- We do not claim the effect generalises across the sepsis literature. One model
  was reproduced; the protocol is offered so others can test the claim.
- We do not use the eICU arm, which failed to reproduce.


---

## §IX — DeSAFE, the deconfounded model (Fig. 6)

| Claim | Value | Source | Script |
|---|---|---|---|
| DeSAFE out-of-fold AUC | 0.8710 | `results/exp_e1_desafe.json` → `auc_full` | `experiments/exp_e1_desafe.py` |
| Physiology channel alone | 0.8422 | same → `auc_P` | same |
| Surveillance channel alone | 0.7914 | same → `auc_M` | same |
| Surveillance share median / IQR | 0.309 / [0.278, 0.335] | same → `sigma_median`, `sigma_iqr` | same |
| σ cases vs controls | 0.291 vs 0.305 | same | same |
| vs entangled baseline (no CMA) | 0.8712 vs 0.8547 | `results/exp_e2_summary.json` → `no_cma` | `experiments/exp_e2_desafe.py` |
| vs entangled baseline (CMA) | 0.8749 vs 0.8547 | same → `cma` | same |
| Brier, no CMA | 0.0189 vs 0.0204 | same | same |
| Net benefit @0.10, no CMA | +0.0082 vs +0.0062 | same | same |
| Shift curve (0.0–0.8) | DeSAFE+CMA ahead at every level, 5/5 folds | `results/exp_e3_summary.json` → `curve`, `paired_deltas` | `experiments/exp_e3_multifold.py` |
| Shift margin | +0.020 → +0.029 (widens under shift) | same → `paired_deltas` | same |
| Policy-flip AUC | 0.8713 vs 0.8435 | same → `policy_flip` | same |
| Degradation slopes | −0.129 / −0.115 / −0.121 | same → `slopes` | same |

**Guards:**
- The combiner is fitted on **out-of-fold** expert logits. Fitting it in-sample
  inflated coefficients to ~2.2, produced in-sample AUC 1.0, and collapsed honest
  AUC to 0.75. Do not "simplify" this away.
- Calibration is inherited from that cross-fitting; mean predicted risk is 0.0294
  against a 0.0255 prevalence with no post-hoc scaling.
- E3 is **5-fold**, not single-fold. An earlier version reported one fold and gave
  0.8886 and 0.8825 on two runs of identical code; `HistGradientBoosting` is not
  bit-reproducible. All quoted E3 numbers are fold means with sd, per-fold values in
  `results/exp_e3_per_fold.csv`.
- With five folds the advantage is both a level gap (+0.020 at zero shift) and a
  slightly flatter slope (−0.121 vs −0.129). Do not quote the slope claim without
  the fold-mean table.
- The shift operator is simulated, not observed. Stated in `DESAFE.md`.

## Design history worth preserving

The first DeSAFE was `f_P + f_M + f_S` only and scored 0.8577 against the
baseline's 0.8773. The 0.02 gap was **not** leakage — it was genuine interaction
signal (a lab that is trended *and* moving). Adding channel `I` recovered it and
DeSAFE reached 0.8710. This is documented because the first design would have
shipped with a plausible story and an unexplained deficit.


## §X — E4, mask surgery (the vulnerability test)

| Claim | Value | Source | Script |
|---|---|---|---|
| Mask-sensitivity index, entangled | 0.0108 | `results/exp_e4_surgery.json` → `msi` | `experiments/exp_e4_surgery.py` |
| MSI, DeSAFE (P+S only) | 0.0007 | same | same |
| MSI, DeSAFE + CMA | 0.0043 | same | same |
| Ratio DeSAFE : entangled | 2.5× less mask-dependent | same | same |
| Surgery curve, all levels | `curve` | same | same |

**Guard:** the surgery operator replaces observation *patterns* with same-label
donor masks while preserving each patient's own value multiset (verified: value
multiset identical at 100% surgery, no-op at 0%). It therefore measures robustness
to **pattern** change, not to a **policy** change. Do not describe MSI as
simulating a different hospital.

**Why this experiment matters more than the AUC numbers:** the fatal review
objection to DeSAFE is "your interaction channel is smuggled surveillance". E4 is
the direct answer and should be prominent in the paper, not buried.
