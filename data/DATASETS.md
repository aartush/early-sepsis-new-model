# Datasets

**No patient data is included in this repository.** Both source databases are
released under PhysioNet credentialed data use agreements that prohibit
redistribution. Downloading them into a public repository breaches those
agreements and can result in permanent loss of PhysioNet access for the account
holder. Do not commit them.

---

## 1. MIMIC-IV v2.2 — required

The cohort comes from MIMIC-IV. Access requires credentialing.

| Step | Detail |
|---|---|
| 1. Account | https://physionet.org/register/ |
| 2. Training | Complete the CITI "Data or Specimens Only Research" course (~2–4 h), then link the completion report to your PhysioNet profile under *Training* |
| 3. Agreement | Request access to MIMIC-IV and sign the credentialed data use agreement (typically approved within a few days) |
| 4. Download | https://physionet.org/content/mimiciv/2.2/ |

Tables used by the pipeline:

- **Hosp module:** `patients`, `admissions`, `diagnoses_icd`, `d_icd_diagnoses`
- **ICU module:** `icustays`, `chartevents`, `d_items`
- **Derived:** the `sepsis3` concept table, built with
  https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/sepsis

`chartevents` alone is roughly 30 GB.

---

## 2. Reference study artefacts — required

This project reproduces Stylianides et al. (IEEE JBHI 2026); it does not
repackage their data. Their repository ships the processed cohort and the 40
balanced splits, which is what makes an exact reproduction possible.

```bash
git clone https://github.com/cstylianides96/sepsis-prediction
cd sepsis-prediction
git lfs pull          # fetches shared_dataset_mimiciv.csv (760 MB) and eicu (1.15 GB)
```

Their licence permits personal, educational and non-commercial use with
attribution. It reserves commercial rights and prohibits claiming ownership of
their code. This repository therefore contains an independent reimplementation of
their *procedure*, not their source files.

Point the code at that checkout:

```bash
export SEPSIS_REFERENCE_ROOT=/absolute/path/to/sepsis-prediction
# Windows PowerShell: $env:SEPSIS_REFERENCE_ROOT="C:\path\to\sepsis-prediction"
```

or place it at `./reference-repo` (gitignored).

Files the code expects inside that checkout:

```
sepsis-prediction/
├── data_processed/
│   ├── train_{1..40}.csv      balanced training splits
│   ├── test_{1..40}.csv       balanced test splits
│   ├── train_selected_feat40.csv
│   ├── val_selected_feat40.csv
│   └── test_selected_feat40.csv
├── models/
│   └── GBM_feat_selection_feat50(40).pkl    ordered list of the 40 features
└── shared_dataset_mimiciv.csv               45,148 × 2,211 full cohort
```

If any of these is missing, the loader raises a `FileNotFoundError` naming the
file and pointing back to this document.

---

## 3. eICU-CRD v2.0 — optional

Used only for the transfer arm, which **does not reproduce** (AUC 0.806 against a
reported 0.88) and is excluded from every analysis in the paper. Same
credentialing process: https://physionet.org/content/eicu-crd/2.0/

The paper explains why the gap is most likely label construction rather than
population difference: eICU prevalence in that release is 0.29% against 2.41% in
MIMIC, and the eICU feature space has roughly half as many columns.

---

## 4. PhysioNet/CinC Challenge 2019 — not used here

Open access (ODbL), no credentialing: https://physionet.org/content/challenge-2019/1.0.0/
Relevant only to the separate prior-work project in this programme.

---

## Reproducing without the data

Three components run from files already in this repository, because they depend on
our own outputs rather than on patient records:

```bash
python -m _audit.experiments.exp_c_prevalence   # needs predictions + labels
python -m _audit.experiments.exp_d_vacuity      # needs the rule list
python -m _audit.experiments.exp_curves         # needs predictions + labels
```

`exp_c_prevalence` and `exp_curves` need test-set *labels*, which come from
`data_processed/`, so they still require the reference checkout. `exp_d_vacuity`
reads only `_reproduction/results/te2rules_summary.json`, so it runs with no data
at all — useful as a smoke test.

---

## Sanity check before pushing

```bash
git status --porcelain | grep -Ei '\.psv|shared_dataset|train_[0-9]+\.csv|test_[0-9]+\.csv' \
  && echo "STOP: patient data staged" || echo "clean: no patient data"
```
