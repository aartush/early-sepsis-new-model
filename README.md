# Early Sepsis Prediction — Reproduction and Measurement-Artefact Audit

**Authors:** Tushar Sharma (2026PAD2515) · Ayush Juneja (2026PAD2514)
M.Tech, 1st year — Netaji Subhas University of Technology, Delhi

We reproduce a published early-sepsis prediction model to the third decimal, then
ask what its reported numbers actually measure. Two properties of the evaluation —
not of the physiology — turn out to account for most of the reported performance.
We quantify both, and build a model that separates them.

---

## Headline results

| Finding | Number |
|---|---|
| Baseline reproduces | AUC **0.8704** vs published 0.87; net benefit **0.4501** vs 0.45 |
| Measurement-only model | AUC **0.8615** vs physiology 0.8773 — **92.2%** of the signal |
| Permutation control | shuffled labels → **0.497** |
| Capacity-matched control | 40 features each → **0.8310** (87.7%) |
| Prevalence correction | net benefit **+0.4501 → −0.0546**; precision **17.7×** overstated |
| Artefact subgroup | **5.6%** of patients contain **23.0%** of all sepsis |
| Rule vacuity | **26.3%** of positive rules carry a clause true for ≥95% of patients |
| DeSAFE vs baseline | AUC **0.8749** vs 0.8547; **2.5× less mask-dependent** |


---

## The argument in one paragraph

The published model is not wrong; it is correctly built and correctly measured.
But it is measured in a population that does not exist: a 50% prevalence instead of
the cohort's 2.41%, and with features that partly encode how often the team drew
blood rather than what the blood showed. Correcting for both removes the reported
clinical benefit and changes its sign. Separating the two signal channels into an
additive, exactly decomposable model recovers the performance while making the
surveillance contribution visible per patient, and makes the model **2.5× less
dependent on observation pattern** than the baseline.

---

## Repository structure

```
.
├
├── _reproduction/             Step 1 — does the published study reproduce?
│   ├── common.py              shared paths + metric routine
│   ├── gbm.py                 gradient boosting, 40 balanced splits
│   ├── dl.py                  LSTM / TCN / 1DCNN / 1DCNN-LSTM
│   ├── ensemble.py            soft voting + net benefit
│   ├── te2rules.py            rule extraction, 40 iterations × 2 classes
│   ├── eicu.py                eICU arm (does not reproduce — see paper §V)
│   ├── prevalence.py          prevalence correction
│   ├── predictions/           per-split model outputs (our results, shipped)
│   └── results/               all metrics as JSON
│
├── _audit/                    Step 2 — what do those numbers mean?
│   ├── common.py              cohort loaders, availability features, metrics
│   ├── desafe.py              the DeSAFE model (4-channel additive + CMA)
│   ├── experiments/
│   │   ├── exp_a_availability.py   null model, measurement-explained fraction
│   │   ├── exp_a_verify.py         permutation / incremental / family controls
│   │   ├── exp_a_capacity.py       capacity-matched control
│   │   ├── exp_b_artefact.py       artefact-conditional performance
│   │   ├── exp_c_prevalence.py     balanced vs true prevalence
│   │   ├── exp_d_vacuity.py        rule vacuity scoring
│   │   ├── exp_e1_desafe.py        decomposition + surveillance share
│   │   ├── exp_e2_desafe.py        DeSAFE vs entangled baseline
│   │   ├── exp_e3_multifold.py     robustness to monitoring shift (5-fold)
│   │   ├── exp_e4_surgery.py       mask-surgery test
│   │   └── exp_curves.py           ROC / PR / calibration / DCA + importances
│   ├── make_figures.py        figures 2–6 from results JSON
│   ├── make_replication_figures.py  figures 1 and 2
│   └── results/               all numbers as JSON/CSV
│
├── data/
│   └── DATASETS.md            how to obtain MIMIC-IV (not redistributed)
│
├── docs/
│   ├── RESOURCES.md           technical stack, libraries, hardware
│   └── GUIDEBOOK.md           full walkthrough of the project
│
├── requirements.txt
└── .gitignore
```

---

## Flow

```
     Published model (MIMIC-IV, 45,148 stays, 2.41% sepsis)
                        ┌─────────────────────────────────┐
                        │  MIMIC-IV, 45,148 ICU stays     │
                        │  1,088 sepsis cases = 2.41%     │
                        └────────────────┬────────────────┘
                                         │
        ╔════════════════════════════════▼════════════════════════════════╗
        ║  STEP 1 — REPRODUCE                                             ║
        ║  "Does the published study actually reproduce?"                  ║
        ╚════════════════════════════════┬════════════════════════════════╝
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
              ┌──────────┐        ┌───────────┐        ┌──────────┐
              │   GBM    │        │  4 × DL   │        │ TE2Rules │
              │ 0.8704   │        │ 0.781–0.800│       │ 42 / 47  │
              └──────────┘        └───────────┘        └──────────┘
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                        ┌─────────────────────────────────┐
                        │  Ensemble  AUC 0.8646           │
                        │  Net benefit @0.1 = 0.4501      │
                        │  (paper claims 0.87 and 0.45)   │
                        └────────────────┬────────────────┘
                                         │
                       ╔═════════════════▼═════════════════╗
                       ║  GATE: 9 of 9 metrics within ±0.01 ║
                       ║  → criticism is now defensible     ║
                       ╚═════════════════┬═════════════════╝
                                         │
        ╔════════════════════════════════▼════════════════════════════════╗
        ║  STEP 2 — AUDIT                                                 ║
        ║  "What do those numbers actually measure?"                       ║
        ╚════════════════════════════════┬════════════════════════════════╝
                                         │
              ┌──────────────┬───────────┴───────────┬──────────────┐
              ▼              ▼                       ▼              ▼
        ┌──────────┐  ┌──────────┐          ┌──────────┐   ┌──────────┐
        │    A     │  │    B     │          │    C     │   │    D     │
        │ null     │  │ artefact │          │ preval-  │   │ rule     │
        │ model    │  │ features │          │ ence     │   │ vacuity  │
        └────┬─────┘  └────┬─────┘          └────┬─────┘   └────┬─────┘
             │             │                     │              │
        MEF 92.2%    23% of sepsis        NB +0.45→−0.05    26.3% of
        (87.7% cap-  from 5.6% of         precision 17.7×    rules have
         matched)     patients             overstated        a dead clause
             │             │                     │              │
             └─────────────┴──────────┬──────────┴──────────────┘
                                      ▼
                 ┌────────────────────────────────────────┐
                 │  Three things the number was hiding    │
                 │  • signal comes from measurement       │
                 │  • utility depends on assumed prior    │
                 │  • explanations are shorter than shown │
                 └───────────────────┬────────────────────┘
                                     │
        ╔════════════════════════════▼════════════════════════════════════╗
        ║  STEP 3 — BUILD THE FIX                                         ║
        ║  "If measurement carries 92%, deleting it is wrong —            ║
        ║   so separate it instead."                                      ║
        ╚════════════════════════════════┬════════════════════════════════╝
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        ┌────────────────────────┐              ┌────────────────────────┐
        │  DeSAFE architecture   │              │  CMA training          │
        │  logit = f_P + f_M     │              │  donor masks from      │
        │        + f_I + f_S     │              │  same label + decile   │
        └───────────┬────────────┘              └───────────┬────────────┘
                    │                                       │
                    └──────────────────┬────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  AUC 0.8749  (baseline 0.8547)       │
                    │  σ median 0.309  (surveillance share)│
                    │  wins 5/5 folds at every shift level │
                    │  MSI 0.0043 vs 0.0108 baseline       │
                    └──────────────────┬───────────────────┘
                                       │
        ╔══════════════════════════════▼══════════════════════════════════╗
        ║  STEP 4 — PACKAGE + STATE THE LIMITS                            ║
        ║  10-page IEEE paper · 8 figures · 11 tables · 2 algorithms      ║
        ╚═════════════════════════════════════════════════════════════════╝
```

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Data is NOT in this repository — see data/DATASETS.md.
# Then point the code at a local checkout of the reference study's artefacts:
export SEPSIS_REFERENCE_ROOT=/path/to/sepsis-prediction
#   (Windows PowerShell: $env:SEPSIS_REFERENCE_ROOT="C:\path\to\sepsis-prediction")

# Step 2: audit experiments (the paper's results)
python -m _audit.experiments.exp_c_prevalence     # seconds
python -m _audit.experiments.exp_d_vacuity        # seconds
python -m _audit.experiments.exp_b_artefact       # ~5 min
python -m _audit.experiments.exp_a_availability   # ~25 min
python -m _audit.experiments.exp_a_verify         # ~20 min
python -m _audit.experiments.exp_a_capacity       # ~15 min

# Step 3: DeSAFE
python -m _audit.experiments.exp_e1_desafe        # ~6 min
python -m _audit.experiments.exp_e2_desafe        # ~12 min
python -m _audit.experiments.exp_e3_multifold     # ~20 min
python -m _audit.experiments.exp_e4_surgery       # ~15 min

# Figures and paper
python -m _audit.make_figures
python -m _audit.experiments.exp_curves
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every experiment writes JSON to `_audit/results/`. No number in the paper is typed
by hand — see the claim-to-file map in [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

For the deep-learning arms, set `KERAS_BACKEND=torch`.

---

## Contact

Tushar Sharma — tushar.sharma.pg26@nsut.ac.in
Ayush Juneja — ayush.juneja.pg26@nsut.ac.in
Supervisor: Dr. Ankush Jain, Assistant Professor (PhD), NSUT Delhi
