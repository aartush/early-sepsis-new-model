# Resources

What the project was built with. Everything listed is publicly available and free
for academic use.

---

## Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| CPU | 24 threads (multi-core, used for `n_jobs=-1` in the searches) |
| RAM | Sufficient for the 760 MB cohort file; the audit loads 45,148 × 2,211 into memory |
| OS | Windows 11, Git Bash |

The deep-learning arms train on CPU or GPU. The GPU shortens a full LSTM run from
roughly 35 minutes to a few, but the pipeline is reproducible without one.

---

## Software

### Core

| Library | Version | Used for |
|---|---|---|
| Python | 3.12 | everything |
| numpy | 2.5 | array operations |
| pandas | 3.0 | cohort handling, CSV I/O |
| scikit-learn | 1.9 | `HistGradientBoosting`, `StratifiedKFold`, metrics |
| PyTorch | 2.11 (CUDA 12.8 build) | tensor backend for Keras |
| Keras | 3.15, torch backend | the four hybrid neural networks |
| imbalanced-learn | 0.14 | SMOTE comparison |
| TE2Rules | 1.0 | rule extraction from the GBM ensemble |
| matplotlib | 3.11 | every figure |

### Document

| Tool | Used for |
|---|---|
| MiKTeX / pdfTeX 4.23 | LaTeX compilation |
| IEEEtran class | manuscript format |
| TikZ | methodology flowchart and project flow diagram |
| algorithm / algpseudocode | the two pseudocode blocks |
| poppler (`pdftotext`, `pdftoppm`) | verifying rendered output |

---

## Data sources

| Source | Access | Role |
|---|---|---|
| MIMIC-IV v2.2 | PhysioNet credentialing | the cohort |
| eICU-CRD v2.0 | PhysioNet credentialing | transfer arm (did not reproduce) |
| Reference study repository | public, non-commercial | processed splits and cohort release |

Full access instructions: [`data/DATASETS.md`](../data/DATASETS.md).

---

## Libraries used inside the reproduced pipeline

| Library | Licence | Note |
|---|---|---|
| TE2Rules (LinkedIn) | **CC BY-NC 4.0** — non-commercial only | Anyone building commercially on this code must obtain a commercial licence or replace this component |
| scikit-learn, numpy, pandas, matplotlib | BSD-3 / PSF | standard academic use |

---

## Literature search

PubMed and Europe PMC were used to verify novelty claims and to confirm each
reference's authorship. Every entry in `paper/references.bib` was checked against a
recorded PubMed ID; the identifiers are kept in the `note` field of each entry so
the check is repeatable.

---

## Compute budget

| Task | Wall time |
|---|---|
| GBM, 40 balanced splits (150-candidate search) | ~55 min |
| Four deep-learning arms, 40 chunks each | ~35 min total |
| TE2Rules, 40 iterations × 2 classes | ~2.2 h |
| Audit experiments (A–D plus controls) | ~50 min |
| DeSAFE (E1–E4) | ~55 min |
| Figures | seconds |
| LaTeX, four passes | seconds |

A full clean reproduction is roughly half a day of compute on the hardware above.
