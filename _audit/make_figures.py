"""Generate the paper's figures from the audit result files.

Every figure reads only from `_audit/results/*.json` — nothing is hand-typed, so
regenerating the experiments regenerates the figures. Output is vector PDF at
300 dpi, sized for a two-column IEEEtran page (3.5 in wide).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _audit import common as C

W = 3.5          # single-column width, inches
DPI = 300
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'font.family': 'serif', 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.0, 'lines.markersize': 3,
})

FIG = C.FIGURES
(FIG / 'png').mkdir(exist_ok=True)


def load(name):
    return json.loads((C.RESULTS / name).read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Fig. 1 — the three evaluation regimes (prevalence discontinuity)
# ---------------------------------------------------------------------------
def fig_prevalence():
    d = load('exp_c_prevalence.json')
    rows = d['honest_analytic']
    thr = [r['threshold'] for r in rows]
    nb_bal = [r['net_benefit_balanced'] for r in rows]
    nb_hon = [r['net_benefit_honest'] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1))

    ax[0].plot(thr, nb_bal, 'o-', color='#1f4e79', label='balanced (0.50)')
    ax[0].plot(thr, nb_hon, 's--', color='#c00000', label='true (0.0241)')
    ax[0].axhline(0, color='k', lw=0.6, ls=':')
    ax[0].set_xlabel('decision threshold')
    ax[0].set_ylabel('net benefit')
    ax[0].set_title('(a) Net benefit changes sign')
    ax[0].legend(frameon=False, loc='lower left')

    ratio = [r['overstatement_factor'] for r in rows]
    ax[1].plot(thr, ratio, 'D-', color='#7f6000')
    ax[1].axhline(1, color='k', lw=0.6, ls=':')
    ax[1].set_xlabel('decision threshold')
    ax[1].set_ylabel('precision overstatement (x)')
    ax[1].set_title('(b) Reported precision is inflated')

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / 'fig1_prevalence.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG / 'png' / 'fig1_prevalence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig1_prevalence.pdf")


# ---------------------------------------------------------------------------
# Fig. 2 — the measurement-explained fraction across cohorts
# ---------------------------------------------------------------------------
def fig_mef():
    d = load('exp_a_summary.json')
    cohorts = [('full', 'full\n(2.41%)'), ('paper_test_split', 'paper test\n(2.41%)'),
               ('balanced_per_split', 'balanced\n(50%)')]
    labels, v, m, a = [], [], [], []
    for key, lab in cohorts:
        r = d.get(key)
        if not r:
            continue
        labels.append(lab)
        v.append(r['V_physiology']['auc'])
        m.append(r['M_measurement']['auc'])
        a.append(r['A_all']['auc'])

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.bar(x - w, v, w, label='V: physiology', color='#1f4e79')
    ax.bar(x, m, w, label='M: measurement', color='#c00000')
    ax.bar(x + w, a, w, label='A: all', color='#548235')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('AUC (5-fold CV)')
    ax.set_ylim(0.40, 1.0)
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.set_title('Availability alone recovers most of the signal')
    for xi, (vv, mm) in enumerate(zip(v, m)):
        ax.text(xi, min(vv, mm) - 0.045, f'{100*(mm-0.5)/(vv-0.5):.0f}%',
                ha='center', fontsize=6, fontweight='bold')
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / 'fig2_measurement_explained.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG / 'png' / 'fig2_measurement_explained.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2_measurement_explained.pdf")


# ---------------------------------------------------------------------------
# Fig. 3 — permutation test and mechanism decomposition
# ---------------------------------------------------------------------------
def fig_verification():
    v = load('exp_a_verification.json')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1))

    perm = v['permutation']
    vals = [perm['real_auc']] + perm['permuted_aucs']
    names = ['real'] + [f'shuf {i+1}' for i in range(len(perm['permuted_aucs']))]
    cols = ['#c00000'] + ['#999999'] * (len(vals) - 1)
    ax[0].bar(names, vals, color=cols)
    ax[0].axhline(0.5, color='k', lw=0.6, ls=':')
    ax[0].set_ylim(0, 1.0)
    ax[0].set_ylabel('AUC')
    ax[0].set_title('(a) Permutation test')
    ax[0].tick_params(axis='x', rotation=30)

    fam = v['family_decomposition']
    keys = ['presence (n_distinct)', 'repetition (n_change)',
            'imputed-run (n_static_hours)', 'timing (first_hour/span)']
    keys = [k for k in keys if k in fam]
    short = ['presence', 'repetition', 'imputed run', 'timing']
    vals2 = [fam[k]['auc'] for k in keys]
    ax[1].barh(short[:len(keys)], vals2, color='#1f4e79')
    ax[1].axvline(0.5, color='k', lw=0.6, ls=':')
    ax[1].set_xlim(0.4, 0.9)
    ax[1].set_xlabel('AUC')
    ax[1].set_title('(b) Which availability family')
    for i, val in enumerate(vals2):
        ax[1].text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=5)

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / 'fig3_verification.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG / 'png' / 'fig3_verification.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig3_verification.pdf")


# ---------------------------------------------------------------------------
# Fig. 4 — artefact-conditional yield
# ---------------------------------------------------------------------------
def fig_artefact():
    b = load('exp_b_summary.json')
    uf, ub = b['union_full'], b['union_balanced']
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1))

    labels = ['full\n(2.41%)', 'balanced\n(50%)']
    cov = [uf['pct_fire'], ub['pct_fire']]
    cap = [100 * uf['captures_fraction_of_sepsis'],
           100 * ub['captures_fraction_of_sepsis']]
    x = np.arange(2); w = 0.35
    ax[0].bar(x - w/2, cov, w, label='patients flagged (%)', color='#c00000')
    ax[0].bar(x + w/2, cap, w, label='sepsis captured (%)', color='#1f4e79')
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel('percent')
    ax[0].legend(frameon=False, fontsize=5)
    ax[0].set_title('(a) Subgroup vs yield', fontsize=7)

    prof = b['artefact_features']
    full = np.array([[r['pct_nonzero'], r['ppv_if_fires'], r['base_rate']]
                     for r in __import__('pandas').read_csv(
                         C.RESULTS / 'exp_b_feature_profile_full.csv').to_dict('records')])
    names = [__import__('pandas').read_csv(
        C.RESULTS / 'exp_b_feature_profile_full.csv')['feature'].tolist()]
    feats = names[0]
    ax[1].scatter(full[:, 0], full[:, 1], s=14, color='#c00000', zorder=3)
    ax[1].axhline(uf['base_rate'], color='k', lw=0.7, ls='--',
                  label=f"base rate {100*uf['base_rate']:.1f}%")
    ax[1].set_xlabel('patients where feature fires (%)')
    ax[1].set_ylabel('precision when it fires')
    ax[1].set_title('(b) Precision when fires', fontsize=7)
    ax[1].legend(frameon=False, fontsize=5)
    for i, f in enumerate(feats):
        ax[1].annotate(f.split('_')[0], (full[i, 0], full[i, 1]),
                       fontsize=4, xytext=(2, 2), textcoords='offset points')

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / 'fig4_artefact.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG / 'png' / 'fig4_artefact.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4_artefact.pdf")


# ---------------------------------------------------------------------------
# Fig. 5 — rule vacuity
# ---------------------------------------------------------------------------
def fig_vacuity():
    import pandas as pd
    d = pd.read_csv(C.RESULTS / 'exp_d_rules_positive_balanced.csv')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1))

    ax[0].hist(d['n_clauses'], bins=range(1, 10), alpha=0.55,
               label='as written', color='#7f6000')
    ax[0].hist(d['effective_length'], bins=range(1, 10), alpha=0.8,
               label='effective', color='#1f4e79')
    ax[0].set_xlabel('clauses per rule')
    ax[0].set_ylabel('rule count')
    ax[0].legend(frameon=False, fontsize=5)
    ax[0].set_title('(a) Vacuity shortens rules')

    ax[1].scatter(d['support_frac'] * 100, d['max_clause_vacuity'] * 100,
                  s=12, color='#c00000', zorder=3)
    ax[1].axhline(95, color='k', lw=0.7, ls='--')
    ax[1].set_xlabel('rule support (% of patients)')
    ax[1].set_ylabel('most vacuous clause (%)')
    ax[1].set_title('(b) Vacuity vs support')

    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / 'fig5_vacuity.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(FIG / 'png' / 'fig5_vacuity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig5_vacuity.pdf")


if __name__ == '__main__':
    print("generating figures ->", FIG)
    fig_prevalence()
    fig_mef()
    fig_verification()
    fig_artefact()
    fig_vacuity()
    print("done")
