#!/usr/bin/env python3
# @status:   maturing
# @process:  P6-revision
# @paper:    paper1
"""
Extended Data Figure: Spatial dynamics of crises by archetype (R1.4).

Reframed (2026-06-20) to lead with the footprint measure, which is robust at the
median, and to report neighbour co-escalation honestly with BOTH median and mean.

Two panels, archetypes ordered by the primary (footprint) measure:
  (a) PRIMARY — Net national footprint ratio (median end/onset of admin1 areas in
      crisis, IPC Phase 3+). Cleanly separates worsening archetypes (1.12-1.33)
      from stable/resolving ones (1.00) at the median.
  (b) SECONDARY — Net neighbour co-escalation (Δ share of adjacent admin1 areas in
      crisis, onset->end). Bars show the MEDIAN; diamonds show the MEAN. For
      escalating/oscillating the signal holds at the median (0.25, 0.33); for
      protracted/entrenched it is mean-driven (median 0), i.e. concentrated in a
      subset of episodes.

The worsening/stable contrast is not explained by duration: escalating (median
12 mo) expands more than the longer prolonged-moderate (18 mo).

Source (canonical per-episode):
  outputs/data/r1_spatial_extent_by_episode.csv     (net_ratio)
  outputs/data/r1_spatial_adjacency_by_episode.csv  (co_escalation_net)

Outputs:
  outputs/figures/ExtDataFig_spatial_dynamics.png / .pdf
  outputs/figures/SourceData_EDFig3.xlsx            (source data)
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent  # code/code -> capsule root (papers/paper1/code/)
OUT = PACKAGE_ROOT / 'outputs' / 'data'
FIG_DIR = PACKAGE_ROOT / 'outputs' / 'figures'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 7, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})

LABELS = {
    'seasonal_crisis': 'Seasonal crisis', 'prolonged_moderate': 'Prolonged moderate',
    'rapid_onset': 'Rapid onset', 'severe_shock': 'Severe shock',
    'entrenched_moderate': 'Entrenched moderate', 'protracted_emergency': 'Protracted emergency',
    'escalating': 'Escalating', 'oscillating': 'Oscillating',
}
STABLE = {'seasonal_crisis', 'prolonged_moderate', 'rapid_onset', 'severe_shock'}
C_STABLE, C_EXPAND = '#4C72B0', '#C44E52'


def _by_archetype(path, value_col):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        v = r.get(value_col, '')
        if v not in ('', 'nan', 'NA', None):
            g[r['archetype']].append(float(v))
    return g


def export_source_data(order, fp_med, co_med, co_mean, n_fp, n_co):
    """Export source data for Extended Data Figure 3 to Excel."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = FIG_DIR / 'SourceData_EDFig3.xlsx'

    # Panel a: net national footprint ratio (median end/onset areas in crisis)
    panel_a = pd.DataFrame({
        'archetype': [LABELS[a] for a in order],
        'net_footprint_ratio_median': [round(fp_med[a], 4) for a in order],
        'n_episodes': [n_fp[a] for a in order],
    })

    # Panel b: net neighbour co-escalation, median bars + mean diamonds
    panel_b = pd.DataFrame({
        'archetype': [LABELS[a] for a in order],
        'co_escalation_net_median': [round(co_med[a], 4) for a in order],
        'co_escalation_net_mean': [round(co_mean[a], 4) for a in order],
        'n_episodes': [n_co[a] for a in order],
    })

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        panel_a.to_excel(writer, sheet_name='panel_a_footprint', index=False)
        panel_b.to_excel(writer, sheet_name='panel_b_co_escalation', index=False)
    print(f"Saved: {xlsx_path}")


def main():
    fp_vals = _by_archetype(OUT / 'r1_spatial_extent_by_episode.csv', 'net_ratio')
    co_vals = _by_archetype(OUT / 'r1_spatial_adjacency_by_episode.csv', 'co_escalation_net')

    archs = [a for a in LABELS if a in fp_vals and a in co_vals]
    fp_med = {a: st.median(fp_vals[a]) for a in archs}
    co_med = {a: st.median(co_vals[a]) for a in archs}
    co_mean = {a: st.mean(co_vals[a]) for a in archs}
    # panel (a) and panel (b) rest on different episode sets: every episode has a
    # footprint, only those with a mapped neighbour set have co-escalation
    n_fp = {a: len(fp_vals[a]) for a in archs}
    n_co = {a: len(co_vals[a]) for a in archs}

    # order by primary measure (footprint median), tie-break by mean co-escalation
    order = sorted(archs, key=lambda a: (fp_med[a], co_mean[a]))
    labels = [LABELS[a] for a in order]
    colors = [C_STABLE if a in STABLE else C_EXPAND for a in order]
    y = np.arange(len(order))

    print(f"{'archetype':22} fp_median  co_median  co_mean  n_fp  n_co")
    for a in order:
        print(f"{a:22} {fp_med[a]:.2f}       {co_med[a]:.2f}      {co_mean[a]:.3f}   "
              f"{n_fp[a]:<5} {n_co[a]}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)

    # Panel a — PRIMARY: footprint median net ratio
    fp = [fp_med[a] for a in order]
    axa.barh(y, fp, color=colors, edgecolor='black', linewidth=0.4)
    axa.set_yticks(y)
    axa.set_yticklabels(labels)
    axa.set_xlabel('Net footprint ratio\n(median end/onset areas in crisis)')
    axa.set_title('a', loc='left', fontweight='bold')
    axa.axvline(1.0, color='black', linewidth=0.5, linestyle='--')
    for yi, a in enumerate(order):
        axa.text(fp_med[a] + 0.012, yi, f'{fp_med[a]:.2f} (n={n_fp[a]})', va='center', fontsize=6.3)
    axa.set_xlim(0.9, max(fp) + 0.16)

    # Panel b — SECONDARY: neighbour co-escalation, median bars + mean diamonds
    com = [co_med[a] for a in order]
    coa = [co_mean[a] for a in order]
    axb.barh(y, com, color=colors, edgecolor='black', linewidth=0.4, alpha=0.85)
    axb.scatter(coa, y, marker='D', s=18, color='black', zorder=5, label='Mean')
    axb.set_xlabel('Net neighbour co-escalation\n(Δ share of adjacent areas in crisis, onset→end)')
    axb.set_title('b', loc='left', fontweight='bold')
    axb.axvline(0, color='black', linewidth=0.5)
    for yi, a in enumerate(order):
        axb.text(max(co_med[a], co_mean[a]) + 0.008, yi,
                 f'med {co_med[a]:.2f} / mean {co_mean[a]:.2f}', va='center', fontsize=6.0)
    axb.set_xlim(-0.02, max(coa) + 0.16)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_EXPAND, ec='black', lw=0.4),
               plt.Rectangle((0, 0), 1, 1, color=C_STABLE, ec='black', lw=0.4),
               plt.Line2D([0], [0], marker='D', color='black', linestyle='None', markersize=5)]
    axa.legend(handles, ['Worsening archetypes', 'Stable / resolving archetypes', 'Mean (panel b)'],
               loc='lower right', frameon=False)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext_ in ('png', 'pdf'):
        fig.savefig(FIG_DIR / f'ExtDataFig_spatial_dynamics.{ext_}', bbox_inches='tight')
    print(f"Saved ExtDataFig_spatial_dynamics.png / .pdf to {FIG_DIR}")

    export_source_data(order, fp_med, co_med, co_mean, n_fp, n_co)


if __name__ == '__main__':
    main()
