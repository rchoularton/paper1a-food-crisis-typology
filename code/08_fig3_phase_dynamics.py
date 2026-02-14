#!/usr/bin/env python3
"""
08_fig3_phase_dynamics.py — Figure 3: Phase Persistence and Transition Dynamics
=================================================================================

2x3 grid:
  (a) Transition probability matrix (heatmap)
  (b) Phase 1 — Minimal: escalation by duration
  (c) Phase 2 — Stressed: recovery vs escalation by duration
  (d) Phase 3 — Crisis: recovery vs escalation + crossover annotation
  (e) Phase 4 — Emergency: recovery vs escalation by duration
  (f) Phase 5 — Famine: recovery (small-n caveat)

Each duration panel shows recovery (green, solid, circles) and escalation
(red, dashed, triangles) with shaded 95% CI bands.

Inputs  (relative to package root):
    outputs/data/full_transition_matrix.json
    outputs/data/phase{1..5}_duration_conditioned.json
    outputs/data/phase3_crossover.json  (optional)

Outputs (relative to package root):
    outputs/figures/Figure3_phase_dynamics.png  (600 dpi)
    outputs/figures/Figure3_phase_dynamics.pdf

Author: Richard Choularton
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ============================================================
# Nature Food style
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# ============================================================
# Constants
# ============================================================
COL_RECOVERY = '#228833'    # green
COL_ESCALATION = '#EE6677'  # red

DURATION_TICK_LABELS = ['1\u20133', '4\u20136', '7\u201312', '13\u201324', '24+']
# Evenly spaced positions so all bins get equal visual weight
DURATION_MIDPOINTS = [1, 2, 3, 4, 5]
BIN_KEYS = ['1-3 mo', '4-6 mo', '7-12 mo', '13-24 mo', '24+ mo']

PHASE_TITLES = {
    1: 'Phase 1 (Minimal)',
    2: 'Phase 2 (Stressed)',
    3: 'Phase 3 (Crisis)',
    4: 'Phase 4 (Emergency)',
    5: 'Phase 5 (Famine)',
}


# ============================================================
# Data loading
# ============================================================

def load_phase_data(phase_num):
    """Load duration-conditioned JSON for a given IPC phase."""
    path = os.path.join(DATA_DIR, f'phase{phase_num}_duration_conditioned.json')
    with open(path) as f:
        return json.load(f)


def extract_series(phase_data):
    """Extract recovery, persistence, escalation arrays + CIs from phase JSON."""
    bins = phase_data['bins']
    rec = [bins[k]['recovery_pct'] for k in BIN_KEYS]
    esc = [bins[k]['escalation_pct'] for k in BIN_KEYS]
    persist = [bins[k]['persistence_pct'] for k in BIN_KEYS]
    n_vals = [bins[k]['n'] for k in BIN_KEYS]

    rec_ci = [bins[k].get('recovery_ci', [rec[i], rec[i]]) for i, k in enumerate(BIN_KEYS)]
    esc_ci = [bins[k].get('escalation_ci', [esc[i], esc[i]]) for i, k in enumerate(BIN_KEYS)]
    persist_ci = [bins[k].get('persistence_ci', [persist[i], persist[i]]) for i, k in enumerate(BIN_KEYS)]

    return {
        'recovery': np.array(rec),
        'escalation': np.array(esc),
        'persistence': np.array(persist),
        'n': np.array(n_vals),
        'recovery_ci_lo': np.array([c[0] for c in rec_ci]),
        'recovery_ci_hi': np.array([c[1] for c in rec_ci]),
        'escalation_ci_lo': np.array([c[0] for c in esc_ci]),
        'escalation_ci_hi': np.array([c[1] for c in esc_ci]),
        'persistence_ci_lo': np.array([c[0] for c in persist_ci]),
        'persistence_ci_hi': np.array([c[1] for c in persist_ci]),
    }


# ============================================================
# Panel plotting
# ============================================================

def plot_duration_panel(ax, series, phase_num, panel_label, crossover_data=None,
                        show_ylabel=False):
    """
    Plot recovery and escalation lines with CI bands on a single axis.

    - Phase 1: skip recovery (always 0)
    - Phase 5: skip escalation (always 0)
    - Phase 3: annotate crossover point with CI
    - Only plot bins where n > 0
    - Colorblind-safe: recovery = solid + circle, escalation = dashed + triangle
    """
    x_all = np.array(DURATION_MIDPOINTS)

    # Mask: only plot bins with observations
    valid = series['n'] > 0
    x = x_all[valid]

    def _plot_recovery(vals, ci_lo, ci_hi):
        v, lo, hi = vals[valid], ci_lo[valid], ci_hi[valid]
        ax.plot(x, v, '-', color=COL_RECOVERY, linewidth=1.5,
                marker='o', markersize=5, zorder=3)
        ax.fill_between(x, lo, hi, color=COL_RECOVERY, alpha=0.12, zorder=1)

    def _plot_escalation(vals, ci_lo, ci_hi):
        v, lo, hi = vals[valid], ci_lo[valid], ci_hi[valid]
        ax.plot(x, v, '--', color=COL_ESCALATION, linewidth=1.5,
                marker='^', markersize=5, zorder=3)
        ax.fill_between(x, lo, hi, color=COL_ESCALATION, alpha=0.12, zorder=1)

    # --- Recovery ---
    if phase_num > 1:
        _plot_recovery(series['recovery'], series['recovery_ci_lo'],
                       series['recovery_ci_hi'])

    # --- Escalation ---
    if phase_num < 5:
        _plot_escalation(series['escalation'], series['escalation_ci_lo'],
                         series['escalation_ci_hi'])

    # --- Phase 3 crossover annotation with CI ---
    if phase_num == 3 and crossover_data:
        crossover_month = crossover_data.get('crossover', {}).get('month', None)
        if crossover_month:
            crossover_ci_range = crossover_data.get('crossover_ci',
                                                     [crossover_month, crossover_month])
            # Shade last bin region
            ax.axvspan(4.5, 5.5, alpha=0.10, color=COL_ESCALATION, zorder=0)
            # Arrow target: last valid escalation point
            last_esc = series['escalation'][valid][-1]
            last_x = x[-1]
            rec_max = max(series['recovery'][valid])
            ci_lo, ci_hi = int(round(crossover_ci_range[0])), int(round(crossover_ci_range[1]))
            ax.annotate(f'Crossover ~{crossover_month:.0f} mo\n(CI: {ci_lo}\u2013{ci_hi})',
                        xy=(last_x, last_esc),
                        xytext=(3.2, rec_max * 0.55),
                        fontsize=6, ha='center',
                        arrowprops=dict(arrowstyle='->', color='#555', lw=0.8,
                                        connectionstyle='arc3,rad=0.15'),
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='#999', alpha=0.9, linewidth=0.5))

    # --- Sample size annotation (all panels) ---
    total_n = int(series['n'].sum())
    ax.text(0.97, 0.97, f'n = {total_n:,}',
            transform=ax.transAxes, fontsize=6, ha='right', va='top',
            fontstyle='italic', color='#888')

    # --- Formatting ---
    ax.set_xticks(x_all)
    ax.set_xticklabels(DURATION_TICK_LABELS, rotation=0)
    ax.set_xlim(0.5, 5.5)

    # Y-axis: auto-scale to plotted lines only (recovery + escalation CIs)
    plotted = []
    if phase_num > 1:
        plotted.append(series['recovery_ci_hi'][valid])
    if phase_num < 5:
        plotted.append(series['escalation_ci_hi'][valid])
    if not plotted:
        plotted.append(series['recovery_ci_hi'][valid])
    all_vals = np.concatenate(plotted)
    y_max = max(all_vals) * 1.3 if len(all_vals) > 0 and max(all_vals) > 0 else 10
    y_max = min(y_max, 105)  # cap at 105% -- probability cannot exceed 100%
    ax.set_ylim(0, y_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # Nature-style panel title
    ax.set_title(f'$\\bf{{{panel_label}}}$  {PHASE_TITLES[phase_num]}',
                 loc='left', fontsize=8, x=-0.02)

    # Y-axis label only on leftmost panels to avoid repetition
    if show_ylabel:
        ax.set_ylabel('Prob. (%)', fontsize=7.5)


# ============================================================
# Main figure
# ============================================================

def create_figure3():
    """
    Figure 3: Phase persistence and transition dynamics (2x3 grid)
    """
    print("Creating Figure 3: Phase persistence & transitions (6-panel)...")

    # Load transition matrix
    matrix_path = os.path.join(DATA_DIR, 'full_transition_matrix.json')
    with open(matrix_path) as f:
        matrix_data = json.load(f)
    trans_data = np.array(matrix_data['matrix_pct'])

    # Load crossover data for Phase 3
    crossover_data = None
    crossover_path = os.path.join(DATA_DIR, 'phase3_crossover.json')
    if os.path.exists(crossover_path):
        with open(crossover_path) as f:
            crossover_data = json.load(f)

    # Load all phase duration data
    phase_series = {}
    for p in range(1, 6):
        try:
            pdata = load_phase_data(p)
            phase_series[p] = extract_series(pdata)
        except FileNotFoundError:
            print(f"  WARNING: phase{p}_duration_conditioned.json not found, skipping")

    # --- Create 2x3 figure ---
    # 180mm wide ~ 7.09 inches; 6.0 inches tall for breathing room
    fig = plt.figure(figsize=(7.1, 6.3))
    gs = GridSpec(2, 3, hspace=0.45, wspace=0.42,
                  left=0.09, right=0.97, bottom=0.13, top=0.92)

    # =========================================
    # Panel A: Transition Matrix Heatmap
    # =========================================
    ax_a = fig.add_subplot(gs[0, 0])

    colors_cmap = ['#FFFFFF', '#E8F4FC', '#B8D4E8', '#6BAED6', '#2171B5', '#084594']
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors_cmap, N=256)

    im = ax_a.imshow(trans_data, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    phase_nums = [1, 2, 3, 4, 5]
    for i in range(5):
        for j in range(5):
            value = trans_data[i, j]
            text_color = 'white' if value > 50 else 'black'
            if value >= 10:
                text = f'{value:.0f}'
            elif value >= 1:
                text = f'{value:.1f}'
            elif value >= 0.1:
                text = f'{value:.1f}'
            else:
                text = f'{value:.2f}'

            weight = 'bold' if i == j else 'normal'
            ax_a.text(j, i, text, ha='center', va='center',
                      color=text_color, fontsize=7, fontweight=weight)

    ax_a.set_xticks(range(5))
    ax_a.set_xticklabels(phase_nums)
    ax_a.set_yticks(range(5))
    ax_a.set_yticklabels(phase_nums)
    ax_a.set_xlabel('To Phase', fontsize=8)
    ax_a.set_ylabel('From Phase', fontsize=8)

    cbar = plt.colorbar(im, ax=ax_a, fraction=0.030, pad=0.03, shrink=0.8)
    cbar.set_label('Prob. (%)', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax_a.set_title(r'$\bf{a}$  Transition matrix', loc='left', fontsize=8, x=0.0)

    # =========================================
    # Duration panels: b-f (Phases 1-5)
    # =========================================
    panel_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    panel_labels = ['b', 'c', 'd', 'e', 'f']

    for idx, (phase_num, (row, col)) in enumerate(zip(range(1, 6), panel_positions)):
        ax = fig.add_subplot(gs[row, col])
        # Y-label only on panel d (row1,col0)
        is_leftmost = (row == 1 and col == 0)
        if phase_num in phase_series:
            plot_duration_panel(
                ax, phase_series[phase_num], phase_num, panel_labels[idx],
                crossover_data=crossover_data if phase_num == 3 else None,
                show_ylabel=is_leftmost,
            )
        else:
            ax.text(0.5, 0.5, f'Phase {phase_num}\ndata not available',
                    transform=ax.transAxes, ha='center', va='center', fontsize=9)
            ax.set_title(panel_labels[idx], loc='left', fontweight='bold', fontsize=10)

    # =========================================
    # Shared x-axis label + per-row y-axis labels
    # =========================================
    fig.supxlabel('Consecutive months at phase', fontsize=9, y=0.06)

    # =========================================
    # Shared legend (horizontal, below x-label)
    # =========================================
    legend_elements = [
        Line2D([0], [0], color=COL_RECOVERY, linewidth=1.5, linestyle='-',
               marker='o', markersize=5, label='Recovery (any lower phase)'),
        Line2D([0], [0], color=COL_ESCALATION, linewidth=1.5, linestyle='--',
               marker='^', markersize=5, label='Escalation (any higher phase)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.52, 0.005))

    # Save -- 600 DPI for line art (Nature Food standard)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, 'Figure3_phase_dynamics.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'Figure3_phase_dynamics.pdf')
    plt.savefig(png_path, dpi=600, facecolor='white', bbox_inches='tight')
    plt.savefig(pdf_path, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Print verification summary
    print("\nVerification summary:")
    for p in range(1, 6):
        if p in phase_series:
            s = phase_series[p]
            for i, k in enumerate(BIN_KEYS):
                total = s['recovery'][i] + s['persistence'][i] + s['escalation'][i]
                if abs(total - 100) > 5:
                    print(f"  WARNING: Phase {p}, {k}: rec+pers+esc = {total:.1f}% (should be ~100%)")
            first_sum = s['recovery'][0] + s['persistence'][0] + s['escalation'][0]
            last_sum = s['recovery'][-1] + s['persistence'][-1] + s['escalation'][-1]
            total_n = int(s['n'].sum())
            print(f"  Phase {p}: n={total_n:,}, bin1 sum={first_sum:.1f}%, "
                  f"bin5 sum={last_sum:.1f}%")

    return png_path


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Figure 3: Phase Persistence and Transition Dynamics")
    print("=" * 60)

    matrix_path = os.path.join(DATA_DIR, 'full_transition_matrix.json')
    if not os.path.exists(matrix_path):
        print(f"ERROR: Transition matrix not found: {matrix_path}")
        print("Run 01_reference_pipeline.py first.")
        sys.exit(1)

    create_figure3()

    print("=" * 60)
    print("Done!")
    print("=" * 60)
