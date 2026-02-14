#!/usr/bin/env python3
"""
09_fig4_recovery.py — Figure 4: Recovery Degradation by Crisis Duration
=========================================================================

Dual-panel figure:
  (a) Phase 4 recovery probability by consecutive months at Phase 4
  (b) Recovery-to-escalation ratio comparison (Phase 3 vs Phase 4)

Inputs  (relative to package root):
    outputs/data/phase3_duration_conditioned.json
    outputs/data/phase4_duration_conditioned.json

Outputs (relative to package root):
    outputs/figures/Figure4_recovery_degradation.png  (300 dpi)
    outputs/figures/Figure4_recovery_degradation.pdf

Author: Richard Choularton
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

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
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


# ============================================================
# Figure generation
# ============================================================

def create_figure4():
    """
    Figure 4: Recovery degradation
    Panel a: Phase 4 recovery probability by duration at Phase 4
    Panel b: Recovery/escalation ratio comparison (Phase 3 vs Phase 4)
    """
    print("Creating Figure 4: Recovery degradation (dual-panel)...")

    # Load Phase 4 duration-conditioned data
    p4_path = os.path.join(DATA_DIR, 'phase4_duration_conditioned.json')
    with open(p4_path) as f:
        p4_data = json.load(f)

    # Load Phase 3 duration-conditioned data
    p3_path = os.path.join(DATA_DIR, 'phase3_duration_conditioned.json')
    with open(p3_path) as f:
        p3_data = json.load(f)

    bin_keys = ['1-3 mo', '4-6 mo', '7-12 mo', '13-24 mo', '24+ mo']
    bin_keys_4 = ['1-3 mo', '4-6 mo', '7-12 mo', '13-24 mo']

    # Phase 4: exclude 24+ bin if sample too small
    p4_bins = p4_data['bins']
    if p4_bins['24+ mo']['n'] < 10:
        duration_labels = ['1\u20133', '4\u20136', '7\u201312', '13\u201324']
        duration_midpoints = [2, 5, 9.5, 18.5]
        active_keys = bin_keys_4
    else:
        duration_labels = ['1\u20133', '4\u20136', '7\u201312', '13\u201324', '24+']
        duration_midpoints = [2, 5, 9.5, 18.5, 30]
        active_keys = bin_keys

    p4_recovery = [p4_bins[k]['recovery_pct'] for k in active_keys]
    p4_recovery_ci_lo = [p4_bins[k].get('recovery_ci', [0, 0])[0] for k in active_keys]
    p4_recovery_ci_hi = [p4_bins[k].get('recovery_ci', [0, 0])[1] for k in active_keys]
    p4_n = [p4_bins[k]['n'] for k in active_keys]
    p4_escalation = [p4_bins[k]['escalation_pct'] for k in active_keys]

    # Phase 3: all 5 bins
    p3_bins = p3_data['bins']
    p3_recovery = [p3_bins[k]['recovery_pct'] for k in bin_keys]
    p3_escalation = [p3_bins[k]['escalation_pct'] for k in bin_keys]
    p3_ratio = [r / e if e > 0 else float('inf') for r, e in zip(p3_recovery, p3_escalation)]

    p4_ratio = [r / e if e > 0 else float('inf') for r, e in zip(p4_recovery, p4_escalation)]

    duration_labels_5 = ['1\u20133', '4\u20136', '7\u201312', '13\u201324', '24+']
    duration_midpoints_5 = [2, 5, 9.5, 18.5, 30]

    # Create figure
    fig = plt.figure(figsize=(10, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    # =========================================
    # Panel A: Phase 4 Recovery Degradation
    # =========================================
    ax1 = fig.add_subplot(gs[0])

    # Plot with confidence intervals
    ax1.fill_between(duration_midpoints, p4_recovery_ci_lo, p4_recovery_ci_hi,
                     alpha=0.2, color='#228833')
    ax1.plot(duration_midpoints, p4_recovery, 'o-', color='#228833', linewidth=2,
             markersize=8, label='P(4 \u2192 3)', zorder=3)

    # Add sample sizes
    for x, y, n in zip(duration_midpoints, p4_recovery, p4_n):
        ax1.text(x, y + 1.8, f'n={n}', ha='center', va='bottom',
                fontsize=7, color='#666666')

    # Add value labels
    for x, y in zip(duration_midpoints, p4_recovery):
        ax1.text(x, y - 2.0, f'{y:.1f}%', ha='center', va='top',
                fontsize=8, fontweight='bold', color='#228833')

    # Add decay annotation
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor='#228833', linewidth=1)
    ax1.text(0.38, 0.92,
             f'Decline: {p4_recovery[0]:.0f}% \u2192 {p4_recovery[-1]:.0f}%\nover {duration_labels[-1]} months',
             ha='center', va='top', fontsize=8, color='#228833',
             bbox=bbox_props, transform=ax1.transAxes)

    # Exponential decay fit line from JSON
    decay_fit = p4_data.get('decay_fit', {})
    if decay_fit:
        a_fit = decay_fit['a']
        b_fit = decay_fit['b']
        r2_fit = decay_fit['r_squared']
        x_fit = np.linspace(1, max(duration_midpoints), 100)
        y_fit = a_fit * np.exp(-b_fit * x_fit)
        ax1.plot(x_fit, y_fit, '--', color='#228833', alpha=0.4, linewidth=1,
                 label=f'Exponential fit (R\u00b2={r2_fit:.2f})')

    ax1.set_xlabel('Consecutive months at Phase 4', fontsize=10)
    ax1.set_ylabel('Monthly recovery probability (%)', fontsize=10)
    ax1.set_xticks(duration_midpoints)
    ax1.set_xticklabels(duration_labels)
    ax1.set_ylim(0, max(p4_recovery_ci_hi) * 1.3)
    ax1.set_xlim(0, max(duration_midpoints) + 2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')

    ax1.set_title('a', loc='left', fontweight='bold', fontsize=13, x=-0.12)
    ax1.text(0.5, -0.18, 'Phase 4 recovery declines\ngradually with duration',
             ha='center', va='top', fontsize=9, style='italic', color='#555555',
             transform=ax1.transAxes)

    # =========================================
    # Panel B: Recovery/Escalation Ratio Comparison
    # =========================================
    ax2 = fig.add_subplot(gs[1])

    # Phase 3 ratio (capped at 10 for display)
    p3_ratio_display = [min(r, 12) for r in p3_ratio]

    ax2.plot(duration_midpoints_5, p3_ratio_display, 'o-', color='#4477AA', linewidth=2,
             markersize=7, label='Phase 3 (recovery/escalation)', zorder=3)

    # Add ratio labels for Phase 3
    for i, (x, r) in enumerate(zip(duration_midpoints_5, p3_ratio)):
        if r < 100:
            y_pos = min(r, 12) + 0.4
            color = '#CC3333' if r < 1 else '#4477AA'
            ax2.text(x, y_pos, f'{r:.1f}:1', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color=color)

    # Horizontal line at ratio = 1 (escalation = recovery)
    ax2.axhline(y=1, color='#CC3333', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(32, 1.3, 'Parity', fontsize=7, color='#CC3333', fontstyle='italic')

    # Shade region where escalation dominates
    ax2.axhspan(0, 1, alpha=0.05, color='#EE6677', zorder=0)
    ax2.text(30, 0.3, 'Escalation\ndominates', ha='center', fontsize=7,
             color='#CC3333', fontstyle='italic', alpha=0.7)

    # Add annotation about Phase 4
    bbox_props2 = dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                       edgecolor='#228833', linewidth=1)
    ax2.text(0.35, 0.92, 'Phase 4: ratio >100:1\nat all durations\n(no crossover)',
             ha='center', va='top', fontsize=8, color='#228833',
             bbox=bbox_props2, transform=ax2.transAxes)

    # Mark crossover using JSON data
    crossover_info = p3_data.get('crossover', {})
    crossover_ci = p3_data.get('crossover_ci', [])
    crossover_month = crossover_info.get('month', 28)
    ci_str = f"\n(95% CI: {crossover_ci[0]:.0f}\u2013{crossover_ci[1]:.0f})" if crossover_ci else ""
    ax2.annotate(f'Crossover ~{crossover_month:.0f} months{ci_str}',
                 xy=(30, p3_ratio[-1]), xytext=(15, 3.5),
                 fontsize=8, ha='center',
                 arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#999999', alpha=0.9))

    ax2.set_xlabel('Consecutive months at target phase', fontsize=10)
    ax2.set_ylabel('Recovery-to-escalation ratio', fontsize=10)
    ax2.set_xticks(duration_midpoints_5)
    ax2.set_xticklabels(duration_labels_5)
    ax2.set_ylim(0, 8)
    ax2.set_xlim(0, 34)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')

    ax2.set_title('b', loc='left', fontweight='bold', fontsize=13, x=-0.12)
    ax2.text(0.5, -0.18, f'Phase 3 ratio inverts after ~{crossover_month:.0f} months;\nPhase 4 ratio remains strongly asymmetric',
             ha='center', va='top', fontsize=9, style='italic', color='#555555',
             transform=ax2.transAxes)

    # Layout
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.93, wspace=0.35)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, 'Figure4_recovery_degradation.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'Figure4_recovery_degradation.pdf')
    plt.savefig(png_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(pdf_path, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    return png_path


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Figure 4: Recovery Degradation")
    print("=" * 60)

    p4_path = os.path.join(DATA_DIR, 'phase4_duration_conditioned.json')
    p3_path = os.path.join(DATA_DIR, 'phase3_duration_conditioned.json')
    for path, label in [(p4_path, 'phase4_duration_conditioned.json'),
                         (p3_path, 'phase3_duration_conditioned.json')]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            print("Run 01_reference_pipeline.py first.")
            sys.exit(1)

    create_figure4()

    print("=" * 60)
    print("Done!")
    print("=" * 60)
