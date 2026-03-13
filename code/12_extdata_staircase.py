#!/usr/bin/env python3
"""
12_extdata_staircase.py -- Extended Data Figure 1: Crisis Staircase
=====================================================================

Generates the simplified staircase diagram (Extended Data Fig 1) showing
the main escalation pathway: seasonal -> prolonged -> protracted, with
cycling, escalation, recovery, and direct-jump arrows.

Inputs (relative to package root):
    outputs/data/episodes.csv
        Produced by 01_reference_pipeline.py
    outputs/data/archetype_transitions.csv
        Produced by 02_generate_transitions.py

Outputs (relative to package root):
    outputs/figures/Figure_crisis_staircase.png        (300 dpi)
    outputs/figures/Figure_crisis_staircase.pdf
    outputs/figures/SourceData_EDFig1.xlsx

Dependencies: pandas, numpy, matplotlib, openpyxl

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================================================
# Paths -- relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
TRANSITIONS_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'archetype_transitions.csv')
FIGURES_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ============================================================
# Nature Food style rcParams
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})

# ============================================================
# Colorblind-friendly palette matching existing figures
# ============================================================
ARCHETYPE_COLORS = {
    'seasonal_crisis': '#4477AA',
    'prolonged_moderate': '#66CCEE',
    'protracted_emergency': '#EE6677',
    'rapid_onset': '#228833',
    'entrenched_moderate': '#CCBB44',
    'oscillating': '#AA3377',
    'severe_shock': '#BBBBBB',
    'escalating': '#EE8866',
}

ARCHETYPE_LABELS = {
    'seasonal_crisis': 'Seasonal\ncrisis',
    'prolonged_moderate': 'Prolonged\nmoderate',
    'protracted_emergency': 'Protracted\nemergency',
    'rapid_onset': 'Rapid\nonset',
    'entrenched_moderate': 'Entrenched\nmoderate',
    'oscillating': 'Oscillating',
    'severe_shock': 'Severe\nshock',
    'escalating': 'Escalating',
}


# ============================================================
# Data loading
# ============================================================

def build_location_summaries(episodes_df, trans_df):
    """Build location summaries with archetype_sequence and unique_archetypes.

    The original source script reads these columns from a pre-built
    location_summaries.csv.  Here we derive them on the fly from the
    episodes and transitions CSVs that the reproducibility pipeline
    produces.
    """
    # Parse start dates for ordering
    episodes = episodes_df.copy()
    dates_first = episodes['dates'].str.split(',').str[0].str.strip()
    episodes['start_dt'] = pd.to_datetime(dates_first)
    episodes = episodes.sort_values(['location', 'start_dt'])

    records = []
    for location, grp in episodes.groupby('location'):
        archetypes = grp['archetype'].tolist()
        seq_str = ' \u2192 '.join(archetypes)
        unique_count = len(set(archetypes))
        records.append({
            'location': location,
            'archetype_sequence': seq_str,
            'unique_archetypes': unique_count,
            'total_episodes': len(grp),
        })

    loc_df = pd.DataFrame(records)
    return loc_df


def load_transition_data(exclude_seasonal_only=True):
    """Load transition data from CSV files.

    Args:
        exclude_seasonal_only: If True, exclude locations that only have
                               seasonal_crisis -> seasonal_crisis transitions
    """
    trans_df = pd.read_csv(TRANSITIONS_PATH)
    episodes_df = pd.read_csv(EPISODES_PATH)

    loc_df = build_location_summaries(episodes_df, trans_df)

    if exclude_seasonal_only:
        # Filter to locations with more than one unique archetype
        loc_df = loc_df[loc_df['unique_archetypes'] > 1]

        # Get list of locations with archetype diversity
        diverse_locations = set(loc_df['location'].tolist())

        # Filter transitions to only include these locations
        trans_df = trans_df[trans_df['location'].isin(diverse_locations)]

        # Also exclude seasonal -> seasonal transitions for cleaner visualization
        trans_df_filtered = trans_df[
            ~((trans_df['from_archetype'] == 'seasonal_crisis') &
              (trans_df['to_archetype'] == 'seasonal_crisis'))
        ]
    else:
        trans_df_filtered = trans_df

    # Build transition matrix from filtered data
    matrix_df = pd.crosstab(
        trans_df_filtered['from_archetype'],
        trans_df_filtered['to_archetype'],
        margins=True
    )

    # Also keep the full transitions for reference
    trans_df_all = trans_df

    return matrix_df, trans_df_filtered, loc_df, trans_df_all


# ============================================================
# Figure: Simplified staircase
# ============================================================

def figure_staircase_simplified():
    """Create a simplified staircase diagram showing the main escalation pathway."""
    print("Creating simplified crisis staircase diagram...")

    matrix_df, trans_df, loc_df, trans_all = load_transition_data(exclude_seasonal_only=True)

    n_locations = len(loc_df)
    n_transitions = len(trans_df)

    def get_count(from_arch, to_arch):
        try:
            return int(matrix_df.loc[from_arch, to_arch])
        except (KeyError, ValueError):
            return 0

    ss_to_pm = get_count('seasonal_crisis', 'prolonged_moderate')
    pm_to_pe = get_count('prolonged_moderate', 'protracted_emergency')
    ss_to_pe = get_count('seasonal_crisis', 'protracted_emergency')
    pm_to_ss = get_count('prolonged_moderate', 'seasonal_crisis')
    pe_to_ss = get_count('protracted_emergency', 'seasonal_crisis')

    other_from_seasonal = (
        get_count('seasonal_crisis', 'rapid_onset') +
        get_count('seasonal_crisis', 'oscillating') +
        get_count('seasonal_crisis', 'entrenched_moderate') +
        get_count('seasonal_crisis', 'severe_shock') +
        get_count('seasonal_crisis', 'escalating')
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    col_x = [1.5, 4.5, 7.5, 11.0]
    box_width = 2.0
    box_height = 1.4

    stages = [
        {'name': 'seasonal_crisis', 'label': 'Seasonal\nCrisis', 'y': 4,
         'desc': 'Entry point\nPredictable, cyclical'},
        {'name': 'seasonal_crisis', 'label': 'Seasonal\nCrisis', 'y': 4,
         'desc': 'Cycling\n(most common)'},
        {'name': 'prolonged_moderate', 'label': 'Prolonged\nModerate', 'y': 4,
         'desc': 'Intermediate stage\n12-36 months'},
        {'name': 'protracted_emergency', 'label': 'Protracted\nEmergency', 'y': 4,
         'desc': 'Severe state\n>36 months, Phase 4+'},
    ]

    for i, stage in enumerate(stages):
        x = col_x[i] - box_width/2
        y = stage['y'] - box_height/2

        box = FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=ARCHETYPE_COLORS[stage['name']],
            edgecolor='none', linewidth=0, alpha=0.9
        )
        ax.add_patch(box)

        ax.text(col_x[i], stage['y'] + 0.1, stage['label'],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if stage['name'] == 'protracted_emergency' else 'black')

        ax.text(col_x[i], stage['y'] - box_height/2 - 0.25, stage['desc'],
                ha='center', va='top', fontsize=7, color='#666666',
                linespacing=1.2)

    pe_to_pm = get_count('protracted_emergency', 'prolonged_moderate')

    def arrow_width(count):
        return max(1.5, min(3.5, 1.5 + (count / 116) * 2))

    y_center = 4
    y_escalation = y_center + 0.25
    y_recovery = y_center - 0.25

    # Cycling arrow between seasonal spike boxes
    ss_to_ss_count = len(trans_all[(trans_all['from_archetype'] == 'seasonal_crisis') &
                                   (trans_all['to_archetype'] == 'seasonal_crisis')])

    cycle_x = (col_x[0] + col_x[1]) / 2
    arrow_y = y_center
    arrow_left = col_x[0] + box_width/2 + 0.15
    arrow_right = col_x[1] - box_width/2 - 0.15

    ax.annotate('', xy=(arrow_right, arrow_y + 0.12),
                xytext=(arrow_left, arrow_y + 0.12),
                arrowprops=dict(arrowstyle='->', mutation_scale=12,
                               lw=2, color='#4477AA'))
    ax.annotate('', xy=(arrow_left, arrow_y - 0.12),
                xytext=(arrow_right, arrow_y - 0.12),
                arrowprops=dict(arrowstyle='->', mutation_scale=12,
                               lw=2, color='#4477AA'))

    ax.text(cycle_x, arrow_y + 0.5, f'{ss_to_ss_count}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#4477AA')
    ax.text(cycle_x, arrow_y - 0.55, 'Cycling',
            ha='center', va='top', fontsize=9, color='#4477AA')

    # Seasonal -> Prolonged escalation
    ax.annotate('', xy=(col_x[2] - box_width/2 - 0.1, y_escalation + 0.15),
                xytext=(col_x[1] + box_width/2 + 0.1, y_escalation + 0.15),
                arrowprops=dict(arrowstyle='->', mutation_scale=14,
                               lw=arrow_width(ss_to_pm), color='#CC0000',
                               connectionstyle='arc3,rad=0'))
    ax.text((col_x[1] + col_x[2])/2, y_escalation + 0.6, f'{ss_to_pm}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#CC0000')

    # Prolonged -> Seasonal recovery
    ax.annotate('', xy=(col_x[1] + box_width/2 + 0.1, y_recovery - 0.15),
                xytext=(col_x[2] - box_width/2 - 0.1, y_recovery - 0.15),
                arrowprops=dict(arrowstyle='->', mutation_scale=12,
                               lw=arrow_width(pm_to_ss), linestyle='--',
                               color='#228833'))
    ax.text((col_x[1] + col_x[2])/2, y_recovery - 0.6, f'{pm_to_ss}',
            ha='center', va='top', fontsize=10, fontweight='bold', color='#228833')

    # Prolonged -> Protracted escalation
    ax.annotate('', xy=(col_x[3] - box_width/2 - 0.1, y_escalation + 0.15),
                xytext=(col_x[2] + box_width/2 + 0.1, y_escalation + 0.15),
                arrowprops=dict(arrowstyle='->', mutation_scale=14,
                               lw=arrow_width(pm_to_pe), color='#CC0000',
                               connectionstyle='arc3,rad=0'))
    ax.text((col_x[2] + col_x[3])/2, y_escalation + 0.6, f'{pm_to_pe}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#CC0000')

    # Protracted -> Prolonged recovery
    ax.annotate('', xy=(col_x[2] + box_width/2 + 0.1, y_recovery - 0.15),
                xytext=(col_x[3] - box_width/2 - 0.1, y_recovery - 0.15),
                arrowprops=dict(arrowstyle='->', mutation_scale=12,
                               lw=arrow_width(pe_to_pm), linestyle='--',
                               color='#228833'))
    ax.text((col_x[2] + col_x[3])/2, y_recovery - 0.6, f'{pe_to_pm}',
            ha='center', va='top', fontsize=10, fontweight='bold', color='#228833')

    # Direct jump: Seasonal -> Protracted
    ax.annotate('', xy=(col_x[3] - box_width/2, 5.3),
                xytext=(col_x[1] + box_width/2, 5.3),
                arrowprops=dict(arrowstyle='->', mutation_scale=12,
                               lw=2, linestyle='--', color='#EE8866',
                               connectionstyle='arc3,rad=-0.2'))
    ax.text((col_x[1] + col_x[3])/2, 6.3, f'Direct jump: {ss_to_pe}',
            ha='center', va='bottom', fontsize=9, color='#EE8866')

    # Rare recovery: Protracted -> Seasonal
    if pe_to_ss > 0:
        ax.annotate('', xy=(col_x[1] + box_width/2, 2.7),
                    xytext=(col_x[3] - box_width/2, 2.7),
                    arrowprops=dict(arrowstyle='->', mutation_scale=12,
                                   lw=1.5, linestyle='--',
                                   color='#228833', connectionstyle='arc3,rad=-0.2'))
        ax.text((col_x[1] + col_x[3])/2, 1.7, f'Rare recovery: {pe_to_ss}',
                ha='center', va='top', fontsize=9, color='#228833')

    # Title and annotations
    ax.text(6.5, 8.0, 'The Crisis Staircase: Escalation Pathways',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(6.5, 7.5,
            f'Transitions in locations with archetype changes (n={n_locations} locations, {n_transitions} transitions)',
            ha='center', va='center', fontsize=9, color='#666666')

    # Legend
    legend_x = 0.3
    legend_y = 1.3
    ax.plot([legend_x, legend_x + 0.5], [legend_y, legend_y], '-', color='#4477AA', lw=2.5)
    ax.text(legend_x + 0.6, legend_y, 'Cycling', fontsize=8, va='center')
    ax.plot([legend_x, legend_x + 0.5], [legend_y - 0.4, legend_y - 0.4], '-', color='#CC0000', lw=2.5)
    ax.text(legend_x + 0.6, legend_y - 0.4, 'Escalation', fontsize=8, va='center')
    ax.plot([legend_x, legend_x + 0.5], [legend_y - 0.8, legend_y - 0.8], '--', color='#228833', lw=2)
    ax.text(legend_x + 0.6, legend_y - 0.8, 'Recovery', fontsize=8, va='center')
    ax.plot([legend_x, legend_x + 0.5], [legend_y - 1.2, legend_y - 1.2], '--', color='#EE8866', lw=2)
    ax.text(legend_x + 0.6, legend_y - 1.2, 'Direct jump (rare)', fontsize=8, va='center')

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath_png = os.path.join(FIGURES_DIR, 'Figure_crisis_staircase.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'Figure_crisis_staircase.pdf')
    plt.savefig(filepath_png, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(filepath_pdf, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")


# ============================================================
# Main
# ============================================================

def export_source_data():
    """Export source data for Extended Data Figure 1 to Excel."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    xlsx_path = os.path.join(FIGURES_DIR, 'SourceData_EDFig1.xlsx')

    matrix_df, trans_df, loc_df, trans_all = load_transition_data(exclude_seasonal_only=True)
    plot_matrix = matrix_df.drop('All', axis=0, errors='ignore').drop('All', axis=1, errors='ignore')

    # Staircase pathway counts
    pathway_rows = []
    for from_arch in plot_matrix.index:
        for to_arch in plot_matrix.columns:
            count = plot_matrix.loc[from_arch, to_arch]
            if count > 0:
                pathway_rows.append({
                    'from_archetype': from_arch,
                    'to_archetype': to_arch,
                    'n_transitions': int(count),
                })
    pathway_df = pd.DataFrame(pathway_rows).sort_values('n_transitions', ascending=False)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        pathway_df.to_excel(writer, sheet_name='transition_counts', index=False)
        plot_matrix.to_excel(writer, sheet_name='transition_matrix')
    print(f"Saved: {xlsx_path}")


def main():
    """Generate Extended Data Figure 1: Crisis Staircase."""
    print("=" * 60)
    print("Generating Extended Data Figure 1: Crisis Staircase")
    print("=" * 60)

    figure_staircase_simplified()
    export_source_data()

    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
