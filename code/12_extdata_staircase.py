#!/usr/bin/env python3
"""
12_extdata_staircase.py -- Extended Data Figure: Crisis Staircase & Alluvial
=============================================================================

Generates three panels illustrating how locations transition between crisis
archetypes over consecutive episodes:

  1. Simplified staircase diagram — seasonal -> prolonged -> protracted pathway
     with cycling, escalation, recovery, and direct-jump arrows.
  2. Full alluvial (Sankey) diagram — all archetype-to-archetype flows
     (excluding seasonal->seasonal) weighted by transition count.
  3. IPC-ordered alluvial — same alluvial with archetypes sorted by peak IPC
     phase (most severe at top) and IPC bracket indicators.

Algorithm is identical to the source script
``papers/paper1a/figures/Figure_crisis_staircase_alluvial.py``; only the
data-loading paths and output paths have been changed to read from the
reproducibility-package ``outputs/data/`` directory.

Inputs (relative to package root):
    outputs/data/episodes.csv
        Produced by 01_reference_pipeline.py
    outputs/data/archetype_transitions.csv
        Produced by 02_generate_transitions.py

Outputs (relative to package root):
    outputs/figures/ExtData_crisis_staircase.png        (300 dpi)
    outputs/figures/ExtData_crisis_staircase.pdf
    outputs/figures/ExtData_alluvial_transitions.png     (300 dpi)
    outputs/figures/ExtData_alluvial_transitions.pdf
    outputs/figures/ExtData_alluvial_ipc_ordered.png     (300 dpi)
    outputs/figures/ExtData_alluvial_ipc_ordered.pdf

Dependencies: pandas, numpy, matplotlib

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

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

ARCHETYPE_SHORT = {
    'seasonal_crisis': 'Seasonal',
    'prolonged_moderate': 'Prolonged',
    'protracted_emergency': 'Protracted',
    'rapid_onset': 'Rapid',
    'entrenched_moderate': 'Entrenched',
    'oscillating': 'Oscillating',
    'severe_shock': 'Severe',
    'escalating': 'Escalating',
}

ARCHETYPE_FULL = {
    'seasonal_crisis': 'Seasonal crisis',
    'prolonged_moderate': 'Prolonged moderate',
    'protracted_emergency': 'Protracted emergency',
    'rapid_onset': 'Rapid onset',
    'entrenched_moderate': 'Entrenched moderate',
    'oscillating': 'Oscillating',
    'severe_shock': 'Severe shock',
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
# Drawing helpers (unchanged from source)
# ============================================================

def draw_curved_arrow(ax, start, end, width, color, alpha=0.6):
    """Draw a curved flow arrow between two points."""
    mid_x = (start[0] + end[0]) / 2

    verts = [
        (start[0], start[1] + width/2),
        (mid_x, start[1] + width/2),
        (mid_x, end[1] + width/2),
        (end[0], end[1] + width/2),
        (end[0], end[1] - width/2),
        (mid_x, end[1] - width/2),
        (mid_x, start[1] - width/2),
        (start[0], start[1] - width/2),
        (start[0], start[1] + width/2),
    ]

    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]

    path = MPath(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=color, edgecolor='white',
                               linewidth=0.3, alpha=alpha)
    ax.add_patch(patch)


def draw_curved_flow(ax, x1, y1_bottom, height1, x2, y2_bottom, height2, color, alpha=0.6):
    """Draw a curved flow band connecting two bars with proper bezier curves."""
    cx = (x1 + x2) / 2

    top_verts = [
        (x1, y1_bottom + height1),
        (cx, y1_bottom + height1),
        (cx, y2_bottom + height2),
        (x2, y2_bottom + height2),
    ]

    bottom_verts = [
        (x2, y2_bottom),
        (cx, y2_bottom),
        (cx, y1_bottom),
        (x1, y1_bottom),
    ]

    verts = top_verts + bottom_verts + [(x1, y1_bottom + height1)]

    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]

    path = MPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none',
                      linewidth=0, alpha=alpha)
    ax.add_patch(patch)


# ============================================================
# Figure 1: Simplified staircase
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
    filepath_png = os.path.join(FIGURES_DIR, 'ExtData_crisis_staircase.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'ExtData_crisis_staircase.pdf')
    plt.savefig(filepath_png, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(filepath_pdf, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")


# ============================================================
# Figure 2: Full alluvial diagram
# ============================================================

def figure_alluvial_full():
    """Create a full alluvial/Sankey diagram showing all archetype transitions."""
    print("Creating full alluvial diagram...")

    matrix_df, trans_df, loc_df, trans_all = load_transition_data(exclude_seasonal_only=True)

    n_locations = len(loc_df)
    n_transitions = len(trans_df)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    plot_matrix = matrix_df.drop('All', axis=0, errors='ignore').drop('All', axis=1, errors='ignore')

    from_totals = {}
    to_totals = {}
    for arch in plot_matrix.index:
        from_totals[arch] = plot_matrix.loc[arch].sum()
    for arch in plot_matrix.columns:
        to_totals[arch] = plot_matrix[arch].sum()

    all_archetypes = list(set(list(from_totals.keys()) + list(to_totals.keys())))

    archetype_order = [
        'seasonal_crisis', 'prolonged_moderate', 'entrenched_moderate',
        'rapid_onset', 'oscillating', 'escalating', 'severe_shock',
        'protracted_emergency',
    ]
    archetypes = [a for a in archetype_order if a in all_archetypes]

    left_x = 3.5
    right_x = 10.5
    bar_width = 0.4

    plot_height = 8.0
    y_start = 1.5
    gap = 0.3

    def calc_positions(totals_dict, archetypes_list):
        positions = {}
        active = [(a, totals_dict.get(a, 0)) for a in archetypes_list if totals_dict.get(a, 0) > 0]
        if not active:
            return positions
        total = sum(t for _, t in active)
        n_bars = len(active)
        total_gap = gap * (n_bars - 1)
        available_height = plot_height - total_gap
        current_y = y_start
        for arch, count in active:
            height = (count / total) * available_height
            height = max(height, 0.3)
            positions[arch] = {
                'y_bottom': current_y, 'height': height,
                'y_center': current_y + height / 2,
                'y_top': current_y + height, 'count': count
            }
            current_y += height + gap
        return positions

    left_positions = calc_positions(from_totals, archetypes)
    right_positions = calc_positions(to_totals, archetypes)

    left_flow_y = {arch: pos['y_bottom'] for arch, pos in left_positions.items()}
    right_flow_y = {arch: pos['y_bottom'] for arch, pos in right_positions.items()}

    flows = []
    for from_arch in archetypes:
        if from_arch not in plot_matrix.index:
            continue
        for to_arch in archetypes:
            if to_arch not in plot_matrix.columns:
                continue
            count = plot_matrix.loc[from_arch, to_arch]
            if count >= 1:
                flows.append((from_arch, to_arch, count))

    flows.sort(key=lambda x: x[2], reverse=True)

    for from_arch, to_arch, count in flows:
        if from_arch not in left_positions or to_arch not in right_positions:
            continue
        from_total = from_totals[from_arch]
        to_total = to_totals[to_arch]
        from_height = (count / from_total) * left_positions[from_arch]['height']
        to_height = (count / to_total) * right_positions[to_arch]['height']
        flow_height = (from_height + to_height) / 2
        flow_height = max(flow_height, 0.06)
        from_y = left_flow_y[from_arch]
        to_y = right_flow_y[to_arch]
        left_flow_y[from_arch] += from_height
        right_flow_y[to_arch] += to_height
        color = ARCHETYPE_COLORS[from_arch]
        alpha = 0.65
        draw_curved_flow(ax,
                         left_x + bar_width/2, from_y, from_height,
                         right_x - bar_width/2, to_y, to_height,
                         color, alpha)

    for arch in archetypes:
        if arch in left_positions:
            pos = left_positions[arch]
            rect = FancyBboxPatch(
                (left_x - bar_width/2, pos['y_bottom']), bar_width, pos['height'],
                boxstyle="round,pad=0.02",
                facecolor=ARCHETYPE_COLORS[arch], edgecolor='none', linewidth=0
            )
            ax.add_patch(rect)
            ax.text(left_x - bar_width/2 - 0.2, pos['y_center'],
                    ARCHETYPE_FULL[arch], ha='right', va='center', fontsize=9,
                    fontweight='bold')
            ax.text(left_x - bar_width/2 - 0.2, pos['y_center'] - 0.35,
                    f'n={int(pos["count"])}', ha='right', va='center',
                    fontsize=7, color='#555555')

    for arch in archetypes:
        if arch in right_positions:
            pos = right_positions[arch]
            rect = FancyBboxPatch(
                (right_x - bar_width/2, pos['y_bottom']), bar_width, pos['height'],
                boxstyle="round,pad=0.02",
                facecolor=ARCHETYPE_COLORS[arch], edgecolor='none', linewidth=0
            )
            ax.add_patch(rect)
            ax.text(right_x + bar_width/2 + 0.2, pos['y_center'],
                    ARCHETYPE_FULL[arch], ha='left', va='center', fontsize=9,
                    fontweight='bold')
            ax.text(right_x + bar_width/2 + 0.2, pos['y_center'] - 0.35,
                    f'n={int(pos["count"])}', ha='left', va='center',
                    fontsize=7, color='#555555')

    ax.text(left_x, 11.0, 'From (Episode N)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(right_x, 11.0, 'To (Episode N+1)', ha='center', va='center',
            fontsize=11, fontweight='bold')

    ax.text(7, 11.6, 'Crisis Archetype Transitions',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7, 11.15,
            f'{n_transitions} transitions across {n_locations} locations (excludes seasonal\u2192seasonal cycling)',
            ha='center', va='center', fontsize=9, color='#666666', style='italic')

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath_png = os.path.join(FIGURES_DIR, 'ExtData_alluvial_transitions.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'ExtData_alluvial_transitions.pdf')
    plt.savefig(filepath_png, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(filepath_pdf, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")


# ============================================================
# Figure 3: IPC-ordered alluvial
# ============================================================

def figure_alluvial_ipc_ordered():
    """Create alluvial diagram with archetypes ordered by IPC severity."""
    print("Creating IPC-ordered alluvial diagram...")

    matrix_df, trans_df, loc_df, trans_all = load_transition_data(exclude_seasonal_only=True)

    n_locations = len(loc_df)
    n_transitions = len(trans_df)

    ARCHETYPE_IPC = {
        'protracted_emergency': '4-5',
        'severe_shock': '4-5',
        'escalating': '4',
        'rapid_onset': '4',
        'oscillating': '3-4',
        'entrenched_moderate': '3',
        'prolonged_moderate': '3',
        'seasonal_crisis': '3',
    }

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    plot_matrix = matrix_df.drop('All', axis=0, errors='ignore').drop('All', axis=1, errors='ignore')

    from_totals = {}
    to_totals = {}
    for arch in plot_matrix.index:
        from_totals[arch] = plot_matrix.loc[arch].sum()
    for arch in plot_matrix.columns:
        to_totals[arch] = plot_matrix[arch].sum()

    all_archetypes = list(set(list(from_totals.keys()) + list(to_totals.keys())))

    archetype_order_by_ipc = [
        'protracted_emergency', 'severe_shock', 'escalating', 'rapid_onset',
        'oscillating', 'entrenched_moderate', 'prolonged_moderate', 'seasonal_crisis',
    ]
    archetypes = [a for a in archetype_order_by_ipc if a in all_archetypes]

    left_x = 4.5
    right_x = 11.5
    bar_width = 0.4

    plot_height = 8.0
    y_start = 1.5
    gap = 0.3

    def calc_positions(totals_dict, archetypes_list):
        positions = {}
        active = [(a, totals_dict.get(a, 0)) for a in archetypes_list if totals_dict.get(a, 0) > 0]
        if not active:
            return positions
        total = sum(t for _, t in active)
        n_bars = len(active)
        total_gap = gap * (n_bars - 1)
        available_height = plot_height - total_gap
        current_y = y_start
        for arch, count in active:
            height = (count / total) * available_height
            height = max(height, 0.3)
            positions[arch] = {
                'y_bottom': current_y, 'height': height,
                'y_center': current_y + height / 2,
                'y_top': current_y + height, 'count': count
            }
            current_y += height + gap
        return positions

    left_positions = calc_positions(from_totals, archetypes)
    right_positions = calc_positions(to_totals, archetypes)

    left_flow_y = {arch: pos['y_bottom'] for arch, pos in left_positions.items()}
    right_flow_y = {arch: pos['y_bottom'] for arch, pos in right_positions.items()}

    flows = []
    for from_arch in archetypes:
        if from_arch not in plot_matrix.index:
            continue
        for to_arch in archetypes:
            if to_arch not in plot_matrix.columns:
                continue
            count = plot_matrix.loc[from_arch, to_arch]
            if count >= 1:
                flows.append((from_arch, to_arch, count))

    flows.sort(key=lambda x: x[2], reverse=True)

    for from_arch, to_arch, count in flows:
        if from_arch not in left_positions or to_arch not in right_positions:
            continue
        from_total = from_totals[from_arch]
        to_total = to_totals[to_arch]
        from_height = (count / from_total) * left_positions[from_arch]['height']
        to_height = (count / to_total) * right_positions[to_arch]['height']
        flow_height = (from_height + to_height) / 2
        flow_height = max(flow_height, 0.06)
        from_y = left_flow_y[from_arch]
        to_y = right_flow_y[to_arch]
        left_flow_y[from_arch] += from_height
        right_flow_y[to_arch] += to_height
        color = ARCHETYPE_COLORS[from_arch]
        alpha = 0.65
        draw_curved_flow(ax,
                         left_x + bar_width/2, from_y, from_height,
                         right_x - bar_width/2, to_y, to_height,
                         color, alpha)

    # Draw bars (no borders)
    for arch in archetypes:
        if arch in left_positions:
            pos = left_positions[arch]
            rect = FancyBboxPatch(
                (left_x - bar_width/2, pos['y_bottom']), bar_width, pos['height'],
                boxstyle="round,pad=0.02",
                facecolor=ARCHETYPE_COLORS[arch], edgecolor='none', linewidth=0
            )
            ax.add_patch(rect)
            ax.text(left_x - bar_width/2 - 0.2, pos['y_center'],
                    ARCHETYPE_FULL[arch], ha='right', va='center', fontsize=9,
                    fontweight='bold')
            ax.text(left_x - bar_width/2 - 0.2, pos['y_center'] - 0.35,
                    f'n={int(pos["count"])}', ha='right', va='center',
                    fontsize=7, color='#555555')

    for arch in archetypes:
        if arch in right_positions:
            pos = right_positions[arch]
            rect = FancyBboxPatch(
                (right_x - bar_width/2, pos['y_bottom']), bar_width, pos['height'],
                boxstyle="round,pad=0.02",
                facecolor=ARCHETYPE_COLORS[arch], edgecolor='none', linewidth=0
            )
            ax.add_patch(rect)
            ax.text(right_x + bar_width/2 + 0.2, pos['y_center'],
                    ARCHETYPE_FULL[arch], ha='left', va='center', fontsize=9,
                    fontweight='bold')
            ax.text(right_x + bar_width/2 + 0.2, pos['y_center'] - 0.35,
                    f'n={int(pos["count"])}', ha='left', va='center',
                    fontsize=7, color='#555555')

    # IPC Phase brackets
    ipc_x_left = 1.8
    ipc_x_right = 14.2

    phase_groups = {
        'Phase 4-5': ['protracted_emergency', 'severe_shock'],
        'Phase 4': ['escalating', 'rapid_onset'],
        'Phase 3-4': ['oscillating'],
        'Phase 3': ['entrenched_moderate', 'prolonged_moderate', 'seasonal_crisis'],
    }

    phase_colors = {
        'Phase 4-5': '#CC0000',
        'Phase 4': '#EE6677',
        'Phase 3-4': '#EE8866',
        'Phase 3': '#4477AA',
    }

    def draw_ipc_bracket(ax, x, y_bottom, y_top, label, color, side='left'):
        bracket_width = 0.15
        if side == 'left':
            ax.plot([x + bracket_width, x, x, x + bracket_width],
                    [y_top, y_top, y_bottom, y_bottom],
                    color=color, linewidth=2, solid_capstyle='round')
            ax.text(x - 0.1, (y_top + y_bottom) / 2, label,
                    ha='right', va='center', fontsize=8, fontweight='bold',
                    color=color, rotation=90)
        else:
            ax.plot([x - bracket_width, x, x, x - bracket_width],
                    [y_top, y_top, y_bottom, y_bottom],
                    color=color, linewidth=2, solid_capstyle='round')
            ax.text(x + 0.1, (y_top + y_bottom) / 2, label,
                    ha='left', va='center', fontsize=8, fontweight='bold',
                    color=color, rotation=270)

    for phase, archs in phase_groups.items():
        y_positions = []
        for arch in archs:
            if arch in left_positions:
                pos = left_positions[arch]
                y_positions.append(pos['y_bottom'])
                y_positions.append(pos['y_top'])
        if y_positions:
            y_min = min(y_positions)
            y_max = max(y_positions)
            draw_ipc_bracket(ax, ipc_x_left, y_min, y_max, phase,
                             phase_colors[phase], 'left')

    for phase, archs in phase_groups.items():
        y_positions = []
        for arch in archs:
            if arch in right_positions:
                pos = right_positions[arch]
                y_positions.append(pos['y_bottom'])
                y_positions.append(pos['y_top'])
        if y_positions:
            y_min = min(y_positions)
            y_max = max(y_positions)
            draw_ipc_bracket(ax, ipc_x_right, y_min, y_max, phase,
                             phase_colors[phase], 'right')

    all_y_positions = []
    for pos in left_positions.values():
        all_y_positions.extend([pos['y_bottom'], pos['y_top']])
    for pos in right_positions.values():
        all_y_positions.extend([pos['y_bottom'], pos['y_top']])

    y_min = min(all_y_positions) if all_y_positions else 1.5
    y_max = max(all_y_positions) if all_y_positions else 9.5

    ax.text(8, 11.2, 'Crisis Archetype Transitions by IPC Phase',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(8, 10.75,
            f'{n_transitions} transitions across {n_locations} locations (excludes seasonal\u2192seasonal cycling)',
            ha='center', va='center', fontsize=9, color='#666666', style='italic')

    header_y = y_max + 0.3
    ax.text(left_x, header_y, 'From (Episode N)', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.text(right_x, header_y, 'To (Episode N+1)', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

    arrow_x = 0.8
    ax.annotate('', xy=(arrow_x, y_min - 0.1), xytext=(arrow_x, y_max + 0.1),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2))
    ax.text(arrow_x - 0.4, (y_min + y_max) / 2, 'Increasing\nSeverity',
            ha='center', va='center', fontsize=8, color='#666666', rotation=90)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath_png = os.path.join(FIGURES_DIR, 'ExtData_alluvial_ipc_ordered.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'ExtData_alluvial_ipc_ordered.pdf')
    plt.savefig(filepath_png, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(filepath_pdf, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    """Generate all crisis staircase extended data figures."""
    print("=" * 60)
    print("Generating Extended Data: Crisis Staircase & Alluvial Diagrams")
    print("=" * 60)

    figure_staircase_simplified()
    figure_alluvial_full()
    figure_alluvial_ipc_ordered()

    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
