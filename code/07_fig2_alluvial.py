#!/usr/bin/env python3
"""
07_fig2_alluvial.py — Figure 2: Annual Archetype Evolution (2016 Cohort)
=========================================================================

Two-panel alluvial diagram:
  Panel (a) — Archetype transitions ordered by IPC severity (Episode N -> N+1)
  Panel (b) — Annual archetype evolution for the 2016 cohort (2011-2023)

Inputs  (relative to package root):
    outputs/data/archetype_transitions.csv   (from 02_generate_transitions.py)
    outputs/data/episodes.csv                (from 01_reference_pipeline.py)
    outputs/data/location_summaries.csv      (from 02_generate_transitions.py)
    data/HFID_hv1.csv                        (raw HFID for 2016 cohort)

Outputs (relative to package root):
    outputs/figures/Figure2_combined_alluvial.png  (600 dpi)
    outputs/figures/Figure2_combined_alluvial.pdf

Author: Richard Choularton
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

# ============================================================
# Paths
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
TRANSITIONS_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'archetype_transitions.csv')
LOCATION_SUMMARIES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'location_summaries.csv')
HFID_PATH = os.path.join(PACKAGE_ROOT, 'data', 'HFID_hv1.csv')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ============================================================
# Nature Food styling
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})

# ============================================================
# Archetype definitions
# ============================================================

# IPC severity order: highest at TOP (protracted) -> lowest at BOTTOM (seasonal)
ARCHETYPE_ORDER_IPC = [
    'protracted_emergency',
    'severe_shock',
    'escalating',
    'rapid_onset',
    'oscillating',
    'entrenched_moderate',
    'prolonged_moderate',
    'seasonal_crisis',
]

# For panel (b) - mild at top, severe at bottom (stacked bars)
ARCHETYPE_ORDER_BARS = [
    'seasonal_crisis',
    'prolonged_moderate',
    'entrenched_moderate',
    'oscillating',
    'rapid_onset',
    'severe_shock',
    'escalating',
    'protracted_emergency',
]

ARCHETYPE_LABELS_FULL = {
    'seasonal_crisis': 'Seasonal crisis',
    'prolonged_moderate': 'Prolonged moderate',
    'protracted_emergency': 'Protracted emergency',
    'rapid_onset': 'Rapid onset',
    'entrenched_moderate': 'Entrenched moderate',
    'oscillating': 'Oscillating',
    'severe_shock': 'Severe shock',
    'escalating': 'Escalating',
}

# Shared colour palette (Tol Bright -- colorblind-safe)
ARCHETYPE_COLORS = {
    'seasonal_crisis':        '#228833',
    'prolonged_moderate':    '#CCBB44',
    'entrenched_moderate':   '#EE6677',
    'oscillating':           '#AA3377',
    'rapid_onset':           '#EE8866',
    'severe_shock':          '#CC3311',
    'escalating':            '#004488',
    'protracted_emergency':  '#332288',
}

SEVERITY_RANK = {a: i for i, a in enumerate(ARCHETYPE_ORDER_BARS)}

# IPC phase groupings for brackets in panel (a)
PHASE_GROUPS = {
    'Phase 4-5': ['protracted_emergency', 'severe_shock'],
    'Phase 4':   ['escalating', 'rapid_onset'],
    'Phase 3-4': ['oscillating'],
    'Phase 3':   ['entrenched_moderate', 'prolonged_moderate', 'seasonal_crisis'],
}

PHASE_COLORS = {
    'Phase 4-5': '#CC0000',
    'Phase 4':   '#EE6677',
    'Phase 3-4': '#EE8866',
    'Phase 3':   '#4477AA',
}


# =====================================================================
# Panel (a) - IPC-ordered archetype transitions (Episode N -> N+1)
# =====================================================================

def load_transition_data():
    """Load transition data, excluding seasonal-only locations and seasonal->seasonal."""
    trans_df = pd.read_csv(TRANSITIONS_PATH)

    # Load location summaries
    loc_df = pd.read_csv(LOCATION_SUMMARIES_PATH)

    # Compute unique archetypes per location from episodes (the reproducibility
    # package's location_summaries.csv may not have unique_archetypes column)
    if 'unique_archetypes' not in loc_df.columns:
        episodes = pd.read_csv(EPISODES_PATH)
        ua = episodes.groupby('location')['archetype'].nunique().reset_index()
        ua.columns = ['location', 'unique_archetypes']
        loc_df = loc_df.merge(ua, on='location', how='left')
        loc_df['unique_archetypes'] = loc_df['unique_archetypes'].fillna(0).astype(int)

    # Keep only locations with >1 unique archetype
    loc_df = loc_df[loc_df['unique_archetypes'] > 1]
    diverse_locations = set(loc_df['location'].tolist())
    trans_df = trans_df[trans_df['location'].isin(diverse_locations)]

    # Exclude seasonal->seasonal for cleaner viz
    trans_filtered = trans_df[
        ~((trans_df['from_archetype'] == 'seasonal_crisis') &
          (trans_df['to_archetype'] == 'seasonal_crisis'))
    ]

    # Build transition matrix
    matrix_df = pd.crosstab(
        trans_filtered['from_archetype'],
        trans_filtered['to_archetype'],
        margins=True,
    )

    n_locations = len(loc_df)
    n_transitions = len(trans_filtered)
    return matrix_df, n_locations, n_transitions


def draw_curved_flow(ax, x1, y1_bottom, height1, x2, y2_bottom, height2, color, alpha=0.6):
    """Draw a curved flow band between two bars."""
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


def draw_panel_a(ax):
    """Draw IPC-ordered alluvial transition diagram on the given axes."""
    matrix_df, n_locations, n_transitions = load_transition_data()

    plot_matrix = matrix_df.drop('All', axis=0, errors='ignore').drop('All', axis=1, errors='ignore')

    from_totals = {arch: plot_matrix.loc[arch].sum() for arch in plot_matrix.index}
    to_totals = {arch: plot_matrix[arch].sum() for arch in plot_matrix.columns}

    all_archetypes = list(set(list(from_totals.keys()) + list(to_totals.keys())))
    archetypes = [a for a in ARCHETYPE_ORDER_IPC if a in all_archetypes]

    # Layout
    left_x = 3.5
    right_x = 9.0
    bar_width = 0.30

    plot_height = 6.5
    y_start = 1.0
    gap = 0.22

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
            height = max(height, 0.25)
            positions[arch] = {
                'y_bottom': current_y,
                'height': height,
                'y_center': current_y + height / 2,
                'y_top': current_y + height,
                'count': count,
            }
            current_y += height + gap
        return positions

    left_positions = calc_positions(from_totals, archetypes)
    right_positions = calc_positions(to_totals, archetypes)

    # Track flow stacking
    left_flow_y = {arch: pos['y_bottom'] for arch, pos in left_positions.items()}
    right_flow_y = {arch: pos['y_bottom'] for arch, pos in right_positions.items()}

    # Collect flows
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

    # Draw flows
    for from_arch, to_arch, count in flows:
        if from_arch not in left_positions or to_arch not in right_positions:
            continue
        from_total = from_totals[from_arch]
        to_total = to_totals[to_arch]
        from_height = (count / from_total) * left_positions[from_arch]['height']
        to_height = (count / to_total) * right_positions[to_arch]['height']
        from_y = left_flow_y[from_arch]
        to_y = right_flow_y[to_arch]
        left_flow_y[from_arch] += from_height
        right_flow_y[to_arch] += to_height
        draw_curved_flow(ax,
                         left_x + bar_width / 2, from_y, from_height,
                         right_x - bar_width / 2, to_y, to_height,
                         ARCHETYPE_COLORS[from_arch], 0.60)

    # Draw bars
    for arch in archetypes:
        if arch in left_positions:
            pos = left_positions[arch]
            rect = FancyBboxPatch(
                (left_x - bar_width / 2, pos['y_bottom']), bar_width, pos['height'],
                boxstyle="round,pad=0.02",
                facecolor=ARCHETYPE_COLORS[arch], edgecolor='none', linewidth=0,
            )
            ax.add_patch(rect)
            ax.text(left_x - bar_width / 2 - 0.15, pos['y_center'],
                    ARCHETYPE_LABELS_FULL[arch], ha='right', va='center',
                    fontsize=6.5, fontweight='bold')
            ax.text(left_x - bar_width / 2 - 0.15, pos['y_center'] - 0.28,
                    f'n={int(pos["count"])}', ha='right', va='center',
                    fontsize=5.5, color='#555555')

    for arch in archetypes:
        if arch in right_positions:
            pos = right_positions[arch]
            rect = FancyBboxPatch(
                (right_x - bar_width / 2, pos['y_bottom']), bar_width, pos['height'],
                boxstyle="round,pad=0.02",
                facecolor=ARCHETYPE_COLORS[arch], edgecolor='none', linewidth=0,
            )
            ax.add_patch(rect)
            ax.text(right_x + bar_width / 2 + 0.15, pos['y_center'],
                    ARCHETYPE_LABELS_FULL[arch], ha='left', va='center',
                    fontsize=6.5, fontweight='bold')
            ax.text(right_x + bar_width / 2 + 0.15, pos['y_center'] - 0.28,
                    f'n={int(pos["count"])}', ha='left', va='center',
                    fontsize=5.5, color='#555555')

    # IPC phase brackets
    ipc_x_left = 1.3
    ipc_x_right = 11.2

    def draw_ipc_bracket(ax, x, y_bottom, y_top, label, color, side='left'):
        bracket_width = 0.12
        if side == 'left':
            ax.plot([x + bracket_width, x, x, x + bracket_width],
                    [y_top, y_top, y_bottom, y_bottom],
                    color=color, linewidth=1.5, solid_capstyle='round')
            ax.text(x - 0.05, (y_top + y_bottom) / 2, label,
                    ha='right', va='center', fontsize=6, fontweight='bold',
                    color=color, rotation=90)
        else:
            ax.plot([x - bracket_width, x, x, x - bracket_width],
                    [y_top, y_top, y_bottom, y_bottom],
                    color=color, linewidth=1.5, solid_capstyle='round')
            ax.text(x + 0.05, (y_top + y_bottom) / 2, label,
                    ha='left', va='center', fontsize=6, fontweight='bold',
                    color=color, rotation=270)

    for phase, archs in PHASE_GROUPS.items():
        for side, positions, x_pos in [('left', left_positions, ipc_x_left),
                                        ('right', right_positions, ipc_x_right)]:
            y_positions = []
            for arch in archs:
                if arch in positions:
                    pos = positions[arch]
                    y_positions.extend([pos['y_bottom'], pos['y_top']])
            if y_positions:
                draw_ipc_bracket(ax, x_pos, min(y_positions), max(y_positions),
                                 phase, PHASE_COLORS[phase], side)

    # Column headers
    all_y_tops = [pos['y_top'] for pos in list(left_positions.values()) + list(right_positions.values())]
    y_max = max(all_y_tops) if all_y_tops else 8
    header_y = y_max + 0.25
    ax.text(left_x, header_y, 'From (Episode N)', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')
    ax.text(right_x, header_y, 'To (Episode N+1)', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')

    # Severity arrow
    all_y_bots = [pos['y_bottom'] for pos in list(left_positions.values()) + list(right_positions.values())]
    y_min = min(all_y_bots) if all_y_bots else 1
    arrow_x = 0.4
    ax.annotate('', xy=(arrow_x, y_min - 0.1), xytext=(arrow_x, y_max + 0.1),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))
    ax.text(arrow_x - 0.2, (y_min + y_max) / 2, 'Increasing\nSeverity',
            ha='center', va='center', fontsize=5.5, color='#666666', rotation=90)

    # Subtitle
    ax.text(6.25, header_y + 0.55,
            f'{n_transitions} transitions across {n_locations} locations '
            f'(excludes seasonal\u2192seasonal cycling)',
            ha='center', va='bottom', fontsize=5.5, color='#666666', style='italic')

    # Axis limits
    ax.set_xlim(-0.1, 12.0)
    ax.set_ylim(y_min - 0.5, header_y + 0.9)
    ax.axis('off')


# =====================================================================
# Panel (b) - Annual archetype evolution (2016 cohort)
# =====================================================================

def ym_to_float(ym_str):
    parts = ym_str.split('-')
    return int(parts[0]) + (int(parts[1]) - 1) / 12.0


def get_2016_cohort():
    """Return set of locations monitored in 2016, reading from raw HFID CSV."""
    print(f"  Loading HFID from: {HFID_PATH}")
    hfid = pd.read_csv(HFID_PATH)
    hfid['ipc_phase_fews'] = pd.to_numeric(hfid['ipc_phase_fews'], errors='coerce')
    hfid['ipc_phase_ipcch'] = pd.to_numeric(hfid['ipc_phase_ipcch'], errors='coerce')
    hfid['ipc_phase'] = hfid['ipc_phase_fews'].fillna(hfid['ipc_phase_ipcch'])
    hfid['location'] = hfid['iso3'] + '_' + hfid['ADMIN1'].fillna('national')
    hfid['year'] = hfid['year_month'].str[:4].astype(int)
    hfid_ipc = hfid[hfid['ipc_phase'].notna()].copy()
    cohort = set(hfid_ipc[hfid_ipc['year'] == 2016]['location'].unique())
    print(f"  2016 cohort: {len(cohort)} locations")
    return cohort


def load_cohort_episodes(cohort):
    """Load episodes filtered to cohort locations."""
    episodes = pd.read_csv(EPISODES_PATH)
    episodes['start'] = episodes['dates'].str.split(',').str[0].str[:7]
    episodes['end'] = episodes['dates'].str.split(',').str[-1].str[:7]
    episodes['start_f'] = episodes['start'].apply(ym_to_float)
    episodes['end_f'] = episodes['end'].apply(ym_to_float)
    episodes['severity_rank'] = episodes['archetype'].map(SEVERITY_RANK)
    return episodes[episodes['location'].isin(cohort)].copy()


def build_annual_data(episodes):
    """Build flows and counts for annual periods."""
    periods = [(str(y), y, y) for y in range(2011, 2024)]
    period_labels = [p[0] for p in periods]

    # Location-period matrix: assign most severe archetype overlapping each year
    rows = []
    for _, ep in episodes.iterrows():
        for label, p_start, p_end in periods:
            period_start_f = float(p_start)
            period_end_f = float(p_end) + 11.0 / 12.0
            if ep['start_f'] <= period_end_f and ep['end_f'] >= period_start_f:
                rows.append({
                    'location': ep['location'],
                    'period': label,
                    'archetype': ep['archetype'],
                    'severity_rank': ep['severity_rank'],
                })

    lp_all = pd.DataFrame(rows)
    idx = lp_all.groupby(['location', 'period'])['severity_rank'].idxmax()
    lp = lp_all.loc[idx, ['location', 'period', 'archetype']].copy()

    pivot = lp.pivot(index='location', columns='period', values='archetype')

    period_counts = {}
    for period in period_labels:
        if period in pivot.columns:
            period_counts[period] = pivot[period].dropna().value_counts().to_dict()
        else:
            period_counts[period] = {}

    flows = {}
    for i in range(len(period_labels) - 1):
        p1, p2 = period_labels[i], period_labels[i + 1]
        if p1 not in pivot.columns or p2 not in pivot.columns:
            continue
        both = pivot[[p1, p2]].dropna()
        flow_counts = defaultdict(int)
        for _, row in both.iterrows():
            flow_counts[(row[p1], row[p2])] += 1
        flows[(p1, p2)] = dict(flow_counts)

    return flows, period_counts, period_labels


def draw_flow(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color, alpha=0.35,
              edgecolor='none', linewidth=0, zorder=2, hatch=None):
    """Draw smooth curved flow band between two block positions."""
    dx = (x1 - x0) * 0.45
    verts = [
        (x0, y0_top),
        (x0 + dx, y0_top),
        (x1 - dx, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (x1 - dx, y1_bot),
        (x0 + dx, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    path = MPath(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor=edgecolor,
                      linewidth=linewidth, alpha=alpha, zorder=zorder,
                      hatch=hatch)
    ax.add_patch(patch)


def draw_panel_b(ax, n_cohort):
    """Draw annual alluvial for 2016 cohort on the given axes."""
    from matplotlib.colors import to_rgba

    cohort = get_2016_cohort()
    episodes = load_cohort_episodes(cohort)
    flows, period_counts, period_labels = build_annual_data(episodes)

    n_periods = len(period_labels)
    archetypes = [a for a in ARCHETYPE_ORDER_BARS if
                  any(a in pc for pc in period_counts.values())]

    # Wider layout for standalone figure
    col_width = 0.14
    col_spacing = 0.52
    block_gap_frac = 0.015

    x_pos = {label: i * col_spacing for i, label in enumerate(period_labels)}

    max_total = 0
    for period in period_labels:
        pc = period_counts.get(period, {})
        total = sum(pc.get(a, 0) for a in archetypes)
        max_total = max(max_total, total)

    if max_total == 0:
        return

    total_height = max_total

    # Block positions
    block_pos = {}
    for period in period_labels:
        pc = period_counts.get(period, {})
        y_cursor = 0
        for archetype in archetypes:
            count = pc.get(archetype, 0)
            if count == 0:
                continue
            block_pos[(period, archetype)] = {
                'x': x_pos[period],
                'y_bot': y_cursor,
                'y_top': y_cursor + count,
                'count': count,
            }
            y_cursor += count + block_gap_frac * total_height

    # Draw flows
    for i in range(n_periods - 1):
        p1, p2 = period_labels[i], period_labels[i + 1]
        if (p1, p2) not in flows:
            continue

        flow_dict = flows[(p1, p2)]
        out_cursor = {}
        in_cursor = {}
        for arch in archetypes:
            if (p1, arch) in block_pos:
                out_cursor[(p1, arch)] = block_pos[(p1, arch)]['y_bot']
            if (p2, arch) in block_pos:
                in_cursor[(p2, arch)] = block_pos[(p2, arch)]['y_bot']

        # Classify flows by type
        persistence_flows = []
        recovery_flows = []
        escalation_flows = []
        for (a1, a2), count in flow_dict.items():
            if count == 0 or (p1, a1) not in block_pos or (p2, a2) not in block_pos:
                continue
            if a1 == a2:
                persistence_flows.append((a1, a2, count))
            elif SEVERITY_RANK.get(a2, 0) > SEVERITY_RANK.get(a1, 0):
                escalation_flows.append((a1, a2, count))
            else:
                recovery_flows.append((a1, a2, count))

        # Draw persistence (faint background)
        for a1, a2, count in sorted(persistence_flows, key=lambda x: -x[2]):
            x0 = block_pos[(p1, a1)]['x'] + col_width / 2
            y0_bot = out_cursor.get((p1, a1), 0)
            y0_top = y0_bot + count
            out_cursor[(p1, a1)] = y0_top
            x1 = block_pos[(p2, a2)]['x'] - col_width / 2
            y1_bot = in_cursor.get((p2, a2), 0)
            y1_top = y1_bot + count
            in_cursor[(p2, a2)] = y1_top
            color = ARCHETYPE_COLORS.get(a1, '#999999')
            # Skip seasonal persistence entirely to remove green wash
            if a1 == 'seasonal_crisis':
                alpha = 0.04
            else:
                alpha = 0.10
            draw_flow(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot,
                      color, alpha=alpha, zorder=1)

        # Draw recovery (moderate visibility)
        for a1, a2, count in sorted(recovery_flows, key=lambda x: -x[2]):
            x0 = block_pos[(p1, a1)]['x'] + col_width / 2
            y0_bot = out_cursor.get((p1, a1), 0)
            y0_top = y0_bot + count
            out_cursor[(p1, a1)] = y0_top
            x1 = block_pos[(p2, a2)]['x'] - col_width / 2
            y1_bot = in_cursor.get((p2, a2), 0)
            y1_top = y1_bot + count
            in_cursor[(p2, a2)] = y1_top
            color = ARCHETYPE_COLORS.get(a1, '#999999')
            draw_flow(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot,
                      color, alpha=0.20, zorder=2)

        # Draw escalation (bold, on top, with diagonal hatching for accessibility)
        for a1, a2, count in sorted(escalation_flows, key=lambda x: -x[2]):
            x0 = block_pos[(p1, a1)]['x'] + col_width / 2
            y0_bot = out_cursor.get((p1, a1), 0)
            y0_top = y0_bot + count
            out_cursor[(p1, a1)] = y0_top
            x1 = block_pos[(p2, a2)]['x'] - col_width / 2
            y1_bot = in_cursor.get((p2, a2), 0)
            y1_top = y1_bot + count
            in_cursor[(p2, a2)] = y1_top
            color = ARCHETYPE_COLORS.get(a2, '#999999')
            edge_rgba = to_rgba(color, alpha=0.85)
            plt.rcParams['hatch.linewidth'] = 0.3
            draw_flow(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot,
                      color, alpha=0.65, edgecolor=edge_rgba,
                      linewidth=0.4, zorder=4, hatch='//')

    # Draw blocks
    for (period, archetype), pos in block_pos.items():
        height = pos['y_top'] - pos['y_bot']
        rect = mpatches.FancyBboxPatch(
            (pos['x'] - col_width / 2, pos['y_bot']),
            col_width, height,
            boxstyle="round,pad=0.01",
            facecolor=ARCHETYPE_COLORS[archetype],
            edgecolor='white', linewidth=0.6,
            alpha=0.95, zorder=5,
        )
        ax.add_patch(rect)

        # Count labels inside blocks
        mid_y = (pos['y_bot'] + pos['y_top']) / 2
        light_bars = {'seasonal_crisis', 'prolonged_moderate'}
        txt_color = '#333333' if archetype in light_bars else 'white'
        if height > total_height * 0.07:
            ax.text(pos['x'], mid_y, f"{pos['count']}",
                    ha='center', va='center',
                    fontsize=6.5, fontweight='bold', color=txt_color, zorder=6)
        elif height > total_height * 0.035:
            ax.text(pos['x'], mid_y, f"{pos['count']}",
                    ha='center', va='center',
                    fontsize=5.5, color=txt_color, zorder=6)

    # Year labels at bottom
    y_label = -total_height * 0.06
    for period in period_labels:
        x = x_pos[period]
        ax.text(x, y_label, period,
                ha='center', va='top',
                fontsize=7, color='#333333')

    # Y-axis
    x_min = -0.35
    x_max = (n_periods - 1) * col_spacing + 0.45
    y_max = max(pos['y_top'] for pos in block_pos.values()) if block_pos else total_height
    y_min_plot = -total_height * 0.10
    y_top = y_max + total_height * 0.06

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_plot, y_top)
    ax.set_ylabel('Number of subnational locations', fontsize=8, labelpad=8)

    # Severity arrow on right side
    arrow_x = x_max - 0.15
    all_bar_tops = [pos['y_top'] for pos in block_pos.values()]
    all_bar_bots = [pos['y_bot'] for pos in block_pos.values()]
    bar_top = max(all_bar_tops) if all_bar_tops else total_height
    bar_bot = min(all_bar_bots) if all_bar_bots else 0
    ax.annotate('', xy=(arrow_x, bar_top), xytext=(arrow_x, bar_bot),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0))
    ax.text(arrow_x + 0.08, (bar_bot + bar_top) / 2, 'Increasing\nseverity',
            ha='left', va='center', fontsize=6, color='#888888', rotation=270)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(axis='y', labelsize=7)

    return archetypes


# =====================================================================
# Combined figure
# =====================================================================

def create_combined_figure():
    """Create standalone Figure 2: annual archetype evolution."""
    print("Creating Figure 2 (standalone alluvial)...")

    # Get 2016 cohort count for caption
    cohort = get_2016_cohort()
    n_cohort = len(cohort)

    # Figure: oversized to compensate for bbox_inches='tight' cropping
    fig, ax = plt.subplots(figsize=(7.87, 4.33))
    fig.patch.set_facecolor('white')

    present_archetypes = draw_panel_b(ax, n_cohort)

    # Cohort sample size label (top-left)
    ax.text(0.01, 0.98, f'2016 cohort (n\u2009=\u2009{n_cohort})',
            transform=ax.transAxes, fontsize=7, color='#555555',
            ha='left', va='top', style='italic')

    # Legend below figure
    if present_archetypes:
        legend_handles = []
        for archetype in present_archetypes:
            handle = mpatches.Patch(
                color=ARCHETYPE_COLORS[archetype],
                label=ARCHETYPE_LABELS_FULL[archetype],
            )
            legend_handles.append(handle)

        ax.legend(handles=legend_handles,
                  loc='upper center',
                  bbox_to_anchor=(0.47, -0.06),
                  ncol=4,
                  fontsize=6.5, title='Crisis Archetype',
                  title_fontsize=7.5, frameon=False,
                  columnspacing=1.0, handletextpad=0.5)

    fig.subplots_adjust(bottom=0.18)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, 'Figure2_combined_alluvial.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'Figure2_combined_alluvial.pdf')
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Convert PNG from RGBA to RGB (journal requirement)
    try:
        from PIL import Image
        img = Image.open(png_path)
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            rgb_img.save(png_path, dpi=(600, 600))
            print(f"  Converted PNG to RGB mode")
    except ImportError:
        print("  Note: PIL not available, skipping RGBA->RGB conversion")

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    return png_path, n_cohort


def export_source_data():
    """Export source data for Figure 2 to Excel."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx_path = os.path.join(OUTPUT_DIR, 'SourceData_Fig2.xlsx')

    # Transition matrix (panel a data)
    trans_df = pd.read_csv(TRANSITIONS_PATH)
    matrix = pd.crosstab(trans_df['from_archetype'], trans_df['to_archetype'])

    # Cohort annual counts (panel b data)
    cohort = get_2016_cohort()
    episodes = load_cohort_episodes(cohort)
    _, period_counts, period_labels = build_annual_data(episodes)

    annual_rows = []
    for period in period_labels:
        pc = period_counts.get(period, {})
        for arch, count in pc.items():
            annual_rows.append({'year': period, 'archetype': arch, 'n_locations': count})
    annual_df = pd.DataFrame(annual_rows)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        matrix.to_excel(writer, sheet_name='transition_matrix')
        annual_df.to_excel(writer, sheet_name='annual_cohort_counts', index=False)
    print(f"Saved: {xlsx_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Figure 2: Combined Alluvial Diagram")
    print("=" * 60)

    for path, label in [(EPISODES_PATH, 'episodes.csv'),
                         (TRANSITIONS_PATH, 'archetype_transitions.csv'),
                         (HFID_PATH, 'HFID_hv1.csv')]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            print("Run 01_reference_pipeline.py and 02_generate_transitions.py first.")
            sys.exit(1)

    png_path, n_cohort = create_combined_figure()
    export_source_data()

    print("=" * 60)
    print("Done!")
    print(f"  PNG: {png_path}")
    print("=" * 60)
