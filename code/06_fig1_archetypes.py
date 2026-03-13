#!/usr/bin/env python3
"""
06_fig1_archetypes.py — Figure 1: Crisis Archetype Scatter Plot
================================================================

Duration (x, log scale) vs Maximum IPC phase (y) with dot size encoding
volatility. Zone-band design with Tol Bright palette.

Addresses coauthor feedback (Per Becker, Krishna Krishnamurthy):
  - Per: use max IPC (peak_phase) on y-axis since that is what the rule-based
    classification uses, making archetypes more clearly distinguished.
  - Krishna: protracted emergency label/box should contain its episodes.

Inputs  (relative to package root):
    outputs/data/episodes.csv

Outputs (relative to package root):
    outputs/figures/Figure1_archetype_scatter.png  (600 dpi)
    outputs/figures/Figure1_archetype_scatter.pdf
    outputs/figures/SourceData_Fig1.xlsx

Author: Richard Choularton
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# ============================================================
# Paths
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
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

# === Tol Bright palette (matches Figure 2) ===
ARCHETYPE_COLORS = {
    'seasonal_crisis':        '#228833',  # Tol bright green
    'prolonged_moderate':    '#CCBB44',  # Tol bright yellow
    'entrenched_moderate':   '#EE6677',  # Tol bright rose
    'oscillating':           '#AA3377',  # Tol bright purple
    'rapid_onset':           '#EE8866',  # Warm coral
    'severe_shock':          '#CC3311',  # Strong red
    'escalating':            '#004488',  # Tol bright blue
    'protracted_emergency':  '#332288',  # Tol bright indigo
}

ARCHETYPE_LABELS = {
    'seasonal_crisis': 'Seasonal crisis',
    'prolonged_moderate': 'Prolonged moderate',
    'protracted_emergency': 'Protracted emergency',
    'rapid_onset': 'Rapid onset',
    'entrenched_moderate': 'Entrenched moderate',
    'oscillating': 'Oscillating',
    'severe_shock': 'Severe shock',
    'escalating': 'Escalating',
}

# Plot order: draw large clusters first, small/important last (on top)
ARCHETYPE_ORDER = [
    'seasonal_crisis', 'prolonged_moderate', 'entrenched_moderate',
    'protracted_emergency', 'rapid_onset', 'oscillating',
    'severe_shock', 'escalating',
]

# Severity order for z-ordering (mild -> severe)
SEVERITY_RANK = {
    'seasonal_crisis': 0,
    'prolonged_moderate': 1,
    'entrenched_moderate': 2,
    'oscillating': 3,
    'rapid_onset': 4,
    'severe_shock': 5,
    'escalating': 6,
    'protracted_emergency': 7,
}


def load_episodes():
    """Load crisis episodes and compute volatility."""
    df = pd.read_csv(EPISODES_PATH)
    df['ipc_std'] = np.sqrt(df['phase_variance'].clip(lower=0))
    print(f"  Loaded {len(df)} episodes")
    return df


def create_figure():
    """Create Figure 1: archetype scatter with volatility encoding."""
    print("Creating Figure 1 (archetype scatter)...")

    df = load_episodes()

    # Taller figure: gives space for Phase 3 density band + clean annotations
    fig, ax = plt.subplots(figsize=(6.69, 5.91))  # ~170mm x 150mm
    fig.patch.set_facecolor('white')

    rng = np.random.RandomState(42)

    # --- Y-axis positioning (peak_phase: discrete 3, 4, or 5) ---
    # Jitter within each phase band to show density
    phase3_mask = df['peak_phase'] == 3
    phase4_mask = df['peak_phase'] == 4
    phase5_mask = df['peak_phase'] == 5

    y_jitter = np.zeros(len(df))
    # Phase 3 band: wide jitter (1,466 episodes need room)
    y_jitter[phase3_mask] = rng.uniform(-0.25, 0.25, size=phase3_mask.sum())
    # Phase 4 band: moderate jitter (187 episodes)
    y_jitter[phase4_mask] = rng.uniform(-0.20, 0.20, size=phase4_mask.sum())
    # Phase 5: small jitter (5 episodes)
    y_jitter[phase5_mask] = rng.uniform(-0.08, 0.08, size=phase5_mask.sum())
    df['y_plot'] = df['peak_phase'] + y_jitter

    # X-direction jitter (multiplicative on log scale)
    x_jitter_factor = rng.uniform(0.85, 1.176, size=len(df))  # symmetric on log
    df['x_plot'] = df['duration_months'] * x_jitter_factor

    # --- Dot sizing (volatility encoding) ---
    size_min = 12
    size_max = 110
    std_max = df['ipc_std'].quantile(0.98)
    df['dot_size'] = size_min + (df['ipc_std'].clip(upper=std_max) / max(std_max, 0.001)) * (size_max - size_min)

    BACKGROUND_ARCHETYPES = {'seasonal_crisis'}

    # --- Differentiated alpha ---
    alpha_map = {
        'seasonal_crisis': 0.40,
        'prolonged_moderate': 0.65,
        'entrenched_moderate': 0.70,
    }
    default_alpha = 0.80

    # --- Zone bands for archetypes ---
    n_seasonal = len(df[df['archetype'] == 'seasonal_crisis'])
    n_prolonged = len(df[df['archetype'] == 'prolonged_moderate'])
    n_entrenched = len(df[df['archetype'] == 'entrenched_moderate'])
    n_protracted = len(df[df['archetype'] == 'protracted_emergency'])

    ylo, yhi = 2.35, 5.25
    def y_to_frac(y):
        return (y - ylo) / (yhi - ylo)

    # Phase 3 zones: span 2.65-3.30
    p3_bot, p3_top = y_to_frac(2.65), y_to_frac(3.30)
    ax.axvspan(0.8, 12, ymin=p3_bot, ymax=p3_top, alpha=0.07,
               color=ARCHETYPE_COLORS['seasonal_crisis'], zorder=0)
    ax.text(np.sqrt(0.8 * 12), 2.62, f'Seasonal crisis\nn = {n_seasonal:,}',
            ha='center', va='top', fontsize=6, fontweight='bold',
            color=ARCHETYPE_COLORS['seasonal_crisis'], zorder=11)

    ax.axvspan(12, 36, ymin=p3_bot, ymax=p3_top, alpha=0.12,
               color=ARCHETYPE_COLORS['prolonged_moderate'], zorder=0)
    ax.text(np.sqrt(12 * 36), 2.62, f'Prolonged moderate\nn = {n_prolonged}',
            ha='center', va='top', fontsize=6, fontweight='bold',
            color=ARCHETYPE_COLORS['prolonged_moderate'], zorder=11)

    ax.axvspan(36, 200, ymin=p3_bot, ymax=p3_top, alpha=0.12,
               color=ARCHETYPE_COLORS['entrenched_moderate'], zorder=0)
    ax.text(np.sqrt(36 * 150), 2.62, f'Entrenched moderate\nn = {n_entrenched}',
            ha='center', va='top', fontsize=6, fontweight='bold',
            color=ARCHETYPE_COLORS['entrenched_moderate'], zorder=11)

    # Phase 4+ zone: protracted emergency
    p4_bot, p4_top = y_to_frac(3.40), y_to_frac(5.15)
    ax.axvspan(12, 200, ymin=p4_bot, ymax=p4_top, alpha=0.06,
               color=ARCHETYPE_COLORS['protracted_emergency'], zorder=0)
    ax.text(150, 3.50, f'Protracted emergency\nn = {n_protracted}',
            ha='right', va='top', fontsize=6, fontweight='bold',
            color=ARCHETYPE_COLORS['protracted_emergency'], zorder=11)

    # --- Draw points ---
    for archetype in ARCHETYPE_ORDER:
        subset = df[df['archetype'] == archetype]
        if len(subset) > 0:
            alpha = alpha_map.get(archetype, default_alpha)
            if archetype in BACKGROUND_ARCHETYPES:
                edge_color = 'white'
                edge_width = 0.3
            else:
                edge_color = ARCHETYPE_COLORS[archetype]
                edge_width = 0.5
            ax.scatter(
                subset['x_plot'],
                subset['y_plot'],
                c=ARCHETYPE_COLORS[archetype],
                s=subset['dot_size'],
                marker='o',
                alpha=alpha,
                edgecolors=edge_color,
                linewidths=edge_width,
                zorder=2 + SEVERITY_RANK.get(archetype, 0),
            )

    # --- Axes ---
    ax.set_xscale('log')
    ax.set_xlabel('Duration (months)', fontsize=8)
    ax.set_ylabel('Maximum IPC phase', fontsize=8)
    ax.set_xlim(0.8, 200)
    ax.set_ylim(2.35, 5.25)

    # Human-readable x-axis ticks
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())

    # IPC phase reference lines
    ax.axhline(y=3, color='#CCCCCC', linestyle='-', linewidth=0.6, zorder=0)
    ax.axhline(y=4, color='#CCCCCC', linestyle='-', linewidth=0.6, zorder=0)
    ax.axhline(y=5, color='#CCCCCC', linestyle='-', linewidth=0.6, zorder=0)

    # Y-axis ticks
    ax.set_yticks([3, 4, 5])
    ax.set_yticklabels(['Phase 3\n(Crisis)', 'Phase 4\n(Emergency)', 'Phase 5\n(Famine)'])

    # Phase 3 annotation
    n_phase3 = int(phase3_mask.sum())
    pct_phase3 = n_phase3 / len(df) * 100
    ax.annotate(
        f'{pct_phase3:.0f}% of episodes peak at Phase 3\n(jittered to show density)',
        xy=(4, 3.18), xytext=(0.9, 3.50),
        fontsize=6, color='#444444', fontstyle='italic',
        ha='left', va='bottom',
        arrowprops=dict(arrowstyle='->', color='#888888', lw=0.6),
        zorder=10,
    )

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Legend ---
    LEGEND_ORDER_PHASE3 = [
        'seasonal_crisis', 'prolonged_moderate', 'entrenched_moderate',
    ]
    LEGEND_ORDER_PHASE4 = [
        'rapid_onset', 'oscillating', 'severe_shock',
        'escalating', 'protracted_emergency',
    ]
    legend_handles = []
    for archetype in LEGEND_ORDER_PHASE3:
        n = len(df[df['archetype'] == archetype])
        if archetype in BACKGROUND_ARCHETYPES:
            edge_c, edge_w = 'white', 0.3
        else:
            edge_c, edge_w = ARCHETYPE_COLORS[archetype], 0.5
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=ARCHETYPE_COLORS[archetype],
                   markeredgecolor=edge_c, markeredgewidth=edge_w,
                   markersize=5, label=f"{ARCHETYPE_LABELS[archetype]} ({n:,})")
        )
    # Blank spacer
    legend_handles.append(
        Line2D([0], [0], marker='None', color='w', linestyle='None', label=' ')
    )
    for archetype in LEGEND_ORDER_PHASE4:
        n = len(df[df['archetype'] == archetype])
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=ARCHETYPE_COLORS[archetype],
                   markeredgecolor=ARCHETYPE_COLORS[archetype],
                   markeredgewidth=0.5,
                   markersize=5, label=f"{ARCHETYPE_LABELS[archetype]} ({n})")
        )
    leg1 = ax.legend(handles=legend_handles,
                     loc='upper left',
                     bbox_to_anchor=(1.02, 1.0),
                     frameon=False, fontsize=6.5, handletextpad=0.4,
                     labelspacing=0.30, borderaxespad=0)
    ax.add_artist(leg1)

    # Volatility size legend
    size_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#888888', markeredgecolor='white',
               markeredgewidth=0.3,
               markersize=np.sqrt(size_min) * 0.8, label='Low'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#888888', markeredgecolor='white',
               markeredgewidth=0.3,
               markersize=np.sqrt(size_max) * 0.8, label='High'),
    ]
    leg2 = ax.legend(handles=size_handles,
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0.0),
                     title='Volatility',
                     title_fontsize=6.5, frameon=False,
                     fontsize=6, handletextpad=0.4,
                     labelspacing=0.5, borderaxespad=0)
    ax.add_artist(leg1)

    fig.subplots_adjust(right=0.72)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, 'Figure1_archetype_scatter.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'Figure1_archetype_scatter.pdf')
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white',
                bbox_extra_artists=[leg1, leg2])
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                bbox_extra_artists=[leg1, leg2])
    plt.close(fig)

    # Convert PNG to RGB
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
    return png_path


def export_source_data():
    """Export source data for Figure 1 to Excel."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx_path = os.path.join(OUTPUT_DIR, 'SourceData_Fig1.xlsx')

    df = pd.read_csv(EPISODES_PATH)
    export_df = df[['location', 'iso3', 'archetype', 'duration_months',
                     'peak_phase']].copy()
    export_df = export_df.sort_values(['archetype', 'location'])
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Figure1_data', index=False)
        # Summary counts
        summary = export_df.groupby('archetype').agg(
            n_episodes=('location', 'count'),
            mean_duration=('duration_months', 'mean'),
            mean_peak=('peak_phase', 'mean'),
        ).round(2).reset_index()
        summary.to_excel(writer, sheet_name='archetype_summary', index=False)
    print(f"Saved: {xlsx_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Figure 1: Crisis Archetype Scatter Plot")
    print("=" * 60)

    if not os.path.exists(EPISODES_PATH):
        print(f"ERROR: Episodes file not found: {EPISODES_PATH}")
        print("Run 01_reference_pipeline.py first.")
        sys.exit(1)

    create_figure()
    export_source_data()

    print("=" * 60)
    print("Done!")
    print("=" * 60)
