#!/usr/bin/env python3
"""
06_fig1_archetypes.py — Figure 1: Eight Archetypes Scatter Plot
================================================================

Duration (x, log scale) vs Peak severity (y, jittered), coloured by archetype,
with convex-hull colour clouds around each cluster.

Inputs  (relative to package root):
    outputs/data/episodes.csv

Outputs (relative to package root):
    outputs/figures/Figure1_archetype_scatter.png  (300 dpi)
    outputs/figures/Figure1_archetype_scatter.pdf

Author: Richard Choularton
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# ============================================================
# Paths
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ============================================================
# Nature Food style
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
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# ============================================================
# Archetype definitions
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

ARCHETYPE_ORDER = [
    'seasonal_crisis', 'prolonged_moderate', 'protracted_emergency',
    'rapid_onset', 'entrenched_moderate', 'oscillating', 'severe_shock', 'escalating'
]

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


# ============================================================
# Figure generation
# ============================================================

def figure1_archetype_scatter(df):
    """
    Figure 1: Eight archetypes scatter plot
    Duration (x, log scale) vs Peak severity (y, jittered), colored by archetype
    With color clouds (convex hulls) around each cluster
    """
    print("Creating Figure 1: Archetype scatter plot...")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Add jitter to peak_phase for visibility (seeded for reproducibility)
    rng = np.random.RandomState(42)
    df = df.copy()
    df['peak_jittered'] = df['peak_phase'] + rng.uniform(-0.3, 0.3, size=len(df))

    # First pass: draw color clouds (convex hulls) behind points
    for archetype in ARCHETYPE_ORDER:
        subset = df[df['archetype'] == archetype]
        if len(subset) >= 3:
            points = np.column_stack([
                np.log10(subset['duration_months'].values.clip(min=0.5)),
                subset['peak_jittered'].values
            ])

            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points_plot = np.column_stack([
                    10**hull_points[:, 0],
                    hull_points[:, 1]
                ])

                polygon = Polygon(
                    hull_points_plot,
                    alpha=0.15,
                    facecolor=ARCHETYPE_COLORS[archetype],
                    edgecolor=ARCHETYPE_COLORS[archetype],
                    linewidth=1,
                    linestyle='-'
                )
                ax.add_patch(polygon)
            except Exception:
                pass

    # Second pass: draw points on top
    for archetype in ARCHETYPE_ORDER:
        subset = df[df['archetype'] == archetype]
        if len(subset) > 0:
            ax.scatter(
                subset['duration_months'],
                subset['peak_jittered'],
                c=ARCHETYPE_COLORS[archetype],
                label=f"{ARCHETYPE_LABELS[archetype]} (n={len(subset)})",
                alpha=0.6,
                s=15,
                edgecolors='white',
                linewidths=0.3
            )

    ax.set_xscale('log')
    ax.set_xlabel('Duration (months)')
    ax.set_ylabel('Peak severity (IPC phase)')
    ax.set_xlim(0.8, 200)
    ax.set_ylim(2.4, 5.6)

    # Duration threshold lines
    ax.axvline(x=12, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=36, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.text(12, 5.5, '12m', ha='center', fontsize=7, color='gray')
    ax.text(36, 5.5, '36m', ha='center', fontsize=7, color='gray')

    # Severity threshold line
    ax.axhline(y=3.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Y-axis ticks at actual IPC phases
    ax.set_yticks([3, 4, 5])
    ax.set_yticklabels(['Phase 3\n(Crisis)', 'Phase 4\n(Emergency)', 'Phase 5\n(Famine)'])

    # Quadrant labels
    ax.text(0.03, 0.25, 'Short\nmoderate', transform=ax.transAxes, fontsize=7,
            color='#888888', fontstyle='italic', va='center')
    ax.text(0.03, 0.75, 'Short\nsevere', transform=ax.transAxes, fontsize=7,
            color='#888888', fontstyle='italic', va='center')
    ax.text(0.75, 0.25, 'Long\nmoderate', transform=ax.transAxes, fontsize=7,
            color='#888888', fontstyle='italic', va='center')
    ax.text(0.75, 0.75, 'Long\nsevere', transform=ax.transAxes, fontsize=7,
            color='#888888', fontstyle='italic', va='center')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, 'Figure1_archetype_scatter.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'Figure1_archetype_scatter.pdf')
    plt.savefig(png_path, dpi=300, facecolor='white')
    plt.savefig(pdf_path, facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Figure 1: Archetype Scatter Plot")
    print("=" * 60)

    if not os.path.exists(EPISODES_PATH):
        print(f"ERROR: Episodes file not found: {EPISODES_PATH}")
        print("Run 01_reference_pipeline.py first.")
        sys.exit(1)

    df = pd.read_csv(EPISODES_PATH)
    print(f"Loaded {len(df)} episodes")
    figure1_archetype_scatter(df)

    print("=" * 60)
    print("Done!")
    print("=" * 60)
