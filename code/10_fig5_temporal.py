#!/usr/bin/env python3
"""
10_fig5_temporal.py — Figure 5: Temporal Trends
=================================================

Creates a two-panel figure showing:
  (a) Annual Phase 4+ rates from the interpolated time series
  (b) Period comparison of archetype distribution (2011–2017 vs 2018–2023)

Inputs
------
- outputs/data/episodes.csv             (from step 01)
- outputs/data/episode_verification.json (from step 01)
- outputs/data/temporal_comparison.json  (from step 01)

Outputs
-------
- outputs/figures/Figure5_temporal_trends.png
- outputs/figures/Figure5_temporal_trends.pdf
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data')
FIG_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ---------------------------------------------------------------------------
# Nature-style formatting
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Archetype definitions
# ---------------------------------------------------------------------------
ARCHETYPE_ORDER = [
    'seasonal_crisis', 'prolonged_moderate', 'protracted_emergency',
    'rapid_onset', 'entrenched_moderate', 'oscillating',
    'severe_shock', 'escalating',
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


def main():
    """Generate Figure 5: Temporal trends."""
    # ------------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------------
    episodes_path = os.path.join(DATA_DIR, 'episodes.csv')
    ev_path = os.path.join(DATA_DIR, 'episode_verification.json')
    tc_path = os.path.join(DATA_DIR, 'temporal_comparison.json')

    for path in [episodes_path, ev_path, tc_path]:
        if not os.path.exists(path):
            print(f"ERROR: Required input not found: {path}")
            print("Run step 01 first.")
            sys.exit(1)

    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(episodes_path)
    print(f"Loaded {len(df)} episodes")

    with open(ev_path) as f:
        ev_data = json.load(f)
    phase4_by_year = ev_data.get('phase4_plus_by_year', {})

    with open(tc_path) as f:
        tc_data = json.load(f)

    # Parse episode start dates
    df = df.copy()
    df['start_date'] = df['dates'].str.split(',').str[0]
    df['start_year'] = pd.to_datetime(df['start_date']).dt.year

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    print("Creating Figure 5: Temporal trends...")

    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, wspace=0.3)

    # (a) Phase 4+ rate by year
    ax1 = fig.add_subplot(gs[0])

    years = sorted([int(y) for y in phase4_by_year.keys()])
    rates = [phase4_by_year[str(y)] for y in years]

    ax1.bar(years, rates, color='#EE6677', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Phase 4+ rate (%)')
    ax1.set_xticks(years)
    ax1.set_xticklabels(years, rotation=45, ha='right')

    # Period marker
    ax1.axvline(x=2017.5, color='black', linestyle=':', linewidth=1)

    # Matched-location annotation
    matched = tc_data.get('matched_locations', {})
    n_matched = matched.get('n_locations', 0)
    early_rate = matched.get('early_phase4_pct', 0)
    late_rate = matched.get('late_phase4_pct', 0)

    ax1.text(0.02, 0.95,
             f'Matched locations (n={n_matched}):\n'
             f'{early_rate}% \u2192 {late_rate}%',
             transform=ax1.transAxes, fontsize=7,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='gray', alpha=0.9))

    ax1.set_title('a  Phase 4+ rate by year', loc='left', fontweight='bold')

    # (b) Period comparison of archetype distribution
    ax2 = fig.add_subplot(gs[1])

    early = df[df['start_year'].between(2011, 2017)]
    recent = df[df['start_year'].between(2018, 2023)]

    early_dist = early['archetype'].value_counts(normalize=True) * 100
    recent_dist = recent['archetype'].value_counts(normalize=True) * 100

    x = np.arange(len(ARCHETYPE_ORDER))
    width = 0.35

    early_vals = [early_dist.get(a, 0) for a in ARCHETYPE_ORDER]
    recent_vals = [recent_dist.get(a, 0) for a in ARCHETYPE_ORDER]

    ax2.barh(x - width / 2, early_vals, width, label='2011\u20132017',
             color='#CCCCCC', edgecolor='black', linewidth=0.5)
    ax2.barh(x + width / 2, recent_vals, width, label='2018\u20132023',
             color='#4477AA', edgecolor='black', linewidth=0.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels([ARCHETYPE_LABELS[a] for a in ARCHETYPE_ORDER])
    ax2.set_xlabel('Proportion of episodes (%)')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 90)

    ax2.set_title('b  Archetype distribution by period', loc='left',
                   fontweight='bold')

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    png_path = os.path.join(FIG_DIR, 'Figure5_temporal_trends.png')
    pdf_path = os.path.join(FIG_DIR, 'Figure5_temporal_trends.pdf')
    plt.savefig(png_path, dpi=300, facecolor='white')
    plt.savefig(pdf_path, facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    main()
