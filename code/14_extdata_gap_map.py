#!/usr/bin/env python3
"""
14_extdata_gap_map.py -- Extended Data Figure: Gap Compression Map
===================================================================

Two-panel map showing the geographic distribution of recurring food crises
and recovery gap compression:

  (a) Locations with 3+ crisis episodes, colored by worst trajectory pattern
      (seasonal only, acute, prolonged/entrenched, escalating, protracted).
  (b) Mean recovery gap duration between episodes at each location, using
      an RdYlGn colourmap (shorter gaps = red).

Algorithm is identical to the source script
``papers/paper1a/figures/ExtDataFig_gap_compression_map.py``; only the
data-loading paths and output paths have been changed to read from the
reproducibility-package directories.  Docx generation is removed.

NOTE: This script requires ``geopandas``, which is an optional dependency.
If geopandas is not installed, the script prints an informational message
and exits with code 0 (success) so it does not block the rest of the
pipeline.

Inputs (relative to package root):
    outputs/data/episodes.csv
        Produced by 01_reference_pipeline.py
    outputs/data/location_summaries.csv    (if exists, from 02_generate_transitions.py)
    data/admin1_centroids.csv
        Admin1 centroid coordinates

Outputs (relative to package root):
    outputs/figures/ExtData_gap_compression_map.png     (300 dpi)
    outputs/figures/ExtData_gap_compression_map.pdf

Dependencies: pandas, numpy, matplotlib, geopandas (optional)

Author: Richard Choularton
"""

import os
import sys

# ============================================================
# Optional geopandas import -- graceful exit if unavailable
# ============================================================
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import urllib.request
import zipfile

# ============================================================
# Paths -- relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
LOCATION_SUMMARIES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data',
                                        'location_summaries.csv')
CENTROID_FILE = os.path.join(PACKAGE_ROOT, 'data', 'admin1_centroids.csv')
FIGURES_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')
NE_DIR = os.path.join(PACKAGE_ROOT, 'data', 'naturalearth')

# ============================================================
# Nature Food style rcParams
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})


# ============================================================
# Helper functions (unchanged from source)
# ============================================================

def get_world_map():
    """Get world country boundaries from Natural Earth."""
    shp_file = os.path.join(NE_DIR, 'ne_110m_admin_0_countries.shp')
    if not os.path.exists(shp_file):
        print("Downloading Natural Earth data...")
        os.makedirs(NE_DIR, exist_ok=True)
        url = ("https://naturalearth.s3.amazonaws.com/110m_cultural/"
               "ne_110m_admin_0_countries.zip")
        zip_path = os.path.join(NE_DIR, "ne_110m_admin_0_countries.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(NE_DIR)
    return gpd.read_file(shp_file)


def classify_pattern(row):
    """Classify location by worst trajectory pattern."""
    seq = row['archetype_sequence']
    if 'protracted_emergency' in seq:
        return 'Protracted emergency'
    elif 'escalating' in seq:
        return 'Escalating'
    elif 'entrenched_moderate' in seq or 'prolonged_moderate' in seq:
        return 'Prolonged/entrenched'
    elif ('oscillating' in seq or 'severe_shock' in seq
          or 'rapid_onset' in seq):
        return 'Acute (rapid/oscillating)'
    else:
        return 'Seasonal only'


def compute_gap_stats(episodes_df, locations_3plus):
    """Compute mean gap duration per location."""
    gap_records = []
    for loc in locations_3plus['location'].unique():
        loc_eps = episodes_df[episodes_df['location'] == loc].copy()
        starts, ends = [], []
        for _, ep in loc_eps.iterrows():
            dates = str(ep['dates']).split(',')
            if len(dates) > 0:
                starts.append(pd.to_datetime(dates[0].strip()))
                ends.append(pd.to_datetime(dates[-1].strip()))
        if len(starts) < 2:
            continue
        sorted_pairs = sorted(zip(starts, ends), key=lambda x: x[0])
        gaps = []
        for i in range(1, len(sorted_pairs)):
            gap_months = ((sorted_pairs[i][0].year
                           - sorted_pairs[i-1][1].year) * 12
                          + (sorted_pairs[i][0].month
                             - sorted_pairs[i-1][1].month))
            if gap_months > 0:
                gaps.append(gap_months)
        if gaps:
            gap_records.append({
                'location': loc,
                'mean_gap_months': np.mean(gaps),
                'min_gap_months': np.min(gaps),
                'n_gaps': len(gaps)
            })
    return pd.DataFrame(gap_records)


def build_location_summaries_from_episodes(episodes_df):
    """Build location summaries with archetype_sequence, unique_archetypes,
    total_episodes, iso3, and admin1 from episodes.csv.

    Used when outputs/data/location_summaries.csv does not contain the
    archetype_sequence column (which is needed for classify_pattern).
    """
    episodes = episodes_df.copy()
    dates_first = episodes['dates'].str.split(',').str[0].str.strip()
    episodes['start_dt'] = pd.to_datetime(dates_first)
    episodes = episodes.sort_values(['location', 'start_dt'])

    records = []
    for location, grp in episodes.groupby('location'):
        archetypes = grp['archetype'].tolist()
        seq_str = ' \u2192 '.join(archetypes)
        unique_count = len(set(archetypes))
        iso3 = grp['iso3'].iloc[0]
        admin1 = location.split('_', 1)[1] if '_' in location else location
        records.append({
            'location': location,
            'iso3': iso3,
            'admin1': admin1,
            'archetype_sequence': seq_str,
            'unique_archetypes': unique_count,
            'total_episodes': len(grp),
        })

    return pd.DataFrame(records)


def size_legend(ax, sizes, labels, loc='lower right', title='Episodes'):
    """Add a bubble size legend."""
    handles = []
    for s, label in zip(sizes, labels):
        handles.append(Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor='#888888', markeredgecolor='black',
            markeredgewidth=0.3, markersize=np.sqrt(s),
            label=label, linestyle='None'))
    leg = ax.legend(handles=handles, loc=loc, fontsize=6, framealpha=0.9,
                    title=title, title_fontsize=7, handletextpad=0.5,
                    borderpad=0.4, labelspacing=0.6)
    return leg


# ============================================================
# Main figure creation
# ============================================================

def create_figure():
    """Create the two-panel gap compression map."""
    print("Creating Extended Data Figure: Gap compression map...")

    # Load episodes
    episodes = pd.read_csv(EPISODES_PATH)

    # Build or load location summaries with archetype_sequence
    if os.path.exists(LOCATION_SUMMARIES_PATH):
        loc_df = pd.read_csv(LOCATION_SUMMARIES_PATH)
        # If the file lacks archetype_sequence, rebuild from episodes
        if 'archetype_sequence' not in loc_df.columns:
            loc_df = build_location_summaries_from_episodes(episodes)
    else:
        loc_df = build_location_summaries_from_episodes(episodes)

    # Load centroids
    centroids = pd.read_csv(CENTROID_FILE)

    # Filter to 3+ episodes
    loc_3plus = loc_df[loc_df['total_episodes'] >= 3].copy()

    # Merge coordinates
    # Try both ADMIN1 and admin1 column names for flexibility
    if 'ADMIN1' in centroids.columns:
        merged = loc_3plus.merge(
            centroids, left_on=['iso3', 'admin1'],
            right_on=['iso3', 'ADMIN1'], how='left'
        )
    elif 'admin1' in centroids.columns:
        merged = loc_3plus.merge(
            centroids, on=['iso3', 'admin1'], how='left'
        )
    else:
        # Fallback: try matching on whatever columns exist
        merged = loc_3plus.merge(centroids, on='iso3', how='left')

    matched = merged.dropna(subset=['centroid_lat', 'centroid_lon']).copy()
    print(f"  Locations with 3+ episodes: {len(loc_3plus)} "
          f"({len(matched)} mapped)")

    # Classify patterns and compute gaps
    matched['pattern'] = matched.apply(classify_pattern, axis=1)
    gap_stats = compute_gap_stats(episodes, loc_3plus)
    matched = matched.merge(gap_stats, on='location', how='left')

    # GeoDataFrame
    gdf = gpd.GeoDataFrame(
        matched,
        geometry=gpd.points_from_xy(matched['centroid_lon'],
                                     matched['centroid_lat']),
        crs="EPSG:4326"
    )

    world = get_world_map()

    # View bounds
    xlim = (-95, 105)
    ylim = (-38, 42)

    scale = 6

    pattern_order = [
        'Seasonal only',
        'Acute (rapid/oscillating)',
        'Prolonged/entrenched',
        'Escalating',
        'Protracted emergency',
    ]
    pattern_colors = {
        'Protracted emergency': '#c0392b',
        'Escalating': '#e67e22',
        'Prolonged/entrenched': '#8e44ad',
        'Acute (rapid/oscillating)': '#27ae60',
        'Seasonal only': '#2980b9',
    }

    # =========================================================
    # Create figure: two panels stacked vertically
    # =========================================================
    fig_width = 7.09  # 180mm
    map_width = fig_width * 0.93
    data_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    panel_height = map_width * data_aspect
    fig_height = panel_height * 2 + 0.55

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1, 0.03],
                  height_ratios=[1, 1],
                  hspace=0.10, wspace=0.02)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.set_visible(False)

    # --- Panel a: Trajectory pattern ---
    world.plot(ax=ax1, color='#f5f5f5', edgecolor='#d0d0d0', linewidth=0.3)

    for pattern in pattern_order:
        subset = gdf[gdf['pattern'] == pattern]
        if len(subset) > 0:
            ax1.scatter(
                subset.geometry.x, subset.geometry.y,
                s=subset['total_episodes'] * scale,
                c=pattern_colors[pattern], alpha=0.75,
                edgecolors='black', linewidth=0.2,
                zorder=pattern_order.index(pattern) + 3
            )

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('')
    ax1.set_ylabel('Latitude', fontsize=8)
    ax1.text(-0.02, 1.02, 'a', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='bottom')

    # Pattern legend
    pattern_handles = []
    for pattern in pattern_order:
        n = len(gdf[gdf['pattern'] == pattern])
        pattern_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=pattern_colors[pattern],
                   markeredgecolor='black', markeredgewidth=0.3,
                   markersize=6, linestyle='None',
                   label=f'{pattern} (n={n})')
        )
    leg1 = ax1.legend(handles=pattern_handles, loc='lower left', fontsize=6.5,
                      framealpha=0.95, title='Worst trajectory reached',
                      title_fontsize=7, borderpad=0.5, handletextpad=0.4)
    ax1.add_artist(leg1)

    size_legend(ax1, [3*scale, 7*scale, 14*scale], ['3', '7', '14'],
                loc='upper left', title='Episodes')

    ax1.tick_params(labelbottom=False)

    # --- Panel b: Gap duration ---
    world.plot(ax=ax2, color='#f5f5f5', edgecolor='#d0d0d0', linewidth=0.3)

    gap_data = gdf.dropna(subset=['mean_gap_months'])
    sc = ax2.scatter(
        gap_data.geometry.x, gap_data.geometry.y,
        s=gap_data['total_episodes'] * scale,
        c=gap_data['mean_gap_months'],
        cmap='RdYlGn', vmin=0, vmax=36,
        alpha=0.85, edgecolors='black', linewidth=0.2,
        zorder=5
    )

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Mean gap between episodes (months)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('Longitude', fontsize=8)
    ax2.set_ylabel('Latitude', fontsize=8)
    ax2.text(-0.02, 1.02, 'b', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='bottom')

    size_legend(ax2, [3*scale, 7*scale, 14*scale], ['3', '7', '14'],
                loc='upper left', title='Episodes')

    # Save
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_png = os.path.join(FIGURES_DIR, 'ExtData_gap_compression_map.png')
    out_pdf = os.path.join(FIGURES_DIR, 'ExtData_gap_compression_map.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close()

    # Caption statistics
    n_locations = len(matched)
    n_countries = matched['iso3'].nunique()
    short = gap_data[gap_data['mean_gap_months'] <= 6]
    n_short = len(short)
    pct_short = 100 * n_short / len(gap_data) if len(gap_data) > 0 else 0
    median_gap = gap_data['mean_gap_months'].median()

    pattern_counts = {}
    for pattern in pattern_order:
        n = len(matched[matched['pattern'] == pattern])
        pattern_counts[pattern] = n

    print(f"\n  Caption statistics:")
    print(f"  - {n_locations} admin1 locations with >=3 episodes "
          f"across {n_countries} countries")
    print(f"  - {n_short} locations ({pct_short:.0f}%) with mean gap "
          f"<=6 months")
    print(f"  - Median gap: {median_gap:.1f} months")
    for pattern in pattern_order:
        n = pattern_counts[pattern]
        print(f"  - {pattern}: {n} ({100*n/n_locations:.1f}%)")


# ============================================================
# Main
# ============================================================

def main():
    """Generate gap compression map extended data figure."""
    if not HAS_GEOPANDAS:
        print("geopandas is not installed. Skipping 14_extdata_gap_map.py.")
        print("To generate this figure, install geopandas:")
        print("  pip install geopandas")
        sys.exit(0)

    print("=" * 60)
    print("Generating Extended Data: Gap Compression Map")
    print("=" * 60)

    create_figure()

    print("=" * 60)
    print(f"Figure saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
