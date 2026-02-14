#!/usr/bin/env python3
"""
13_extdata_gap_compression.py -- Extended Data Figure: Gap Compression Over Time
=================================================================================

Generates three visualisations of how recovery windows between consecutive
food security crisis episodes have compressed:

  1. Standalone mean/median gap trend with IQR band (overview figure).
  2. Dual-panel figure:
     (a) Gap compression time trend  (b) IPC phase escalation rate by gap bin.
  3. Robustness panel: all locations vs consistent locations (monitored in
     both early [2011-2015] AND late [2020-2023] periods).

Algorithm is identical to the source script
``papers/paper1a/figures/Figure_gap_compression.py``; only the data-loading
paths and output paths have been changed to read from the reproducibility-
package ``outputs/data/`` directory.  Docx generation is removed.

Inputs (relative to package root):
    outputs/data/archetype_transitions.csv
        Produced by 02_generate_transitions.py

Outputs (relative to package root):
    outputs/figures/ExtData_gap_compression.png          (300 dpi)
    outputs/figures/ExtData_gap_compression.pdf
    outputs/figures/ExtData_gap_compression_dual.png      (300 dpi)
    outputs/figures/ExtData_gap_compression_dual.pdf
    outputs/figures/ExtData_gap_compression_consistent.png (300 dpi)
    outputs/figures/ExtData_gap_compression_consistent.pdf

Dependencies: pandas, numpy, matplotlib

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ============================================================
# Paths -- relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSITIONS_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'archetype_transitions.csv')
FIGURES_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ============================================================
# Nature Food style rcParams
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
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# Tol bright palette (consistent with main figures)
COL_MEAN = '#EE6677'     # red
COL_MEDIAN = '#4477AA'   # blue
COL_IQR = '#A8DADC'      # light teal for IQR band
COL_BAR = '#4477AA'      # blue (Tol bright)


# ============================================================
# Load data and compute yearly gap statistics
# ============================================================

def load_data():
    """Load transitions and compute yearly gap aggregates."""
    df = pd.read_csv(TRANSITIONS_PATH)

    # Convert dates and extract year
    df['from_start_date'] = pd.to_datetime(df['from_start'])
    df['year'] = df['from_start_date'].dt.year

    # Calculate yearly gap statistics
    yearly_gaps = df.groupby('year').agg({
        'gap_months': ['mean', 'median', 'std', 'count',
                       lambda x: x.quantile(0.25),
                       lambda x: x.quantile(0.75)]
    }).round(1)
    yearly_gaps.columns = ['mean_gap', 'median_gap', 'std_gap',
                           'n_transitions', 'q25', 'q75']
    yearly_gaps = yearly_gaps.reset_index()

    # Filter to years with sufficient data (exclude years with <10 observations)
    yearly_gaps = yearly_gaps[yearly_gaps['n_transitions'] >= 10]

    print("Yearly gap statistics:")
    print(yearly_gaps.to_string())

    return df, yearly_gaps


# ============================================================
# Figure 1: Standalone gap compression trend
# ============================================================

def figure_gap_compression(df, yearly_gaps):
    """Main figure showing gap compression over time."""

    fig, ax = plt.subplots(figsize=(10, 6))

    years = yearly_gaps['year'].values
    mean_gaps = yearly_gaps['mean_gap'].values
    median_gaps = yearly_gaps['median_gap'].values
    q25 = yearly_gaps['q25'].values
    q75 = yearly_gaps['q75'].values
    n_trans = yearly_gaps['n_transitions'].values

    color_mean = '#E63946'
    color_median = '#1D3557'
    color_band = '#A8DADC'

    # Plot IQR band
    ax.fill_between(years, q25, q75, alpha=0.3, color=color_band,
                    label='Interquartile range')

    # Plot mean and median lines
    ax.plot(years, mean_gaps, 'o-', color=color_mean, linewidth=2.5,
            markersize=8, label='Mean gap')
    ax.plot(years, median_gaps, 's--', color=color_median, linewidth=2.5,
            markersize=7, label='Median gap')

    # Reference lines
    ax.axhline(y=12, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.text(2023.5, 12.5, '1 year', fontsize=9, color='gray', ha='left')
    ax.axhline(y=6, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.text(2023.5, 6.5, '6 months', fontsize=9, color='gray', ha='left')

    # Start annotation
    ax.annotate(f'{mean_gaps[0]:.0f} mo',
                xy=(years[0], mean_gaps[0]),
                xytext=(years[0]-0.3, mean_gaps[0]+8),
                fontsize=10, fontweight='bold', color=color_mean, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_mean, lw=1.5))

    # End annotation
    last_idx = len(years) - 1
    ax.annotate(f'{mean_gaps[last_idx]:.1f} mo',
                xy=(years[last_idx], mean_gaps[last_idx]),
                xytext=(years[last_idx]+0.3, mean_gaps[last_idx]+8),
                fontsize=10, fontweight='bold', color=color_mean, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_mean, lw=1.5))

    # Sample sizes
    for i, (year, n) in enumerate(zip(years, n_trans)):
        if i % 2 == 0:
            ax.text(year, -3, f'n={n}', fontsize=8, ha='center',
                    color='gray', alpha=0.8)

    ax.set_xlabel('Year of Transition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap Duration (months)', fontsize=12, fontweight='bold')
    ax.set_title('Recovery Windows Between Crises Are Compressing',
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(2010.5, 2024)
    ax.set_ylim(-5, 55)
    ax.set_xticks(range(2011, 2024, 2))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Summary text box (data-driven)
    first_gap = mean_gaps[0]
    last_gap = mean_gaps[len(years) - 1]
    fold = first_gap / last_gap
    textstr = (f'Gap compression:\n{first_gap:.0f} months \u2192 '
               f'{last_gap:.0f} months\n({fold:.0f}x reduction)')
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD',
                 edgecolor='#856404', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-')

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath_png = os.path.join(FIGURES_DIR, 'ExtData_gap_compression.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'ExtData_gap_compression.pdf')
    fig.savefig(filepath_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(filepath_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")
    plt.close()


# ============================================================
# Figure 2: Dual panel (a) time trend  (b) escalation risk
# ============================================================

def figure_gap_compression_dual(df, yearly_gaps):
    """Dual panel: (a) time trend, (b) escalation risk by gap duration."""

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # === Panel A: Gap compression over time ===
    ax = axes[0]

    years = yearly_gaps['year'].values
    mean_gaps = yearly_gaps['mean_gap'].values
    median_gaps = yearly_gaps['median_gap'].values
    q25 = yearly_gaps['q25'].values
    q75 = yearly_gaps['q75'].values

    ax.fill_between(years, q25, q75, alpha=0.15, color=COL_IQR,
                    label='Interquartile range')
    ax.plot(years, mean_gaps, 'o-', color=COL_MEAN, linewidth=1.5,
            markersize=4, label='Mean gap')
    ax.plot(years, median_gaps, 's--', color=COL_MEDIAN, linewidth=1.5,
            markersize=3.5, label='Median gap')

    ax.axhline(y=12, color='gray', linestyle=':', alpha=0.5, linewidth=0.6)
    ax.text(2023.8, 12.5, '12 mo', fontsize=6, color='gray', ha='left')
    ax.axhline(y=6, color='gray', linestyle=':', alpha=0.5, linewidth=0.6)
    ax.text(2023.8, 6.5, '6 mo', fontsize=6, color='gray', ha='left')

    ax.annotate(f'{mean_gaps[0]:.0f} mo',
                xy=(years[0], mean_gaps[0]),
                xytext=(2012.2, 57),
                fontsize=6.5, color='#333333', ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.7))

    last_idx = len(years) - 1
    ax.annotate(f'{mean_gaps[last_idx]:.1f} mo',
                xy=(years[last_idx], mean_gaps[last_idx]),
                xytext=(2021.2, 10),
                fontsize=6.5, color='#333333', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.7))

    ax.set_xlabel('Year')
    ax.set_ylabel('Gap duration (months)')
    ax.set_title('a', loc='left', fontweight='bold', fontsize=8, x=-0.12)

    ax.set_xlim(2010.5, 2024.5)
    ax.set_ylim(0, 60)
    ax.set_xticks(range(2011, 2024, 2))
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # === Panel B: Escalation risk by gap duration ===
    ax = axes[1]

    gap_bins = [0, 3, 6, 12, 24, float('inf')]
    gap_labels = ['0\u20133', '4\u20136', '7\u201312', '13\u201324', '24+']
    df['gap_category'] = pd.cut(df['gap_months'], bins=gap_bins,
                                labels=gap_labels, include_lowest=True)

    escalation_data = df.groupby('gap_category', observed=True).agg({
        'severity_change': [lambda x: (x > 0).mean() * 100, 'count',
                            lambda x: (x > 0).sum()]
    })
    escalation_data.columns = ['escalation_rate', 'n_transitions', 'n_escalated']
    escalation_data = escalation_data.reset_index()

    # Wilson confidence intervals (95%)
    z = 1.96
    ci_lo_list, ci_hi_list = [], []
    for _, row in escalation_data.iterrows():
        n = row['n_transitions']
        p = row['escalation_rate'] / 100
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        ci_lo_list.append(max(0, centre - margin) * 100)
        ci_hi_list.append(min(1, centre + margin) * 100)
    escalation_data['ci_lo'] = ci_lo_list
    escalation_data['ci_hi'] = ci_hi_list

    bars = ax.bar(escalation_data['gap_category'],
                  escalation_data['escalation_rate'],
                  color=COL_BAR, edgecolor='none', width=0.65)

    # Error bars (Wilson 95% CI)
    rates = escalation_data['escalation_rate'].values
    yerr_lo = rates - escalation_data['ci_lo'].values
    yerr_hi = escalation_data['ci_hi'].values - rates
    x_positions = [bar.get_x() + bar.get_width() / 2. for bar in bars]
    ax.errorbar(x_positions, rates, yerr=[yerr_lo, yerr_hi],
                fmt='none', ecolor='#333333', elinewidth=0.8,
                capsize=2, capthick=0.8)

    # Value labels above bars
    for bar, rate in zip(bars, rates):
        ci_hi = escalation_data.loc[
            escalation_data['escalation_rate'] == rate, 'ci_hi'].values[0]
        ax.text(bar.get_x() + bar.get_width() / 2., ci_hi + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=7)

    # Sample sizes inside bars
    for bar, n in zip(bars, escalation_data['n_transitions']):
        ax.text(bar.get_x() + bar.get_width() / 2., 0.5,
                f'n={n}', ha='center', va='bottom', fontsize=6, color='white')

    # Annotation: 0-3 month bin caveat
    short_same_pct = df[df['gap_months'] <= 3]['same_archetype'].mean() * 100
    ax.text(x_positions[0], -0.12,
            f'({short_same_pct:.0f}% seasonal\ncycling)',
            ha='center', va='top', fontsize=5.5, color='#888888',
            style='italic', transform=ax.get_xaxis_transform())

    ax.set_xlabel('Gap duration (months)')
    ax.set_ylabel('IPC phase escalation rate (%)')
    ax.set_title('b', loc='left', fontweight='bold', fontsize=8, x=-0.12)

    ax.set_ylim(0, max(escalation_data['ci_hi']) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.22, top=0.90,
                        wspace=0.35)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath_png = os.path.join(FIGURES_DIR, 'ExtData_gap_compression_dual.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'ExtData_gap_compression_dual.pdf')
    fig.savefig(filepath_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(filepath_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")
    plt.close()


# ============================================================
# Figure 3: Consistent-locations robustness check
# ============================================================

def figure_gap_compression_consistent(df, yearly_gaps):
    """Robustness: gap compression for consistent locations only."""

    # Find consistent locations (monitored in both early AND late periods)
    early_locations = set(df[df['year'] <= 2015]['location'].unique())
    late_locations = set(df[df['year'] >= 2020]['location'].unique())
    consistent_locations = early_locations & late_locations

    print(f"\nConsistent locations (monitored early AND late): "
          f"{len(consistent_locations)}")

    df_consistent = df[df['location'].isin(consistent_locations)]

    consistent_yearly = df_consistent.groupby('year').agg({
        'gap_months': ['mean', 'median', 'count',
                       lambda x: x.quantile(0.25),
                       lambda x: x.quantile(0.75)]
    }).round(1)
    consistent_yearly.columns = ['mean_gap', 'median_gap', 'n_transitions',
                                 'q25', 'q75']
    consistent_yearly = consistent_yearly.reset_index()
    consistent_yearly = consistent_yearly[consistent_yearly['n_transitions'] >= 5]

    all_yearly = yearly_gaps.copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # === Panel A: All locations ===
    ax = axes[0]

    years_all = all_yearly['year'].values
    mean_all = all_yearly['mean_gap'].values
    median_all = all_yearly['median_gap'].values
    q25_all = all_yearly['q25'].values
    q75_all = all_yearly['q75'].values

    color_mean = '#E63946'
    color_median = '#1D3557'
    color_band = '#A8DADC'

    ax.fill_between(years_all, q25_all, q75_all, alpha=0.3, color=color_band,
                    label='Interquartile range')
    ax.plot(years_all, mean_all, 'o-', color=color_mean, linewidth=2.5,
            markersize=8, label='Mean gap')
    ax.plot(years_all, median_all, 's--', color=color_median, linewidth=2.5,
            markersize=7, label='Median gap')

    ax.axhline(y=12, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.axhline(y=6, color='gray', linestyle=':', alpha=0.7, linewidth=1)

    ax.annotate(f'{mean_all[0]:.0f} mo',
                xy=(years_all[0], mean_all[0]),
                xytext=(years_all[0]+0.5, mean_all[0]+6),
                fontsize=10, fontweight='bold', color=color_mean, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_mean, lw=1.5))

    last_idx = len(years_all) - 1
    ax.annotate(f'{mean_all[last_idx]:.1f} mo',
                xy=(years_all[last_idx], mean_all[last_idx]),
                xytext=(years_all[last_idx]-0.5, mean_all[last_idx]+8),
                fontsize=10, fontweight='bold', color=color_mean, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_mean, lw=1.5))

    ax.set_xlabel('Year of Transition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gap Duration (months)', fontsize=11, fontweight='bold')
    n_all_locs = df['location'].nunique()
    ax.set_title(f'(A) All Locations (n={n_all_locs})',
                 fontsize=12, fontweight='bold', pad=10)

    ax.set_xlim(2010.5, 2024)
    ax.set_ylim(-2, 65)
    ax.set_xticks(range(2011, 2024, 2))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    first_all = mean_all[0]
    last_all = mean_all[len(years_all) - 1]
    fold_all = first_all / last_all
    textstr = (f'All locations\n{first_all:.0f} \u2192 {last_all:.0f} months'
               f'\n({fold_all:.0f}x reduction)')
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD',
                 edgecolor='#856404', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')

    # === Panel B: Consistent locations only ===
    ax = axes[1]

    years_con = consistent_yearly['year'].values
    mean_con = consistent_yearly['mean_gap'].values
    median_con = consistent_yearly['median_gap'].values
    q25_con = consistent_yearly['q25'].values
    q75_con = consistent_yearly['q75'].values

    ax.fill_between(years_con, q25_con, q75_con, alpha=0.3, color=color_band,
                    label='Interquartile range')
    ax.plot(years_con, mean_con, 'o-', color=color_mean, linewidth=2.5,
            markersize=8, label='Mean gap')
    ax.plot(years_con, median_con, 's--', color=color_median, linewidth=2.5,
            markersize=7, label='Median gap')

    ax.axhline(y=12, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.axhline(y=6, color='gray', linestyle=':', alpha=0.7, linewidth=1)

    ax.annotate(f'{mean_con[0]:.0f} mo',
                xy=(years_con[0], mean_con[0]),
                xytext=(years_con[0]+0.5, mean_con[0]+6),
                fontsize=10, fontweight='bold', color=color_mean, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_mean, lw=1.5))

    last_idx = len(years_con) - 1
    ax.annotate(f'{mean_con[last_idx]:.1f} mo',
                xy=(years_con[last_idx], mean_con[last_idx]),
                xytext=(years_con[last_idx]-0.5, mean_con[last_idx]+8),
                fontsize=10, fontweight='bold', color=color_mean, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_mean, lw=1.5))

    ax.set_xlabel('Year of Transition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gap Duration (months)', fontsize=11, fontweight='bold')
    ax.set_title(f'(B) Consistent Locations Only (n={len(consistent_locations)})',
                 fontsize=12, fontweight='bold', pad=10)

    ax.set_xlim(2010.5, 2024)
    ax.set_ylim(-2, 65)
    ax.set_xticks(range(2011, 2024, 2))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    first_con = mean_con[0]
    last_con = mean_con[len(years_con) - 1]
    fold_con = first_con / last_con
    textstr = (f'Same locations\n{first_con:.0f} \u2192 {last_con:.0f} months'
               f'\n({fold_con:.0f}x reduction)')
    props = dict(boxstyle='round,pad=0.5', facecolor='#D4EDDA',
                 edgecolor='#155724', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')

    fig.suptitle('Gap Compression: Robustness Check', fontsize=14,
                 fontweight='bold', y=1.02)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath_png = os.path.join(FIGURES_DIR,
                                'ExtData_gap_compression_consistent.png')
    filepath_pdf = os.path.join(FIGURES_DIR,
                                'ExtData_gap_compression_consistent.pdf')
    fig.savefig(filepath_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(filepath_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    """Generate all gap compression extended data figures."""
    print("=" * 60)
    print("Generating Extended Data: Gap Compression Figures")
    print("=" * 60)

    df, yearly_gaps = load_data()

    figure_gap_compression(df, yearly_gaps)
    figure_gap_compression_dual(df, yearly_gaps)
    figure_gap_compression_consistent(df, yearly_gaps)

    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
