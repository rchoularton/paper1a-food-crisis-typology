#!/usr/bin/env python3
"""
13_extdata_gap_compression.py -- Figure 4: Gap Compression Dual-Panel
=======================================================================

Generates Figure 4 of the paper: a dual-panel figure showing
  (a) Gap compression time trend  (b) IPC phase escalation rate by gap bin.

Inputs (relative to package root):
    outputs/data/archetype_transitions.csv
        Produced by 02_generate_transitions.py

Outputs (relative to package root):
    outputs/figures/Figure4_gap_compression.png            (300 dpi)
    outputs/figures/Figure4_gap_compression.pdf
    outputs/figures/SourceData_Fig4.xlsx                   (source data)

Dependencies: pandas, numpy, matplotlib, openpyxl

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# Figure 4: Dual panel (a) time trend  (b) escalation risk
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
    filepath_png = os.path.join(FIGURES_DIR, 'Figure4_gap_compression.png')
    filepath_pdf = os.path.join(FIGURES_DIR, 'Figure4_gap_compression.pdf')
    fig.savefig(filepath_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(filepath_pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")
    plt.close()


# ============================================================
# Main
# ============================================================

def export_source_data(df, yearly_gaps):
    """Export source data for Figure 4 to Excel."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    xlsx_path = os.path.join(FIGURES_DIR, 'SourceData_Fig4.xlsx')

    # Panel a: yearly gap statistics
    yearly_export = yearly_gaps.copy()

    # Panel b: escalation risk by gap bin
    gap_bins = [0, 3, 6, 12, 24, float('inf')]
    gap_labels = ['0\u20133', '4\u20136', '7\u201312', '13\u201324', '24+']
    df_copy = df.copy()
    df_copy['gap_category'] = pd.cut(df_copy['gap_months'], bins=gap_bins,
                                      labels=gap_labels, include_lowest=True)
    escalation = df_copy.groupby('gap_category', observed=True).agg(
        n_transitions=('gap_months', 'count'),
        n_escalated=('severity_change', lambda x: (x > 0).sum()),
        escalation_rate=('severity_change', lambda x: (x > 0).mean() * 100),
    ).round(2).reset_index()

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        yearly_export.to_excel(writer, sheet_name='panel_a_yearly_gaps', index=False)
        escalation.to_excel(writer, sheet_name='panel_b_escalation_risk', index=False)
    print(f"Saved: {xlsx_path}")


def main():
    """Generate Figure 4: Gap Compression dual-panel."""
    print("=" * 60)
    print("Generating Figure 4: Gap Compression Dual-Panel")
    print("=" * 60)

    df, yearly_gaps = load_data()

    figure_gap_compression_dual(df, yearly_gaps)
    export_source_data(df, yearly_gaps)

    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
