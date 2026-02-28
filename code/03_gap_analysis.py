#!/usr/bin/env python3
"""
03_gap_analysis.py — Comprehensive Gap Analysis Between Crisis Episodes
========================================================================

Analyzes the periods between consecutive food security crisis episodes to
understand recovery patterns, vulnerability accumulation, and escalation risk.

Twelve analysis sections:
  1. Gap duration distribution
  2. Gap duration and escalation risk
  3. Gap patterns by from-archetype
  4. Gap patterns by to-archetype
  5. Cumulative gap patterns by location
  6. Rapid cycling locations (high vulnerability)
  7. Gap duration and next episode severity
  8. Gap duration and next episode duration
  9. Country-level gap patterns
 10. Gap compression over time
 11. Specific transition gap patterns
 12. Summary findings

Inputs (all relative to package root):
    outputs/data/archetype_transitions.csv
        Produced by 02_generate_transitions.py

Outputs (all relative to package root):
    outputs/data/location_gap_patterns.csv
    outputs/data/country_gap_patterns.csv
    outputs/data/escalation_by_gap_duration.csv
    outputs/data/rapid_cycling_locations.csv   (if any qualify)

Dependencies: pandas, numpy

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# Paths — relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSITIONS_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'archetype_transitions.csv')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load transition data
    df = pd.read_csv(TRANSITIONS_PATH)

    print("=" * 70)
    print("COMPREHENSIVE GAP ANALYSIS BETWEEN CRISIS EPISODES")
    print("=" * 70)
    print(f"\nTotal transitions analyzed: {len(df)}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Countries: {df['iso3'].nunique()}")

    # =================================================================
    # 1. GAP DURATION DISTRIBUTION
    # =================================================================
    print("\n" + "=" * 70)
    print("1. GAP DURATION DISTRIBUTION")
    print("=" * 70)

    gap_stats = df['gap_months'].describe()
    print(f"\nGap Duration Statistics (months):")
    print(f"  Mean:   {gap_stats['mean']:.1f}")
    print(f"  Median: {gap_stats['50%']:.1f}")
    print(f"  Std:    {gap_stats['std']:.1f}")
    print(f"  Min:    {gap_stats['min']:.0f}")
    print(f"  Max:    {gap_stats['max']:.0f}")

    # Gap duration categories
    gap_bins = [0, 3, 6, 12, 24, float('inf')]
    gap_labels = ['Very short (0-3)', 'Short (4-6)', 'Medium (7-12)',
                  'Long (13-24)', 'Very long (24+)']
    df['gap_category'] = pd.cut(df['gap_months'], bins=gap_bins,
                                labels=gap_labels, include_lowest=True)

    gap_dist = df['gap_category'].value_counts().sort_index()
    print(f"\nGap Duration Distribution:")
    for cat, count in gap_dist.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # =================================================================
    # 2. GAP DURATION AND ESCALATION RISK
    # =================================================================
    print("\n" + "=" * 70)
    print("2. GAP DURATION AND ESCALATION RISK")
    print("=" * 70)

    escalation_by_gap = df.groupby('gap_category', observed=True).agg({
        'severity_change': ['mean', 'count',
                            lambda x: (x > 0).sum(),
                            lambda x: (x < 0).sum()]
    }).round(3)
    escalation_by_gap.columns = ['mean_severity_change', 'n_transitions',
                                 'escalations', 'de_escalations']
    escalation_by_gap['escalation_rate'] = (
        escalation_by_gap['escalations'] / escalation_by_gap['n_transitions'])
    escalation_by_gap['de_escalation_rate'] = (
        escalation_by_gap['de_escalations'] / escalation_by_gap['n_transitions'])

    print("\nEscalation Risk by Gap Duration:")
    print(escalation_by_gap.to_string())

    short_gaps = df[df['gap_months'] <= 6]['severity_change']
    long_gaps = df[df['gap_months'] > 6]['severity_change']
    print(f"\nShort gaps (<=6 months): mean severity change = {short_gaps.mean():.3f}")
    print(f"Long gaps (>6 months): mean severity change = {long_gaps.mean():.3f}")

    # =================================================================
    # 3. GAP PATTERNS BY FROM-ARCHETYPE
    # =================================================================
    print("\n" + "=" * 70)
    print("3. GAP PATTERNS BY ARCHETYPE (EXITING FROM)")
    print("=" * 70)

    archetype_gaps = df.groupby('from_archetype').agg({
        'gap_months': ['mean', 'median', 'std', 'count']
    }).round(1)
    archetype_gaps.columns = ['mean_gap', 'median_gap', 'std_gap', 'n_transitions']
    archetype_gaps = archetype_gaps.sort_values('mean_gap')

    print("\nMean Gap Duration by Exiting Archetype:")
    print(archetype_gaps.to_string())

    # =================================================================
    # 4. GAP PATTERNS BY TO-ARCHETYPE
    # =================================================================
    print("\n" + "=" * 70)
    print("4. GAP PATTERNS BY ARCHETYPE (ENTERING INTO)")
    print("=" * 70)

    to_archetype_gaps = df.groupby('to_archetype').agg({
        'gap_months': ['mean', 'median', 'count']
    }).round(1)
    to_archetype_gaps.columns = ['mean_gap', 'median_gap', 'n_transitions']
    to_archetype_gaps = to_archetype_gaps.sort_values('mean_gap')

    print("\nMean Gap Duration Before Entering Archetype:")
    print(to_archetype_gaps.to_string())

    # =================================================================
    # 5. CUMULATIVE GAP ANALYSIS BY LOCATION
    # =================================================================
    print("\n" + "=" * 70)
    print("5. CUMULATIVE GAP PATTERNS BY LOCATION")
    print("=" * 70)

    location_gaps = df.groupby('location').agg({
        'gap_months': ['mean', 'min', 'max', 'sum', 'count'],
        'severity_change': 'sum',
        'iso3': 'first'
    }).round(1)
    location_gaps.columns = ['mean_gap', 'min_gap', 'max_gap',
                             'total_gap_months', 'n_transitions',
                             'cumulative_severity_change', 'iso3']

    # Categorize locations by gap patterns
    location_gaps['gap_pattern'] = pd.cut(
        location_gaps['mean_gap'],
        bins=[0, 4, 8, 15, float('inf')],
        labels=['Rapid cycling (<4mo)', 'Frequent (4-8mo)',
                'Moderate (8-15mo)', 'Infrequent (>15mo)']
    )

    gap_pattern_dist = location_gaps['gap_pattern'].value_counts()
    print("\nLocation Gap Pattern Distribution:")
    for pattern, count in gap_pattern_dist.items():
        pct = count / len(location_gaps) * 100
        print(f"  {pattern}: {count} locations ({pct:.1f}%)")

    # Relationship between gap pattern and cumulative severity change
    gap_severity = location_gaps.groupby('gap_pattern', observed=True).agg({
        'cumulative_severity_change': 'mean',
        'n_transitions': ['mean', 'sum']
    }).round(2)
    gap_severity.columns = ['mean_cumulative_severity', 'mean_transitions',
                            'total_transitions']
    gap_severity['n_locations'] = location_gaps.groupby(
        'gap_pattern', observed=True).size()

    print("\nCumulative Severity Change by Location Gap Pattern:")
    print(gap_severity.to_string())

    # =================================================================
    # 6. RAPID CYCLING LOCATIONS
    # =================================================================
    print("\n" + "=" * 70)
    print("6. RAPID CYCLING LOCATIONS (Potential High Vulnerability)")
    print("=" * 70)

    rapid_cyclers = location_gaps[
        (location_gaps['mean_gap'] < 4) &
        (location_gaps['n_transitions'] >= 3)
    ].sort_values('mean_gap')

    print(f"\nLocations with rapid cycling (mean gap < 4 months, 3+ transitions):")
    print(f"Total: {len(rapid_cyclers)} locations")
    if len(rapid_cyclers) > 0:
        print("\nTop 15 rapid cycling locations:")
        display_cols = ['iso3', 'mean_gap', 'n_transitions',
                        'cumulative_severity_change']
        print(rapid_cyclers[display_cols].head(15).to_string())

    # =================================================================
    # 7. GAP DURATION AND NEXT EPISODE SEVERITY
    # =================================================================
    print("\n" + "=" * 70)
    print("7. GAP DURATION AND NEXT EPISODE PEAK SEVERITY")
    print("=" * 70)

    gap_severity_corr = df[['gap_months', 'to_peak']].corr().iloc[0, 1]
    print(f"\nCorrelation between gap duration and next episode peak: "
          f"{gap_severity_corr:.3f}")

    peak_by_gap = df.groupby('gap_category', observed=True)['to_peak'].agg(
        ['mean', 'std', 'count']).round(2)
    print("\nMean Next Episode Peak Severity by Gap Duration:")
    print(peak_by_gap.to_string())

    # =================================================================
    # 8. GAP DURATION AND NEXT EPISODE DURATION
    # =================================================================
    print("\n" + "=" * 70)
    print("8. GAP DURATION AND NEXT EPISODE DURATION")
    print("=" * 70)

    gap_duration_corr = df[['gap_months', 'to_duration']].corr().iloc[0, 1]
    print(f"\nCorrelation between gap duration and next episode duration: "
          f"{gap_duration_corr:.3f}")

    duration_by_gap = df.groupby('gap_category', observed=True)['to_duration'].agg(
        ['mean', 'std', 'count']).round(1)
    print("\nMean Next Episode Duration (months) by Gap Duration:")
    print(duration_by_gap.to_string())

    # =================================================================
    # 9. COUNTRY-LEVEL GAP PATTERNS
    # =================================================================
    print("\n" + "=" * 70)
    print("9. COUNTRY-LEVEL GAP PATTERNS")
    print("=" * 70)

    country_gaps = df.groupby('iso3').agg({
        'gap_months': ['mean', 'median', 'count'],
        'severity_change': 'mean',
        'location': 'nunique'
    }).round(2)
    country_gaps.columns = ['mean_gap', 'median_gap', 'n_transitions',
                            'mean_severity_change', 'n_locations']
    country_gaps = country_gaps.sort_values('mean_gap')

    print("\nCountries by Mean Gap Duration (shortest = most frequent crises):")
    print(country_gaps.head(15).to_string())

    print("\nCountries by Mean Gap Duration (longest = least frequent crises):")
    print(country_gaps.tail(15).to_string())

    # =================================================================
    # 10. GAP COMPRESSION OVER TIME
    # =================================================================
    print("\n" + "=" * 70)
    print("10. GAP COMPRESSION OVER TIME (Are gaps getting shorter?)")
    print("=" * 70)

    df['from_start_date'] = pd.to_datetime(df['from_start'])
    df['year'] = df['from_start_date'].dt.year

    yearly_gaps = df.groupby('year').agg({
        'gap_months': ['mean', 'median', 'count']
    }).round(1)
    yearly_gaps.columns = ['mean_gap', 'median_gap', 'n_transitions']

    print("\nGap Duration Trends Over Time:")
    print(yearly_gaps.to_string())

    # =================================================================
    # 11. SPECIFIC TRANSITION GAP PATTERNS
    # =================================================================
    print("\n" + "=" * 70)
    print("11. GAP PATTERNS FOR SPECIFIC TRANSITION TYPES")
    print("=" * 70)

    key_transitions = [
        ('seasonal_crisis', 'seasonal_crisis'),
        ('seasonal_crisis', 'prolonged_moderate'),
        ('seasonal_crisis', 'protracted_emergency'),
        ('prolonged_moderate', 'protracted_emergency'),
        ('prolonged_moderate', 'seasonal_crisis'),
    ]

    print("\nGap Statistics for Key Transition Types:")
    for from_arch, to_arch in key_transitions:
        subset = df[(df['from_archetype'] == from_arch) &
                    (df['to_archetype'] == to_arch)]
        if len(subset) > 0:
            print(f"\n  {from_arch} -> {to_arch} (n={len(subset)}):")
            print(f"    Mean gap: {subset['gap_months'].mean():.1f} months")
            print(f"    Median gap: {subset['gap_months'].median():.1f} months")
            print(f"    Range: {subset['gap_months'].min():.0f} - "
                  f"{subset['gap_months'].max():.0f} months")

    # =================================================================
    # 12. SAVE OUTPUTS
    # =================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Save location-level gap analysis
    loc_path = os.path.join(OUTPUT_DIR, 'location_gap_patterns.csv')
    location_gaps.to_csv(loc_path)
    print(f"Saved: {loc_path}")

    # Save country-level gap analysis
    country_path = os.path.join(OUTPUT_DIR, 'country_gap_patterns.csv')
    country_gaps.to_csv(country_path)
    print(f"Saved: {country_path}")

    # Save escalation by gap category
    esc_path = os.path.join(OUTPUT_DIR, 'escalation_by_gap_duration.csv')
    escalation_by_gap.to_csv(esc_path)
    print(f"Saved: {esc_path}")

    # Save rapid cyclers list
    if len(rapid_cyclers) > 0:
        rapid_path = os.path.join(OUTPUT_DIR, 'rapid_cycling_locations.csv')
        rapid_cyclers.to_csv(rapid_path)
        print(f"Saved: {rapid_path}")

    # =================================================================
    # 13. SUMMARY FINDINGS
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS ON GAP PATTERNS")
    print("=" * 70)

    print(f"""
1. OVERALL GAP DISTRIBUTION:
   - Mean gap between crises: {gap_stats['mean']:.1f} months
   - Median gap: {gap_stats['50%']:.1f} months
   - {(df['gap_months'] <= 6).sum()} transitions ({(df['gap_months'] <= 6).mean()*100:.1f}%) have gaps <=6 months

2. GAP DURATION AND ESCALATION:
   - Short gaps (<=6 mo) mean severity change: {short_gaps.mean():.3f}
   - Long gaps (>6 mo) mean severity change: {long_gaps.mean():.3f}
   - {'Shorter gaps associated with HIGHER escalation' if short_gaps.mean() > long_gaps.mean() else 'Gap duration shows weak relationship with escalation'}

3. RAPID CYCLING LOCATIONS:
   - {len(rapid_cyclers)} locations cycle rapidly (mean gap <4 mo, 3+ episodes)
   - These represent highest vulnerability accumulation risk

4. NEXT EPISODE CHARACTERISTICS:
   - Gap-to-peak correlation: {gap_severity_corr:.3f} ({'negative' if gap_severity_corr < 0 else 'positive'})
   - Gap-to-duration correlation: {gap_duration_corr:.3f} ({'negative' if gap_duration_corr < 0 else 'positive'})

5. COUNTRY PATTERNS:
   - Shortest mean gaps: {', '.join(country_gaps.head(3).index.tolist())}
   - Longest mean gaps: {', '.join(country_gaps.tail(3).index.tolist())}
""")

    print("\nAnalysis complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
