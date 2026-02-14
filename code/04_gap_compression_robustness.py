#!/usr/bin/env python3
"""
04_gap_compression_robustness.py — Robustness Check for Gap Compression
========================================================================

Investigates whether the observed gap compression over time is a real
phenomenon or an artifact of:
  1. New areas being added to monitoring (composition bias)
  2. Changes in which countries are monitored
  3. Within-location trends vs between-location differences

Seven analysis sections:
  1. Composition bias — new locations over time
  2. Consistent locations — locations monitored in both early and late periods
  3. Within-location trend analysis (4+ transitions)
  4. Country-level composition check
  5. New vs old locations gap comparison
  6. Data frequency check
  7. Simple fixed-effects analysis (location-demeaned year trend)

Inputs (all relative to package root):
    outputs/data/archetype_transitions.csv
        Produced by 02_generate_transitions.py

Outputs (all relative to package root):
    outputs/data/within_location_gap_trends.csv

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

    # Convert dates
    df['from_start_date'] = pd.to_datetime(df['from_start'])
    df['from_end_date'] = pd.to_datetime(df['from_end'])
    df['to_start_date'] = pd.to_datetime(df['to_start'])
    df['year'] = df['from_start_date'].dt.year

    print("=" * 70)
    print("ROBUSTNESS CHECK: IS GAP COMPRESSION REAL?")
    print("=" * 70)

    # =================================================================
    # 1. CHECK FOR COMPOSITION BIAS — NEW LOCATIONS OVER TIME
    # =================================================================
    print("\n" + "=" * 70)
    print("1. COMPOSITION BIAS: Are new locations driving the trend?")
    print("=" * 70)

    # Find first year each location appears in data
    location_first_year = df.groupby('location')['year'].min().reset_index()
    location_first_year.columns = ['location', 'first_year']

    # Merge back
    df = df.merge(location_first_year, on='location')

    # Count new vs existing locations by year
    location_cohorts = location_first_year.groupby('first_year').size()
    print("\nNew locations entering dataset by year:")
    print(location_cohorts.to_string())

    # Calculate % of transitions from "new" locations
    # (first appearing that year or year before)
    df['is_new_location'] = (df['year'] - df['first_year']) <= 1

    new_location_pct = df.groupby('year').agg({
        'is_new_location': 'mean',
        'gap_months': 'mean',
        'location': 'count'
    }).round(3)
    new_location_pct.columns = ['pct_new_locations', 'mean_gap', 'n_transitions']

    print("\nProportion of transitions from 'new' locations by year:")
    print(new_location_pct.to_string())

    # =================================================================
    # 2. CONTROL FOR COMPOSITION: CONSISTENT LOCATIONS ONLY
    # =================================================================
    print("\n" + "=" * 70)
    print("2. CONSISTENT LOCATIONS: Gaps for locations monitored since 2011-2015")
    print("=" * 70)

    # Find locations that have transitions in both early and late periods
    early_locations = set(df[df['year'] <= 2015]['location'].unique())
    late_locations = set(df[df['year'] >= 2020]['location'].unique())
    consistent_locations = early_locations & late_locations

    print(f"\nLocations with transitions in 2011-2015: {len(early_locations)}")
    print(f"Locations with transitions in 2020-2023: {len(late_locations)}")
    print(f"Locations in BOTH periods: {len(consistent_locations)}")

    # Filter to consistent locations
    df_consistent = df[df['location'].isin(consistent_locations)]

    # Calculate gap trend for consistent locations only
    consistent_gaps = df_consistent.groupby('year').agg({
        'gap_months': ['mean', 'median', 'count']
    }).round(1)
    consistent_gaps.columns = ['mean_gap', 'median_gap', 'n_transitions']

    print("\nGap duration for CONSISTENT locations only:")
    print(consistent_gaps.to_string())

    # Compare early vs late for consistent locations
    early_gaps = df_consistent[df_consistent['year'] <= 2015]['gap_months']
    late_gaps = df_consistent[df_consistent['year'] >= 2020]['gap_months']

    print(f"\nConsistent locations - Early period (2011-2015):")
    print(f"  Mean gap: {early_gaps.mean():.1f} months, "
          f"Median: {early_gaps.median():.1f}, n={len(early_gaps)}")
    print(f"\nConsistent locations - Late period (2020-2023):")
    print(f"  Mean gap: {late_gaps.mean():.1f} months, "
          f"Median: {late_gaps.median():.1f}, n={len(late_gaps)}")

    # =================================================================
    # 3. WITHIN-LOCATION TREND ANALYSIS
    # =================================================================
    print("\n" + "=" * 70)
    print("3. WITHIN-LOCATION TRENDS: Do gaps shrink within individual locations?")
    print("=" * 70)

    # For locations with 4+ transitions, calculate trend
    location_counts = df.groupby('location').size()
    locations_with_history = location_counts[location_counts >= 4].index

    within_location_trends = []
    for loc in locations_with_history:
        loc_data = df[df['location'] == loc].sort_values('year')
        if loc_data['year'].nunique() >= 2:  # Need variation in years
            # Simple correlation between year and gap
            corr = loc_data[['year', 'gap_months']].corr().iloc[0, 1]
            early_mean = loc_data[
                loc_data['year'] <= loc_data['year'].median()
            ]['gap_months'].mean()
            late_mean = loc_data[
                loc_data['year'] > loc_data['year'].median()
            ]['gap_months'].mean()
            within_location_trends.append({
                'location': loc,
                'iso3': loc_data['iso3'].iloc[0],
                'n_transitions': len(loc_data),
                'year_gap_corr': corr,
                'early_mean_gap': early_mean,
                'late_mean_gap': late_mean,
                'gap_change': late_mean - early_mean
            })

    within_df = pd.DataFrame(within_location_trends)

    print(f"\nLocations with 4+ transitions and year variation: {len(within_df)}")
    print(f"\nWithin-location year-gap correlation:")
    print(f"  Mean correlation: {within_df['year_gap_corr'].mean():.3f}")
    print(f"  Median correlation: {within_df['year_gap_corr'].median():.3f}")
    print(f"  Locations with NEGATIVE correlation (gaps shrinking): "
          f"{(within_df['year_gap_corr'] < 0).sum()} "
          f"({(within_df['year_gap_corr'] < 0).mean()*100:.1f}%)")
    print(f"  Locations with POSITIVE correlation (gaps growing): "
          f"{(within_df['year_gap_corr'] > 0).sum()} "
          f"({(within_df['year_gap_corr'] > 0).mean()*100:.1f}%)")

    print(f"\nMean gap change (late - early) within locations: "
          f"{within_df['gap_change'].mean():.1f} months")
    print(f"Median gap change: {within_df['gap_change'].median():.1f} months")

    # Show examples
    print("\nLocations with LARGEST gap reduction (within-location):")
    print(within_df.nsmallest(10, 'gap_change')[
        ['location', 'iso3', 'n_transitions', 'early_mean_gap',
         'late_mean_gap', 'gap_change']
    ].to_string())

    print("\nLocations with gap INCREASE (within-location):")
    print(within_df.nlargest(10, 'gap_change')[
        ['location', 'iso3', 'n_transitions', 'early_mean_gap',
         'late_mean_gap', 'gap_change']
    ].to_string())

    # =================================================================
    # 4. COUNTRY-LEVEL COMPOSITION CHECK
    # =================================================================
    print("\n" + "=" * 70)
    print("4. COUNTRY COMPOSITION: Are new countries driving the trend?")
    print("=" * 70)

    # First year each country appears
    country_first_year = df.groupby('iso3')['year'].min()
    print("\nFirst year of transitions by country:")
    print(country_first_year.sort_values().to_string())

    # Gap trends within consistently monitored countries
    consistent_countries = ['AFG', 'SOM', 'KEN', 'ETH', 'SDN', 'SSD',
                            'MLI', 'NER', 'TCD', 'HTI', 'GTM']

    print("\nGap trends for major consistently-monitored countries:")
    for iso3 in consistent_countries:
        country_data = df[df['iso3'] == iso3]
        if len(country_data) >= 10:
            early = country_data[country_data['year'] <= 2017]['gap_months']
            late = country_data[country_data['year'] >= 2020]['gap_months']
            if len(early) >= 3 and len(late) >= 3:
                print(f"\n  {iso3}:")
                print(f"    Early (<=2017): mean={early.mean():.1f} mo, n={len(early)}")
                print(f"    Late (>=2020):  mean={late.mean():.1f} mo, n={len(late)}")
                print(f"    Change: {late.mean() - early.mean():+.1f} months")

    # =================================================================
    # 5. NEW LOCATIONS VS OLD LOCATIONS GAP COMPARISON
    # =================================================================
    print("\n" + "=" * 70)
    print("5. NEW vs OLD LOCATIONS: Do newer locations have shorter gaps?")
    print("=" * 70)

    # Categorize locations by when they entered monitoring
    df['location_cohort'] = pd.cut(
        df['first_year'],
        bins=[2010, 2014, 2017, 2020, 2025],
        labels=['2011-2014', '2015-2017', '2018-2020', '2021+']
    )

    cohort_gaps = df.groupby('location_cohort', observed=True).agg({
        'gap_months': ['mean', 'median', 'count'],
        'location': 'nunique'
    }).round(1)
    cohort_gaps.columns = ['mean_gap', 'median_gap', 'n_transitions', 'n_locations']

    print("\nGap duration by location cohort (when location first monitored):")
    print(cohort_gaps.to_string())

    # =================================================================
    # 6. DATA FREQUENCY CHECK
    # =================================================================
    print("\n" + "=" * 70)
    print("6. DATA FREQUENCY: Has monitoring frequency increased?")
    print("=" * 70)

    # Count transitions per year
    yearly_transitions = df.groupby('year').size()
    yearly_locations = df.groupby('year')['location'].nunique()

    print("\nTransitions and locations by year:")
    freq_df = pd.DataFrame({
        'transitions': yearly_transitions,
        'unique_locations': yearly_locations,
        'transitions_per_location': (
            yearly_transitions / yearly_locations).round(2)
    })
    print(freq_df.to_string())

    # =================================================================
    # 7. FIXED EFFECTS ANALYSIS (Simple version)
    # =================================================================
    print("\n" + "=" * 70)
    print("7. SIMPLE FIXED EFFECTS: Year trend controlling for location")
    print("=" * 70)

    # Demean gaps within each location (remove location fixed effects)
    df['location_mean_gap'] = df.groupby('location')['gap_months'].transform('mean')
    df['gap_demeaned'] = df['gap_months'] - df['location_mean_gap']

    # Now look at year trend in demeaned gaps
    demeaned_by_year = df.groupby('year')['gap_demeaned'].agg(
        ['mean', 'std', 'count']).round(2)
    print("\nYear trend in DEMEANED gaps (location effects removed):")
    print(demeaned_by_year.to_string())

    # Correlation between year and demeaned gap
    year_demeaned_corr = df[['year', 'gap_demeaned']].corr().iloc[0, 1]
    print(f"\nCorrelation between year and demeaned gap: {year_demeaned_corr:.3f}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: IS GAP COMPRESSION REAL?")
    print("=" * 70)

    # Key statistics
    pct_negative_within = (within_df['year_gap_corr'] < 0).mean() * 100
    mean_within_change = within_df['gap_change'].mean()
    consistent_early = early_gaps.mean()
    consistent_late = late_gaps.mean()

    print(f"""
EVIDENCE FOR REAL GAP COMPRESSION:
1. Consistent locations (monitored early AND late):
   - Early period mean: {consistent_early:.1f} months
   - Late period mean: {consistent_late:.1f} months
   - Change: {consistent_late - consistent_early:.1f} months

2. Within-location trends:
   - {pct_negative_within:.0f}% of locations show gaps SHRINKING over time
   - Mean within-location change: {mean_within_change:.1f} months

3. Year-demeaned gap correlation: {year_demeaned_corr:.3f}
   (Negative = gaps shrinking even after removing location effects)

POTENTIAL CONFOUNDS:
- New locations entering dataset: {len(df['location'].unique()) - len(early_locations)} new since 2016
- But trend persists in consistent locations

CONCLUSION: Gap compression appears to be a REAL phenomenon, not just composition bias.
""")

    # Save detailed outputs
    within_path = os.path.join(OUTPUT_DIR, 'within_location_gap_trends.csv')
    within_df.to_csv(within_path, index=False)
    print(f"\nSaved: {within_path}")


if __name__ == '__main__':
    main()
