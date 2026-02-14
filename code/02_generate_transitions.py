#!/usr/bin/env python3
"""
02_generate_transitions.py — Archetype Transition Generation
=============================================================

Reads the crisis episodes produced by 01_reference_pipeline.py and generates
transition records between consecutive episodes at each location.  A
"transition" pairs a FROM-episode with the next TO-episode at the same
location, recording the gap duration (months), severity change, and both
archetype labels.

A separate location-level summary is also produced.

Inputs  (all relative to package root):
    outputs/data/episodes.csv
        Columns used: location, iso3, archetype, peak_phase, duration_months,
                      dates (comma-separated YYYY-MM-DD strings)

Outputs (all relative to package root):
    outputs/data/archetype_transitions.csv
        One row per consecutive-episode pair, with gap_months, severity_change,
        from_archetype, to_archetype, etc.

    outputs/data/location_summaries.csv
        One row per location with episode count, mean gap, total gap months,
        and cumulative severity change.

Dependencies: pandas, numpy  (standard scientific-Python stack)

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# Paths — relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data')


def parse_episode_dates(dates_str):
    """Parse comma-separated date string and return first and last date as YYYY-MM."""
    dates = [d.strip() for d in dates_str.split(',')]
    first = dates[0][:7]   # YYYY-MM
    last = dates[-1][:7]   # YYYY-MM
    return first, last


def generate_transitions():
    """Generate transitions between consecutive episodes at each location."""

    # Load pipeline episodes
    episodes = pd.read_csv(EPISODES_PATH)
    print(f"Loaded {len(episodes)} episodes from pipeline")
    print(f"  Locations: {episodes['location'].nunique()}")
    print(f"  Countries: {episodes['iso3'].nunique()}")

    # Parse start and end dates for each episode
    starts = []
    ends = []
    for _, row in episodes.iterrows():
        start, end = parse_episode_dates(row['dates'])
        starts.append(start)
        ends.append(end)

    episodes['start_date'] = starts
    episodes['end_date'] = ends
    episodes['start_dt'] = pd.to_datetime(episodes['start_date'])

    # Sort by location and start date
    episodes = episodes.sort_values(['location', 'start_dt'])

    # Extract admin1 name from location (format: ISO3_AdminName)
    episodes['admin1'] = episodes['location'].str.split('_', n=1).str[1]

    # Generate transitions
    transitions = []

    for location, group in episodes.groupby('location'):
        group = group.sort_values('start_dt')
        rows = group.reset_index(drop=True)

        for i in range(len(rows) - 1):
            from_ep = rows.iloc[i]
            to_ep = rows.iloc[i + 1]

            # Calculate gap in months
            from_end_dt = pd.to_datetime(from_ep['end_date'])
            to_start_dt = pd.to_datetime(to_ep['start_date'])
            gap_months = (to_start_dt.year - from_end_dt.year) * 12 + (to_start_dt.month - from_end_dt.month)

            # Severity change based on peak phase
            severity_change = int(to_ep['peak_phase']) - int(from_ep['peak_phase'])

            transitions.append({
                'location': location,
                'iso3': from_ep['iso3'],
                'admin1': from_ep['admin1'],
                'from_archetype': from_ep['archetype'],
                'to_archetype': to_ep['archetype'],
                'from_start': from_ep['start_date'],
                'from_end': from_ep['end_date'],
                'to_start': to_ep['start_date'],
                'to_end': to_ep['end_date'],
                'gap_months': gap_months,
                'from_duration': int(from_ep['duration_months']),
                'to_duration': int(to_ep['duration_months']),
                'from_peak': int(from_ep['peak_phase']),
                'to_peak': int(to_ep['peak_phase']),
                'same_archetype': from_ep['archetype'] == to_ep['archetype'],
                'severity_change': severity_change
            })

    df = pd.DataFrame(transitions)

    # Summary statistics
    print(f"\nGenerated {len(df)} transitions")
    print(f"  Locations with transitions: {df['location'].nunique()}")
    print(f"  Countries: {df['iso3'].nunique()}")
    print(f"\nGap statistics:")
    print(f"  Mean: {df['gap_months'].mean():.1f} months")
    print(f"  Median: {df['gap_months'].median():.1f} months")
    print(f"  Min: {df['gap_months'].min()}, Max: {df['gap_months'].max()}")

    # Transition type counts
    print(f"\nTop 10 transition types:")
    type_counts = df.groupby(['from_archetype', 'to_archetype']).size().sort_values(ascending=False)
    for (from_a, to_a), count in type_counts.head(10).items():
        print(f"  {from_a} -> {to_a}: {count}")

    # Same archetype stats
    same_pct = df['same_archetype'].mean() * 100
    print(f"\nSame archetype transitions: {df['same_archetype'].sum()} ({same_pct:.1f}%)")

    # Escalation stats
    esc = (df['severity_change'] > 0).sum()
    deesc = (df['severity_change'] < 0).sum()
    same = (df['severity_change'] == 0).sum()
    print(f"Escalation (peak phase increase): {esc} ({esc/len(df)*100:.1f}%)")
    print(f"De-escalation: {deesc} ({deesc/len(df)*100:.1f}%)")
    print(f"Same peak: {same} ({same/len(df)*100:.1f}%)")

    # Save transitions
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transitions_path = os.path.join(OUTPUT_DIR, 'archetype_transitions.csv')
    df.to_csv(transitions_path, index=False)
    print(f"\nSaved: {transitions_path}")

    # ----------------------------------------------------------------
    # Location-level summaries
    # ----------------------------------------------------------------
    location_summaries = []
    for location, grp in df.groupby('location'):
        location_summaries.append({
            'location': location,
            'iso3': grp['iso3'].iloc[0],
            'n_transitions': len(grp),
            'mean_gap_months': round(grp['gap_months'].mean(), 1),
            'median_gap_months': round(grp['gap_months'].median(), 1),
            'min_gap_months': int(grp['gap_months'].min()),
            'max_gap_months': int(grp['gap_months'].max()),
            'total_gap_months': int(grp['gap_months'].sum()),
            'cumulative_severity_change': int(grp['severity_change'].sum()),
            'n_escalations': int((grp['severity_change'] > 0).sum()),
            'n_deescalations': int((grp['severity_change'] < 0).sum()),
        })

    loc_df = pd.DataFrame(location_summaries)
    loc_path = os.path.join(OUTPUT_DIR, 'location_summaries.csv')
    loc_df.to_csv(loc_path, index=False)
    print(f"Saved: {loc_path}")

    return df


if __name__ == '__main__':
    generate_transitions()
