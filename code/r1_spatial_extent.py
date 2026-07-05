#!/usr/bin/env python3
# @status:   canonical
# @process:  P6-revision
# @paper:    paper1
"""
R1.4 spatial dynamics: crisis spatial-extent (footprint) by archetype.
======================================================================

Addresses Nature Food Reviewer 1 comment R1.4: "what about spatial dynamics?
[...] Can you say anything about how crises seem to cluster spatially over time?"

This script measures how the *spatial extent* of crises behaves over the crisis
life cycle, by archetype. For each episode we measure the national crisis
footprint — the number and share of admin1 areas in the same country at IPC
Phase 3+ — at the episode's onset, peak, and end month, and summarise expansion
(peak/onset) and recovery contraction (end/peak) by archetype.

Built on the frozen primary panel (FEWS priority, MAX aggregation, 12-month
interpolation) via the canonical pipeline functions — does NOT re-derive the
methodology. Additive analysis: does not touch transition probabilities.

NOTE (rigor): the seasonal-stable / other-archetypes-expand pattern is a
HYPOTHESIS to be tested here, not assumed. Report what the data show.

Outputs:
  outputs/data/r1_spatial_extent_by_archetype.json
  outputs/data/r1_spatial_extent_by_episode.csv

Run:
  python3 code/r1_spatial_extent.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from reference_transition_analysis import (  # noqa: E402
    load_hfid, preprocess, interpolate,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'data')
EPISODES_CSV = os.path.join(OUTPUT_DIR, 'episodes.csv')
CRISIS_THRESHOLD = 3


def build_footprint_index(df_interp):
    """
    For each (iso3, year_month): count admin1 locations at Phase 3+ (crisis
    footprint) and total monitored locations (denominator for the share).
    Returns two dicts keyed by (iso3, 'YYYY-MM').
    """
    crisis = df_interp[df_interp['ipc_phase'] >= CRISIS_THRESHOLD]
    foot = crisis.groupby(['iso3', 'year_month'])['location'].nunique()
    total = df_interp.groupby(['iso3', 'year_month'])['location'].nunique()
    return foot.to_dict(), total.to_dict()


def episode_yms(dates_str):
    """Ordered list of YYYY-MM for every month in the episode."""
    return [d.strip()[:7] for d in str(dates_str).split(',') if d.strip()]


def main():
    print("=" * 70)
    print("R1.4 SPATIAL EXTENT (FOOTPRINT) BY ARCHETYPE")
    print("=" * 70)

    df_raw = load_hfid()
    df_pre = preprocess(df_raw, priority='fews', aggregation='max')
    df_interp = interpolate(df_pre, max_gap=12)
    foot, total = build_footprint_index(df_interp)

    ep = pd.read_csv(EPISODES_CSV)
    rows = []
    for _, r in ep.iterrows():
        yms = episode_yms(r['dates'])
        if not yms:
            continue
        iso3 = r['iso3']
        # Footprint at EVERY month of the episode (avoids the severity-peak
        # artefact: expansion is measured over the whole life cycle, so a
        # flat-severity episode can still show — or not show — spatial growth).
        series = [foot.get((iso3, ym), np.nan) for ym in yms]
        valid = [(ym, v) for ym, v in zip(yms, series) if v == v]
        if not valid:
            continue
        f_on = valid[0][1]
        f_en = valid[-1][1]
        max_ym, f_max = max(valid, key=lambda kv: kv[1])
        rows.append({
            'crisis_id': r['crisis_id'], 'iso3': iso3, 'location': r['location'],
            'archetype': r['archetype'], 'duration_months': r['duration_months'],
            'onset_ym': valid[0][0], 'max_ym': max_ym, 'end_ym': valid[-1][0],
            'footprint_onset': f_on, 'footprint_max': f_max, 'footprint_end': f_en,
            'share_onset': f_on / total.get((iso3, valid[0][0]), np.nan)
            if total.get((iso3, valid[0][0])) else np.nan,
            'share_max': f_max / total.get((iso3, max_ym), np.nan)
            if total.get((iso3, max_ym)) else np.nan,
            # expansion over the life cycle = peak extent reached / onset extent
            'expansion_ratio': f_max / f_on if f_on and f_on > 0 else np.nan,
            # end vs onset = net spatial change by the time the episode resolves
            'net_ratio': f_en / f_on if f_on and f_on > 0 else np.nan,
        })

    epx = pd.DataFrame(rows)
    epx.to_csv(os.path.join(OUTPUT_DIR, 'r1_spatial_extent_by_episode.csv'),
               index=False)

    # Aggregate by archetype
    agg = {}
    order = ['seasonal_crisis', 'prolonged_moderate', 'entrenched_moderate',
             'oscillating', 'rapid_onset', 'severe_shock', 'escalating',
             'protracted_emergency']
    present = [a for a in order if a in epx['archetype'].unique()]
    present += [a for a in epx['archetype'].unique() if a not in present]

    print(f"\n{'archetype':<22} {'n':>5} {'dur':>5} {'foot_on':>8} {'foot_max':>8} "
          f"{'expand':>7} {'net':>6}")
    for a in present:
        s = epx[epx['archetype'] == a]
        rec = {
            'n_episodes': int(len(s)),
            'median_duration': float(s['duration_months'].median()),
            'mean_footprint_onset': round(float(s['footprint_onset'].mean()), 2),
            'mean_footprint_max': round(float(s['footprint_max'].mean()), 2),
            'mean_footprint_end': round(float(s['footprint_end'].mean()), 2),
            'mean_share_onset': round(float(s['share_onset'].mean()), 3),
            'mean_share_max': round(float(s['share_max'].mean()), 3),
            'median_expansion_ratio': round(float(s['expansion_ratio'].median()), 2),
            'mean_expansion_ratio': round(float(s['expansion_ratio'].mean()), 2),
            'median_net_ratio': round(float(s['net_ratio'].median()), 2),
            'mean_net_ratio': round(float(s['net_ratio'].mean()), 2),
        }
        agg[a] = rec
        print(f"{a:<22} {rec['n_episodes']:>5} {rec['median_duration']:>5} "
              f"{rec['mean_footprint_onset']:>8} {rec['mean_footprint_max']:>8} "
              f"{rec['median_expansion_ratio']:>7} {rec['median_net_ratio']:>6}")

    out = {
        'description': ('R1.4 crisis spatial extent (national admin1 footprint at '
                        'Phase 3+) by archetype, measured over the episode life '
                        'cycle: onset, max-during-episode, and end (NOT the '
                        'severity-peak month, which would force a null for '
                        'flat-severity episodes). expansion_ratio=max/onset; '
                        'net_ratio=end/onset. Frozen primary panel (FEWS/MAX/12mo).'),
        'crisis_threshold': CRISIS_THRESHOLD,
        'footprint_definition': ('count of admin1 areas in the same country at IPC '
                                 'Phase 3+ in the given month; share = that count / '
                                 'total monitored admin1 areas in the country-month'),
        'by_archetype': agg,
    }
    with open(os.path.join(OUTPUT_DIR, 'r1_spatial_extent_by_archetype.json'),
              'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'r1_spatial_extent_by_archetype.json')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'r1_spatial_extent_by_episode.csv')}")


if __name__ == '__main__':
    main()
