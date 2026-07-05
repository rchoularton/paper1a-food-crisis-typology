#!/usr/bin/env python3
# @status:   canonical
# @process:  P6-revision
# @paper:    paper1
"""
R1.4 spatial dynamics: admin1 neighbour-spread (co-escalation) by archetype.
============================================================================

Companion to r1_spatial_extent.py. Where the footprint analysis measures the
national crisis extent, this measures direct *neighbour spread*: for each
episode, how many of its ADJACENT admin1 areas are themselves in crisis
(IPC Phase 3+) at the episode's onset, peak, and end — i.e. do crises spread to
neighbouring areas (R1: "how crises seem to cluster spatially over time").

Adjacency is built from HFID's own geometry (data/hfid/simplified_hfid_geom.gpkg),
dissolved to admin1, within-country shared boundaries only. The iso3<->ADMIN0
map is taken from the HFID dataset itself, so the location keys (iso3_ADMIN1)
match the frozen primary panel exactly. Match rate is reported.

Additive analysis (no transition-probability change). Frozen panel (FEWS/MAX/12mo).

Outputs:
  outputs/data/r1_admin1_adjacency.csv
  outputs/data/r1_spatial_adjacency_by_archetype.json
  outputs/data/r1_spatial_adjacency_by_episode.csv

Run:
  python3 code/r1_spatial_adjacency.py
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from reference_transition_analysis import (  # noqa: E402
    load_hfid, preprocess, interpolate,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'data')
GEOM_PATH = os.path.join(PROJECT_ROOT, 'data', 'hfid', 'simplified_hfid_geom.gpkg')
EPISODES_CSV = os.path.join(OUTPUT_DIR, 'episodes.csv')
CRISIS_THRESHOLD = 3
# Adjacency tolerance (degrees). Simplified polygons have slivers/gaps, so
# strict touches() under-detects shared borders; we treat admin1 areas within
# ADJ_EPS of each other (same country) as neighbours.
ADJ_EPS = 0.05


def build_admin1_adjacency(admin0_to_iso3):
    """Within-country admin1 adjacency from HFID geometry → dict location->set.

    Repairs invalid geometry with make_valid() and dissolves admin2→admin1
    per country (smaller, more robust unions). Countries whose geometry cannot
    be dissolved are skipped and reported.
    """
    import geopandas as gpd
    print("  Loading geometry + repairing invalid polygons...")
    g = gpd.read_file(GEOM_PATH)
    g = g.dropna(subset=['ADMIN0', 'ADMIN1']).copy()
    g['iso3'] = g['ADMIN0'].map(admin0_to_iso3)
    g = g.dropna(subset=['iso3']).copy()
    g['geometry'] = g.geometry.make_valid()

    adj = defaultdict(set)
    n_admin1 = 0
    skipped = []
    for iso, sub in g.groupby('iso3'):
        try:
            a1 = sub.dissolve(by='ADMIN1').reset_index()
        except Exception:
            try:
                a1 = (sub.assign(geometry=sub.geometry.buffer(0))
                      .dissolve(by='ADMIN1').reset_index())
            except Exception as e:
                skipped.append((iso, str(e)[:60]))
                continue
        a1 = a1.reset_index(drop=True)
        a1['geometry'] = a1.geometry.make_valid()
        a1['location'] = iso + '_' + a1['ADMIN1'].astype(str)
        n_admin1 += len(a1)
        sidx = a1.sindex
        buf = a1.geometry.buffer(ADJ_EPS)  # bridge simplification slivers
        for i in range(len(a1)):
            try:
                cand = sidx.query(buf.iloc[i], predicate='intersects')
            except Exception:
                cand = [j for j in range(len(a1)) if j != i
                        and buf.iloc[i].intersects(a1.geometry.iloc[j])]
            for j in cand:
                if j != i:
                    adj[a1.loc[i, 'location']].add(a1.loc[j, 'location'])
    n_nb = [len(v) for v in adj.values()]
    mean_nb = sum(n_nb) / len(n_nb) if n_nb else 0
    print(f"  Admin1 polygons: {n_admin1}; locations with >=1 neighbour: "
          f"{len(adj)} (mean neighbours {mean_nb:.1f}, eps={ADJ_EPS})")
    if skipped:
        print(f"  Skipped {len(skipped)} countries (geometry): "
              f"{[s[0] for s in skipped]}")
    return adj, skipped


def episode_yms(dates_str):
    return [d.strip()[:7] for d in str(dates_str).split(',') if d.strip()]


def main():
    print("=" * 70)
    print("R1.4 ADMIN1 NEIGHBOUR-SPREAD BY ARCHETYPE")
    print("=" * 70)

    df_raw = load_hfid()
    # iso3 <-> ADMIN0 map from HFID itself (guarantees key consistency)
    admin0_to_iso3 = (df_raw.dropna(subset=['ADMIN0', 'iso3'])
                      .groupby('ADMIN0')['iso3']
                      .agg(lambda s: s.value_counts().index[0]).to_dict())

    df_pre = preprocess(df_raw, priority='fews', aggregation='max')
    df_interp = interpolate(df_pre, max_gap=12)
    # phase lookup keyed by (location, ym)
    phase_lookup = {(r.location, r.year_month): r.ipc_phase
                    for r in df_interp.itertuples(index=False)}

    adj, skipped = build_admin1_adjacency(admin0_to_iso3)

    # Save adjacency edge list
    edges = [{'location': a, 'neighbor_location': b}
             for a, nbrs in adj.items() for b in nbrs]
    pd.DataFrame(edges).to_csv(
        os.path.join(OUTPUT_DIR, 'r1_admin1_adjacency.csv'), index=False)

    ep = pd.read_csv(EPISODES_CSV)
    ep_locs = set(ep['location'].unique())
    matched = sum(1 for loc in ep_locs if loc in adj)
    print(f"\n  Episode locations: {len(ep_locs)}; with adjacency: {matched} "
          f"({100*matched/len(ep_locs):.1f}%)")

    def nb_in_crisis(location, ym):
        nbrs = adj.get(location)
        if not nbrs:
            return np.nan, np.nan
        n = len(nbrs)
        c = sum(1 for nb in nbrs
                if phase_lookup.get((nb, ym), 0) >= CRISIS_THRESHOLD)
        return c, c / n

    rows = []
    for _, r in ep.iterrows():
        loc = r['location']
        if loc not in adj:
            continue
        yms = episode_yms(r['dates'])
        shares = [(ym, nb_in_crisis(loc, ym)[1]) for ym in yms]
        shares = [(ym, s) for ym, s in shares if s == s]
        if not shares:
            continue
        s_on = shares[0][1]
        s_en = shares[-1][1]
        s_max = max(s for _, s in shares)
        rows.append({
            'crisis_id': r['crisis_id'], 'iso3': r['iso3'],
            'location': loc, 'archetype': r['archetype'],
            'n_neighbors': len(adj.get(loc, [])),
            'nb_share_onset': s_on, 'nb_share_max': s_max, 'nb_share_end': s_en,
            # net co-escalation over the life cycle (end vs onset) — non-mechanical
            'co_escalation_net': s_en - s_on,
            # peak neighbour spread reached during the episode
            'co_escalation_peak': s_max - s_on,
        })
    epx = pd.DataFrame(rows)
    epx.to_csv(os.path.join(OUTPUT_DIR, 'r1_spatial_adjacency_by_episode.csv'),
               index=False)

    order = ['seasonal_crisis', 'prolonged_moderate', 'entrenched_moderate',
             'oscillating', 'rapid_onset', 'severe_shock', 'escalating',
             'protracted_emergency']
    present = [a for a in order if a in epx['archetype'].unique()]

    print(f"\n{'archetype':<22} {'n':>5} {'nb':>5} {'sh_on':>6} {'sh_end':>6} "
          f"{'co_net':>7} {'co_pk':>6}")
    agg = {}
    for a in present:
        s = epx[epx['archetype'] == a].dropna(subset=['nb_share_onset'])
        rec = {
            'n_episodes': int(len(s)),
            'mean_n_neighbors': round(float(s['n_neighbors'].mean()), 1),
            'mean_nb_share_onset': round(float(s['nb_share_onset'].mean()), 3),
            'mean_nb_share_end': round(float(s['nb_share_end'].mean()), 3),
            'mean_co_escalation_net': round(float(s['co_escalation_net'].mean()), 3),
            'mean_co_escalation_peak': round(float(s['co_escalation_peak'].mean()), 3),
        }
        agg[a] = rec
        print(f"{a:<22} {rec['n_episodes']:>5} {rec['mean_n_neighbors']:>5} "
              f"{rec['mean_nb_share_onset']:>6} {rec['mean_nb_share_end']:>6} "
              f"{rec['mean_co_escalation_net']:>7} {rec['mean_co_escalation_peak']:>6}")

    out = {
        'description': ('R1.4 admin1 neighbour-spread: share of adjacent admin1 '
                        'areas at IPC Phase 3+ over the episode life cycle (onset, '
                        'max, end), by archetype. co_escalation_net=end-onset; '
                        'co_escalation_peak=max-onset. Adjacency from HFID geometry '
                        '(within-country, ~5km buffer). Frozen panel (FEWS/MAX/12mo).'),
        'crisis_threshold': CRISIS_THRESHOLD,
        'episode_locations': len(ep_locs),
        'episode_locations_with_adjacency': matched,
        'match_rate_pct': round(100 * matched / len(ep_locs), 1),
        'co_escalation_definition': ('net = mean(nb_share_end - nb_share_onset); '
                                     'peak = mean(nb_share_max - nb_share_onset)'),
        'by_archetype': agg,
    }
    with open(os.path.join(OUTPUT_DIR, 'r1_spatial_adjacency_by_archetype.json'),
              'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'r1_spatial_adjacency_by_archetype.json')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'r1_spatial_adjacency_by_episode.csv')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'r1_admin1_adjacency.csv')}")


if __name__ == '__main__':
    main()
