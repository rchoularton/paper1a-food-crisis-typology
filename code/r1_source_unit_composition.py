#!/usr/bin/env python3
# @status:   canonical
# @process:  P6-revision
# @paper:    paper1
"""
R3.7 + R3.9 disclosure: source mix + analytical-unit composition.
=================================================================

Quantifies, for the frozen primary panel (FEWS priority, MAX aggregation,
12-month interpolation):

  R3.7 — source mix behind the monthly transitions:
    * share of observed admin1-month classifications sourced from FEWS NET
      vs IPC/CH (under the FEWS-priority rule used in the analysis);
    * share of the interpolated panel that is carried-forward (= how the
      static IPC/CH projection windows are handled).

  R3.9 — analytical-unit composition:
    * admin1 (primary) and admin2 (sensitivity) unit counts;
    * units per country (size/heterogeneity of the analytical grid).

These feed the Methods disclosure requested by R3 (paragraphs [64]/[66]/[82]).
Built on canonical pipeline functions; verification query (does not alter any
frozen output).

Output:
  outputs/data/r1_source_unit_composition.json

Run:
  python3 code/r1_source_unit_composition.py
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
    load_hfid, preprocess, preprocess_admin2, interpolate, compute_transitions,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'data')


def source_mix(df_raw):
    """Source of each observed admin1-month classification (FEWS vs IPC/CH)."""
    df = df_raw.copy()
    df['ipc_phase'] = df['ipc_phase_fews'].fillna(df['ipc_phase_ipcch'])
    df.loc[df['ipc_phase'] == 6, 'ipc_phase'] = np.nan
    df = df[df['ipc_phase'].notna()].copy()
    df['location'] = df['iso3'] + '_' + df['ADMIN1'].fillna('national')
    df['fews_ok'] = df['ipc_phase_fews'].notna() & (df['ipc_phase_fews'] != 6)
    df['ipcch_ok'] = df['ipc_phase_ipcch'].notna() & (df['ipc_phase_ipcch'] != 6)

    a1 = df.groupby(['location', 'year_month']).agg(
        fews=('fews_ok', 'max'), ipcch=('ipcch_ok', 'max')).reset_index()
    a1['source'] = np.where(a1['fews'], 'fews',
                            np.where(a1['ipcch'], 'ipcch', 'none'))
    n = len(a1)
    vc = a1['source'].value_counts()
    return {
        'observed_admin1_months': int(n),
        'fews_sourced': int(vc.get('fews', 0)),
        'ipcch_only_sourced': int(vc.get('ipcch', 0)),
        'fews_sourced_pct': round(100 * vc.get('fews', 0) / n, 1),
        'ipcch_only_pct': round(100 * vc.get('ipcch', 0) / n, 1),
        'both_available_pct': round(100 * (a1['fews'] & a1['ipcch']).sum() / n, 1),
    }


def carry_forward_share(df_interp):
    """Share of the interpolated panel that is carried-forward, and of
    consecutive transition pairs that involve a carried-forward endpoint."""
    n = len(df_interp)
    n_interp = int(df_interp['is_interpolated'].sum())

    pairs_total = 0
    pairs_with_interp = 0
    for loc, g in df_interp.groupby('location'):
        g = g.sort_values('date')
        dates = g['date'].values
        interp = g['is_interpolated'].values
        for i in range(len(dates) - 1):
            gap = (dates[i + 1] - dates[i]) / np.timedelta64(1, 'D')
            if gap < 25 or gap > 35:
                continue
            pairs_total += 1
            if interp[i] or interp[i + 1]:
                pairs_with_interp += 1
    return {
        'panel_records': int(n),
        'carried_forward_records': n_interp,
        'carried_forward_pct': round(100 * n_interp / n, 1),
        'transition_pairs': int(pairs_total),
        'pairs_with_carryforward_endpoint': int(pairs_with_interp),
        'pairs_with_carryforward_pct': round(100 * pairs_with_interp / pairs_total, 1)
        if pairs_total else 0.0,
    }


def unit_composition(df_raw):
    a1 = preprocess(df_raw, priority='fews', aggregation='max')
    a2 = preprocess_admin2(df_raw, priority='fews')

    a1_per_country = a1.groupby('iso3')['location'].nunique()
    a2_per_a1 = (a2.groupby(['iso3', 'ADMIN1'])['location'].nunique())

    return {
        'admin1_locations': int(a1['location'].nunique()),
        'admin1_countries': int(a1['iso3'].nunique()),
        'admin1_per_country_median': float(a1_per_country.median()),
        'admin1_per_country_min': int(a1_per_country.min()),
        'admin1_per_country_max': int(a1_per_country.max()),
        'admin2_locations': int(a2['location'].nunique()),
        'admin2_countries': int(a2['iso3'].nunique()),
        'admin2_per_admin1_median': float(a2_per_a1.median()),
        'admin2_per_admin1_max': int(a2_per_a1.max()),
    }


def main():
    print("=" * 70)
    print("R3.7 + R3.9 — SOURCE MIX + ANALYTICAL-UNIT COMPOSITION")
    print("=" * 70)

    df_raw = load_hfid()

    sm = source_mix(df_raw)
    print("\n[R3.7] Source mix (observed admin1-months):")
    for k, v in sm.items():
        print(f"  {k}: {v}")

    df_pre = preprocess(df_raw, priority='fews', aggregation='max')
    df_interp = interpolate(df_pre, max_gap=12)
    cf = carry_forward_share(df_interp)
    print("\n[R3.7] Carry-forward (interpolation) share:")
    for k, v in cf.items():
        print(f"  {k}: {v}")

    uc = unit_composition(df_raw)
    print("\n[R3.9] Analytical-unit composition:")
    for k, v in uc.items():
        print(f"  {k}: {v}")

    out = {
        'description': ('R3.7/R3.9 disclosure: FEWS-vs-IPC/CH source mix, '
                        'carry-forward share, and analytical-unit composition '
                        'for the frozen primary panel (FEWS/MAX/12mo).'),
        'source_mix_R3_7': sm,
        'carry_forward_R3_7': cf,
        'unit_composition_R3_9': uc,
    }
    with open(os.path.join(OUTPUT_DIR, 'r1_source_unit_composition.json'),
              'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'r1_source_unit_composition.json')}")


if __name__ == '__main__':
    main()
