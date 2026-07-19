#!/usr/bin/env python3
# @status:   maturing
# @process:  P6-audit-v24
# @paper:    paper1
"""
18_monitoring_expansion.py — Monitoring-expansion share of crisis-location growth
=================================================================================

Computes how much of the growth in crisis location counts between two periods
is attributable to monitoring expansion (new locations entering the dataset)
rather than genuine escalation, using the fixed cohort of locations already
monitored in 2016 as the control.

    share = (full-set growth − cohort growth) / full-set growth

Added per the v24 audit (finding A3, AUTHOR DECISION 2026-07-18): the published
figure previously had no computational home anywhere in the codebase, which is
why it drifted undetected through three audits. This script makes the window
and subject EXPLICIT PARAMETERS and emits them alongside the result, so the
share cannot be quoted again without its definition.

A location counts within a window if at least one of its crisis episodes
overlaps the window. The 2016 cohort is every location carrying at least one
non-null IPC/FEWS phase observation dated 2016.

Inputs:
  outputs/data/episodes.csv  — produced by 01_reference_pipeline.py --phase core
  data/HFID_hv1.csv          — HFID v1.1.1 (for the 2016 cohort)

Output:
  outputs/data/monitoring_expansion_share.json

Reference values (v24 audit, independently recomputed 2026-07-18):
  all locations 117 -> 408 (+291); 2016 cohort 104 -> 228 (+124);
  share (291-124)/291 = 57.4%; cohort size exactly 395.
"""

import json
import os

import pandas as pd

# ============================================================
# Paths
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
HFID_PATH = os.path.join(PACKAGE_ROOT, 'data', 'HFID_hv1.csv')
OUTPUT_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data',
                           'monitoring_expansion_share.json')

# Manuscript window (Results L97 paragraph): every other figure in that
# paragraph uses 2011-13 vs 2020-22, so the share must too (audit A3).
DEFAULT_WINDOW_EARLY = (2011, 2013)
DEFAULT_WINDOW_LATE = (2020, 2022)
COHORT_YEAR = 2016

SUBJECT_DEFINITION = (
    'All crisis locations: a location counts within a window if at least one '
    'of its crisis episodes (IPC Phase 3+, FEWS priority, MAX aggregation, '
    '12-month interpolation) overlaps the window.'
)
COHORT_DEFINITION = (
    'Locations with at least one non-null IPC/FEWS phase observation '
    f'dated {COHORT_YEAR} in HFID (matches get_2016_cohort() in '
    '07_fig2_alluvial.py).'
)


def get_cohort(hfid_path, cohort_year):
    """Return the set of locations monitored in cohort_year (raw HFID)."""
    hfid = pd.read_csv(hfid_path)
    hfid['ipc_phase_fews'] = pd.to_numeric(hfid['ipc_phase_fews'], errors='coerce')
    hfid['ipc_phase_ipcch'] = pd.to_numeric(hfid['ipc_phase_ipcch'], errors='coerce')
    hfid['ipc_phase'] = hfid['ipc_phase_fews'].fillna(hfid['ipc_phase_ipcch'])
    hfid['location'] = hfid['iso3'] + '_' + hfid['ADMIN1'].fillna('national')
    hfid['year'] = hfid['year_month'].str[:4].astype(int)
    with_ipc = hfid[hfid['ipc_phase'].notna()]
    return set(with_ipc[with_ipc['year'] == cohort_year]['location'].unique())


def locations_in_window(episodes, year_start, year_end):
    """Set of locations with >=1 episode overlapping [year_start, year_end]."""
    lo, hi = f'{year_start}-01', f'{year_end}-12'
    overlap = (episodes['start'] <= hi) & (episodes['end'] >= lo)
    return set(episodes[overlap]['location'])


def compute_monitoring_expansion_share(episodes, cohort,
                                       window_early, window_late):
    """
    Decompose crisis-location growth into monitoring expansion vs escalation.

    Window and subject travel with the result by construction: the returned
    dict embeds the window bounds, the subject and cohort definitions, all
    four counts, both growths, and the share.
    """
    early_all = locations_in_window(episodes, *window_early)
    late_all = locations_in_window(episodes, *window_late)
    early_cohort = early_all & cohort
    late_cohort = late_all & cohort

    growth_all = len(late_all) - len(early_all)
    growth_cohort = len(late_cohort) - len(early_cohort)
    share_pct = ((growth_all - growth_cohort) / growth_all * 100
                 if growth_all > 0 else None)

    return {
        'window_early': {'start_year': window_early[0], 'end_year': window_early[1]},
        'window_late': {'start_year': window_late[0], 'end_year': window_late[1]},
        'subject_definition': SUBJECT_DEFINITION,
        'cohort_definition': COHORT_DEFINITION,
        'cohort_year': COHORT_YEAR,
        'cohort_size': len(cohort),
        'full_set': {
            'early_count': len(early_all),
            'late_count': len(late_all),
            'growth': growth_all,
        },
        'cohort': {
            'early_count': len(early_cohort),
            'late_count': len(late_cohort),
            'growth': growth_cohort,
        },
        'monitoring_expansion_share_pct': round(share_pct, 1) if share_pct is not None else None,
        'formula': '(full_set.growth - cohort.growth) / full_set.growth * 100',
    }


def main():
    print('=' * 70)
    print('  MONITORING-EXPANSION SHARE (v24 audit A3)')
    print('=' * 70)

    episodes = pd.read_csv(EPISODES_PATH)
    episodes['start'] = episodes['dates'].str.split(',').str[0].str[:7]
    episodes['end'] = episodes['dates'].str.split(',').str[-1].str[:7]
    print(f'  Episodes: {len(episodes)}')

    cohort = get_cohort(HFID_PATH, COHORT_YEAR)
    print(f'  {COHORT_YEAR} cohort: {len(cohort)} locations')

    result = compute_monitoring_expansion_share(
        episodes, cohort, DEFAULT_WINDOW_EARLY, DEFAULT_WINDOW_LATE)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {OUTPUT_PATH}')

    fs, co = result['full_set'], result['cohort']
    print(f"  Full set:  {fs['early_count']} -> {fs['late_count']} "
          f"(+{fs['growth']})")
    print(f"  Cohort:    {co['early_count']} -> {co['late_count']} "
          f"(+{co['growth']})")
    print(f"  Monitoring-expansion share: "
          f"{result['monitoring_expansion_share_pct']}%")


if __name__ == '__main__':
    main()
