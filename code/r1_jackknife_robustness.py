#!/usr/bin/env python3
# @status:   canonical
# @process:  P6-revision
# @paper:    paper1
"""
R1.2 robustness: leave-one-country-out jackknife + archetype-merge check.
=========================================================================

Addresses Nature Food Reviewer 1 comment R1.2 (statistical power / small,
concentrated cells): "conduct some robustness checks that exclude influential
countries, bootstrap your confidence intervals, or aggregate similar trajectory
types."

This script does NOT re-derive the methodology — it imports the canonical
pipeline functions from `reference_transition_analysis.py` and re-applies them
to country-subsetted data, so the jackknife is guaranteed consistent with the
frozen primary analysis (FEWS priority, MAX aggregation, 12-month interpolation).

Three robustness analyses:
  A. Recovery-asymmetry jackknife — recompute the headline ratios
     (Phase 4->3 / 3->4 = 7.0:1; Phase 3->2 / 2->3 = 1.2:1) excluding each
     country one at a time, by summing per-location transition counts.
  B. Gap-compression jackknife — recompute the early(2011) vs late(2023) mean
     inter-episode gap and the linear trend, excluding each country.
  C. Archetype-merge check — collapse the two smallest severe archetypes
     (severe_shock + rapid_onset) into one class and report the distribution;
     note that the transition asymmetry and crisis-staircase findings are
     computed from phase transitions / seasonal-prolonged-protracted sequences
     and are therefore invariant to this relabeling.

Outputs:
  outputs/data/r1_jackknife_robustness.json
  outputs/data/r1_jackknife_by_country.csv

Run:
  python3 code/r1_jackknife_robustness.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Make the canonical pipeline + its config importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from reference_transition_analysis import (  # noqa: E402
    load_hfid, preprocess, interpolate, compute_transitions, compute_key_ratios,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'data')
TRANSITIONS_CSV = os.path.join(OUTPUT_DIR, 'archetype_transitions.csv')
EPISODES_CSV = os.path.join(OUTPUT_DIR, 'episodes.csv')


def _country_of(location):
    """Location key is 'ISO3_ADMIN1' — country is the ISO3 prefix."""
    return location.split('_')[0]


# ============================================================
# A. Recovery-asymmetry jackknife
# ============================================================

def jackknife_recovery_asymmetry():
    print("\n" + "=" * 70)
    print("A. RECOVERY-ASYMMETRY JACKKNIFE (leave-one-country-out)")
    print("=" * 70)

    df_raw = load_hfid()
    df_pre = preprocess(df_raw, priority='fews', aggregation='max')
    df_interp = interpolate(df_pre, max_gap=12)
    trans = compute_transitions(df_interp)
    per_loc = dict(trans['per_location_counts'])

    full = compute_key_ratios(trans['raw_counts'], trans['row_totals'])
    print(f"\n  FULL SAMPLE: ratio_4to3/3to4 = {full['ratio_4to3_over_3to4']} "
          f"(P43={full['P_4to3']}, P34={full['P_3to4']}, "
          f"n43={full['n_4to3']}, n34={full['n_3to4']})")
    print(f"  FULL SAMPLE: ratio_3to2/2to3 = {full['ratio_3to2_over_2to3']}")

    countries = sorted({_country_of(loc) for loc in per_loc})
    rows = []
    for c in countries:
        agg = np.zeros((5, 5), dtype=int)
        n_locs = 0
        for loc, m in per_loc.items():
            if _country_of(loc) != c:
                agg += m
            else:
                n_locs += 1
        rt = agg.sum(axis=1)
        kr = compute_key_ratios(agg, rt)
        rows.append({
            'excluded_country': c,
            'n_locations_excluded': n_locs,
            'ratio_4to3': kr['ratio_4to3_over_3to4'],
            'ratio_3to2': kr['ratio_3to2_over_2to3'],
            'P_4to3': kr['P_4to3'], 'P_3to4': kr['P_3to4'],
            'n_4to3': kr['n_4to3'], 'n_3to4': kr['n_3to4'],
        })

    jk = pd.DataFrame(rows)
    # Finite ratios only (exclude any inf if a removal zeroed a denominator)
    r43 = jk['ratio_4to3'].replace([np.inf, -np.inf], np.nan).dropna()
    r32 = jk['ratio_3to2'].replace([np.inf, -np.inf], np.nan).dropna()

    min43 = jk.loc[r43.idxmin()]
    max43 = jk.loc[r43.idxmax()]
    print(f"\n  Leave-one-country-out range (ratio 4to3): "
          f"{r43.min():.2f} (excl {min43['excluded_country']}) "
          f"to {r43.max():.2f} (excl {max43['excluded_country']})")
    print(f"  Mean across jackknife: {r43.mean():.2f}; "
          f"all {len(r43)} leave-one-out estimates > parity: "
          f"{bool((r43 > 1).all())}")
    print(f"  Leave-one-country-out range (ratio 3to2): "
          f"{r32.min():.2f} to {r32.max():.2f} (mean {r32.mean():.2f})")

    summary = {
        'full_ratio_4to3': full['ratio_4to3_over_3to4'],
        'full_ratio_3to2': full['ratio_3to2_over_2to3'],
        'full_P_4to3': full['P_4to3'], 'full_P_3to4': full['P_3to4'],
        'n_countries': len(countries),
        'ratio_4to3_jackknife_min': round(float(r43.min()), 2),
        'ratio_4to3_jackknife_max': round(float(r43.max()), 2),
        'ratio_4to3_jackknife_mean': round(float(r43.mean()), 2),
        'ratio_4to3_min_excluding': min43['excluded_country'],
        'ratio_4to3_max_excluding': max43['excluded_country'],
        'ratio_4to3_all_above_parity': bool((r43 > 1).all()),
        'ratio_3to2_jackknife_min': round(float(r32.min()), 2),
        'ratio_3to2_jackknife_max': round(float(r32.max()), 2),
        'ratio_3to2_jackknife_mean': round(float(r32.mean()), 2),
    }
    return summary, jk


# ============================================================
# B. Gap-compression jackknife
# ============================================================

def _linfit_slope(years, values):
    """Least-squares slope (months per year); guards tiny samples."""
    if len(years) < 2:
        return float('nan')
    return float(np.polyfit(years, values, 1)[0])


def _gap_endpoints(df, early_year=2011, late_year=2023):
    yearly = df.groupby('year')['gap_months'].mean()
    early = float(yearly.get(early_year, np.nan))
    late = float(yearly.get(late_year, np.nan))
    slope = _linfit_slope(yearly.index.values.astype(float), yearly.values)
    return early, late, slope, yearly


def jackknife_gap_compression():
    print("\n" + "=" * 70)
    print("B. GAP-COMPRESSION JACKKNIFE (leave-one-country-out)")
    print("=" * 70)

    df = pd.read_csv(TRANSITIONS_CSV)
    df['year'] = pd.to_datetime(df['from_start']).dt.year

    early, late, slope, yearly = _gap_endpoints(df)
    print(f"\n  FULL SAMPLE: mean gap {early:.1f} mo (2011) -> {late:.1f} mo (2023); "
          f"trend slope {slope:.2f} mo/yr")
    print("  Yearly mean gap (full sample):")
    print(yearly.round(1).to_string())

    countries = sorted(df['iso3'].dropna().unique())
    rows = []
    for c in countries:
        sub = df[df['iso3'] != c]
        e, l, s, _ = _gap_endpoints(sub)
        rows.append({'excluded_country': c, 'early_2011': round(e, 1),
                     'late_2023': round(l, 1), 'slope_mo_per_yr': round(s, 2)})
    jk = pd.DataFrame(rows)

    e_series = jk['early_2011'].replace([np.inf, -np.inf], np.nan).dropna()
    l_series = jk['late_2023'].replace([np.inf, -np.inf], np.nan).dropna()
    s_series = jk['slope_mo_per_yr'].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\n  Leave-one-out early(2011) range: {e_series.min():.1f}-{e_series.max():.1f}")
    print(f"  Leave-one-out late(2023) range:  {l_series.min():.1f}-{l_series.max():.1f}")
    print(f"  Leave-one-out slope range: {s_series.min():.2f} to {s_series.max():.2f} mo/yr "
          f"(all negative = compression robust: {bool((s_series < 0).all())})")

    summary = {
        'full_early_2011': round(early, 1),
        'full_late_2023': round(late, 1),
        'full_slope_mo_per_yr': round(slope, 2),
        'jackknife_early_min': round(float(e_series.min()), 1),
        'jackknife_early_max': round(float(e_series.max()), 1),
        'jackknife_late_min': round(float(l_series.min()), 1),
        'jackknife_late_max': round(float(l_series.max()), 1),
        'jackknife_slope_min': round(float(s_series.min()), 2),
        'jackknife_slope_max': round(float(s_series.max()), 2),
        'compression_robust_all_negative_slope': bool((s_series < 0).all()),
    }
    return summary, jk


# ============================================================
# C. Archetype-merge robustness
# ============================================================

def archetype_merge_check():
    print("\n" + "=" * 70)
    print("C. ARCHETYPE-MERGE CHECK (severe_shock + rapid_onset)")
    print("=" * 70)

    ep = pd.read_csv(EPISODES_CSV)
    n = len(ep)
    orig = ep['archetype'].value_counts()
    orig_pct = (orig / n * 100).round(1)

    merged = ep['archetype'].replace({
        'severe_shock': 'severe_acute',
        'rapid_onset': 'severe_acute',
    })
    mg = merged.value_counts()
    mg_pct = (mg / n * 100).round(1)

    print(f"\n  Episodes: {n}")
    print("\n  Original distribution:")
    for k in orig.index:
        print(f"    {k}: {orig[k]} ({orig_pct[k]}%)")
    print("\n  After merging severe_shock + rapid_onset -> severe_acute:")
    for k in mg.index:
        print(f"    {k}: {mg[k]} ({mg_pct[k]}%)")

    combined_n = int(orig.get('severe_shock', 0) + orig.get('rapid_onset', 0))
    print(f"\n  Combined severe_acute class: {combined_n} episodes "
          f"({round(combined_n / n * 100, 1)}%)")
    print("\n  NOTE: the headline recovery asymmetry (7.0:1) is computed from "
          "monthly PHASE transitions and the crisis staircase from "
          "seasonal->prolonged->protracted archetype sequences. Neither uses "
          "the severe_shock / rapid_onset labels, so both are invariant to "
          "this merge. The merge only affects the typology distribution above.")

    return {
        'n_episodes': int(n),
        'severe_shock_n': int(orig.get('severe_shock', 0)),
        'rapid_onset_n': int(orig.get('rapid_onset', 0)),
        'merged_severe_acute_n': combined_n,
        'merged_severe_acute_pct': round(combined_n / n * 100, 1),
        'distribution_original': {k: int(v) for k, v in orig.items()},
        'distribution_merged': {k: int(v) for k, v in mg.items()},
        'note': ('Recovery asymmetry and crisis staircase are computed from '
                 'phase transitions / seasonal-prolonged-protracted sequences '
                 'and are invariant to the severe_shock+rapid_onset merge.'),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("R1.2 ROBUSTNESS — JACKKNIFE + ARCHETYPE MERGE")
    print("=" * 70)

    a_summary, a_table = jackknife_recovery_asymmetry()
    b_summary, b_table = jackknife_gap_compression()
    c_summary = archetype_merge_check()

    # Save per-country jackknife table (asymmetry + gap merged on country)
    a_table = a_table.rename(columns={'excluded_country': 'iso3'})
    b_table = b_table.rename(columns={'excluded_country': 'iso3'})
    by_country = a_table.merge(b_table, on='iso3', how='outer')
    by_country.to_csv(
        os.path.join(OUTPUT_DIR, 'r1_jackknife_by_country.csv'), index=False)

    out = {
        'description': ('R1.2 robustness: leave-one-country-out jackknife of '
                        'headline ratios + gap compression, and archetype-merge '
                        'check. Consistent with primary analysis (FEWS priority, '
                        'MAX aggregation, 12-month interpolation).'),
        'recovery_asymmetry_jackknife': a_summary,
        'gap_compression_jackknife': b_summary,
        'archetype_merge': c_summary,
    }
    out_path = os.path.join(OUTPUT_DIR, 'r1_jackknife_robustness.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved: {out_path}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'r1_jackknife_by_country.csv')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
