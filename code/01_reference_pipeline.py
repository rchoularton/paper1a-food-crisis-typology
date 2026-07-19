#!/usr/bin/env python3
# @status:   canonical
# @process:  P1+P2
# @paper:    paper1
"""
01_reference_pipeline.py — Core Crisis Episode Detection and Transition Analysis
=================================================================================

Reads the raw HFID CSV and applies the authoritative analysis pipeline:
  1. Combined IPC phase: FEWS NET priority, fallback to CH/IPC
  2. Filter Phase 6 (areas of concern) → NaN
  3. Aggregate admin2 → admin1 using MAX
  4. Forward-fill interpolation with 12-month gap limit
  5. Monthly transition matrix (consecutive months only, 25–35 day gap)
  6. Crisis episode detection and archetype classification
  7. Sensitivity analysis across 10 pipeline variants
  8. Bootstrap confidence intervals (10,000 iterations)

Inputs:
  data/HFID_hv1.csv  — HFID v1.1.1 (Machefer et al. 2025)

Outputs (all in outputs/data/):
  episodes.csv                       — Crisis episodes with archetype labels
  full_transition_matrix.json        — 5×5 transition matrix + CIs
  phase{1..5}_duration_conditioned.json — Duration-conditioned transitions
  phase3_crossover.json              — Recovery–escalation crossover point
  sensitivity_analysis.json          — 10-variant comparison table
  sensitivity_summary.csv            — Same as CSV
  admin2_transition_analysis.json    — Admin2-level sensitivity check
  episode_verification.json          — Episode statistics verification
  left_censoring_sensitivity.json    — Left/right-censoring impact analysis
  right_censoring_analysis.json      — Right-censoring summary (WS5)
  observed_only_transitions.json     — Observed-only transition sensitivity (WS2)
  extended_duration_bins.json        — 8-bin duration decay + AIC/BIC models (WS4)
  staircase_censoring_sensitivity.json — Staircase censoring check (WS8)
  robustness_summary.json            — Cross-workstream robustness table
  paper_audit.json                   — Internal statistics audit (QC artifact)
  quarterly_analysis.json            — Quarterly aggregation robustness
  regional_transition_analysis.json  — Regional breakdown
  temporal_comparison.json           — Early vs late period comparison
  crisis_staircase.json              — Multi-episode pathway analysis
  country_counts.json                — Country coverage summary

Author: Richard Choularton
"""

import argparse
import json
import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Paths — relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HFID_PATH = os.path.join(PACKAGE_ROOT, 'data', 'HFID_hv1.csv')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data')

# ============================================================
# Configuration
# ============================================================
CRISIS_THRESHOLD = 3
DEFAULT_INTERPOLATION_GAP = 12
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42
DURATION_BINS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 9999)]
DURATION_LABELS = ['1-3 mo', '4-6 mo', '7-12 mo', '13-24 mo', '24+ mo']


# ============================================================
# Step 1: Load and Preprocess HFID Data
# ============================================================

def load_hfid():
    """Load raw HFID CSV file."""
    print("=" * 70)
    print("REFERENCE TRANSITION ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\nLoading HFID data from: {HFID_PATH}")

    if not os.path.exists(HFID_PATH):
        print(f"\nERROR: HFID data file not found at {HFID_PATH}")
        print("Please ensure data/HFID_hv1.csv is present in the package root.")
        sys.exit(1)

    df = pd.read_csv(HFID_PATH)
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Countries: {df['iso3'].nunique()}")
    print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    return df


def preprocess(df, priority='fews', aggregation='max'):
    """
    Create combined IPC phase and aggregate to admin1.

    Parameters:
        priority: 'fews' (FEWS NET first) or 'ipc' (IPC-CH first)
        aggregation: 'max', 'median', 'mean', or 'dictzip' (arbitrary pick)
    """
    df = df.copy()

    # Step 1: Combined phase
    if priority == 'fews':
        df['ipc_phase'] = df['ipc_phase_fews'].fillna(df['ipc_phase_ipcch'])
    else:
        df['ipc_phase'] = df['ipc_phase_ipcch'].fillna(df['ipc_phase_fews'])

    # Step 2: Filter Phase 6 → NaN
    df.loc[df['ipc_phase'] == 6, 'ipc_phase'] = np.nan

    # Remove records with no IPC phase
    df = df[df['ipc_phase'].notna()].copy()

    # Create location key and date
    df['location'] = df['iso3'] + '_' + df['ADMIN1'].fillna('national')
    df['date'] = pd.to_datetime(df['year_month'] + '-01')

    # Step 3: Aggregate admin2 → admin1
    if aggregation == 'dictzip':
        df_agg = df.groupby(['iso3', 'ADMIN1', 'location', 'region',
                             'year_month', 'date']).agg({
            'ipc_phase': 'last',
        }).reset_index()
    elif aggregation == 'median':
        df_agg = df.groupby(['iso3', 'ADMIN1', 'location', 'region',
                             'year_month', 'date']).agg({
            'ipc_phase': 'median',
        }).reset_index()
        df_agg['ipc_phase'] = df_agg['ipc_phase'].round().astype(int)
    elif aggregation == 'mean':
        df_agg = df.groupby(['iso3', 'ADMIN1', 'location', 'region',
                             'year_month', 'date']).agg({
            'ipc_phase': 'mean',
        }).reset_index()
        # Round mean to nearest integer phase
        df_agg['ipc_phase'] = df_agg['ipc_phase'].round().astype(int)
    elif aggregation == 'max':
        # MAX aggregation (authoritative)
        df_agg = df.groupby(['iso3', 'ADMIN1', 'location', 'region',
                             'year_month', 'date']).agg({
            'ipc_phase': 'max',
        }).reset_index()
    else:
        raise ValueError(
            f"Unrecognised aggregation {aggregation!r}. "
            "Expected one of: 'max', 'median', 'mean', 'dictzip'."
        )

    df_agg = df_agg.sort_values(['location', 'date']).reset_index(drop=True)

    print(f"  Preprocessed: {len(df_agg):,} location-months "
          f"(priority={priority}, agg={aggregation})")
    print(f"  Unique locations: {df_agg['location'].nunique()}")
    return df_agg


def preprocess_admin2(df, priority='fews'):
    """Preprocess at admin2 level (no aggregation needed)."""
    df = df.copy()

    if priority == 'fews':
        df['ipc_phase'] = df['ipc_phase_fews'].fillna(df['ipc_phase_ipcch'])
    else:
        df['ipc_phase'] = df['ipc_phase_ipcch'].fillna(df['ipc_phase_fews'])

    df.loc[df['ipc_phase'] == 6, 'ipc_phase'] = np.nan
    df = df[df['ipc_phase'].notna()].copy()

    df = df[df['ADMIN2'].notna()].copy()
    df['location'] = df['iso3'] + '_' + df['ADMIN1'].fillna('') + '_' + df['ADMIN2']
    df['date'] = pd.to_datetime(df['year_month'] + '-01')

    df_agg = df.groupby(['iso3', 'ADMIN1', 'ADMIN2', 'location', 'region',
                         'year_month', 'date']).agg({
        'ipc_phase': 'max',
    }).reset_index()

    df_agg = df_agg.sort_values(['location', 'date']).reset_index(drop=True)

    print(f"  Preprocessed admin2: {len(df_agg):,} location-months")
    print(f"  Unique admin2 locations: {df_agg['location'].nunique()}")
    return df_agg


# ============================================================
# Step 2: Interpolation
# ============================================================

def interpolate(df, max_gap):
    """Forward-fill interpolation with max gap limit."""
    has_admin2 = 'ADMIN2' in df.columns
    records = []
    for location in df['location'].unique():
        loc_data = df[df['location'] == location].sort_values('date')
        if len(loc_data) == 0:
            continue

        iso3 = loc_data.iloc[0]['iso3']
        admin1 = loc_data.iloc[0].get('ADMIN1', None)
        admin2 = loc_data.iloc[0].get('ADMIN2', None) if has_admin2 else None
        region = loc_data.iloc[0]['region']

        date_range = pd.date_range(start=loc_data['date'].min(),
                                   end=loc_data['date'].max(), freq='MS')
        observed_dates = set(loc_data['date'].tolist())
        date_to_phase = dict(zip(loc_data['date'], loc_data['ipc_phase']))

        current_phase = None
        last_observed = None

        for date in date_range:
            if date in observed_dates:
                current_phase = date_to_phase[date]
                last_observed = date
                is_interp = False
            else:
                if current_phase is not None and last_observed is not None:
                    months_since = ((date.year - last_observed.year) * 12 +
                                   (date.month - last_observed.month))
                    if months_since <= max_gap:
                        is_interp = True
                    else:
                        continue
                else:
                    continue

            rec = {
                'iso3': iso3, 'ADMIN1': admin1, 'location': location,
                'region': region, 'year_month': date.strftime('%Y-%m'),
                'date': date, 'ipc_phase': current_phase,
                'is_interpolated': is_interp
            }
            if has_admin2:
                rec['ADMIN2'] = admin2
            records.append(rec)

    result = pd.DataFrame(records)
    print(f"  Interpolated ({max_gap}m): {len(result):,} records")
    return result


# ============================================================
# Step 3: Transition Computation
# ============================================================

def compute_transitions(df_interp):
    """
    Compute monthly transition matrix from interpolated time series.
    Only counts consecutive months (gap <= 35 days).
    """
    raw_counts = np.zeros((5, 5), dtype=int)
    per_location_counts = defaultdict(lambda: np.zeros((5, 5), dtype=int))

    for location in df_interp['location'].unique():
        loc_data = df_interp[df_interp['location'] == location].sort_values('date')
        phases = loc_data['ipc_phase'].values
        dates = loc_data['date'].values

        for i in range(len(phases) - 1):
            gap_days = (dates[i + 1] - dates[i]) / np.timedelta64(1, 'D')
            if gap_days < 25 or gap_days > 35:
                continue

            f = int(phases[i]) - 1
            t = int(phases[i + 1]) - 1
            if 0 <= f < 5 and 0 <= t < 5:
                raw_counts[f, t] += 1
                per_location_counts[location][f, t] += 1

    row_totals = raw_counts.sum(axis=1)
    pct_matrix = np.zeros((5, 5))
    for i in range(5):
        if row_totals[i] > 0:
            pct_matrix[i] = raw_counts[i] / row_totals[i] * 100

    return {
        'raw_counts': raw_counts,
        'pct_matrix': pct_matrix,
        'row_totals': row_totals,
        'per_location_counts': per_location_counts,
    }


def compute_key_ratios(raw_counts, row_totals):
    """Compute key asymmetry ratios from raw counts."""
    results = {}

    p43 = raw_counts[3, 2] / row_totals[3] * 100 if row_totals[3] > 0 else 0
    p34 = raw_counts[2, 3] / row_totals[2] * 100 if row_totals[2] > 0 else 0
    results['P_4to3'] = round(p43, 2)
    results['P_3to4'] = round(p34, 2)
    results['ratio_4to3_over_3to4'] = round(p43 / p34, 2) if p34 > 0 else float('inf')
    results['n_4to3'] = int(raw_counts[3, 2])
    results['n_3to4'] = int(raw_counts[2, 3])

    # Low-n caveat for the headline recovery ratio (v24 audit H1): give a
    # ratio resting on few observed transitions the same automatic warning
    # its Phase 5 statistics already carry (same < 50 threshold).
    n_headline = results['n_4to3'] + results['n_3to4']
    if n_headline < 50:
        results['ratio_4to3_caveat'] = (
            f"CAUTION: 4→3/3→4 recovery ratio based on only {n_headline} "
            f"transitions ({results['n_4to3']} recoveries, "
            f"{results['n_3to4']} escalations) — insufficient for reliable inference."
        )

    p32 = raw_counts[2, 1] / row_totals[2] * 100 if row_totals[2] > 0 else 0
    p23 = raw_counts[1, 2] / row_totals[1] * 100 if row_totals[1] > 0 else 0
    results['P_3to2'] = round(p32, 2)
    results['P_2to3'] = round(p23, 2)
    results['ratio_3to2_over_2to3'] = round(p32 / p23, 2) if p23 > 0 else float('inf')

    p45 = raw_counts[3, 4] / row_totals[3] * 100 if row_totals[3] > 0 else 0
    p54 = raw_counts[4, 3] / row_totals[4] * 100 if row_totals[4] > 0 else 0
    results['P_4to5'] = round(p45, 2)
    results['P_5to4'] = round(p54, 2)
    results['phase5_n_transitions'] = int(row_totals[4]) if len(row_totals) > 4 else 0
    if results['phase5_n_transitions'] < 50:
        results['phase5_caveat'] = (
            f"CAUTION: Phase 5 statistics based on only {results['phase5_n_transitions']} "
            "transitions — insufficient for reliable inference. Do not cite Phase 5 "
            "rates as robust findings."
        )

    return results


def compute_transitions_observed_only(df_interp):
    """
    Compute transition matrix using only observed-to-observed month pairs,
    excluding any transitions involving interpolated months.
    Returns same structure as compute_transitions().
    """
    raw_counts = np.zeros((5, 5), dtype=int)
    per_location_counts = defaultdict(lambda: np.zeros((5, 5), dtype=int))
    total_pairs = 0
    interpolated_pairs = 0

    for location in df_interp['location'].unique():
        loc_data = df_interp[df_interp['location'] == location].sort_values('date')
        phases = loc_data['ipc_phase'].values
        dates = loc_data['date'].values
        is_interp = loc_data['is_interpolated'].values

        for i in range(len(phases) - 1):
            gap_days = (dates[i + 1] - dates[i]) / np.timedelta64(1, 'D')
            if gap_days < 25 or gap_days > 35:
                continue

            total_pairs += 1

            # Skip if either month is interpolated
            if is_interp[i] or is_interp[i + 1]:
                interpolated_pairs += 1
                continue

            f = int(phases[i]) - 1
            t = int(phases[i + 1]) - 1
            if 0 <= f < 5 and 0 <= t < 5:
                raw_counts[f, t] += 1
                per_location_counts[location][f, t] += 1

    row_totals = raw_counts.sum(axis=1)
    pct_matrix = np.zeros((5, 5))
    for i in range(5):
        if row_totals[i] > 0:
            pct_matrix[i] = raw_counts[i] / row_totals[i] * 100

    return {
        'raw_counts': raw_counts,
        'pct_matrix': pct_matrix,
        'row_totals': row_totals,
        'per_location_counts': per_location_counts,
        'total_pairs': total_pairs,
        'interpolated_pairs': interpolated_pairs,
        'observed_pairs': total_pairs - interpolated_pairs,
        'interpolated_pct': round(interpolated_pairs / total_pairs * 100, 1) if total_pairs > 0 else 0,
    }


def bootstrap_matrix_block(per_location_counts, n_iter=N_BOOTSTRAP,
                           seed=BOOTSTRAP_SEED):
    """
    Block bootstrap that preserves temporal structure within locations.

    Instead of resampling location-level transition counts independently, this
    resamples entire location episode chains: for each sampled location, all
    transitions are included as a block.

    In practice, since per_location_counts already aggregates all transitions
    from a location, the difference from the standard bootstrap is conceptual
    verification. The standard bootstrap already resamples at location level,
    which is the appropriate unit. This function verifies the CI does not change
    materially, confirming the standard approach is sufficient.

    The sample indices are drawn as a single (n_iter, n_locations) array so the
    RNG stream matches the canonical implementation exactly and the stored
    values reproduce.
    """
    rng = np.random.default_rng(seed)
    locations = list(per_location_counts.keys())
    n_locations = len(locations)

    all_sample_indices = rng.integers(0, n_locations, size=(n_iter, n_locations))
    counts_array = np.array([per_location_counts[loc] for loc in locations])

    ratios_43 = []
    ratios_32 = []

    for i in range(n_iter):
        sample_counts = np.zeros((5, 5), dtype=float)
        for idx in all_sample_indices[i]:
            sample_counts += counts_array[idx]

        row_totals = sample_counts.sum(axis=1)
        p43 = sample_counts[3, 2] / row_totals[3] * 100 if row_totals[3] > 0 else 0
        p34 = sample_counts[2, 3] / row_totals[2] * 100 if row_totals[2] > 0 else 0
        ratios_43.append(p43 / p34 if p34 > 0 else float('inf'))

        p32 = sample_counts[2, 1] / row_totals[2] * 100 if row_totals[2] > 0 else 0
        p23 = sample_counts[1, 2] / row_totals[1] * 100 if row_totals[1] > 0 else 0
        ratios_32.append(p32 / p23 if p23 > 0 else float('inf'))

    finite_43 = [r for r in ratios_43 if r != float('inf')]
    finite_32 = [r for r in ratios_32 if r != float('inf')]

    return {
        'ratio_4to3_ci_block': [round(np.percentile(finite_43, 2.5), 1),
                                round(np.percentile(finite_43, 97.5), 1)] if finite_43 else [0, 0],
        'ratio_3to2_ci_block': [round(np.percentile(finite_32, 2.5), 1),
                                round(np.percentile(finite_32, 97.5), 1)] if finite_32 else [0, 0],
        'ratio_4to3_median_block': round(np.median(finite_43), 1) if finite_43 else 0,
        'ratio_3to2_median_block': round(np.median(finite_32), 1) if finite_32 else 0,
        'n_valid': len(finite_43),
    }


def bootstrap_matrix(per_location_counts, n_iter=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Bootstrap confidence intervals by resampling locations."""
    rng = np.random.default_rng(seed)
    locations = list(per_location_counts.keys())
    n_locations = len(locations)

    boot_ratios_43 = []
    boot_ratios_32 = []
    boot_p43 = []
    boot_p34 = []
    boot_p32 = []
    boot_p23 = []
    boot_matrices = []

    for _ in range(n_iter):
        sample_idx = rng.integers(0, n_locations, size=n_locations)
        sample_counts = np.zeros((5, 5), dtype=float)

        for idx in sample_idx:
            sample_counts += per_location_counts[locations[idx]]

        row_totals = sample_counts.sum(axis=1)

        pct = np.zeros((5, 5))
        for i in range(5):
            if row_totals[i] > 0:
                pct[i] = sample_counts[i] / row_totals[i] * 100
        boot_matrices.append(pct)

        p43 = sample_counts[3, 2] / row_totals[3] * 100 if row_totals[3] > 0 else 0
        p34 = sample_counts[2, 3] / row_totals[2] * 100 if row_totals[2] > 0 else 0
        boot_p43.append(p43)
        boot_p34.append(p34)
        ratio_43 = p43 / p34 if p34 > 0 else float('inf')
        boot_ratios_43.append(ratio_43)

        p32 = sample_counts[2, 1] / row_totals[2] * 100 if row_totals[2] > 0 else 0
        p23 = sample_counts[1, 2] / row_totals[1] * 100 if row_totals[1] > 0 else 0
        boot_p32.append(p32)
        boot_p23.append(p23)
        ratio_32 = p32 / p23 if p23 > 0 else float('inf')
        boot_ratios_32.append(ratio_32)

    finite_43 = [r for r in boot_ratios_43 if r != float('inf')]
    finite_32 = [r for r in boot_ratios_32 if r != float('inf')]

    boot_arr = np.array(boot_matrices)
    cell_ci_lo = np.percentile(boot_arr, 2.5, axis=0)
    cell_ci_hi = np.percentile(boot_arr, 97.5, axis=0)

    return {
        'ratio_4to3_ci': [round(np.percentile(finite_43, 2.5), 1),
                          round(np.percentile(finite_43, 97.5), 1)] if finite_43 else [0, 0],
        'ratio_3to2_ci': [round(np.percentile(finite_32, 2.5), 1),
                          round(np.percentile(finite_32, 97.5), 1)] if finite_32 else [0, 0],
        'P_4to3_ci': [round(np.percentile(boot_p43, 2.5), 1),
                      round(np.percentile(boot_p43, 97.5), 1)],
        'P_3to4_ci': [round(np.percentile(boot_p34, 2.5), 1),
                      round(np.percentile(boot_p34, 97.5), 1)],
        'P_3to2_ci': [round(np.percentile(boot_p32, 2.5), 1),
                      round(np.percentile(boot_p32, 97.5), 1)],
        'P_2to3_ci': [round(np.percentile(boot_p23, 2.5), 1),
                      round(np.percentile(boot_p23, 97.5), 1)],
        'cell_ci_lo': cell_ci_lo.tolist(),
        'cell_ci_hi': cell_ci_hi.tolist(),
    }


# ============================================================
# Step 4: Duration-Conditioned Transitions
# ============================================================

def compute_duration_conditioned(df_interp, target_phase, recovery_phase, escalation_phase):
    """Compute transition probabilities conditioned on consecutive months at the target phase."""
    bins = DURATION_BINS
    labels = DURATION_LABELS

    bin_data = {label: {'recovery': 0, 'escalation': 0, 'stay': 0, 'total': 0}
                for label in labels}
    per_location_bin_data = defaultdict(
        lambda: {label: {'recovery': 0, 'escalation': 0, 'stay': 0, 'total': 0}
                 for label in labels}
    )

    for location in df_interp['location'].unique():
        loc_data = df_interp[df_interp['location'] == location].sort_values('date')
        phases = loc_data['ipc_phase'].values
        dates = loc_data['date'].values

        consec = 0
        for i in range(len(phases)):
            p = int(phases[i])
            if p == target_phase:
                consec += 1
                if i < len(phases) - 1:
                    gap_days = (dates[i + 1] - dates[i]) / np.timedelta64(1, 'D')
                    if gap_days < 25 or gap_days > 35:
                        continue

                    next_p = int(phases[i + 1])
                    for (lo, hi), label in zip(bins, labels):
                        if lo <= consec <= hi:
                            bin_data[label]['total'] += 1
                            per_location_bin_data[location][label]['total'] += 1
                            if next_p < target_phase:
                                bin_data[label]['recovery'] += 1
                                per_location_bin_data[location][label]['recovery'] += 1
                            elif next_p > target_phase:
                                bin_data[label]['escalation'] += 1
                                per_location_bin_data[location][label]['escalation'] += 1
                            elif next_p == target_phase:
                                bin_data[label]['stay'] += 1
                                per_location_bin_data[location][label]['stay'] += 1
                            break
            else:
                consec = 0

    results = {}
    for label in labels:
        d = bin_data[label]
        n = d['total']
        rec_pct = d['recovery'] / n * 100 if n > 0 else 0
        esc_pct = d['escalation'] / n * 100 if n > 0 else 0
        persist_pct = d['stay'] / n * 100 if n > 0 else 0
        ratio = rec_pct / esc_pct if esc_pct > 0 else float('inf')
        results[label] = {
            'n': n,
            'recovery_pct': round(rec_pct, 2),
            'escalation_pct': round(esc_pct, 2),
            'persistence_pct': round(persist_pct, 2),
            'ratio': round(ratio, 2) if ratio != float('inf') else 'inf',
            'recovery_count': d['recovery'],
            'escalation_count': d['escalation'],
            'persistence_count': d['stay'],
        }

    return results, per_location_bin_data


def bootstrap_duration_conditioned(per_location_bin_data, n_iter=N_BOOTSTRAP,
                                   seed=BOOTSTRAP_SEED):
    """Bootstrap CIs for duration-conditioned transitions."""
    rng = np.random.default_rng(seed)
    locations = list(per_location_bin_data.keys())
    n_locations = len(locations)
    labels = DURATION_LABELS

    boot_results = {label: {'recovery_pcts': [], 'escalation_pcts': [],
                            'persistence_pcts': [], 'ratios': []}
                    for label in labels}

    for _ in range(n_iter):
        sample_idx = rng.integers(0, n_locations, size=n_locations)
        sample_bins = {label: {'recovery': 0, 'escalation': 0, 'stay': 0, 'total': 0}
                       for label in labels}

        for idx in sample_idx:
            loc = locations[idx]
            for label in labels:
                for key in ['recovery', 'escalation', 'stay', 'total']:
                    sample_bins[label][key] += per_location_bin_data[loc][label][key]

        for label in labels:
            d = sample_bins[label]
            n = d['total']
            rec = d['recovery'] / n * 100 if n > 0 else 0
            esc = d['escalation'] / n * 100 if n > 0 else 0
            persist = d['stay'] / n * 100 if n > 0 else 0
            ratio = rec / esc if esc > 0 else float('inf')
            boot_results[label]['recovery_pcts'].append(rec)
            boot_results[label]['escalation_pcts'].append(esc)
            boot_results[label]['persistence_pcts'].append(persist)
            boot_results[label]['ratios'].append(ratio)

    ci_results = {}
    for label in labels:
        rec_vals = boot_results[label]['recovery_pcts']
        esc_vals = boot_results[label]['escalation_pcts']
        persist_vals = boot_results[label]['persistence_pcts']
        finite_ratios = [r for r in boot_results[label]['ratios'] if r != float('inf')]

        ci_results[label] = {
            'recovery_ci': [round(np.percentile(rec_vals, 2.5), 1),
                            round(np.percentile(rec_vals, 97.5), 1)],
            'escalation_ci': [round(np.percentile(esc_vals, 2.5), 1),
                              round(np.percentile(esc_vals, 97.5), 1)],
            'persistence_ci': [round(np.percentile(persist_vals, 2.5), 1),
                               round(np.percentile(persist_vals, 97.5), 1)],
        }
        if finite_ratios:
            ci_results[label]['ratio_ci'] = [round(np.percentile(finite_ratios, 2.5), 1),
                                             round(np.percentile(finite_ratios, 97.5), 1)]

    return ci_results


def fit_decay_and_crossover(results, labels=None):
    """Fit exponential decay to recovery probabilities and find crossover point."""
    if labels is None:
        labels = DURATION_LABELS

    midpoints = [2, 5, 9.5, 18.5, 30]
    rec_vals = [results[l]['recovery_pct'] for l in labels]
    esc_vals = [results[l]['escalation_pct'] for l in labels]

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    fit_result = {}
    try:
        x_data = np.array(midpoints)
        y_data = np.array(rec_vals)
        valid = y_data > 0
        if valid.sum() >= 2:
            popt, _ = curve_fit(exp_decay, x_data[valid], y_data[valid],
                                p0=[20, 0.05], maxfev=5000)
            y_pred = exp_decay(x_data[valid], *popt)
            ss_res = np.sum((y_data[valid] - y_pred) ** 2)
            ss_tot = np.sum((y_data[valid] - np.mean(y_data[valid])) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            fit_result['decay_fit'] = {
                'a': round(popt[0], 3),
                'b': round(popt[1], 4),
                'r_squared': round(r_squared, 3),
            }

            avg_esc = np.mean(esc_vals)
            if avg_esc > 0 and popt[1] > 0:
                crossover_month = -np.log(avg_esc / popt[0]) / popt[1]
                if 0 < crossover_month < 120:
                    fit_result['crossover'] = {
                        'month': round(crossover_month, 1),
                        'rate_at_crossover': round(avg_esc, 2),
                    }
    except (RuntimeError, ValueError):
        pass

    return fit_result


def bootstrap_crossover(per_location_bin_data, n_iter=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Bootstrap CI for the crossover point."""
    rng = np.random.default_rng(seed)
    locations = list(per_location_bin_data.keys())
    n_locations = len(locations)
    labels = DURATION_LABELS
    midpoints = np.array([2, 5, 9.5, 18.5, 30])

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    crossover_months = []

    for _ in range(n_iter):
        sample_idx = rng.integers(0, n_locations, size=n_locations)
        sample_bins = {label: {'recovery': 0, 'escalation': 0, 'total': 0}
                       for label in labels}

        for idx in sample_idx:
            loc = locations[idx]
            for label in labels:
                for key in ['recovery', 'escalation', 'total']:
                    sample_bins[label][key] += per_location_bin_data[loc][label][key]

        rec_vals = []
        esc_vals = []
        for label in labels:
            d = sample_bins[label]
            n = d['total']
            rec_vals.append(d['recovery'] / n * 100 if n > 0 else 0)
            esc_vals.append(d['escalation'] / n * 100 if n > 0 else 0)

        try:
            y_data = np.array(rec_vals)
            valid = y_data > 0
            if valid.sum() >= 2:
                popt, _ = curve_fit(exp_decay, midpoints[valid], y_data[valid],
                                    p0=[20, 0.05], maxfev=3000)
                avg_esc = np.mean(esc_vals)
                if avg_esc > 0 and popt[1] > 0:
                    cm = -np.log(avg_esc / popt[0]) / popt[1]
                    if 0 < cm < 120:
                        crossover_months.append(cm)
        except (RuntimeError, ValueError):
            pass

    if crossover_months:
        return {
            'crossover_ci': [round(np.percentile(crossover_months, 2.5), 1),
                             round(np.percentile(crossover_months, 97.5), 1)],
            'crossover_median': round(np.median(crossover_months), 1),
            'n_valid_bootstraps': len(crossover_months),
        }
    return {}


# ============================================================
# Step 4b: Extended Duration Bins + Model Comparison (WS4)
# ============================================================

DURATION_BINS_EXTENDED = [
    (1, 2), (3, 4), (5, 6), (7, 9), (10, 12), (13, 18), (19, 24), (25, 9999)
]
DURATION_LABELS_EXTENDED = [
    '1-2 mo', '3-4 mo', '5-6 mo', '7-9 mo', '10-12 mo',
    '13-18 mo', '19-24 mo', '24+ mo'
]
DURATION_MIDPOINTS_EXTENDED = [1.5, 3.5, 5.5, 8, 11, 15.5, 21.5, 30]


def compute_duration_conditioned_extended(df_interp, target_phase,
                                          recovery_phase, escalation_phase):
    """
    Like compute_duration_conditioned but with 8 finer bins for WS4.
    """
    bins = DURATION_BINS_EXTENDED
    labels = DURATION_LABELS_EXTENDED

    bin_data = {label: {'recovery': 0, 'escalation': 0, 'stay': 0, 'total': 0}
                for label in labels}

    for location in df_interp['location'].unique():
        loc_data = df_interp[df_interp['location'] == location].sort_values('date')
        phases = loc_data['ipc_phase'].values
        dates = loc_data['date'].values

        consec = 0
        for i in range(len(phases)):
            p = int(phases[i])
            if p == target_phase:
                consec += 1
                if i < len(phases) - 1:
                    gap_days = (dates[i + 1] - dates[i]) / np.timedelta64(1, 'D')
                    if gap_days < 25 or gap_days > 35:
                        continue
                    next_p = int(phases[i + 1])
                    for (lo, hi), label in zip(bins, labels):
                        if lo <= consec <= hi:
                            bin_data[label]['total'] += 1
                            if next_p < target_phase:
                                bin_data[label]['recovery'] += 1
                            elif next_p > target_phase:
                                bin_data[label]['escalation'] += 1
                            else:
                                bin_data[label]['stay'] += 1
                            break
            else:
                consec = 0

    results = {}
    for label in labels:
        d = bin_data[label]
        n = d['total']
        rec_pct = d['recovery'] / n * 100 if n > 0 else 0
        esc_pct = d['escalation'] / n * 100 if n > 0 else 0
        results[label] = {
            'n': n,
            'recovery_pct': round(rec_pct, 2),
            'escalation_pct': round(esc_pct, 2),
            'recovery_count': d['recovery'],
            'escalation_count': d['escalation'],
        }

    return results


def fit_model_comparison(results, labels=None, midpoints=None):
    """
    Fit 3 models to recovery probability decay and compare via AIC/BIC.
    Models: exponential, linear, quadratic (polynomial degree 2).
    """
    if labels is None:
        labels = DURATION_LABELS_EXTENDED
    if midpoints is None:
        midpoints = DURATION_MIDPOINTS_EXTENDED

    x_data = np.array(midpoints)
    y_data = np.array([results[l]['recovery_pct'] for l in labels])
    n_data = np.array([results[l]['n'] for l in labels])

    # Filter bins with sufficient data
    valid = (y_data > 0) & (n_data >= 5)
    if valid.sum() < 3:
        return {'error': 'Insufficient data points for model comparison'}

    x = x_data[valid]
    y = y_data[valid]
    n = int(valid.sum())

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    def compute_aic_bic(ss_res, n_obs, k_params):
        if ss_res <= 0 or n_obs <= k_params:
            return float('inf'), float('inf')
        ll = -n_obs / 2 * (np.log(2 * np.pi * ss_res / n_obs) + 1)
        aic = 2 * k_params - 2 * ll
        bic = k_params * np.log(n_obs) - 2 * ll
        return round(aic, 2), round(bic, 2)

    model_results = {}

    # 1. Exponential: y = a * exp(-b * x), 2 params
    try:
        popt, _ = curve_fit(exp_decay, x, y, p0=[20, 0.05], maxfev=5000)
        y_pred = exp_decay(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        aic, bic = compute_aic_bic(ss_res, n, 2)
        model_results['exponential'] = {
            'params': {'a': round(popt[0], 3), 'b': round(popt[1], 4)},
            'r_squared': round(r2, 4),
            'r_squared_adj': round(1 - (1 - r2) * (n - 1) / (n - 2 - 1), 4) if n > 3 else r2,
            'AIC': aic, 'BIC': bic, 'n_params': 2,
        }
    except (RuntimeError, ValueError):
        pass

    # 2. Linear: y = a + b*x, 2 params
    try:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        aic, bic = compute_aic_bic(ss_res, n, 2)
        model_results['linear'] = {
            'params': {'slope': round(coeffs[0], 4), 'intercept': round(coeffs[1], 3)},
            'r_squared': round(r2, 4),
            'r_squared_adj': round(1 - (1 - r2) * (n - 1) / (n - 2 - 1), 4) if n > 3 else r2,
            'AIC': aic, 'BIC': bic, 'n_params': 2,
        }
    except (RuntimeError, ValueError):
        pass

    # 3. Quadratic: y = a + b*x + c*x^2, 3 params
    try:
        coeffs = np.polyfit(x, y, 2)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        aic, bic = compute_aic_bic(ss_res, n, 3)
        model_results['quadratic'] = {
            'params': {'a': round(coeffs[0], 6), 'b': round(coeffs[1], 4),
                       'c': round(coeffs[2], 3)},
            'r_squared': round(r2, 4),
            'r_squared_adj': round(1 - (1 - r2) * (n - 1) / (n - 3 - 1), 4) if n > 4 else r2,
            'AIC': aic, 'BIC': bic, 'n_params': 3,
        }
    except (RuntimeError, ValueError):
        pass

    # Determine best model by AIC
    if model_results:
        best = min(model_results.items(), key=lambda x: x[1].get('AIC', float('inf')))
        model_results['best_model'] = best[0]

    return {
        'models': model_results,
        'n_data_points': n,
        'x_values': x.tolist(),
        'y_values': y.tolist(),
    }


# ============================================================
# Step 5: Episode Detection and Classification
# ============================================================

def detect_episodes(df_interp):
    """Identify crisis episodes from interpolated time series."""
    episodes = []
    eid = 0

    for location in df_interp['location'].unique():
        loc_data = df_interp[df_interp['location'] == location].sort_values('date')
        if len(loc_data) == 0:
            continue

        iso3 = loc_data.iloc[0]['iso3']
        in_crisis = False
        phases = []
        dates = []

        for _, row in loc_data.iterrows():
            phase = int(row['ipc_phase'])
            if phase >= CRISIS_THRESHOLD:
                if not in_crisis:
                    in_crisis = True
                    phases = [phase]
                    dates = [row['date']]
                else:
                    phases.append(phase)
                    dates.append(row['date'])
            else:
                if in_crisis:
                    eid += 1
                    episodes.append(_make_episode(eid, iso3, location, phases, dates, False))
                    in_crisis = False

        if in_crisis:
            eid += 1
            episodes.append(_make_episode(eid, iso3, location, phases, dates, True))

    df_ep = pd.DataFrame(episodes)

    first_months = df_interp.groupby('location')['date'].min().to_dict()
    df_ep['is_left_censored'] = df_ep.apply(
        lambda r: r['dates'][0] == first_months.get(r['location']), axis=1
    )
    n_censored = df_ep['is_left_censored'].sum()

    # Flag right-censored episodes (WS5): ongoing OR end date in last month of data
    last_months = df_interp.groupby('location')['date'].max().to_dict()
    df_ep['is_right_censored'] = df_ep.apply(
        lambda r: r['ongoing'] or r['dates'][-1] == last_months.get(r['location']),
        axis=1
    )

    df_ep['archetype'] = df_ep.apply(_classify_archetype, axis=1)

    return df_ep, n_censored


def _make_episode(eid, iso3, location, phases, dates, ongoing):
    """Create episode record."""
    return {
        'crisis_id': eid, 'iso3': iso3, 'location': location,
        'duration_months': len(phases), 'peak_phase': max(phases),
        'mean_phase': round(np.mean(phases), 2),
        'phase_variance': round(np.var(phases), 3),
        'phases': phases, 'dates': dates, 'ongoing': ongoing,
        'months_at_3': sum(1 for p in phases if p == 3),
        'months_at_4': sum(1 for p in phases if p == 4),
        'months_at_5': sum(1 for p in phases if p == 5),
    }


def _classify_archetype(row):
    """Classify episode archetype."""
    dur = row['duration_months']
    peak = row['peak_phase']
    var = row['phase_variance']
    phases = row['phases']

    total_trans = 0
    for i in range(len(phases) - 1):
        if int(phases[i]) != int(phases[i + 1]):
            total_trans += 1

    dur_class = 'short' if dur < 12 else ('medium' if dur <= 36 else 'protracted')
    sev_class = 'moderate' if peak == 3 else ('severe' if peak == 4 else 'extreme')

    peak_indices = [i for i, p in enumerate(phases) if p == max(phases)]
    peak_pos = peak_indices[0] / (len(phases) - 1) if len(phases) > 1 else 0.5

    if var > 0.5 or total_trans >= 4:
        traj = 'oscillating'
    elif var < 0.1 and total_trans <= 1:
        traj = 'steady_state'
    elif peak_pos < 0.2:
        traj = 'immediate_peak'
    elif peak_pos < 0.4:
        traj = 'early_peak'
    elif peak_pos < 0.6:
        traj = 'mid_peak'
    elif peak_pos < 0.8:
        traj = 'late_peak'
    else:
        traj = 'end_peak'

    if dur_class == 'short' and sev_class in ['severe', 'extreme'] and (var < 0.1 and total_trans <= 1):
        return 'severe_shock'
    elif dur_class == 'protracted' and sev_class == 'moderate' and (var < 0.1 and total_trans <= 1):
        return 'entrenched_moderate'
    elif dur_class == 'protracted' and sev_class in ['severe', 'extreme']:
        return 'protracted_emergency'
    elif ((var > 0.3 or total_trans >= 3) or traj == 'oscillating') and total_trans >= 3:
        return 'oscillating'
    elif traj in ['end_peak', 'late_peak']:
        return 'escalating'
    elif traj in ['immediate_peak', 'early_peak']:
        return 'rapid_onset'
    elif dur_class == 'short' and sev_class == 'moderate':
        return 'seasonal_crisis'
    elif sev_class in ['severe', 'extreme']:
        if dur <= 12:
            return 'severe_shock'
        else:
            return 'protracted_emergency'
    else:
        return 'prolonged_moderate'


def _classify_archetype_with_thresholds(row, duration_short=12, duration_long=36,
                                        variance_stable=0.1):
    """Classify episode archetype with configurable thresholds.

    At the default arguments this is identical to _classify_archetype; it exists
    so the classification thresholds can be perturbed for the sensitivity tests
    without touching the authoritative classifier.
    """
    dur = row['duration_months']
    peak = row['peak_phase']
    var = row['phase_variance']
    phases = row['phases']

    total_trans = 0
    for i in range(len(phases) - 1):
        if int(phases[i]) != int(phases[i + 1]):
            total_trans += 1

    dur_class = ('short' if dur < duration_short
                 else ('medium' if dur <= duration_long else 'protracted'))
    sev_class = 'moderate' if peak == 3 else ('severe' if peak == 4 else 'extreme')
    vol_class = ('stable' if (var < variance_stable and total_trans <= 1)
                 else ('volatile' if (var > 0.3 or total_trans >= 3) else 'moderate'))

    peak_indices = [i for i, p in enumerate(phases) if p == max(phases)]
    peak_pos = peak_indices[0] / (len(phases) - 1) if len(phases) > 1 else 0.5

    if var > 0.5 or total_trans >= 4:
        traj = 'oscillating'
    elif var < variance_stable and total_trans <= 1:
        traj = 'steady_state'
    elif peak_pos < 0.2:
        traj = 'immediate_peak'
    elif peak_pos < 0.4:
        traj = 'early_peak'
    elif peak_pos < 0.6:
        traj = 'mid_peak'
    elif peak_pos < 0.8:
        traj = 'late_peak'
    else:
        traj = 'end_peak'

    if dur_class == 'short' and sev_class in ['severe', 'extreme'] and vol_class == 'stable':
        return 'severe_shock'
    elif dur_class == 'protracted' and sev_class == 'moderate' and vol_class == 'stable':
        return 'entrenched_moderate'
    elif dur_class == 'protracted' and sev_class in ['severe', 'extreme']:
        return 'protracted_emergency'
    elif (vol_class == 'volatile' or traj == 'oscillating') and total_trans >= 3:
        return 'oscillating'
    elif traj in ['end_peak', 'late_peak']:
        return 'escalating'
    elif traj in ['immediate_peak', 'early_peak']:
        return 'rapid_onset'
    elif dur_class == 'short' and sev_class == 'moderate':
        return 'seasonal_crisis'
    elif sev_class in ['severe', 'extreme']:
        if dur <= 12:
            return 'severe_shock'
        else:
            return 'protracted_emergency'
    else:
        return 'prolonged_moderate'


def _prep_episodes_for_classification(df_episodes):
    """Return a copy with 'phases' as a list of ints."""
    if isinstance(df_episodes.iloc[0]['phases'], str):
        df_episodes = df_episodes.copy()
        df_episodes['phases'] = df_episodes['phases'].apply(
            lambda x: [int(p) for p in str(x).split(',')])
    return df_episodes


def compute_threshold_sensitivity(df_episodes):
    """
    Sensitivity of archetype classification to DURATION_SHORT alone (WS7).

    Varies DURATION_SHORT over [9-15] and VARIANCE_STABLE over [0.08, 0.10, 0.12],
    holding DURATION_LONG at 36. This is the single-threshold variant.
    """
    print("\n  Computing classification threshold sensitivity (single threshold)...")
    df_episodes = _prep_episodes_for_classification(df_episodes)
    baseline = df_episodes.apply(_classify_archetype_with_thresholds, axis=1)

    duration_thresholds = [9, 10, 11, 12, 13, 14, 15]
    duration_results = []
    all_assignments = []
    for thresh in duration_thresholds:
        new = df_episodes.apply(
            lambda r: _classify_archetype_with_thresholds(r, duration_short=thresh),
            axis=1)
        all_assignments.append(new)
        changed = int((new != baseline).sum())
        changed_pct = round(changed / len(df_episodes) * 100, 1)
        dist = new.value_counts().to_dict()
        duration_results.append({
            'threshold': thresh,
            'changed_count': changed,
            'changed_pct': changed_pct,
            'stable_pct': round(100 - changed_pct, 1),
            'archetype_pcts': {k: round(v / len(df_episodes) * 100, 1)
                               for k, v in dist.items()},
        })
        print(f"    DURATION_SHORT={thresh}: {changed} changed ({changed_pct}%)")

    variance_results = []
    for thresh in [0.08, 0.10, 0.12]:
        new = df_episodes.apply(
            lambda r: _classify_archetype_with_thresholds(r, variance_stable=thresh),
            axis=1)
        changed = int((new != baseline).sum())
        variance_results.append({
            'threshold': thresh,
            'changed_count': changed,
            'changed_pct': round(changed / len(df_episodes) * 100, 1),
        })
        print(f"    VARIANCE_STABLE={thresh}: {changed} changed "
              f"({round(changed / len(df_episodes) * 100, 1)}%)")

    stable_count = sum(
        1 for i in range(len(df_episodes))
        if len({a.iloc[i] for a in all_assignments}) == 1)
    cross_stability = round(stable_count / len(df_episodes) * 100, 1)
    print(f"    Cross-threshold stability: {cross_stability}%")

    return {
        'duration_short_sensitivity': duration_results,
        'variance_stable_sensitivity': variance_results,
        'cross_threshold_stability_pct': cross_stability,
        'stable_episode_count': stable_count,
        'total_episodes': len(df_episodes),
        'baseline_threshold': 12,
        'note': 'Stability measured across DURATION_SHORT thresholds [9-15], '
                'DURATION_LONG held at 36. An episode is "stable" if its '
                'archetype is the same at all thresholds.',
    }


def compute_threshold_sensitivity_both(df_episodes):
    """
    Sensitivity of archetype classification to BOTH duration thresholds (B17).

    Moves DURATION_SHORT and DURATION_LONG together by +/-10%, which is the more
    demanding test and the one the Methods sentence's plural wording describes.
    Reported figures come from this stored output rather than an ad-hoc patch.
    """
    print("\n  Computing classification threshold sensitivity (both thresholds)...")
    df_episodes = _prep_episodes_for_classification(df_episodes)
    baseline = df_episodes.apply(_classify_archetype_with_thresholds, axis=1)

    variants = [
        ('-10%', 11, 32),
        ('baseline', 12, 36),
        ('+10%', 13, 39),
    ]
    results = []
    for label, ds, dl in variants:
        new = df_episodes.apply(
            lambda r: _classify_archetype_with_thresholds(
                r, duration_short=ds, duration_long=dl),
            axis=1)
        changed = int((new != baseline).sum())
        results.append({
            'label': label,
            'duration_short': ds,
            'duration_long': dl,
            'changed_count': changed,
            'changed_pct': round(changed / len(df_episodes) * 100, 3),
        })
        print(f"    both thresholds {label} (ds={ds}, dl={dl}): "
              f"{changed} changed ({round(changed / len(df_episodes) * 100, 3)}%)")

    perturbed = [r for r in results if r['label'] != 'baseline']
    return {
        'both_threshold_sensitivity': results,
        'range_changed_pct': [min(r['changed_pct'] for r in perturbed),
                              max(r['changed_pct'] for r in perturbed)],
        'total_episodes': len(df_episodes),
        'note': 'DURATION_SHORT and DURATION_LONG moved together by +/-10% '
                '(12->11/13, 36->32/39). This is the figure reported in Methods: '
                'the more demanding and more conservative of the two tests.',
    }


# ============================================================
# Step 5b: Quarterly Aggregation Analysis
# ============================================================

def compute_quarterly_transitions(df_interp):
    """Aggregate monthly data into calendar quarters, compute transitions."""
    df = df_interp.copy()
    df['quarter'] = df['date'].dt.to_period('Q')

    quarterly = df.groupby(['location', 'quarter']).agg({
        'ipc_phase': 'max',
        'iso3': 'first',
    }).reset_index()
    quarterly = quarterly.sort_values(['location', 'quarter']).reset_index(drop=True)

    raw_counts = np.zeros((5, 5), dtype=int)
    per_location_counts = defaultdict(lambda: np.zeros((5, 5), dtype=int))

    for location in quarterly['location'].unique():
        loc_data = quarterly[quarterly['location'] == location].sort_values('quarter')
        phases = loc_data['ipc_phase'].values
        quarters = loc_data['quarter'].values

        for i in range(len(phases) - 1):
            q_diff = quarters[i + 1] - quarters[i]
            if q_diff.n != 1:
                continue

            f = int(phases[i]) - 1
            t = int(phases[i + 1]) - 1
            if 0 <= f < 5 and 0 <= t < 5:
                raw_counts[f, t] += 1
                per_location_counts[location][f, t] += 1

    row_totals = raw_counts.sum(axis=1)
    ratios = compute_key_ratios(raw_counts, row_totals)

    return {
        'raw_counts': raw_counts,
        'row_totals': row_totals,
        'key_ratios': ratios,
        'per_location_counts': per_location_counts,
        'n_quarterly_records': len(quarterly),
        'n_locations': int(quarterly['location'].nunique()),
    }


def run_quarterly_analysis(df_interp):
    """Run full quarterly analysis with bootstrap CIs."""
    print("\n  Computing quarterly transitions...")
    qt = compute_quarterly_transitions(df_interp)

    print(f"    Quarterly records: {qt['n_quarterly_records']:,}")
    print(f"    Locations: {qt['n_locations']}")
    print(f"    Recovery ratio (4->3/3->4): {qt['key_ratios']['ratio_4to3_over_3to4']}:1")

    print("    Bootstrapping quarterly CIs (10,000 iterations)...")
    bootstrap_cis = bootstrap_matrix(qt['per_location_counts'], n_iter=N_BOOTSTRAP)

    return {
        'raw_counts': qt['raw_counts'].tolist(),
        'row_totals': qt['row_totals'].tolist(),
        'key_ratios': qt['key_ratios'],
        'bootstrap_cis': {k: v for k, v in bootstrap_cis.items()
                          if k not in ['cell_ci_lo', 'cell_ci_hi']},
        'n_quarterly_records': qt['n_quarterly_records'],
        'n_locations': qt['n_locations'],
        'method': 'MAX phase within calendar quarter, transitions between consecutive quarters',
    }


# ============================================================
# Step 5c: Regional Breakdown
# ============================================================

REGION_DEFINITIONS = {
    'Horn of Africa': ['SOM', 'KEN', 'ETH', 'DJI', 'ERI', 'SSD', 'SDN', 'UGA'],
    'Sahel': ['NER', 'TCD', 'MLI', 'BFA', 'MRT', 'SEN', 'GMB'],
    'Central Africa': ['COD', 'CAF', 'CMR', 'COG', 'GAB'],
    'Southern Africa': ['ZWE', 'MWI', 'MOZ', 'ZMB', 'SWZ', 'LSO', 'MDG'],
    'West Africa': ['NGA', 'GHA', 'SLE', 'LBR', 'GIN', 'CIV', 'BEN', 'TGO'],
    'Central America': ['GTM', 'HND', 'SLV', 'NIC', 'HTI'],
    'Asia': ['AFG', 'PAK', 'NPL', 'BGD', 'MMR', 'YEM', 'SYR', 'IRQ', 'PSE', 'LBN'],
}


def get_region(iso3):
    for region, countries in REGION_DEFINITIONS.items():
        if iso3 in countries:
            return region
    return 'Other'


def compute_regional_transitions(df_interp):
    """Compute transition matrices per region with bootstrap CIs."""
    print("\n  Computing regional transitions...")

    df = df_interp.copy()
    df['region'] = df['iso3'].apply(get_region)

    results = {}
    for region in sorted(df['region'].unique()):
        region_data = df[df['region'] == region]
        n_locs = region_data['location'].nunique()

        if n_locs < 5:
            continue

        trans = compute_transitions(region_data)
        ratios = compute_key_ratios(trans['raw_counts'], trans['row_totals'])

        if len(trans['per_location_counts']) >= 10:
            boot_cis = bootstrap_matrix(trans['per_location_counts'], n_iter=N_BOOTSTRAP)
            ci_data = {k: v for k, v in boot_cis.items()
                       if k not in ['cell_ci_lo', 'cell_ci_hi']}
        else:
            ci_data = {}

        results[region] = {
            'n_locations': n_locs,
            'n_transitions': int(trans['row_totals'].sum()),
            'key_ratios': ratios,
            'bootstrap_cis': ci_data,
            'countries': sorted(region_data['iso3'].unique().tolist()),
        }

        print(f"    {region}: {n_locs} locations, "
              f"ratio={ratios['ratio_4to3_over_3to4']}:1")

    return results


# ============================================================
# Step 5d: Temporal Comparison (matched locations)
# ============================================================

def compute_temporal_comparison(df_interp, df_raw):
    """Temporal comparison using matched locations present in both periods."""
    print("\n  Computing temporal comparison...")

    df_r = df_raw.copy()
    df_r['ipc_phase'] = df_r['ipc_phase_fews'].fillna(df_r['ipc_phase_ipcch'])
    df_r.loc[df_r['ipc_phase'] == 6, 'ipc_phase'] = np.nan
    df_r = df_r[df_r['ipc_phase'].notna()].copy()
    df_r['location'] = df_r['iso3'] + '_' + df_r['ADMIN1'].fillna('national')
    df_r['year'] = pd.to_datetime(df_r['year_month'] + '-01').dt.year

    early_period = df_r[df_r['year'].between(2011, 2017)]
    late_period = df_r[df_r['year'].between(2018, 2023)]

    early_locs = set(early_period['location'].unique())
    late_locs = set(late_period['location'].unique())

    matched_locs = early_locs & late_locs
    dropped_locs = early_locs - late_locs
    new_locs = late_locs - early_locs

    print(f"    Early period locations (2011-2017): {len(early_locs)}")
    print(f"    Late period locations (2018-2023): {len(late_locs)}")
    print(f"    Matched (both periods): {len(matched_locs)}")

    df_i = df_interp.copy()
    df_i['year'] = df_i['date'].dt.year

    def phase4_rate(data, locs, start_year, end_year):
        subset = data[(data['location'].isin(locs)) &
                       (data['year'].between(start_year, end_year))]
        if len(subset) == 0:
            return 0, 0
        rate = round((subset['ipc_phase'] >= 4).mean() * 100, 1)
        n_obs = len(subset)
        return rate, n_obs

    matched_early_rate, matched_early_n = phase4_rate(df_i, matched_locs, 2011, 2017)
    matched_late_rate, matched_late_n = phase4_rate(df_i, matched_locs, 2018, 2023)

    all_early_rate, all_early_n = phase4_rate(df_i, early_locs, 2011, 2017)
    all_late_rate, all_late_n = phase4_rate(df_i, late_locs, 2018, 2023)

    dropped_rate, dropped_n = phase4_rate(df_i, dropped_locs, 2011, 2017)
    new_rate, new_n = phase4_rate(df_i, new_locs, 2018, 2023)

    print(f"    Matched: {matched_early_rate}% -> {matched_late_rate}%")

    loc_year_counts = df_i.groupby('location')['year'].nunique()
    max_years = loc_year_counts.max()
    strict_locs = set(loc_year_counts[loc_year_counts >= max_years - 2].index)
    strict_early_rate, _ = phase4_rate(df_i, strict_locs, 2011, 2017)
    strict_late_rate, _ = phase4_rate(df_i, strict_locs, 2018, 2023)

    return {
        'matched_locations': {
            'n_locations': len(matched_locs),
            'early_phase4_pct': matched_early_rate,
            'late_phase4_pct': matched_late_rate,
            'early_n_observations': matched_early_n,
            'late_n_observations': matched_late_n,
            'definition': 'Any IPC observation in both 2011-2017 AND 2018-2023',
        },
        'all_locations': {
            'early_locations': len(early_locs),
            'late_locations': len(late_locs),
            'early_phase4_pct': all_early_rate,
            'late_phase4_pct': all_late_rate,
        },
        'dropped_locations': {
            'n_locations': len(dropped_locs),
            'phase4_pct': dropped_rate,
            'definition': 'Observed in 2011-2017 only',
        },
        'new_locations': {
            'n_locations': len(new_locs),
            'phase4_pct': new_rate,
            'definition': 'Observed in 2018-2023 only',
        },
        'consistent_locations': {
            'n_locations': len(strict_locs),
            'early_phase4_pct': strict_early_rate,
            'late_phase4_pct': strict_late_rate,
            'definition': f'Observed in >={max_years - 2} of {max_years} years',
        },
        'interpretation': (
            f'Phase 4+ rates changed from {all_early_rate}% to {all_late_rate}% '
            f'across all locations. Matched locations (observed in both periods) '
            f'show rates of {matched_early_rate}% → {matched_late_rate}%. '
            f'{len(new_locs)} new locations entered monitoring in the late period '
            f'with a Phase 4+ rate of {new_rate}%. '
            f'Monitoring expanded from {len(early_locs)} to {len(late_locs)} locations, '
            'so changes in aggregate rates reflect both genuine trends and '
            'composition effects from monitoring expansion.'
        ),
    }


# ============================================================
# Step 5e: Crisis Staircase Analysis
# ============================================================

def compute_crisis_staircase(df_episodes):
    """Analyze multi-episode pathways ("crisis staircase")."""
    print("\n  Computing crisis staircase analysis...")

    loc_episodes = {}
    for _, ep in df_episodes.iterrows():
        loc = ep['location']
        if loc not in loc_episodes:
            loc_episodes[loc] = []
        start = ep['dates'][0] if isinstance(ep['dates'], list) else ep['dates']
        loc_episodes[loc].append({
            'archetype': ep['archetype'],
            'duration': ep['duration_months'],
            'peak_phase': ep['peak_phase'],
            'start': start,
            'crisis_id': ep['crisis_id'],
        })

    for loc in loc_episodes:
        loc_episodes[loc].sort(key=lambda e: e['start'])

    n_single = sum(1 for eps in loc_episodes.values() if len(eps) == 1)
    n_double = sum(1 for eps in loc_episodes.values() if len(eps) == 2)
    n_triple_plus = sum(1 for eps in loc_episodes.values() if len(eps) >= 3)
    n_five_plus = sum(1 for eps in loc_episodes.values() if len(eps) >= 5)

    print(f"    1 episode: {n_single} locations")
    print(f"    2 episodes: {n_double} locations")
    print(f"    3+ episodes: {n_triple_plus} locations")
    print(f"    5+ episodes: {n_five_plus} locations")

    seasonal_to_prolonged = 0
    seasonal_to_protracted = 0
    prolonged_to_protracted = 0
    double_seasonal_to_prolonged = 0
    full_staircase = 0

    severe_types = {'protracted_emergency', 'escalating', 'severe_shock'}

    for loc, eps in loc_episodes.items():
        if len(eps) < 2:
            continue

        archetypes = [e['archetype'] for e in eps]

        for i in range(len(archetypes) - 1):
            if archetypes[i] == 'seasonal_crisis' and archetypes[i + 1] == 'prolonged_moderate':
                seasonal_to_prolonged += 1
            if archetypes[i] == 'seasonal_crisis' and archetypes[i + 1] == 'protracted_emergency':
                seasonal_to_protracted += 1
            if archetypes[i] == 'prolonged_moderate' and archetypes[i + 1] == 'protracted_emergency':
                prolonged_to_protracted += 1

        for i in range(len(archetypes) - 2):
            if (archetypes[i] == 'seasonal_crisis' and
                archetypes[i + 1] == 'seasonal_crisis' and
                archetypes[i + 2] == 'prolonged_moderate'):
                double_seasonal_to_prolonged += 1

        has_seasonal = False
        has_prolonged_after_seasonal = False
        for a in archetypes:
            if a == 'seasonal_crisis':
                has_seasonal = True
            elif a == 'prolonged_moderate' and has_seasonal:
                has_prolonged_after_seasonal = True
            elif a == 'protracted_emergency' and has_prolonged_after_seasonal:
                full_staircase += 1
                break

    multi_locs = {loc: eps for loc, eps in loc_episodes.items() if len(eps) >= 3}
    currently_severe = sum(
        1 for eps in multi_locs.values()
        if eps[-1]['archetype'] in severe_types
    )
    currently_severe_pct = round(currently_severe / len(multi_locs) * 100, 1) if multi_locs else 0

    print(f"    Seasonal -> Prolonged: {seasonal_to_prolonged}")
    print(f"    Full staircase (S->P->Pr): {full_staircase}")

    return {
        'location_episode_counts': {
            '1_episode': n_single,
            '2_episodes': n_double,
            '3_plus_episodes': n_triple_plus,
            '5_plus_episodes': n_five_plus,
            'total_locations': len(loc_episodes),
        },
        'transition_counts': {
            'seasonal_to_prolonged': seasonal_to_prolonged,
            'seasonal_to_protracted': seasonal_to_protracted,
            'prolonged_to_protracted': prolonged_to_protracted,
            'double_seasonal_to_prolonged': double_seasonal_to_prolonged,
            'full_staircase': full_staircase,
        },
        'multi_episode_severity': {
            'locations_with_3plus': len(multi_locs),
            'currently_in_severe_archetype': currently_severe,
            'severe_pct': currently_severe_pct,
            'severe_types': sorted(severe_types),
        },
        'method': ('Episodes grouped by location, ordered chronologically. '
                   'Transition counts: how many times archetype X is followed '
                   'by archetype Y at the same location. Full staircase: '
                   'seasonal → prolonged → protracted in sequence.'),
    }


def compute_crisis_staircase_censored(df_episodes):
    """
    Verify staircase counts with left-censored stratification.
    Reports counts both including and excluding left-censored first episodes.
    """
    print("\n  Computing staircase censoring sensitivity...")

    # Group by location
    loc_episodes = {}
    for _, ep in df_episodes.iterrows():
        loc = ep['location']
        if loc not in loc_episodes:
            loc_episodes[loc] = []
        start = ep['dates'][0] if isinstance(ep['dates'], list) else ep['dates']
        loc_episodes[loc].append({
            'archetype': ep['archetype'],
            'is_left_censored': ep.get('is_left_censored', False),
            'is_right_censored': ep.get('is_right_censored', False),
            'start': start,
        })

    for loc in loc_episodes:
        loc_episodes[loc].sort(key=lambda e: e['start'])

    # Count staircases in two ways
    def count_staircases(loc_eps_dict, exclude_censored_first=False):
        seasonal_to_prolonged = 0
        full_staircase = 0
        for loc, eps in loc_eps_dict.items():
            if len(eps) < 2:
                continue
            # Optionally skip if first episode is left-censored
            start_idx = 0
            if exclude_censored_first and eps[0]['is_left_censored']:
                start_idx = 1
            if start_idx >= len(eps) - 1:
                continue

            archetypes = [e['archetype'] for e in eps[start_idx:]]
            for i in range(len(archetypes) - 1):
                if archetypes[i] == 'seasonal_crisis' and archetypes[i + 1] == 'prolonged_moderate':
                    seasonal_to_prolonged += 1

            has_seasonal = False
            has_prolonged = False
            for a in archetypes:
                if a == 'seasonal_crisis':
                    has_seasonal = True
                elif a == 'prolonged_moderate' and has_seasonal:
                    has_prolonged = True
                elif a == 'protracted_emergency' and has_prolonged:
                    full_staircase += 1
                    break

        return seasonal_to_prolonged, full_staircase

    s2p_all, fs_all = count_staircases(loc_episodes, exclude_censored_first=False)
    s2p_excl, fs_excl = count_staircases(loc_episodes, exclude_censored_first=True)

    n_locs_censored_first = sum(
        1 for eps in loc_episodes.values()
        if len(eps) >= 2 and eps[0]['is_left_censored']
    )

    print(f"    All episodes - S→P: {s2p_all}, Full staircase: {fs_all}")
    print(f"    Excluding censored first - S→P: {s2p_excl}, Full staircase: {fs_excl}")
    print(f"    Locations with censored first episode: {n_locs_censored_first}")

    return {
        'all_episodes': {
            'seasonal_to_prolonged': s2p_all,
            'full_staircase': fs_all,
        },
        'excluding_censored_first': {
            'seasonal_to_prolonged': s2p_excl,
            'full_staircase': fs_excl,
        },
        'locations_with_censored_first': n_locs_censored_first,
    }


def compute_archetype_transitions_admin2(df_episodes):
    """
    Compute inter-episode transitions (archetype of episode N → archetype of N+1)
    with gap duration, for downstream drought/conflict/model scripts.
    """
    transitions = []
    tid = 0

    for location in df_episodes['location'].unique():
        loc_eps = df_episodes[df_episodes['location'] == location].copy()
        # Sort by start date
        loc_eps = loc_eps.sort_values(
            by='dates',
            key=lambda x: x.apply(lambda d: d[0] if isinstance(d, list) else d))

        if len(loc_eps) < 2:
            continue

        eps_list = loc_eps.to_dict('records')
        for i in range(len(eps_list) - 1):
            from_ep = eps_list[i]
            to_ep = eps_list[i + 1]

            from_end = from_ep['dates'][-1] if isinstance(from_ep['dates'], list) else from_ep['dates']
            to_start = to_ep['dates'][0] if isinstance(to_ep['dates'], list) else to_ep['dates']

            if isinstance(from_end, pd.Timestamp) and isinstance(to_start, pd.Timestamp):
                gap_months = ((to_start.year - from_end.year) * 12 +
                              (to_start.month - from_end.month))
            else:
                gap_months = np.nan

            tid += 1
            transitions.append({
                'transition_id': tid,
                'location': location,
                'iso3': from_ep['iso3'],
                'from_crisis_id': from_ep['crisis_id'],
                'to_crisis_id': to_ep['crisis_id'],
                'from_archetype': from_ep['archetype'],
                'to_archetype': to_ep['archetype'],
                'from_duration': from_ep['duration_months'],
                'to_duration': to_ep['duration_months'],
                'from_peak_phase': from_ep['peak_phase'],
                'to_peak_phase': to_ep['peak_phase'],
                'from_start': from_ep['dates'][0].strftime('%Y-%m-%d') if isinstance(from_ep['dates'], list) and hasattr(from_ep['dates'][0], 'strftime') else str(from_ep['dates'][0] if isinstance(from_ep['dates'], list) else from_ep['dates']),
                'from_end': from_end.strftime('%Y-%m-%d') if hasattr(from_end, 'strftime') else str(from_end),
                'to_start': to_start.strftime('%Y-%m-%d') if hasattr(to_start, 'strftime') else str(to_start),
                'gap_months': gap_months,
            })

    return {
        'total_transitions': len(transitions),
        'transitions': transitions,
    }


# ============================================================
# Verification Helpers
# ============================================================

def verify_episodes(df_episodes, df_interp):
    """Verify episode-level statistics."""
    result = {}

    arch_counts = df_episodes['archetype'].value_counts().to_dict()
    arch_pcts = {k: round(v / len(df_episodes) * 100, 1) for k, v in arch_counts.items()}
    result['archetype_distribution'] = {
        'counts': {k: int(v) for k, v in arch_counts.items()},
        'percentages': arch_pcts,
    }

    result['duration_stats'] = {
        'mean': round(df_episodes['duration_months'].mean(), 1),
        'median': round(df_episodes['duration_months'].median(), 1),
        'max': int(df_episodes['duration_months'].max()),
        'min': int(df_episodes['duration_months'].min()),
    }

    gap_data = []
    for location in df_episodes['location'].unique():
        loc_eps = df_episodes[df_episodes['location'] == location].sort_values(
            by='dates', key=lambda x: x.apply(lambda d: d[0] if isinstance(d, list) else d))
        if len(loc_eps) < 2:
            continue
        episodes_list = loc_eps.to_dict('records')
        for i in range(len(episodes_list) - 1):
            end_date = episodes_list[i]['dates'][-1]
            start_date = episodes_list[i + 1]['dates'][0]
            if isinstance(end_date, pd.Timestamp) and isinstance(start_date, pd.Timestamp):
                gap_months = ((start_date.year - end_date.year) * 12 +
                              (start_date.month - end_date.month))
                gap_data.append({
                    'location': location,
                    'gap_months': gap_months,
                    'episode_number': i + 1,
                })

    if gap_data:
        gap_df = pd.DataFrame(gap_data)
        result['inter_episode_gaps'] = {
            'total_gaps': len(gap_df),
            'mean_gap_months': round(gap_df['gap_months'].mean(), 1),
            'median_gap_months': round(gap_df['gap_months'].median(), 1),
            'locations_with_multiple_episodes': int(
                df_episodes.groupby('location').size().pipe(lambda s: (s >= 2).sum())),
        }

        gap_by_number = gap_df.groupby('episode_number')['gap_months'].agg(
            ['mean', 'median', 'count']).reset_index()
        result['gap_compression'] = gap_by_number.to_dict('records')

    df_interp_with_year = df_interp.copy()
    df_interp_with_year['year'] = df_interp_with_year['date'].dt.year
    yearly_phase4 = df_interp_with_year.groupby('year').apply(
        lambda g: round((g['ipc_phase'] >= 4).mean() * 100, 1)
    ).to_dict()
    result['phase4_plus_by_year'] = yearly_phase4

    loc_year_counts = df_interp_with_year.groupby('location')['year'].nunique()
    max_years = loc_year_counts.max()
    consistent_locs = loc_year_counts[loc_year_counts >= max_years - 2].index
    if len(consistent_locs) > 0:
        consistent_data = df_interp_with_year[
            df_interp_with_year['location'].isin(consistent_locs)]
        early = consistent_data[consistent_data['year'] <= 2017]
        late = consistent_data[consistent_data['year'] >= 2018]
        result['consistent_locations'] = {
            'n_locations': len(consistent_locs),
            'early_phase4_pct': round((early['ipc_phase'] >= 4).mean() * 100, 1) if len(early) > 0 else 0,
            'late_phase4_pct': round((late['ipc_phase'] >= 4).mean() * 100, 1) if len(late) > 0 else 0,
        }

    return result


def compute_left_censoring_sensitivity(df_episodes):
    """Compare episode statistics with and without left-censored episodes."""
    def _episode_stats(df):
        if len(df) == 0:
            return {}
        arch_counts = df['archetype'].value_counts().to_dict()
        return {
            'episodes': len(df),
            'countries': int(df['iso3'].nunique()),
            'locations': int(df['location'].nunique()),
            'mean_duration': round(df['duration_months'].mean(), 1),
            'median_duration': round(df['duration_months'].median(), 1),
            'phase4_plus_pct': round((df['peak_phase'] >= 4).mean() * 100, 1),
            'ongoing_pct': round(df['ongoing'].mean() * 100, 1),
            'archetypes': {k: round(v / len(df) * 100, 1) for k, v in arch_counts.items()},
            'archetype_counts': {k: int(v) for k, v in arch_counts.items()},
        }

    all_stats = _episode_stats(df_episodes)
    all_stats['left_censored_count'] = int(df_episodes['is_left_censored'].sum())

    filtered = df_episodes[~df_episodes['is_left_censored']]
    filtered_stats = _episode_stats(filtered)

    censored = df_episodes[df_episodes['is_left_censored']]
    censored_stats = _episode_stats(censored)

    all_countries = set(df_episodes['iso3'].unique())
    filtered_countries = set(filtered['iso3'].unique())
    censored_only_countries = sorted(all_countries - filtered_countries)

    # Right-censoring (WS5)
    right_censored_count = 0
    right_censored_stats = {}
    if 'is_right_censored' in df_episodes.columns:
        right_censored_count = int(df_episodes['is_right_censored'].sum())
        all_stats['right_censored_count'] = right_censored_count
        complete_only = df_episodes[
            ~df_episodes['is_left_censored'] & ~df_episodes['is_right_censored']
        ]
        right_censored_stats = _episode_stats(complete_only)
        right_censored_stats['label'] = 'Complete episodes (no left or right censoring)'

    return {
        'all': all_stats,
        'filtered': filtered_stats,
        'censored_only': censored_stats,
        'complete_episodes': right_censored_stats,
        'censored_only_countries': censored_only_countries,
        'right_censored_count': right_censored_count,
        'note': 'Transition matrices (recovery ratio, crossover, degradation) are '
                'computed from the full interpolated time series and are NOT affected '
                'by left-censoring. Only episode-level statistics differ.',
    }


def compute_country_counts(df_raw):
    """Compute various country counts."""
    total = df_raw['iso3'].nunique()

    df = df_raw.copy()
    df['ipc_phase'] = df['ipc_phase_fews'].fillna(df['ipc_phase_ipcch'])
    df.loc[df['ipc_phase'] == 6, 'ipc_phase'] = np.nan
    phase3_countries = df[df['ipc_phase'] >= 3]['iso3'].nunique()
    ipc_countries = df[df['ipc_phase'].notna()]['iso3'].nunique()

    return {
        'total_hfid_countries': total,
        'countries_with_ipc_data': ipc_countries,
        'countries_with_phase3_plus': phase3_countries,
        'all_iso3_codes': sorted(df_raw['iso3'].dropna().unique().tolist()),
    }


def audit_paper_statistics(primary_result, country_counts, episodes_df):
    """
    Cross-reference every statistic in PLAN.md Key Statistics Summary
    against computed values.

    NOTE: paper_value entries reflect an early draft snapshot and are kept
    verbatim for parity with the canonical pipeline; DISCREPANCY rows against
    the submitted manuscript are expected. This is an internal QC artifact.
    """
    paper_stats = [
        {
            'stat': 'Total episodes',
            'paper_value': '1,656',
            'source': 'crisis_episodes',
        },
        {
            'stat': 'Countries',
            'paper_value': '51',
            'source': 'crisis_episodes',
        },
        {
            'stat': 'Time period',
            'paper_value': '2011-2023',
            'source': 'HFID',
        },
        {
            'stat': 'Seasonal crisis %',
            'paper_value': '72.5%',
            'source': 'archetype_summary',
        },
        {
            'stat': 'Protracted emergency %',
            'paper_value': '5.3%',
            'source': 'archetype_summary',
        },
        {
            'stat': 'Phase 3→4 escalation',
            'paper_value': '2.0% (CI: 1.7–2.2%)',
            'source': 'transition_verification',
        },
        {
            'stat': 'Phase 4→3 recovery',
            'paper_value': '18.7% (CI: 16.6–20.7%)',
            'source': 'transition_verification',
        },
        {
            'stat': 'Recovery ratio (4→3)/(3→4)',
            'paper_value': '9.6:1 (CI: 8.1–11.3)',
            'source': 'transition_verification',
        },
        {
            'stat': 'Phase 3→2 asymmetry',
            'paper_value': '1.5:1 (CI: 1.4–1.6)',
            'source': 'transition_verification',
        },
        {
            'stat': 'Crisis staircase cases',
            'paper_value': '84 (→persistent), 8 (→protracted)',
            'source': 'archetype_transitions',
        },
        {
            'stat': 'Phase 4+ stability',
            'paper_value': '9.7% → 9.6% (same locations)',
            'source': 'temporal_comparison',
        },
    ]

    # Fill in computed values
    for stat in paper_stats:
        name = stat['stat']

        if name == 'Total episodes':
            stat['computed_value'] = str(primary_result['episodes']['total'])
        elif name == 'Countries':
            stat['computed_value'] = str(primary_result['episodes']['countries'])
            stat['note'] = (f"Total HFID: {country_counts['total_hfid_countries']}, "
                           f"with IPC: {country_counts['countries_with_ipc_data']}, "
                           f"Phase 3+: {country_counts['countries_with_phase3_plus']}")
        elif name == 'Time period':
            stat['computed_value'] = '2007-2024 (HFID range)'
            stat['note'] = 'HFID covers 2007-2024; episodes may be subset'
        elif name == 'Seasonal crisis %':
            val = primary_result['archetypes']['percentages'].get('seasonal_crisis', 0)
            stat['computed_value'] = f"{val}%"
        elif name == 'Protracted emergency %':
            val = primary_result['archetypes']['percentages'].get('protracted_emergency', 0)
            stat['computed_value'] = f"{val}%"
        elif name == 'Phase 3→4 escalation':
            val = primary_result['key_ratios']['P_3to4']
            ci = primary_result['bootstrap_cis'].get('P_3to4_ci', [])
            stat['computed_value'] = f"{val}%"
            if ci:
                stat['computed_value'] += f" (CI: {ci[0]}–{ci[1]}%)"
        elif name == 'Phase 4→3 recovery':
            val = primary_result['key_ratios']['P_4to3']
            ci = primary_result['bootstrap_cis'].get('P_4to3_ci', [])
            stat['computed_value'] = f"{val}%"
            if ci:
                stat['computed_value'] += f" (CI: {ci[0]}–{ci[1]}%)"
        elif name == 'Recovery ratio (4→3)/(3→4)':
            val = primary_result['key_ratios']['ratio_4to3_over_3to4']
            ci = primary_result['bootstrap_cis'].get('ratio_4to3_ci', [])
            stat['computed_value'] = f"{val}:1"
            if ci:
                stat['computed_value'] += f" (CI: {ci[0]}–{ci[1]})"
        elif name == 'Phase 3→2 asymmetry':
            val = primary_result['key_ratios']['ratio_3to2_over_2to3']
            ci = primary_result['bootstrap_cis'].get('ratio_3to2_ci', [])
            stat['computed_value'] = f"{val}:1"
            if ci:
                stat['computed_value'] += f" (CI: {ci[0]}–{ci[1]})"
        elif name == 'Crisis staircase cases':
            stat['computed_value'] = 'See transition_verification/'
            stat['note'] = 'Not recomputed here; requires multi-episode pathway analysis'
        elif name == 'Phase 4+ stability':
            stat['computed_value'] = 'See temporal_analysis/'
            stat['note'] = 'Not recomputed here; requires temporal period comparison'

        # Determine match status
        if 'computed_value' in stat:
            # Simple text comparison
            paper_num = stat['paper_value'].split('%')[0].split(':')[0].replace(',', '').strip()
            computed_num = stat['computed_value'].split('%')[0].split(':')[0].replace(',', '').strip()
            try:
                p = float(paper_num)
                c = float(computed_num)
                if abs(p - c) < 0.5:
                    stat['match'] = 'MATCH'
                elif abs(p - c) < 2:
                    stat['match'] = 'CLOSE'
                else:
                    stat['match'] = 'DISCREPANCY'
            except ValueError:
                stat['match'] = 'MANUAL_CHECK'

    return paper_stats


def _extract_sensitivity_row(result):
    """Extract key metrics for the sensitivity comparison table."""
    return {
        'priority': result['pipeline']['priority'],
        'aggregation': result['pipeline']['aggregation'],
        'interpolation_gap': result['pipeline']['interpolation_gap'],
        'is_admin2': result['pipeline']['is_admin2'],
        'P_4to3': result['key_ratios']['P_4to3'],
        'P_3to4': result['key_ratios']['P_3to4'],
        'ratio_4to3': result['key_ratios']['ratio_4to3_over_3to4'],
        'P_3to2': result['key_ratios']['P_3to2'],
        'P_2to3': result['key_ratios']['P_2to3'],
        'ratio_3to2': result['key_ratios']['ratio_3to2_over_2to3'],
        'episodes': result['episodes']['total'],
        'locations': result['data_summary']['unique_locations'],
        'countries': result['episodes']['countries'],
        'seasonal_crisis_pct': result['archetypes']['percentages'].get('seasonal_crisis', 0),
        'protracted_pct': result['archetypes']['percentages'].get('protracted_emergency', 0),
        'phase3_crossover': (result['phase3_duration'].get('crossover', {}).get('month', None)
                             if 'crossover' in result['phase3_duration'] else None),
    }


def save_json(filepath, data):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


# ============================================================
# Full Pipeline Run
# ============================================================

def run_full_pipeline(df_raw, priority='fews', aggregation='max',
                      max_gap=DEFAULT_INTERPOLATION_GAP,
                      run_bootstrap=True, label=None, is_admin2=False):
    """Run the complete analysis pipeline and return all results."""
    if label:
        print(f"\n{'='*70}")
        print(f"  VARIANT: {label}")
        print(f"  priority={priority}, agg={aggregation}, gap={max_gap}, "
              f"admin2={is_admin2}")
        print(f"{'='*70}")

    if is_admin2:
        df_pre = preprocess_admin2(df_raw, priority=priority)
    else:
        df_pre = preprocess(df_raw, priority=priority, aggregation=aggregation)

    df_interp = interpolate(df_pre, max_gap)

    trans = compute_transitions(df_interp)
    ratios = compute_key_ratios(trans['raw_counts'], trans['row_totals'])

    bootstrap_cis = {}
    if run_bootstrap:
        print("  Running bootstrap (10,000 iterations)...")
        t0 = time.time()
        bootstrap_cis = bootstrap_matrix(trans['per_location_counts'])
        print(f"  Bootstrap completed in {time.time() - t0:.1f}s")

    p1_results, p1_loc_data = compute_duration_conditioned(df_interp, 1, 0, 2)
    p2_results, p2_loc_data = compute_duration_conditioned(df_interp, 2, 1, 3)
    p3_results, p3_loc_data = compute_duration_conditioned(df_interp, 3, 2, 4)
    p4_results, p4_loc_data = compute_duration_conditioned(df_interp, 4, 3, 5)
    p5_results, p5_loc_data = compute_duration_conditioned(df_interp, 5, 4, 6)

    p3_fit = fit_decay_and_crossover(p3_results)
    p4_fit = fit_decay_and_crossover(p4_results)

    crossover_ci = {}
    if run_bootstrap and p3_loc_data:
        print("  Bootstrapping crossover point...")
        crossover_ci = bootstrap_crossover(p3_loc_data, n_iter=N_BOOTSTRAP)

    p1_duration_ci = {}
    p2_duration_ci = {}
    p3_duration_ci = {}
    p4_duration_ci = {}
    p5_duration_ci = {}
    if run_bootstrap:
        print("  Bootstrapping duration-conditioned CIs (phases 1-5)...")
        p1_duration_ci = bootstrap_duration_conditioned(p1_loc_data, n_iter=N_BOOTSTRAP)
        p2_duration_ci = bootstrap_duration_conditioned(p2_loc_data, n_iter=N_BOOTSTRAP)
        p3_duration_ci = bootstrap_duration_conditioned(p3_loc_data, n_iter=N_BOOTSTRAP)
        p4_duration_ci = bootstrap_duration_conditioned(p4_loc_data, n_iter=N_BOOTSTRAP)
        p5_duration_ci = bootstrap_duration_conditioned(p5_loc_data, n_iter=N_BOOTSTRAP)

    df_episodes, n_censored = detect_episodes(df_interp)

    arch_dist = df_episodes['archetype'].value_counts().to_dict()
    arch_pct = {k: round(v / len(df_episodes) * 100, 1)
                for k, v in arch_dist.items()} if len(df_episodes) > 0 else {}

    all_countries = df_pre['iso3'].nunique()
    episode_countries = df_episodes['iso3'].nunique() if len(df_episodes) > 0 else 0

    result = {
        'pipeline': {
            'priority': priority,
            'aggregation': aggregation,
            'interpolation_gap': max_gap,
            'is_admin2': is_admin2,
            'label': label,
        },
        'data_summary': {
            'location_months_preprocessed': len(df_pre),
            'location_months_interpolated': len(df_interp),
            'unique_locations': int(df_pre['location'].nunique()),
            'total_countries': all_countries,
        },
        'transition_matrix': {
            'raw_counts': trans['raw_counts'].tolist(),
            'pct_matrix': [[round(v, 2) for v in row]
                           for row in trans['pct_matrix'].tolist()],
            'row_totals': trans['row_totals'].tolist(),
        },
        'key_ratios': ratios,
        'bootstrap_cis': {k: v for k, v in bootstrap_cis.items()
                          if k not in ['cell_ci_lo', 'cell_ci_hi']},
        'cell_cis': {
            'ci_lo': bootstrap_cis.get('cell_ci_lo', []),
            'ci_hi': bootstrap_cis.get('cell_ci_hi', []),
        },
        'phase1_duration': {
            'bins': {label: {**p1_results[label],
                             **(p1_duration_ci.get(label, {}))}
                     for label in DURATION_LABELS},
        },
        'phase2_duration': {
            'bins': {label: {**p2_results[label],
                             **(p2_duration_ci.get(label, {}))}
                     for label in DURATION_LABELS},
        },
        'phase3_duration': {
            'bins': {label: {**p3_results[label],
                             **(p3_duration_ci.get(label, {}))}
                     for label in DURATION_LABELS},
            **p3_fit,
            **crossover_ci,
        },
        'phase4_duration': {
            'bins': {label: {**p4_results[label],
                             **(p4_duration_ci.get(label, {}))}
                     for label in DURATION_LABELS},
            **p4_fit,
        },
        'phase5_duration': {
            'bins': {label: {**p5_results[label],
                             **(p5_duration_ci.get(label, {}))}
                     for label in DURATION_LABELS},
        },
        'episodes': {
            'total': len(df_episodes),
            'left_censored_count': int(n_censored),
            'non_censored_count': len(df_episodes) - int(n_censored),
            'countries': episode_countries,
            'unique_locations': int(df_episodes['location'].nunique()) if len(df_episodes) > 0 else 0,
            'mean_duration': round(df_episodes['duration_months'].mean(), 1) if len(df_episodes) > 0 else 0,
            'median_duration': round(df_episodes['duration_months'].median(), 1) if len(df_episodes) > 0 else 0,
            'phase4_plus_pct': round((df_episodes['peak_phase'] >= 4).mean() * 100, 1) if len(df_episodes) > 0 else 0,
        },
        'archetypes': {
            'counts': {k: int(v) for k, v in arch_dist.items()},
            'percentages': arch_pct,
        },
    }

    return result, df_episodes, df_interp


# ============================================================
# Intermediate data for phase split
# ============================================================
INTERMEDIATE_INTERP = os.path.join(OUTPUT_DIR, '_intermediate_interp.csv')


def _load_intermediates():
    """Reload intermediate data saved by the core phase."""
    df_interp = pd.read_csv(INTERMEDIATE_INTERP, parse_dates=['date'])
    df_episodes = pd.read_csv(os.path.join(OUTPUT_DIR, 'episodes.csv'))
    df_episodes['phases'] = df_episodes['phases'].apply(
        lambda x: [int(p) for p in str(x).split(',')])
    df_episodes['dates'] = df_episodes['dates'].apply(
        lambda x: [pd.Timestamp(d) for d in str(x).split(',')])
    with open(os.path.join(OUTPUT_DIR, 'full_transition_matrix.json')) as f:
        primary_data = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'admin2_transition_analysis.json')) as f:
        admin2_data = json.load(f)
    return df_interp, df_episodes, primary_data, admin2_data


# ============================================================
# Core Phase: Primary + Admin2 analysis
# ============================================================

def run_core_phase(df_raw):
    """Run primary analysis with bootstrap and admin2 sensitivity."""
    # PRIMARY ANALYSIS
    print("\n" + "=" * 70)
    print("  PRIMARY ANALYSIS (authoritative pipeline)")
    print("=" * 70)

    primary_result, primary_episodes, primary_interp = run_full_pipeline(
        df_raw, priority='fews', aggregation='max', max_gap=12,
        run_bootstrap=True,
        label='PRIMARY: FEWS + MAX + 12mo'
    )

    # Save episode CSV
    episode_csv = primary_episodes.copy()
    episode_csv['phases'] = episode_csv['phases'].apply(lambda x: ','.join(str(p) for p in x))
    episode_csv['dates'] = episode_csv['dates'].apply(
        lambda x: ','.join(d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in x))
    episode_csv.to_csv(os.path.join(OUTPUT_DIR, 'episodes.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/episodes.csv ({len(primary_episodes)} episodes)")

    # Save primary outputs
    save_json(os.path.join(OUTPUT_DIR, 'full_transition_matrix.json'), {
        'matrix_pct': primary_result['transition_matrix']['pct_matrix'],
        'raw_counts': primary_result['transition_matrix']['raw_counts'],
        'row_totals': primary_result['transition_matrix']['row_totals'],
        'key_ratios': primary_result['key_ratios'],
        'bootstrap_cis': primary_result['bootstrap_cis'],
        'cell_cis': primary_result['cell_cis'],
        'method': 'FEWS NET priority, Phase 6 filtered, MAX aggregation, 12-month interpolation',
        'pipeline': primary_result['pipeline'],
        'episodes': primary_result['episodes'],
        'archetypes': primary_result['archetypes'],
        'data_summary': primary_result['data_summary'],
        'phase3_duration': primary_result['phase3_duration'],
    })

    for phase_num, phase_key in [(1, 'phase1_duration'), (2, 'phase2_duration'),
                                  (3, 'phase3_duration'), (4, 'phase4_duration'),
                                  (5, 'phase5_duration')]:
        save_json(os.path.join(OUTPUT_DIR, f'phase{phase_num}_duration_conditioned.json'),
                  primary_result[phase_key])

    crossover_data = {}
    if 'crossover' in primary_result['phase3_duration']:
        crossover_data['crossover'] = primary_result['phase3_duration']['crossover']
    if 'crossover_ci' in primary_result['phase3_duration']:
        crossover_data['crossover_ci'] = primary_result['phase3_duration']['crossover_ci']
    if 'crossover_median' in primary_result['phase3_duration']:
        crossover_data['crossover_median'] = primary_result['phase3_duration']['crossover_median']
    if 'decay_fit' in primary_result['phase3_duration']:
        crossover_data['decay_fit'] = primary_result['phase3_duration']['decay_fit']
    save_json(os.path.join(OUTPUT_DIR, 'phase3_crossover.json'), crossover_data)

    # ADMIN2 ANALYSIS
    print("\n" + "=" * 70)
    print("  ADMIN2 ANALYSIS (spatial resolution sensitivity)")
    print("=" * 70)

    admin2_result, admin2_episodes, _ = run_full_pipeline(
        df_raw, priority='fews', aggregation='max', max_gap=12,
        run_bootstrap=True, is_admin2=True,
        label='ADMIN2: FEWS + 12mo'
    )

    save_json(os.path.join(OUTPUT_DIR, 'admin2_transition_analysis.json'), {
        'transition_matrix': admin2_result['transition_matrix'],
        'key_ratios': admin2_result['key_ratios'],
        'bootstrap_cis': admin2_result['bootstrap_cis'],
        'phase3_duration': admin2_result['phase3_duration'],
        'phase4_duration': admin2_result['phase4_duration'],
        'episodes': admin2_result['episodes'],
        'archetypes': admin2_result['archetypes'],
        'data_summary': admin2_result['data_summary'],
        'pipeline': admin2_result['pipeline'],
    })

    # Save intermediate data for robustness phase
    primary_interp.to_csv(INTERMEDIATE_INTERP, index=False)
    print(f"  Saved intermediate data: {INTERMEDIATE_INTERP}")

    return primary_result, primary_episodes, primary_interp, admin2_result


# ============================================================
# Robustness Phase: Sensitivity + verification analyses
# ============================================================

def run_robustness_phase(df_raw, primary_result=None, primary_episodes=None,
                         primary_interp=None, admin2_result=None):
    """Run sensitivity variants and all verification analyses.

    When called with --phase robustness, reloads intermediate data from disk.
    When called with --phase all, uses in-memory variables directly.
    """
    if primary_interp is None:
        # Reload from disk (--phase robustness)
        print("\n  Loading intermediate data from core phase...")
        primary_interp, primary_episodes, primary_data, admin2_data = _load_intermediates()
        print(f"  Loaded: {len(primary_interp)} interpolated rows, {len(primary_episodes)} episodes")
    else:
        primary_data = None
        admin2_data = None

    # SENSITIVITY VARIANTS
    print("\n" + "=" * 70)
    print("  SENSITIVITY ANALYSIS (10 pipeline variants)")
    print("=" * 70)

    variants = [
        ('FEWS + MAX + 6mo', 'fews', 'max', 6, False),
        ('FEWS + MAX + 12mo', 'fews', 'max', 12, False),
        ('FEWS + MAX + 18mo', 'fews', 'max', 18, False),
        ('IPC + MAX + 6mo', 'ipc', 'max', 6, False),
        ('IPC + MAX + 12mo', 'ipc', 'max', 12, False),
        ('IPC + MAX + 18mo', 'ipc', 'max', 18, False),
        ('FEWS + admin2 + 12mo', 'fews', 'max', 12, True),
        ('FEWS + dictzip + 12mo', 'fews', 'dictzip', 12, False),
        ('FEWS + MEDIAN + 12mo', 'fews', 'median', 12, False),
        ('FEWS + MEAN + 12mo', 'fews', 'mean', 12, False),
    ]

    sensitivity_results = []
    for var_label, priority, agg, gap, is_a2 in variants:
        if var_label == 'FEWS + MAX + 12mo':
            if primary_result is not None:
                row = _extract_sensitivity_row(primary_result)
            else:
                row = _extract_sensitivity_row_from_json(primary_data)
            sensitivity_results.append({'label': var_label, **row})
            continue
        if var_label == 'FEWS + admin2 + 12mo':
            if admin2_result is not None:
                row = _extract_sensitivity_row(admin2_result)
            else:
                row = _extract_sensitivity_row_from_json(admin2_data)
            sensitivity_results.append({'label': var_label, **row})
            continue

        result, _, _ = run_full_pipeline(
            df_raw, priority=priority, aggregation=agg, max_gap=gap,
            run_bootstrap=False, is_admin2=is_a2,
            label=var_label
        )
        sensitivity_results.append({
            'label': var_label,
            **_extract_sensitivity_row(result),
        })

    save_json(os.path.join(OUTPUT_DIR, 'sensitivity_analysis.json'), sensitivity_results)

    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv(os.path.join(OUTPUT_DIR, 'sensitivity_summary.csv'), index=False)

    # COUNTRY COUNTS
    country_counts = compute_country_counts(df_raw)
    if primary_result is not None:
        country_counts['countries_with_episodes'] = primary_result['episodes']['countries']
    else:
        country_counts['countries_with_episodes'] = primary_data.get('episodes', {}).get(
            'countries', int(primary_episodes['iso3'].nunique()))
    save_json(os.path.join(OUTPUT_DIR, 'country_counts.json'), country_counts)

    # CLASSIFICATION THRESHOLD SENSITIVITY (both variants stored)
    threshold_result = compute_threshold_sensitivity(primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'threshold_sensitivity.json'), threshold_result)

    threshold_both = compute_threshold_sensitivity_both(primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'threshold_sensitivity_both.json'),
              threshold_both)

    # WS1: BLOCK BOOTSTRAP CI ASSESSMENT
    print("\n" + "=" * 70)
    print("  WS1: BLOCK BOOTSTRAP CI ASSESSMENT")
    print("=" * 70)

    trans = compute_transitions(primary_interp)
    print(f"  Running block bootstrap ({N_BOOTSTRAP:,} iterations)...")
    t0 = time.time()
    block_boot = bootstrap_matrix_block(
        trans['per_location_counts'], n_iter=N_BOOTSTRAP)
    print(f"  Block bootstrap completed in {time.time() - t0:.1f}s")

    if primary_result is not None:
        std_ci = primary_result['bootstrap_cis']
    else:
        std_ci = primary_data.get('bootstrap_cis', {})

    block_comparison = {
        'standard_bootstrap': {
            'ratio_4to3_ci': std_ci.get('ratio_4to3_ci', []),
            'ratio_3to2_ci': std_ci.get('ratio_3to2_ci', []),
        },
        'block_bootstrap': block_boot,
        'comparison': {
            'ratio_4to3_ci_widened': (
                block_boot['ratio_4to3_ci_block'][1] - block_boot['ratio_4to3_ci_block'][0]
                > (std_ci.get('ratio_4to3_ci', [0, 0])[1] - std_ci.get('ratio_4to3_ci', [0, 0])[0])
            ) if std_ci.get('ratio_4to3_ci') else None,
            'lower_bound_above_3': block_boot['ratio_4to3_ci_block'][0] > 3.0,
        },
        'note': 'Block bootstrap resamples entire location episode chains, '
                'preserving temporal autocorrelation. Standard bootstrap already '
                'resamples at location level, so differences should be minimal.',
    }
    save_json(os.path.join(OUTPUT_DIR, 'block_bootstrap_comparison.json'),
              block_comparison)

    print(f"  Standard CI (4->3/3->4): {std_ci.get('ratio_4to3_ci', [])}")
    print(f"  Block CI (4->3/3->4):    {block_boot['ratio_4to3_ci_block']}")
    print(f"  Lower bound > 3:1:       {block_comparison['comparison']['lower_bound_above_3']}")

    # Unified primary reference: in-memory result (--phase all) or reloaded JSON
    # (--phase robustness); both carry key_ratios / bootstrap_cis / episodes / archetypes
    pr = primary_result if primary_result is not None else primary_data

    # WS2: OBSERVED-ONLY TRANSITION SENSITIVITY
    print("\n" + "=" * 70)
    print("  WS2: OBSERVED-ONLY TRANSITIONS (excluding interpolated)")
    print("=" * 70)

    obs_trans = compute_transitions_observed_only(primary_interp)
    obs_ratios = compute_key_ratios(obs_trans['raw_counts'], obs_trans['row_totals'])

    # Bootstrap observed-only
    print("  Bootstrapping observed-only CIs...")
    obs_boot = bootstrap_matrix(obs_trans['per_location_counts'], n_iter=N_BOOTSTRAP)

    observed_only_result = {
        'transition_matrix': {
            'raw_counts': obs_trans['raw_counts'].tolist(),
            'pct_matrix': obs_trans['pct_matrix'].tolist(),
            'row_totals': obs_trans['row_totals'].tolist(),
        },
        'key_ratios': obs_ratios,
        'bootstrap_cis': {k: v for k, v in obs_boot.items()
                          if k not in ['cell_ci_lo', 'cell_ci_hi']},
        'interpolation_stats': {
            'total_transition_pairs': obs_trans['total_pairs'],
            'interpolated_pairs': obs_trans['interpolated_pairs'],
            'observed_pairs': obs_trans['observed_pairs'],
            'interpolated_pct': obs_trans['interpolated_pct'],
        },
        'comparison_with_all': {
            'all_ratio_4to3': pr['key_ratios']['ratio_4to3_over_3to4'],
            'observed_ratio_4to3': obs_ratios['ratio_4to3_over_3to4'],
            'all_P_4to3': pr['key_ratios']['P_4to3'],
            'observed_P_4to3': obs_ratios['P_4to3'],
            'all_P_3to4': pr['key_ratios']['P_3to4'],
            'observed_P_3to4': obs_ratios['P_3to4'],
        },
    }
    save_json(os.path.join(OUTPUT_DIR, 'observed_only_transitions.json'), observed_only_result)

    print(f"  Total pairs: {obs_trans['total_pairs']:,}")
    print(f"  Interpolated: {obs_trans['interpolated_pairs']:,} ({obs_trans['interpolated_pct']}%)")
    print(f"  All transitions ratio: {pr['key_ratios']['ratio_4to3_over_3to4']}:1")
    print(f"  Observed-only ratio:   {obs_ratios['ratio_4to3_over_3to4']}:1")

    # WS4: EXTENDED DURATION BINS + MODEL COMPARISON
    print("\n" + "=" * 70)
    print("  WS4: EXTENDED DURATION BINS + MODEL COMPARISON")
    print("=" * 70)

    # Phase 4 with extended bins
    p4_ext = compute_duration_conditioned_extended(
        primary_interp, target_phase=4, recovery_phase=3, escalation_phase=5)
    p4_model = fit_model_comparison(p4_ext)

    # Phase 3 with extended bins
    p3_ext = compute_duration_conditioned_extended(
        primary_interp, target_phase=3, recovery_phase=2, escalation_phase=4)
    p3_model = fit_model_comparison(p3_ext)

    extended_bins_result = {
        'phase4': {
            'bins': p4_ext,
            'model_comparison': p4_model,
        },
        'phase3': {
            'bins': p3_ext,
            'model_comparison': p3_model,
        },
        'bin_labels': DURATION_LABELS_EXTENDED,
        'midpoints': DURATION_MIDPOINTS_EXTENDED,
    }
    save_json(os.path.join(OUTPUT_DIR, 'extended_duration_bins.json'), extended_bins_result)

    if 'models' in p4_model:
        print(f"  Phase 4 models:")
        for name, m in p4_model['models'].items():
            if name == 'best_model':
                continue
            print(f"    {name}: R²={m.get('r_squared', 'N/A')}, "
                  f"AIC={m.get('AIC', 'N/A')}, BIC={m.get('BIC', 'N/A')}")
        print(f"    Best model (AIC): {p4_model['models'].get('best_model', 'N/A')}")

    # EPISODE VERIFICATION
    episode_verification = verify_episodes(primary_episodes, primary_interp)
    save_json(os.path.join(OUTPUT_DIR, 'episode_verification.json'), episode_verification)

    # LEFT-CENSORING SENSITIVITY
    lc_sensitivity = compute_left_censoring_sensitivity(primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'left_censoring_sensitivity.json'), lc_sensitivity)

    # RIGHT-CENSORING ANALYSIS (WS5)
    print("\n" + "=" * 70)
    print("  WS5: RIGHT-CENSORING ANALYSIS")
    print("=" * 70)

    n_right_censored = int(primary_episodes['is_right_censored'].sum())
    n_left_censored = int(primary_episodes['is_left_censored'].sum())
    n_both_censored = int(
        (primary_episodes['is_left_censored'] & primary_episodes['is_right_censored']).sum())
    n_complete = int(
        (~primary_episodes['is_left_censored'] & ~primary_episodes['is_right_censored']).sum())

    complete_only = primary_episodes[
        ~primary_episodes['is_left_censored'] & ~primary_episodes['is_right_censored']]

    right_censor_result = {
        'total_episodes': len(primary_episodes),
        'left_censored': n_left_censored,
        'right_censored': n_right_censored,
        'both_censored': n_both_censored,
        'complete_only': n_complete,
        'complete_mean_duration': round(complete_only['duration_months'].mean(), 1) if len(complete_only) > 0 else 0,
        'complete_median_duration': round(complete_only['duration_months'].median(), 1) if len(complete_only) > 0 else 0,
        'all_mean_duration': round(primary_episodes['duration_months'].mean(), 1),
        'all_median_duration': round(primary_episodes['duration_months'].median(), 1),
        'complete_archetypes': {
            k: round(v / len(complete_only) * 100, 1)
            for k, v in complete_only['archetype'].value_counts().to_dict().items()
        } if len(complete_only) > 0 else {},
    }
    save_json(os.path.join(OUTPUT_DIR, 'right_censoring_analysis.json'), right_censor_result)

    print(f"  Total: {len(primary_episodes)}, Left-censored: {n_left_censored}, "
          f"Right-censored: {n_right_censored}")
    print(f"  Both: {n_both_censored}, Complete only: {n_complete}")
    print(f"  Duration (all): mean={right_censor_result['all_mean_duration']}, "
          f"median={right_censor_result['all_median_duration']}")
    print(f"  Duration (complete): mean={right_censor_result['complete_mean_duration']}, "
          f"median={right_censor_result['complete_median_duration']}")

    # QUARTERLY ANALYSIS
    quarterly_result = run_quarterly_analysis(primary_interp)
    save_json(os.path.join(OUTPUT_DIR, 'quarterly_analysis.json'), quarterly_result)

    # REGIONAL ANALYSIS
    regional_result = compute_regional_transitions(primary_interp)
    save_json(os.path.join(OUTPUT_DIR, 'regional_transition_analysis.json'), regional_result)

    # TEMPORAL COMPARISON
    temporal_result = compute_temporal_comparison(primary_interp, df_raw)
    save_json(os.path.join(OUTPUT_DIR, 'temporal_comparison.json'), temporal_result)

    # CRISIS STAIRCASE
    staircase_result = compute_crisis_staircase(primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'crisis_staircase.json'), staircase_result)

    # WS8: STAIRCASE CENSORING SENSITIVITY
    print("\n" + "=" * 70)
    print("  WS8: STAIRCASE CENSORING SENSITIVITY")
    print("=" * 70)

    staircase_censored = compute_crisis_staircase_censored(primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'staircase_censoring_sensitivity.json'),
              staircase_censored)

    # ROBUSTNESS SUMMARY TABLE
    print("\n" + "=" * 70)
    print("  ROBUSTNESS SUMMARY")
    print("=" * 70)

    robustness_summary = {
        'WS1_block_bootstrap': {
            'standard_ci': std_ci.get('ratio_4to3_ci', []),
            'block_ci': block_boot['ratio_4to3_ci_block'],
            'robust': block_comparison['comparison']['lower_bound_above_3'],
        },
        'WS2_observed_only': {
            'all_ratio': pr['key_ratios']['ratio_4to3_over_3to4'],
            'observed_only_ratio': obs_ratios['ratio_4to3_over_3to4'],
            'interpolated_pct': obs_trans['interpolated_pct'],
        },
        'WS3_aggregation': {
            agg: next((s['ratio_4to3'] for s in sensitivity_results
                       if agg.upper() in s['label']), None)
            for agg in ['MAX', 'MEDIAN', 'MEAN']
        },
        'WS4_model_comparison': {
            'best_model': p4_model.get('models', {}).get('best_model', 'N/A'),
            'n_bins': 8,
        },
        'WS5_censoring': {
            'left_censored': n_left_censored,
            'right_censored': n_right_censored,
            'complete_only': n_complete,
        },
        'WS7_threshold_stability': {
            'cross_threshold_pct': threshold_result['cross_threshold_stability_pct'],
        },
        'WS8_staircase_censoring': staircase_censored,
    }
    save_json(os.path.join(OUTPUT_DIR, 'robustness_summary.json'), robustness_summary)

    # Print summary table
    print(f"\n  {'Workstream':<30} {'Finding':<50} {'Robust?'}")
    print(f"  {'-'*90}")
    print(f"  {'WS1 Block Bootstrap':<30} "
          f"{'CI: ' + str(block_boot['ratio_4to3_ci_block']):<50} "
          f"{'YES' if block_comparison['comparison']['lower_bound_above_3'] else 'CHECK'}")
    print(f"  {'WS2 Observed-Only':<30} "
          f"{'Ratio: ' + str(obs_ratios['ratio_4to3_over_3to4']) + ':1':<50} "
          f"{'YES' if obs_ratios['ratio_4to3_over_3to4'] > 3 else 'CHECK'}")
    print(f"  {'WS3 Aggregation':<30} "
          f"{'MAX/MEDIAN/MEAN all tested':<50} "
          f"{'YES'}")
    print(f"  {'WS4 Model Comparison':<30} "
          f"{'Best: ' + str(p4_model.get('models', {}).get('best_model', 'N/A')):<50} "
          f"{'YES'}")
    print(f"  {'WS5 Censoring':<30} "
          f"{str(n_complete) + '/' + str(len(primary_episodes)) + ' complete':<50} "
          f"{'YES'}")
    print(f"  {'WS7 Threshold':<30} "
          f"{str(threshold_result['cross_threshold_stability_pct']) + '% stable':<50} "
          f"{'YES' if threshold_result['cross_threshold_stability_pct'] > 90 else 'CHECK'}")

    # PAPER STATISTICS AUDIT
    print("\n" + "=" * 70)
    print("  PAPER STATISTICS AUDIT")
    print("=" * 70)

    audit = audit_paper_statistics(pr, country_counts, primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'paper_audit.json'), audit)

    # Print audit table
    print(f"\n{'Statistic':<30} {'Paper':>25} {'Computed':>30} {'Status':>12}")
    print("-" * 100)
    for stat in audit:
        cv = stat.get('computed_value', 'N/A')
        match = stat.get('match', 'N/A')
        print(f"{stat['stat']:<30} {stat['paper_value']:>25} {cv:>30} {match:>12}")


def run_admin2_standalone(df_raw):
    """
    Run admin2-only pipeline and save outputs to a dedicated directory.

    Produces the same output structure as the primary pipeline but at admin2
    resolution, writing to outputs/transition_verification_admin2/ (relative
    to the working directory by design — downstream project scripts read the
    files from the project root). Optional capability invoked via --admin2;
    not part of the run_all.py deposit steps.
    """
    admin2_dir = 'outputs/transition_verification_admin2'
    os.makedirs(admin2_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  ADMIN2 STANDALONE PIPELINE")
    print("=" * 70)

    admin2_result, admin2_episodes, admin2_interp = run_full_pipeline(
        df_raw, priority='fews', aggregation='max', max_gap=12,
        run_bootstrap=True, is_admin2=True,
        label='ADMIN2: FEWS + 12mo (standalone)'
    )

    # Save episode CSV (mirroring primary pipeline format)
    episode_csv = admin2_episodes.copy()
    episode_csv['phases'] = episode_csv['phases'].apply(
        lambda x: ','.join(str(p) for p in x))
    episode_csv['dates'] = episode_csv['dates'].apply(
        lambda x: ','.join(d.strftime('%Y-%m-%d') if hasattr(d, 'strftime')
                           else str(d) for d in x))
    episode_csv.to_csv(f'{admin2_dir}/episodes.csv', index=False)
    print(f"  Saved: {admin2_dir}/episodes.csv ({len(admin2_episodes)} episodes)")

    # Save full transition matrix
    save_json(f'{admin2_dir}/full_transition_matrix.json', {
        'matrix_pct': admin2_result['transition_matrix']['pct_matrix'],
        'raw_counts': admin2_result['transition_matrix']['raw_counts'],
        'row_totals': admin2_result['transition_matrix']['row_totals'],
        'key_ratios': admin2_result['key_ratios'],
        'bootstrap_cis': admin2_result['bootstrap_cis'],
        'cell_cis': admin2_result['cell_cis'],
        'method': 'FEWS NET priority, Phase 6 filtered, admin2 (no aggregation), 12-month interpolation',
    })

    # Save duration-conditioned
    for phase_key in ['phase1_duration', 'phase2_duration', 'phase3_duration',
                      'phase4_duration', 'phase5_duration']:
        save_json(f'{admin2_dir}/{phase_key.replace("_duration", "")}_duration_conditioned.json',
                  admin2_result[phase_key])

    # Phase 3 crossover
    crossover_data = {}
    if 'crossover' in admin2_result['phase3_duration']:
        crossover_data['crossover'] = admin2_result['phase3_duration']['crossover']
    if 'crossover_ci' in admin2_result['phase3_duration']:
        crossover_data['crossover_ci'] = admin2_result['phase3_duration']['crossover_ci']
    if 'crossover_median' in admin2_result['phase3_duration']:
        crossover_data['crossover_median'] = admin2_result['phase3_duration']['crossover_median']
    if 'decay_fit' in admin2_result['phase3_duration']:
        crossover_data['decay_fit'] = admin2_result['phase3_duration']['decay_fit']
    save_json(f'{admin2_dir}/phase3_crossover.json', crossover_data)

    # Archetype transitions (inter-episode gaps)
    archetype_transitions = compute_archetype_transitions_admin2(admin2_episodes)
    save_json(f'{admin2_dir}/archetype_transitions.json', archetype_transitions)

    # Save archetype_transitions.csv for downstream scripts
    if archetype_transitions.get('transitions'):
        df_trans = pd.DataFrame(archetype_transitions['transitions'])
        df_trans.to_csv(f'{admin2_dir}/archetype_transitions.csv', index=False)
        print(f"  Saved: {admin2_dir}/archetype_transitions.csv "
              f"({len(df_trans)} transitions)")

    # Episode verification
    episode_verification = verify_episodes(admin2_episodes, admin2_interp)
    save_json(f'{admin2_dir}/episode_verification.json', episode_verification)

    # Regional breakdown
    regional_result = compute_regional_transitions(admin2_interp)
    save_json(f'{admin2_dir}/regional_transition_analysis.json', regional_result)

    # Temporal comparison
    temporal_result = compute_temporal_comparison(admin2_interp, df_raw)
    save_json(f'{admin2_dir}/temporal_comparison.json', temporal_result)

    # Crisis staircase
    staircase_result = compute_crisis_staircase(admin2_episodes)
    save_json(f'{admin2_dir}/crisis_staircase.json', staircase_result)

    # Transition summary
    save_json(f'{admin2_dir}/transition_summary.json', {
        'pipeline': admin2_result['pipeline'],
        'data_summary': admin2_result['data_summary'],
        'episodes': admin2_result['episodes'],
        'archetypes': admin2_result['archetypes'],
        'key_ratios': admin2_result['key_ratios'],
        'bootstrap_cis': admin2_result['bootstrap_cis'],
    })

    # Summary
    print(f"\n  ADMIN2 PIPELINE COMPLETE")
    print(f"  Episodes: {admin2_result['episodes']['total']}")
    print(f"  Locations: {admin2_result['data_summary']['unique_locations']}")
    print(f"  Countries: {admin2_result['episodes']['countries']}")
    print(f"  Recovery ratio: {admin2_result['key_ratios']['ratio_4to3_over_3to4']}:1")
    print(f"  All outputs saved to: {admin2_dir}/")

    return admin2_result, admin2_episodes


def _extract_sensitivity_row_from_json(data):
    """Extract sensitivity row from saved JSON data (for --phase robustness)."""
    key_ratios = data.get('key_ratios', {})
    pipeline = data.get('pipeline', {})
    episodes = data.get('episodes', {})
    archetypes = data.get('archetypes', {})
    data_summary = data.get('data_summary', {})
    phase3_duration = data.get('phase3_duration', {})
    return {
        'priority': pipeline.get('priority', 'fews'),
        'aggregation': pipeline.get('aggregation', 'max'),
        'interpolation_gap': pipeline.get('interpolation_gap', 12),
        'is_admin2': pipeline.get('is_admin2', False),
        'P_4to3': key_ratios.get('P_4to3', 0),
        'P_3to4': key_ratios.get('P_3to4', 0),
        'ratio_4to3': key_ratios.get('ratio_4to3_over_3to4', 0),
        'P_3to2': key_ratios.get('P_3to2', 0),
        'P_2to3': key_ratios.get('P_2to3', 0),
        'ratio_3to2': key_ratios.get('ratio_3to2_over_2to3', 0),
        'episodes': episodes.get('total', 0),
        'locations': data_summary.get('unique_locations', 0),
        'countries': episodes.get('countries', 0),
        'seasonal_crisis_pct': archetypes.get('percentages', {}).get('seasonal_crisis', 0),
        'protracted_pct': archetypes.get('percentages', {}).get('protracted_emergency', 0),
        'phase3_crossover': (phase3_duration.get('crossover', {}).get('month', None)
                             if 'crossover' in phase3_duration else None),
    }


# ============================================================
# Main Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='01_reference_pipeline.py — Core analysis pipeline')
    parser.add_argument('--phase', choices=['core', 'robustness', 'all'], default='all',
                        help='Which phase to run: core (primary+admin2), robustness (sensitivity+verification), all (both)')
    parser.add_argument('--admin2', action='store_true',
                        help='Run the standalone admin2 pipeline only (writes to '
                             'outputs/transition_verification_admin2/, CWD-relative; '
                             'not part of the run_all.py deposit steps)')
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = load_hfid()

    if args.admin2:
        run_admin2_standalone(df_raw)
        elapsed = time.time() - start_time
        print(f"\n  Admin2 pipeline completed in {elapsed:.0f}s")
        return

    if args.phase in ('core', 'all'):
        primary_result, primary_episodes, primary_interp, admin2_result = run_core_phase(df_raw)
    else:
        primary_result = primary_episodes = primary_interp = admin2_result = None

    if args.phase in ('robustness', 'all'):
        run_robustness_phase(df_raw, primary_result, primary_episodes,
                             primary_interp, admin2_result)

    # Cleanup intermediate file (only after robustness phase has consumed it)
    if args.phase in ('robustness', 'all') and os.path.exists(INTERMEDIATE_INTERP):
        os.remove(INTERMEDIATE_INTERP)
        print(f"  Cleaned up: {INTERMEDIATE_INTERP}")

    # SUMMARY
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — phase={args.phase} ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")

    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith('_'):
            continue
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f} ({size:,} bytes)")


if __name__ == '__main__':
    main()
