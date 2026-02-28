#!/usr/bin/env python3
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
  7. Sensitivity analysis across 9 pipeline variants
  8. Bootstrap confidence intervals (10,000 iterations)

Inputs:
  data/HFID_hv1.csv  — HFID v1.1.1 (Machefer et al. 2025)

Outputs (all in outputs/data/):
  episodes.csv                       — Crisis episodes with archetype labels
  full_transition_matrix.json        — 5×5 transition matrix + CIs
  phase{1..5}_duration_conditioned.json — Duration-conditioned transitions
  phase3_crossover.json              — Recovery–escalation crossover point
  sensitivity_analysis.json          — 9-variant comparison table
  sensitivity_summary.csv            — Same as CSV
  admin2_transition_analysis.json    — Admin2-level sensitivity check
  episode_verification.json          — Episode statistics verification
  left_censoring_sensitivity.json    — Left-censoring impact analysis
  quarterly_analysis.json            — Quarterly aggregation robustness
  regional_transition_analysis.json  — Regional breakdown
  temporal_comparison.json           — Early vs late period comparison
  crisis_staircase.json              — Multi-episode pathway analysis
  country_counts.json                — Country coverage summary

Author: Richard Choularton
"""

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
        aggregation: 'max', 'median', or 'dictzip' (arbitrary pick)
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
    else:
        df_agg = df.groupby(['iso3', 'ADMIN1', 'location', 'region',
                             'year_month', 'date']).agg({
            'ipc_phase': 'max',
        }).reset_index()

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
    records = []
    for location in df['location'].unique():
        loc_data = df[df['location'] == location].sort_values('date')
        if len(loc_data) == 0:
            continue

        iso3 = loc_data.iloc[0]['iso3']
        admin1 = loc_data.iloc[0].get('ADMIN1', None)
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

            records.append({
                'iso3': iso3, 'ADMIN1': admin1, 'location': location,
                'region': region, 'year_month': date.strftime('%Y-%m'),
                'date': date, 'ipc_phase': current_phase,
                'is_interpolated': is_interp
            })

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
            "transitions — insufficient for reliable inference."
        )

    return results


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

    return {
        'all': all_stats,
        'filtered': filtered_stats,
        'censored_only': censored_stats,
        'censored_only_countries': censored_only_countries,
        'note': 'Transition matrices are computed from the full interpolated time series '
                'and are NOT affected by left-censoring.',
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
# Main Execution
# ============================================================

def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = load_hfid()

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

    # SENSITIVITY VARIANTS
    print("\n" + "=" * 70)
    print("  SENSITIVITY ANALYSIS (9 pipeline variants)")
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
    ]

    sensitivity_results = []
    for var_label, priority, agg, gap, is_a2 in variants:
        if var_label == 'FEWS + MAX + 12mo':
            sensitivity_results.append({
                'label': var_label,
                **_extract_sensitivity_row(primary_result),
            })
            continue
        if var_label == 'FEWS + admin2 + 12mo':
            sensitivity_results.append({
                'label': var_label,
                **_extract_sensitivity_row(admin2_result),
            })
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
    country_counts['countries_with_episodes'] = primary_result['episodes']['countries']
    save_json(os.path.join(OUTPUT_DIR, 'country_counts.json'), country_counts)

    # EPISODE VERIFICATION
    episode_verification = verify_episodes(primary_episodes, primary_interp)
    save_json(os.path.join(OUTPUT_DIR, 'episode_verification.json'), episode_verification)

    # LEFT-CENSORING SENSITIVITY
    lc_sensitivity = compute_left_censoring_sensitivity(primary_episodes)
    save_json(os.path.join(OUTPUT_DIR, 'left_censoring_sensitivity.json'), lc_sensitivity)

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

    # SUMMARY
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")

    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f} ({size:,} bytes)")


if __name__ == '__main__':
    main()
