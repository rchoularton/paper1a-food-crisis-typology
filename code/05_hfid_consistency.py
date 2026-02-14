#!/usr/bin/env python3
"""
05_hfid_consistency.py — HFID Data Consistency Analysis
========================================================

Assesses internal consistency of the Harmonized Food Insecurity Dataset
(HFID v1.1.1, Machefer et al. 2025) used throughout this paper.

Three analyses:

  1. FEWS NET vs CH/IPC classification consistency
     Where both classification systems provide a phase for the same
     area-month, measures exact agreement, within-1 agreement, direction
     of disagreement, and Cohen's kappa (computed without sklearn).

  2. IPC phase vs food security input indicators
     Compares the assigned IPC phase against the implied phase from
     FCS (Food Consumption Score) and rCSI (Reduced Coping Strategy
     Index) using IPC Technical Manual v3.1 reference thresholds.

  3. Data source coverage patterns
     Documents how FEWS NET and CH/IPC coverage varies across countries,
     time, and indicator availability.

Inputs (relative to package root):
    data/HFID_hv1.csv  — Raw HFID v1.1.1

Outputs (relative to package root, written to outputs/data/hfid_consistency/):
    fewsnet_vs_ch_confusion_matrix.csv   (if overlapping records exist)
    fewsnet_vs_ch_by_country.csv         (if overlapping records exist)
    ipc_vs_fcs_lit_confusion_matrix.csv
    ipc_vs_fcs_lit_by_country.csv
    fcs_lit_by_phase.csv
    ipc_vs_rcsi_lit_confusion_matrix.csv
    ipc_vs_rcsi_lit_by_country.csv
    rcsi_lit_by_phase.csv
    source_coverage_by_year.csv

Dependencies: pandas, numpy  (standard library + scientific-Python stack)

Author: Richard Choularton
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Paths — relative to package root
# ============================================================
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HFID_PATH = os.path.join(PACKAGE_ROOT, 'data', 'HFID_hv1.csv')
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'hfid_consistency')


# ============================================================
# IPC BENCHMARK THRESHOLDS
# ============================================================
# These are the IPC Technical Manual v3.1 reference thresholds.
# FCS and rCSI are key IPC input indicators.

# Food Consumption Score (FCS) — higher = better food consumption
# IPC benchmarks (proportion with poor/borderline consumption):
#   Phase 1: <5% poor+borderline
#   Phase 2: 5-20%
#   Phase 3: 20-40%
#   Phase 4: 40-60%
#   Phase 5: >60%
# In HFID, fcs_lit represents proportion with insufficient consumption (0-1 scale)
FCS_THRESHOLDS = {
    1: (0.0, 0.05),   # <5% insufficient
    2: (0.05, 0.20),  # 5-20%
    3: (0.20, 0.40),  # 20-40%
    4: (0.40, 0.60),  # 40-60%
    5: (0.60, 1.00),  # >60%
}

# Reduced Coping Strategy Index (rCSI) — higher = worse coping
# IPC benchmarks (proportion using crisis-level coping):
#   Phase 1: <5%
#   Phase 2: 5-20%
#   Phase 3: 20-40%
#   Phase 4: 40-60%
#   Phase 5: >60%
# In HFID, rcsi_lit represents proportion in crisis coping (0-1 scale)
RCSI_THRESHOLDS = {
    1: (0.0, 0.05),
    2: (0.05, 0.20),
    3: (0.20, 0.40),
    4: (0.40, 0.60),
    5: (0.60, 1.00),
}


# ============================================================
# Cohen's kappa (manual implementation — no sklearn dependency)
# ============================================================

def _cohens_kappa(y1, y2, weights=None):
    """
    Compute Cohen's kappa between two integer arrays.

    Parameters
    ----------
    y1, y2 : array-like of int
        Two raters' classifications (same length).
    weights : None or 'linear'
        None  = unweighted (exact agreement only).
        'linear' = linearly weighted.

    Returns
    -------
    float  kappa statistic
    """
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    labels = np.union1d(y1, y2)
    n_labels = len(labels)
    label_to_idx = {lab: idx for idx, lab in enumerate(labels)}

    # Build observed confusion matrix
    conf = np.zeros((n_labels, n_labels), dtype=float)
    for a, b in zip(y1, y2):
        conf[label_to_idx[a], label_to_idx[b]] += 1

    n = conf.sum()
    if n == 0:
        return 0.0

    # Build weight matrix
    w = np.zeros((n_labels, n_labels), dtype=float)
    if weights is None:
        # Unweighted: 0 on diagonal, 1 elsewhere
        w = np.ones((n_labels, n_labels)) - np.eye(n_labels)
    elif weights == 'linear':
        for i in range(n_labels):
            for j in range(n_labels):
                w[i, j] = abs(labels[i] - labels[j]) / (labels[-1] - labels[0]) if labels[-1] != labels[0] else 0
    else:
        raise ValueError(f"Unknown weights: {weights}")

    # Expected matrix under independence
    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    # Weighted observed and expected disagreement
    obs_disagreement = np.sum(w * conf) / n
    exp_disagreement = np.sum(w * expected) / n

    if exp_disagreement == 0:
        return 1.0  # Perfect agreement

    kappa = 1.0 - obs_disagreement / exp_disagreement
    return kappa


# ============================================================
# Helper: implied IPC phase from indicator value
# ============================================================

def get_implied_phase(value, thresholds):
    """Given an indicator value, return the implied IPC phase."""
    if pd.isna(value):
        return np.nan
    for phase, (low, high) in thresholds.items():
        if low <= value < high:
            return phase
    if value >= 1.0:
        return 5
    return np.nan


# ============================================================
# Data loading
# ============================================================

def load_data():
    """Load raw HFID data from CSV.

    The raw CSV uses column names with spaces (e.g. 'fcs_rt mean').
    We rename to underscore-separated names for consistency with the
    Directus field names used in the original analysis.
    """
    print(f"Loading HFID data from: {HFID_PATH}")
    df = pd.read_csv(HFID_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} records")

    # Rename columns with spaces to underscores
    rename_map = {
        'ADMIN0': 'admin0',
        'ADMIN1': 'admin_level_1',
        'ADMIN2': 'admin_level_2',
        'fcs_rt mean': 'fcs_rt_mean',
        'fcs_rt max': 'fcs_rt_max',
        'fcs_rt min': 'fcs_rt_min',
        'rcsi_rt mean': 'rcsi_rt_mean',
        'rcsi_rt max': 'rcsi_rt_max',
        'rcsi_rt min': 'rcsi_rt_min',
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert IPC phase columns to numeric (they may contain non-numeric values)
    for col in ['ipc_phase_fews', 'ipc_phase_ipcch']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert indicator columns to numeric
    for col in ['fcs_lit', 'rcsi_lit', 'fcs_rt_mean', 'fcs_rt_max',
                'fcs_rt_min', 'rcsi_rt_mean', 'rcsi_rt_max', 'rcsi_rt_min',
                'ha_fews', 'ha_ipcch']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ============================================================
# ANALYSIS 1: FEWS NET vs CH/IPC CONSISTENCY
# ============================================================

def analyze_fewsnet_vs_ch(df):
    """Analyze consistency where both FEWS NET and CH/IPC classifications exist."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: FEWS NET vs CH/IPC CLASSIFICATION CONSISTENCY")
    print("=" * 70)

    # Filter to records where BOTH sources have data
    both = df[
        df['ipc_phase_fews'].notna() &
        df['ipc_phase_ipcch'].notna() &
        (df['ipc_phase_fews'] != 6) &
        (df['ipc_phase_ipcch'] != 6)
    ].copy()

    both['fews'] = both['ipc_phase_fews'].astype(int)
    both['ch'] = both['ipc_phase_ipcch'].astype(int)

    print(f"\nRecords with BOTH FEWS NET and CH/IPC data: {len(both):,}")
    print(f"  (out of {len(df):,} total records = {100*len(both)/len(df):.1f}%)")

    if len(both) == 0:
        print("\n  NO OVERLAPPING RECORDS FOUND.")
        print("  FEWS NET and CH/IPC appear to cover different area-months.")
        # Show coverage breakdown
        fews_only = df['ipc_phase_fews'].notna().sum()
        ch_only = df['ipc_phase_ipcch'].notna().sum()
        print(f"\n  Records with FEWS NET only: {fews_only:,}")
        print(f"  Records with CH/IPC only: {ch_only:,}")

        # Check country overlap
        fews_countries = set(df[df['ipc_phase_fews'].notna()]['iso3'].unique())
        ch_countries = set(df[df['ipc_phase_ipcch'].notna()]['iso3'].unique())
        overlap_countries = fews_countries & ch_countries
        print(f"\n  Countries with FEWS NET data: {len(fews_countries)}")
        print(f"  Countries with CH/IPC data: {len(ch_countries)}")
        print(f"  Countries with BOTH: {len(overlap_countries)}")
        if overlap_countries:
            print(f"  Overlap countries: {sorted(overlap_countries)}")

            # For overlap countries, check temporal overlap
            print(f"\n  Checking temporal overlap in shared countries...")
            for iso3 in sorted(overlap_countries)[:10]:
                country_data = df[df['iso3'] == iso3]
                fews_dates = set(
                    country_data[country_data['ipc_phase_fews'].notna()]['year_month'])
                ch_dates = set(
                    country_data[country_data['ipc_phase_ipcch'].notna()]['year_month'])
                overlap_dates = fews_dates & ch_dates
                print(f"    {iso3}: FEWS={len(fews_dates)} months, "
                      f"CH={len(ch_dates)} months, "
                      f"overlap={len(overlap_dates)} months")

                # Check admin1 overlap within shared months
                if overlap_dates:
                    for ym in sorted(overlap_dates)[:3]:
                        month_data = country_data[country_data['year_month'] == ym]
                        fews_areas = set(
                            month_data[month_data['ipc_phase_fews'].notna()][
                                'admin_level_1'].dropna())
                        ch_areas = set(
                            month_data[month_data['ipc_phase_ipcch'].notna()][
                                'admin_level_1'].dropna())
                        area_overlap = fews_areas & ch_areas
                        print(f"      {ym}: FEWS areas={len(fews_areas)}, "
                              f"CH areas={len(ch_areas)}, "
                              f"SAME areas={len(area_overlap)}")

        return both

    # --- If we DO have overlapping records ---

    # Exact agreement
    exact_match = (both['fews'] == both['ch']).sum()
    within_one = (abs(both['fews'] - both['ch']) <= 1).sum()
    disagree_2plus = (abs(both['fews'] - both['ch']) >= 2).sum()

    print(f"\n--- Agreement Summary ---")
    print(f"  Exact match:        {exact_match:,} ({100*exact_match/len(both):.1f}%)")
    print(f"  Within +/-1 phase:  {within_one:,} ({100*within_one/len(both):.1f}%)")
    print(f"  Disagree by >=2:    {disagree_2plus:,} ({100*disagree_2plus/len(both):.1f}%)")

    # Direction of disagreement
    fews_higher = (both['fews'] > both['ch']).sum()
    ch_higher = (both['ch'] > both['fews']).sum()
    print(f"\n--- Direction of Disagreement ---")
    print(f"  FEWS NET higher (more severe): {fews_higher:,} "
          f"({100*fews_higher/len(both):.1f}%)")
    print(f"  CH/IPC higher (more severe):   {ch_higher:,} "
          f"({100*ch_higher/len(both):.1f}%)")

    # Confusion matrix
    print(f"\n--- Confusion Matrix (rows=FEWS NET, cols=CH/IPC) ---")
    matrix = pd.crosstab(both['fews'], both['ch'], margins=True)
    print(matrix.to_string())

    # Save confusion matrix
    matrix.to_csv(os.path.join(OUTPUT_DIR, 'fewsnet_vs_ch_confusion_matrix.csv'))

    # Cohen's kappa (manual implementation)
    kappa = _cohens_kappa(both['fews'].values, both['ch'].values, weights=None)
    kappa_weighted = _cohens_kappa(both['fews'].values, both['ch'].values,
                                   weights='linear')
    print(f"\n--- Agreement Statistics ---")
    print(f"  Cohen's kappa (unweighted): {kappa:.3f}")
    print(f"  Cohen's kappa (linear weighted): {kappa_weighted:.3f}")
    print(f"  Interpretation: ", end="")
    if kappa_weighted >= 0.81:
        print("Almost perfect agreement")
    elif kappa_weighted >= 0.61:
        print("Substantial agreement")
    elif kappa_weighted >= 0.41:
        print("Moderate agreement")
    elif kappa_weighted >= 0.21:
        print("Fair agreement")
    else:
        print("Slight/poor agreement")

    # Mean difference
    mean_diff = (both['fews'] - both['ch']).mean()
    std_diff = (both['fews'] - both['ch']).std()
    print(f"\n  Mean difference (FEWS - CH): {mean_diff:.3f} +/- {std_diff:.3f}")

    # By country
    print(f"\n--- Agreement by Country (top 20 by record count) ---")
    country_stats = []
    for iso3, group in both.groupby('iso3'):
        exact = (group['fews'] == group['ch']).mean()
        within1 = (abs(group['fews'] - group['ch']) <= 1).mean()
        mean_d = (group['fews'] - group['ch']).mean()
        country_stats.append({
            'iso3': iso3,
            'n_records': len(group),
            'exact_match_pct': round(100 * exact, 1),
            'within_1_pct': round(100 * within1, 1),
            'mean_diff': round(mean_d, 3),
            'fews_higher_pct': round(100 * (group['fews'] > group['ch']).mean(), 1),
            'ch_higher_pct': round(100 * (group['ch'] > group['fews']).mean(), 1)
        })

    country_df = pd.DataFrame(country_stats).sort_values('n_records', ascending=False)
    print(country_df.head(20).to_string(index=False))
    country_df.to_csv(os.path.join(OUTPUT_DIR, 'fewsnet_vs_ch_by_country.csv'),
                      index=False)

    # By year
    both['year'] = both['year_month'].str[:4]
    print(f"\n--- Agreement by Year ---")
    for year, group in both.groupby('year'):
        exact = (group['fews'] == group['ch']).mean()
        mean_d = (group['fews'] - group['ch']).mean()
        print(f"  {year}: n={len(group):,}, exact={100*exact:.1f}%, "
              f"mean_diff={mean_d:+.3f}")

    # Critical disagreements: one says Phase 2 (OK), other says Phase 4+ (emergency)
    critical = both[
        ((both['fews'] <= 2) & (both['ch'] >= 4)) |
        ((both['ch'] <= 2) & (both['fews'] >= 4))
    ]
    print(f"\n--- Critical Disagreements (Phase <=2 vs Phase >=4) ---")
    print(f"  Count: {len(critical):,} ({100*len(critical)/len(both):.2f}%)")
    if len(critical) > 0:
        print(f"  Countries: {critical['iso3'].value_counts().head(10).to_dict()}")

    return both


# ============================================================
# ANALYSIS 2: IPC PHASE vs FOOD SECURITY INDICATORS
# ============================================================

def analyze_ipc_vs_indicators(df):
    """Check IPC phase consistency against FCS and rCSI indicators."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: IPC PHASE vs FOOD SECURITY INDICATOR CONSISTENCY")
    print("=" * 70)

    # Create combined IPC phase
    df['ipc_phase'] = df['ipc_phase_fews'].fillna(df['ipc_phase_ipcch'])
    df.loc[df['ipc_phase'] == 6, 'ipc_phase'] = np.nan

    # Check available indicators
    indicators = {
        'fcs_lit': ('Food Consumption Score (literature)', FCS_THRESHOLDS),
        'rcsi_lit': ('Reduced Coping Strategy Index (literature)', RCSI_THRESHOLDS),
    }

    # Also check real-time indicators if available
    rt_indicators = ['fcs_rt_mean', 'rcsi_rt_mean']
    for col in rt_indicators:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col}: {n:,} records available")

    for indicator_col, (indicator_name, thresholds) in indicators.items():
        print(f"\n{'=' * 70}")
        print(f"  INDICATOR: {indicator_name} ({indicator_col})")
        print(f"{'=' * 70}")

        if indicator_col not in df.columns:
            print(f"  Column '{indicator_col}' not found in dataset. Skipping.")
            continue

        # Filter to records with both IPC phase and this indicator
        valid = df[df['ipc_phase'].notna() & df[indicator_col].notna()].copy()
        valid['phase'] = valid['ipc_phase'].astype(int)
        valid['indicator_value'] = valid[indicator_col].astype(float)

        print(f"  Records with both IPC phase and {indicator_col}: {len(valid):,}")
        print(f"  (out of {df['ipc_phase'].notna().sum():,} with IPC phase)")

        if len(valid) == 0:
            print("  No overlapping records. Skipping.")
            continue

        # Coverage by phase
        print(f"\n  Coverage by IPC Phase:")
        for phase in range(1, 6):
            n_phase = (valid['phase'] == phase).sum()
            n_total = (df['ipc_phase'] == phase).sum()
            pct = 100 * n_phase / n_total if n_total > 0 else 0
            print(f"    Phase {phase}: {n_phase:,} of {n_total:,} ({pct:.1f}% coverage)")

        # Descriptive statistics by phase
        print(f"\n  {indicator_col} Statistics by IPC Phase:")
        print(f"  {'Phase':>6} {'N':>8} {'Mean':>8} {'Median':>8} "
              f"{'Std':>8} {'Min':>8} {'Max':>8}")
        phase_stats = []
        for phase in range(1, 6):
            subset = valid[valid['phase'] == phase]['indicator_value']
            if len(subset) > 0:
                stats_row = {
                    'phase': phase,
                    'n': len(subset),
                    'mean': subset.mean(),
                    'median': subset.median(),
                    'std': subset.std(),
                    'min': subset.min(),
                    'max': subset.max()
                }
                phase_stats.append(stats_row)
                print(f"  {phase:>6} {len(subset):>8,} {subset.mean():>8.3f} "
                      f"{subset.median():>8.3f} {subset.std():>8.3f} "
                      f"{subset.min():>8.3f} {subset.max():>8.3f}")

        # Implied phase from indicator
        valid['implied_phase'] = valid['indicator_value'].apply(
            lambda v: get_implied_phase(v, thresholds)
        )
        valid_with_implied = valid[valid['implied_phase'].notna()].copy()
        valid_with_implied['implied_phase'] = valid_with_implied['implied_phase'].astype(int)

        # Agreement analysis
        exact = (valid_with_implied['phase'] == valid_with_implied['implied_phase']).sum()
        within1 = (abs(valid_with_implied['phase'] - valid_with_implied['implied_phase']) <= 1).sum()
        total = len(valid_with_implied)

        print(f"\n  --- {indicator_col} vs IPC Phase Agreement ---")
        print(f"  Total comparable records: {total:,}")
        print(f"  Exact match:     {exact:,} ({100*exact/total:.1f}%)")
        print(f"  Within +/-1 phase: {within1:,} ({100*within1/total:.1f}%)")

        # Direction
        ipc_higher = (valid_with_implied['phase'] > valid_with_implied['implied_phase']).sum()
        indicator_higher = (valid_with_implied['implied_phase'] > valid_with_implied['phase']).sum()
        print(f"  IPC phase higher than {indicator_col} implies: "
              f"{ipc_higher:,} ({100*ipc_higher/total:.1f}%)")
        print(f"  {indicator_col} implies higher than IPC phase: "
              f"{indicator_higher:,} ({100*indicator_higher/total:.1f}%)")

        # Confusion matrix
        print(f"\n  --- Confusion Matrix (rows=IPC Phase, "
              f"cols=Implied from {indicator_col}) ---")
        matrix = pd.crosstab(
            valid_with_implied['phase'],
            valid_with_implied['implied_phase'],
            rownames=['IPC Phase'],
            colnames=[f'Implied from {indicator_col}'],
            margins=True
        )
        print(matrix.to_string())
        matrix.to_csv(os.path.join(OUTPUT_DIR,
                                   f'ipc_vs_{indicator_col}_confusion_matrix.csv'))

        # Mean difference by phase
        print(f"\n  --- Mean Difference (IPC - Implied) by Phase ---")
        for phase in range(1, 6):
            subset = valid_with_implied[valid_with_implied['phase'] == phase]
            if len(subset) > 0:
                diff = (subset['phase'] - subset['implied_phase']).mean()
                print(f"    Phase {phase}: mean_diff = {diff:+.2f} (n={len(subset):,})")

        # Flag systematic biases
        overall_diff = (valid_with_implied['phase'] - valid_with_implied['implied_phase']).mean()
        print(f"\n  Overall mean difference (IPC - Implied): {overall_diff:+.3f}")
        if abs(overall_diff) > 0.3:
            direction = "MORE severe" if overall_diff > 0 else "LESS severe"
            print(f"  SYSTEMATIC BIAS: IPC classifications tend to be {direction} "
                  f"than {indicator_col} implies")

        # By country
        print(f"\n  --- Agreement by Country (top 15) ---")
        country_stats = []
        for iso3, group in valid_with_implied.groupby('iso3'):
            ex = (group['phase'] == group['implied_phase']).mean()
            w1 = (abs(group['phase'] - group['implied_phase']) <= 1).mean()
            md = (group['phase'] - group['implied_phase']).mean()
            country_stats.append({
                'iso3': iso3,
                'n': len(group),
                'exact_pct': round(100 * ex, 1),
                'within1_pct': round(100 * w1, 1),
                'mean_diff': round(md, 3)
            })
        cs_df = pd.DataFrame(country_stats).sort_values('n', ascending=False)
        print(cs_df.head(15).to_string(index=False))
        cs_df.to_csv(os.path.join(OUTPUT_DIR,
                                  f'ipc_vs_{indicator_col}_by_country.csv'), index=False)

        # Monotonicity check: does the indicator increase/decrease
        # consistently with phase?
        print(f"\n  --- Monotonicity Check ---")
        means = [valid[valid['phase'] == p]['indicator_value'].mean()
                 for p in range(1, 6) if len(valid[valid['phase'] == p]) > 0]
        is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
        print(f"  Phase means: {[f'{m:.3f}' for m in means]}")
        print(f"  Monotonically increasing with phase: "
              f"{'YES' if is_monotonic else 'NO'}")

        # Save phase stats
        pd.DataFrame(phase_stats).to_csv(
            os.path.join(OUTPUT_DIR, f'{indicator_col}_by_phase.csv'), index=False)

    # Real-time indicators analysis
    for rt_col in rt_indicators:
        if rt_col not in df.columns:
            continue
        valid_rt = df[df['ipc_phase'].notna() & df[rt_col].notna()].copy()
        if len(valid_rt) < 100:
            continue

        print(f"\n{'=' * 70}")
        print(f"  REAL-TIME INDICATOR: {rt_col}")
        print(f"{'=' * 70}")
        valid_rt['phase'] = valid_rt['ipc_phase'].astype(int)

        # Use same thresholds as literature version
        thresholds = FCS_THRESHOLDS if 'fcs' in rt_col else RCSI_THRESHOLDS

        print(f"  Records: {len(valid_rt):,}")
        print(f"\n  Statistics by IPC Phase:")
        print(f"  {'Phase':>6} {'N':>8} {'Mean':>8} {'Median':>8} {'Std':>8}")
        for phase in range(1, 6):
            subset = valid_rt[valid_rt['phase'] == phase][rt_col].astype(float)
            if len(subset) > 0:
                print(f"  {phase:>6} {len(subset):>8,} {subset.mean():>8.3f} "
                      f"{subset.median():>8.3f} {subset.std():>8.3f}")

        # Implied phase
        valid_rt['implied_phase'] = valid_rt[rt_col].astype(float).apply(
            lambda v: get_implied_phase(v, thresholds)
        )
        vri = valid_rt[valid_rt['implied_phase'].notna()].copy()
        if len(vri) > 0:
            vri['implied_phase'] = vri['implied_phase'].astype(int)
            exact = (vri['phase'] == vri['implied_phase']).mean()
            within1 = (abs(vri['phase'] - vri['implied_phase']) <= 1).mean()
            print(f"\n  Exact match: {100*exact:.1f}%")
            print(f"  Within +/-1:   {100*within1:.1f}%")


# ============================================================
# ANALYSIS 3: CROSS-SOURCE COVERAGE PATTERNS
# ============================================================

def analyze_coverage_patterns(df):
    """Analyze how data source coverage varies across countries and time."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: DATA SOURCE COVERAGE PATTERNS")
    print("=" * 70)

    df['has_fews'] = df['ipc_phase_fews'].notna()
    df['has_ch'] = df['ipc_phase_ipcch'].notna()
    df['has_fcs'] = df['fcs_lit'].notna() if 'fcs_lit' in df.columns else False
    df['has_rcsi'] = df['rcsi_lit'].notna() if 'rcsi_lit' in df.columns else False

    print(f"\n  Overall coverage:")
    print(f"    FEWS NET:  {df['has_fews'].sum():>8,} records "
          f"({100*df['has_fews'].mean():.1f}%)")
    print(f"    CH/IPC:    {df['has_ch'].sum():>8,} records "
          f"({100*df['has_ch'].mean():.1f}%)")
    print(f"    FCS (lit): {df['has_fcs'].sum():>8,} records "
          f"({100*df['has_fcs'].mean():.1f}%)")
    print(f"    rCSI (lit):{df['has_rcsi'].sum():>8,} records "
          f"({100*df['has_rcsi'].mean():.1f}%)")

    # Source combinations
    print(f"\n  Source combinations:")
    df['source_combo'] = ''
    df.loc[df['has_fews'] & ~df['has_ch'], 'source_combo'] = 'FEWS only'
    df.loc[~df['has_fews'] & df['has_ch'], 'source_combo'] = 'CH/IPC only'
    df.loc[df['has_fews'] & df['has_ch'], 'source_combo'] = 'Both FEWS+CH'
    df.loc[~df['has_fews'] & ~df['has_ch'], 'source_combo'] = 'Neither'
    combo_counts = df['source_combo'].value_counts()
    for combo, count in combo_counts.items():
        print(f"    {combo:>15}: {count:>8,} ({100*count/len(df):.1f}%)")

    # By country
    print(f"\n  Source by Country (countries with both sources):")
    country_source = df.groupby('iso3').agg({
        'has_fews': 'sum',
        'has_ch': 'sum'
    }).reset_index()
    both_countries = country_source[
        (country_source['has_fews'] > 0) & (country_source['has_ch'] > 0)
    ].sort_values('has_fews', ascending=False)
    if len(both_countries) > 0:
        for _, row in both_countries.iterrows():
            print(f"    {row['iso3']}: FEWS={int(row['has_fews']):,}, "
                  f"CH={int(row['has_ch']):,}")
    else:
        print(f"    No countries have both FEWS NET and CH/IPC data")

    # Temporal evolution of sources
    print(f"\n  Source Coverage Over Time:")
    df['year'] = df['year_month'].str[:4]
    yearly = df.groupby('year').agg({
        'has_fews': 'sum',
        'has_ch': 'sum',
        'has_fcs': 'sum',
        'has_rcsi': 'sum'
    }).reset_index()
    print(f"  {'Year':>6} {'FEWS':>8} {'CH/IPC':>8} {'FCS':>8} {'rCSI':>8}")
    for _, row in yearly.iterrows():
        print(f"  {row['year']:>6} {int(row['has_fews']):>8,} "
              f"{int(row['has_ch']):>8,} {int(row['has_fcs']):>8,} "
              f"{int(row['has_rcsi']):>8,}")

    yearly.to_csv(os.path.join(OUTPUT_DIR, 'source_coverage_by_year.csv'), index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("HFID DATA CONSISTENCY ANALYSIS")
    print("=" * 70)

    df = load_data()

    # Show available columns
    print(f"\n  Available columns: {sorted(df.columns.tolist())}")

    # Analysis 1: FEWS NET vs CH
    analyze_fewsnet_vs_ch(df)

    # Analysis 2: IPC vs indicators
    analyze_ipc_vs_indicators(df)

    # Analysis 3: Coverage patterns
    analyze_coverage_patterns(df)

    print(f"\n{'=' * 70}")
    print(f"Analysis complete. Results saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
