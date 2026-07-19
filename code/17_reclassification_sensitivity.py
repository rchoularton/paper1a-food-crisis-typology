#!/usr/bin/env python3
# @status:   canonical
# @process:  P6-audit-v24
# @paper:    paper1
"""
17_reclassification_sensitivity.py — Per-episode reclassification across variants
=================================================================================

Matches crisis episodes across pipeline variants on LOCATION + ONSET MONTH and
reports how many matched episodes change archetype relative to the primary
baseline (FEWS + MAX + 12-month interpolation).

Added per the v24 audit (finding A2 Part 2, AUTHOR DECISION 2026-07-18): no
reclassification calculation existed anywhere in the codebase — the pipeline
computed each variant independently and never joined across them, which is how
the manuscript came to print rates ("4.2% and 2.8%") that were actually
archetype shares transcribed from the adjacent table. Per the audit's
recommended scope, the rate is computed for ALL sensitivity variants, not only
the interpolation-gap ones, so the figure can never again be quoted without a
defined denominator.

The admin2 variant is included for completeness but cannot match the admin1
baseline by construction (different location universe); its row reports the
unmatched counts explicitly.

Input:
  data/HFID_hv1.csv — HFID v1.1.1

Output:
  outputs/data/reclassification_sensitivity.json

Reference values (v24 audit, verified 2026-07-18):
  6-month gap:  31 of 1,658 reclassified = 1.87% (excl. left-censored 1.85%)
  18-month gap:  0 of 1,658 reclassified = 0.00% (excl. left-censored 0.00%)
  1,658 episodes match at every gap, zero unmatched.
"""

import importlib.util
import json
import os

# ============================================================
# Load the canonical pipeline by file path (module name starts
# with a digit; same pattern as reference_transition_analysis.py)
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_PATH = os.path.join(_HERE, '01_reference_pipeline.py')
_spec = importlib.util.spec_from_file_location('_reference_pipeline_impl', _PIPELINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_hfid = _mod.load_hfid
run_full_pipeline = _mod.run_full_pipeline

PACKAGE_ROOT = os.path.dirname(_HERE)
OUTPUT_PATH = os.path.join(PACKAGE_ROOT, 'outputs', 'data',
                           'reclassification_sensitivity.json')

BASELINE = ('FEWS + MAX + 12mo', 'fews', 'max', 12, False)

# Same variant list as the sensitivity analysis in 01_reference_pipeline.py,
# minus the baseline itself.
VARIANTS = [
    ('FEWS + MAX + 6mo', 'fews', 'max', 6, False),
    ('FEWS + MAX + 18mo', 'fews', 'max', 18, False),
    ('IPC + MAX + 6mo', 'ipc', 'max', 6, False),
    ('IPC + MAX + 12mo', 'ipc', 'max', 12, False),
    ('IPC + MAX + 18mo', 'ipc', 'max', 18, False),
    ('FEWS + admin2 + 12mo', 'fews', 'max', 12, True),
    ('FEWS + dictzip + 12mo', 'fews', 'dictzip', 12, False),
    ('FEWS + MEDIAN + 12mo', 'fews', 'median', 12, False),
    ('FEWS + MEAN + 12mo', 'fews', 'mean', 12, False),
]


def episode_keys(df_episodes):
    """Map (location, onset year-month) -> (archetype, is_left_censored)."""
    keyed = {}
    for _, ep in df_episodes.iterrows():
        onset = ep['dates'][0]
        onset_ym = onset.strftime('%Y-%m') if hasattr(onset, 'strftime') else str(onset)[:7]
        keyed[(ep['location'], onset_ym)] = (
            ep['archetype'], bool(ep.get('is_left_censored', False)))
    return keyed


def compare_to_baseline(base_keys, var_keys):
    """Match on location + onset month; count archetype changes."""
    matched = 0
    reclassified = 0
    matched_non_lc = 0
    reclassified_non_lc = 0

    for key, (base_arch, base_lc) in base_keys.items():
        if key not in var_keys:
            continue
        matched += 1
        changed = var_keys[key][0] != base_arch
        if changed:
            reclassified += 1
        if not base_lc:
            matched_non_lc += 1
            if changed:
                reclassified_non_lc += 1

    return {
        'matched': matched,
        'unmatched_baseline': len(base_keys) - matched,
        'unmatched_variant': len(var_keys) - matched,
        'reclassified': reclassified,
        'reclassification_rate_pct': (
            round(reclassified / matched * 100, 2) if matched > 0 else None),
        'reclassified_excl_left_censored': reclassified_non_lc,
        'reclassification_rate_excl_left_censored_pct': (
            round(reclassified_non_lc / matched_non_lc * 100, 2)
            if matched_non_lc > 0 else None),
    }


def main():
    print('=' * 70)
    print('  RECLASSIFICATION SENSITIVITY (v24 audit A2 Part 2)')
    print('=' * 70)

    df_raw = load_hfid()

    base_label, base_priority, base_agg, base_gap, base_a2 = BASELINE
    _, base_episodes, _ = run_full_pipeline(
        df_raw, priority=base_priority, aggregation=base_agg,
        max_gap=base_gap, run_bootstrap=False, is_admin2=base_a2,
        label=f'BASELINE: {base_label}')
    base_keys = episode_keys(base_episodes)
    print(f'\n  Baseline episodes: {len(base_keys)}')

    results = []
    for label, priority, agg, gap, is_a2 in VARIANTS:
        _, var_episodes, _ = run_full_pipeline(
            df_raw, priority=priority, aggregation=agg, max_gap=gap,
            run_bootstrap=False, is_admin2=is_a2, label=label)
        row = {
            'label': label,
            'priority': priority,
            'aggregation': agg,
            'interpolation_gap': gap,
            'is_admin2': is_a2,
            **compare_to_baseline(base_keys, episode_keys(var_episodes)),
        }
        if row['matched'] == 0:
            row['note'] = ('No episodes match the admin1 baseline: this variant '
                           'uses a different location universe.')
        results.append(row)

    output = {
        'baseline': {
            'label': base_label,
            'priority': base_priority,
            'aggregation': base_agg,
            'interpolation_gap': base_gap,
            'episodes': len(base_keys),
        },
        'match_definition': ('Episodes matched across variants on location + '
                             'onset month (YYYY-MM of first episode month). '
                             'Reclassified = matched episode whose archetype '
                             'differs from the baseline archetype.'),
        'variants': results,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Saved: {OUTPUT_PATH}')

    print(f"\n  {'Variant':<24} {'Matched':>8} {'Reclass.':>9} {'Rate':>8}")
    print('  ' + '-' * 55)
    for r in results:
        rate = (f"{r['reclassification_rate_pct']}%"
                if r['reclassification_rate_pct'] is not None else 'n/a')
        print(f"  {r['label']:<24} {r['matched']:>8} {r['reclassified']:>9} {rate:>8}")


if __name__ == '__main__':
    main()
