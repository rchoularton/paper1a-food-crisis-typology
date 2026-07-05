#!/usr/bin/env python3
"""
run_all.py — Master Orchestration Script
==========================================

Runs the full Paper 1A analysis pipeline from raw HFID data to publication
figures. Each step is executed as a subprocess so failures are isolated.

Usage:
    python run_all.py              # Run all 17 steps
    python run_all.py --step 6     # Run only step 6
    python run_all.py --analysis-only   # Steps 01-10 only
    python run_all.py --figures-only    # Steps 11-17 only

Expected runtime: ~20-30 minutes total on a modern laptop.
"""

import argparse
import os
import subprocess
import sys
import time

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PACKAGE_ROOT, 'code')
DATA_FILE = os.path.join(PACKAGE_ROOT, 'data', 'HFID_hv1.csv')

STEPS = [
    # (step_number, filename, description, is_optional, extra_args)
    # --- Analysis (steps 1-10) ---
    (1,  '01_reference_pipeline.py',       'Core pipeline: episodes + transitions + admin2',        False, ['--phase', 'core']),
    (2,  '01_reference_pipeline.py',       'Robustness: sensitivity + verification analyses',       False, ['--phase', 'robustness']),
    (3,  '02_generate_transitions.py',     'Archetype transition sequences',                        False, []),
    (4,  '03_gap_analysis.py',             'Recovery gap analysis',                                 False, []),
    (5,  '04_gap_compression_robustness.py', 'Gap compression robustness checks',                   False, []),
    (6,  '05_hfid_consistency.py',         'FEWS vs CH/IPC source agreement',                       False, []),
    (7,  'r1_jackknife_robustness.py',     'R1.2: leave-one-country-out jackknife (ED Table 6)',    False, []),
    (8,  'r1_spatial_extent.py',           'R1.4: national footprint by archetype (ED Fig 3a)',     False, []),
    (9,  'r1_spatial_adjacency.py',        'R1.4: neighbour co-escalation (ED Fig 3b; needs geopandas)', False, []),
    (10, 'r1_source_unit_composition.py',  'R3.7/R3.9: source mix + admin1/admin2 unit composition', False, []),
    # --- Figures (steps 11-17) ---
    (11, '06_fig1_archetypes.py',          'Figure 1: archetype scatter plot',                      False, []),
    (12, '07_fig2_alluvial.py',            'Figure 2: alluvial transitions',                        False, []),
    (13, '08_fig3_phase_dynamics.py',      'Figure 3: recovery asymmetry (6-panel)',                 False, []),
    (14, '12_extdata_staircase.py',        'Extended Data Fig 1: crisis staircase',                 False, []),
    (15, '13_extdata_gap_compression.py',  'Figure 4: gap compression dual-panel',                   False, []),
    (16, '14_extdata_gap_map.py',          'Extended Data Fig 2: geographic map (optional geopandas)', True,  []),
    (17, '15_extdata_spatial.py',          'Extended Data Fig 3: spatial dynamics by archetype',    False, []),
]


def check_data():
    """Check that input data exists."""
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: Input data not found at:")
        print(f"  {DATA_FILE}")
        print(f"\nPlease ensure the HFID CSV is in the data/ directory.")
        print(f"See data/README.md for download instructions.")
        sys.exit(1)
    size_mb = os.path.getsize(DATA_FILE) / (1024 * 1024)
    print(f"  Input data: {DATA_FILE} ({size_mb:.1f} MB)")


def run_step(step_num, filename, description, is_optional, extra_args=None):
    """Run a single pipeline step as a subprocess."""
    script_path = os.path.join(CODE_DIR, filename)

    if not os.path.exists(script_path):
        print(f"  WARNING: Script not found: {script_path}")
        return False, 0

    print(f"\n{'='*70}")
    print(f"  Step {step_num:02d}: {description}")
    print(f"  Script: {filename}" + (f" {' '.join(extra_args)}" if extra_args else ""))
    print(f"{'='*70}")

    cmd = [sys.executable, script_path] + (extra_args or [])

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=PACKAGE_ROOT,
            capture_output=False,
            timeout=3600,  # 60 minute timeout per step
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"\n  PASSED ({elapsed:.1f}s)")
            return True, elapsed
        else:
            if is_optional:
                print(f"\n  SKIPPED (optional step, returned code {result.returncode}) ({elapsed:.1f}s)")
                return True, elapsed
            else:
                print(f"\n  FAILED (exit code {result.returncode}) ({elapsed:.1f}s)")
                return False, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\n  TIMEOUT after {elapsed:.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ERROR: {e} ({elapsed:.1f}s)")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description='Run Paper 1A analysis pipeline')
    parser.add_argument('--step', type=int, help='Run only this step number (1-17)')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Run analysis steps only (01-10)')
    parser.add_argument('--figures-only', action='store_true',
                        help='Run figure steps only (11-17)')
    args = parser.parse_args()

    print("=" * 70)
    print("  Paper 1A: Food Security Crisis Typology")
    print("  Reproducibility Pipeline")
    print("=" * 70)

    # Check data
    check_data()

    # Determine which steps to run
    if args.step:
        steps_to_run = [s for s in STEPS if s[0] == args.step]
        if not steps_to_run:
            print(f"\nERROR: Step {args.step} not found. Valid steps: 1-17")
            sys.exit(1)
    elif args.analysis_only:
        steps_to_run = [s for s in STEPS if s[0] <= 10]
    elif args.figures_only:
        steps_to_run = [s for s in STEPS if s[0] >= 11]
    else:
        steps_to_run = STEPS

    print(f"\n  Running {len(steps_to_run)} step(s)...")

    # Run steps
    total_start = time.time()
    results = []
    for step_num, filename, description, is_optional, extra_args in steps_to_run:
        passed, elapsed = run_step(step_num, filename, description, is_optional, extra_args)
        results.append((step_num, filename, passed, elapsed))

    # Summary
    total_elapsed = time.time() - total_start
    n_passed = sum(1 for _, _, p, _ in results if p)
    n_failed = sum(1 for _, _, p, _ in results if not p)

    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Step':<6} {'Status':<10} {'Time':>8}  Description")
    print(f"  {'-'*60}")
    for step_num, filename, passed, elapsed in results:
        status = "PASSED" if passed else "FAILED"
        desc = next(d for s, _, d, _, _ in STEPS if s == step_num)
        print(f"  {step_num:02d}     {status:<10} {elapsed:>7.1f}s  {desc}")

    print(f"\n  Total: {n_passed} passed, {n_failed} failed ({total_elapsed:.0f}s)")

    if n_failed > 0:
        print(f"\n  WARNING: {n_failed} step(s) failed. Check output above for details.")
        sys.exit(1)
    else:
        print(f"\n  All steps completed successfully!")
        print(f"  Outputs: outputs/data/ (analysis) and outputs/figures/ (figures)")


if __name__ == '__main__':
    main()
