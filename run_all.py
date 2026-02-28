#!/usr/bin/env python3
"""
run_all.py — Master Orchestration Script
==========================================

Runs the full Paper 1A analysis pipeline from raw HFID data to publication
figures. Each step is executed as a subprocess so failures are isolated.

Usage:
    python run_all.py              # Run all 14 steps
    python run_all.py --step 6     # Run only step 6
    python run_all.py --analysis-only   # Steps 01-05 only
    python run_all.py --figures-only    # Steps 06-14 only
    python run_all.py --paper-only     # Paper figures only (skips supplementary)

Expected runtime: ~15-25 minutes total on a modern laptop.
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
    # (step_number, filename, description, is_optional)
    (1,  '01_reference_pipeline.py',       'Core pipeline: episodes + transitions + sensitivity', False),
    (2,  '02_generate_transitions.py',     'Archetype transition sequences', False),
    (3,  '03_gap_analysis.py',             'Recovery gap analysis', False),
    (4,  '04_gap_compression_robustness.py', 'Gap compression robustness checks', False),
    (5,  '05_hfid_consistency.py',         'FEWS vs CH/IPC source agreement', False),
    (6,  '06_fig1_archetypes.py',          'Figure 1: archetype scatter plot', False),
    (7,  '07_fig2_alluvial.py',            'Figure 2: alluvial transitions', False),
    (8,  '08_fig3_phase_dynamics.py',      'Figure 3: 6-panel phase dynamics', False),
    (9,  '09_fig4_recovery.py',            'Supplementary: recovery degradation curves', True),
    (10, '10_fig5_temporal.py',            'Supplementary: temporal trends', True),
    (11, '11_fig6_framework.py',           'Supplementary: conceptual framework', True),
    (12, '12_extdata_staircase.py',        'Extended Data Fig 1: crisis staircase', False),
    (13, '13_extdata_gap_compression.py',  'Figure 4 + Extended Data: gap compression', False),
    (14, '14_extdata_gap_map.py',          'Extended Data: geographic map (optional geopandas)', True),
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


def run_step(step_num, filename, description, is_optional):
    """Run a single pipeline step as a subprocess."""
    script_path = os.path.join(CODE_DIR, filename)

    if not os.path.exists(script_path):
        print(f"  WARNING: Script not found: {script_path}")
        return False, 0

    print(f"\n{'='*70}")
    print(f"  Step {step_num:02d}: {description}")
    print(f"  Script: {filename}")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PACKAGE_ROOT,
            capture_output=False,
            timeout=1800,  # 30 minute timeout per step
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
    parser.add_argument('--step', type=int, help='Run only this step number (1-14)')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Run analysis steps only (01-05)')
    parser.add_argument('--figures-only', action='store_true',
                        help='Run figure steps only (06-14)')
    parser.add_argument('--paper-only', action='store_true',
                        help='Run paper figures only (skips supplementary steps 09-11)')
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
            print(f"\nERROR: Step {args.step} not found. Valid steps: 1-14")
            sys.exit(1)
    elif args.analysis_only:
        steps_to_run = [s for s in STEPS if s[0] <= 5]
    elif args.figures_only:
        steps_to_run = [s for s in STEPS if s[0] >= 6]
    elif args.paper_only:
        # Steps 01-08, 12-14 (skip supplementary steps 09-11)
        supplementary = {9, 10, 11}
        steps_to_run = [s for s in STEPS if s[0] not in supplementary]
    else:
        steps_to_run = STEPS

    print(f"\n  Running {len(steps_to_run)} step(s)...")

    # Run steps
    total_start = time.time()
    results = []
    for step_num, filename, description, is_optional in steps_to_run:
        passed, elapsed = run_step(step_num, filename, description, is_optional)
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
        desc = next(d for s, _, d, _ in STEPS if s == step_num)
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
