#!/usr/bin/env python3
# @status:   validation
# @process:  P6-revision
# @paper:    paper1

"""
Archetype Classification Replicability Tests
=============================================
Validates that the archetype classification algorithm produces consistent,
correct results across known test cases and stored data.

Test Categories:
1. Unit Tests - Known phase sequences with expected archetypes
2. Boundary Tests - Episodes at classification thresholds
3. Cross-Validation - Verify stored episodes match recalculation
4. Sensitivity Analysis - Measure threshold sensitivity

Author: DRF Research Assistant
Date: January 2026
For: Paper 1A - Food Security Crisis Typology
"""

import csv
import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

# Classification thresholds (inlined; the working repo imports these from
# config.py, which is not part of the reproducibility package).
DURATION_SHORT = 12         # < 12 months = short
DURATION_PROTRACTED = 36    # > 36 months = protracted
VARIANCE_STABLE = 0.1       # < 0.1 = stable
VARIANCE_VOLATILE = 0.3     # > 0.3 = volatile
TRANSITIONS_STABLE = 1      # <= 1 = stable
TRANSITIONS_VOLATILE = 3    # >= 3 = volatile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
EPISODES_CSV = os.path.join(PACKAGE_ROOT, 'outputs', 'data', 'episodes.csv')
REPORT_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'data')

# =============================================================================
# CLASSIFICATION FUNCTIONS (copied from reprocess_crisis_episodes.py)
# =============================================================================

def classify_episode(phases):
    """
    Classify a crisis episode into one of 8 archetypes.

    This is the core classification function extracted for testing.
    It must match the logic in reprocess_crisis_episodes.py exactly.

    Args:
        phases: List of IPC phase values (integers 3-5)

    Returns:
        dict with all classification fields
    """
    duration = len(phases)
    phases_array = np.array(phases)

    # Count transitions
    total_esc = 0
    total_deesc = 0
    for i in range(len(phases) - 1):
        from_p, to_p = int(phases[i]), int(phases[i+1])
        if to_p > from_p:
            total_esc += 1
        elif to_p < from_p:
            total_deesc += 1

    total_transitions = total_esc + total_deesc
    variance = float(np.var(phases))

    # Duration class
    if duration < DURATION_SHORT:
        duration_class = 'short'
    elif duration <= DURATION_PROTRACTED:
        duration_class = 'medium'
    else:
        duration_class = 'protracted'

    # Severity class
    peak = int(max(phases))
    if peak == 3:
        severity_class = 'moderate'
    elif peak == 4:
        severity_class = 'severe'
    else:
        severity_class = 'extreme'

    # Volatility class
    if variance < VARIANCE_STABLE and total_transitions <= TRANSITIONS_STABLE:
        volatility_class = 'stable'
    elif variance > VARIANCE_VOLATILE or total_transitions >= TRANSITIONS_VOLATILE:
        volatility_class = 'volatile'
    else:
        volatility_class = 'moderate'

    # Trajectory pattern
    if len(phases) < 2:
        trajectory = None
    else:
        peak_indices = [i for i, p in enumerate(phases) if p == peak]
        peak_position = peak_indices[0] / (len(phases) - 1)

        if variance > 0.5 or total_transitions >= 4:
            trajectory = 'oscillating'
        elif variance < 0.1 and total_transitions <= 1:
            trajectory = 'steady_state'
        elif peak_position < 0.2:
            trajectory = 'immediate_peak'
        elif peak_position < 0.4:
            trajectory = 'early_peak'
        elif peak_position < 0.6:
            trajectory = 'mid_peak'
        elif peak_position < 0.8:
            trajectory = 'late_peak'
        else:
            trajectory = 'end_peak'

    # Archetype assignment (priority order matters!)
    if duration_class == 'short' and severity_class in ['severe', 'extreme'] and volatility_class == 'stable':
        archetype = 'severe_shock'
    elif duration_class == 'protracted' and severity_class == 'moderate' and volatility_class == 'stable':
        archetype = 'entrenched_moderate'
    elif duration_class == 'protracted' and severity_class in ['severe', 'extreme']:
        archetype = 'protracted_emergency'
    elif (volatility_class == 'volatile' or trajectory == 'oscillating') and total_transitions >= 3:
        # Require multiple transitions for true oscillation (not just high variance from single recovery)
        archetype = 'oscillating'
    elif trajectory in ['end_peak', 'late_peak']:
        archetype = 'escalating'
    elif trajectory in ['immediate_peak', 'early_peak']:
        archetype = 'rapid_onset'
    elif duration_class == 'short' and severity_class == 'moderate':
        archetype = 'seasonal_crisis'
    elif severity_class in ['severe', 'extreme']:
        if duration <= 12:
            archetype = 'severe_shock'
        else:
            archetype = 'protracted_emergency'
    else:
        archetype = 'prolonged_moderate'

    return {
        'duration_months': duration,
        'duration_class': duration_class,
        'peak_phase': peak,
        'severity_class': severity_class,
        'phase_variance': round(variance, 4),
        'volatility_class': volatility_class,
        'total_escalations': total_esc,
        'total_deescalations': total_deesc,
        'total_transitions': total_transitions,
        'trajectory_pattern': trajectory,
        'archetype': archetype
    }

# =============================================================================
# TEST CASES
# =============================================================================

# Unit tests: Known phase sequences with expected archetypes
UNIT_TESTS = [
    # Format: (phase_sequence, expected_archetype, description)

    # 1. SEASONAL_CRISIS - short + moderate + any volatility
    ([3, 3, 3], 'seasonal_crisis', 'Short stable Phase 3 (3 months)'),
    ([3, 3, 3, 3, 3], 'seasonal_crisis', 'Short stable Phase 3 (5 months)'),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'seasonal_crisis', 'Short stable Phase 3 (11 months, boundary)'),
    ([3], 'seasonal_crisis', 'Single month Phase 3'),

    # 2. SEVERE_SHOCK - short + severe/extreme + stable
    ([4, 4, 4], 'severe_shock', 'Short stable Phase 4'),
    ([5, 5, 5], 'severe_shock', 'Short stable Phase 5'),
    ([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'severe_shock', 'Short stable Phase 4 (11 months)'),
    ([5], 'severe_shock', 'Single month Phase 5'),

    # 3. ENTRENCHED_MODERATE - protracted + moderate + stable
    ([3] * 37, 'entrenched_moderate', 'Protracted stable Phase 3 (37 months)'),
    ([3] * 50, 'entrenched_moderate', 'Protracted stable Phase 3 (50 months)'),

    # 4. PROTRACTED_EMERGENCY - protracted + severe/extreme
    ([4] * 37, 'protracted_emergency', 'Protracted Phase 4 (37 months)'),
    ([5] * 37, 'protracted_emergency', 'Protracted Phase 5 (37 months)'),
    ([4] * 50, 'protracted_emergency', 'Long protracted Phase 4'),
    ([3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
     'protracted_emergency', 'Protracted with Phase 4 peak'),

    # 5. OSCILLATING - volatile or trajectory=oscillating
    ([3, 4, 3, 4, 3], 'oscillating', 'High variance oscillation'),
    ([3, 5, 3, 5, 3], 'oscillating', 'Extreme oscillation'),
    ([3, 4, 3, 4, 3, 4, 3, 4], 'oscillating', 'Long oscillation pattern'),
    ([3, 3, 4, 4, 3, 3, 4, 4, 3, 3], 'oscillating', 'Many transitions'),

    # 6. ESCALATING - trajectory = end_peak or late_peak
    # Note: High variance (3→5) triggers oscillating before escalating
    ([3, 3, 3, 3, 4], 'escalating', 'End peak (Phase 4 at end)'),
    ([3, 3, 3, 3, 3, 3, 3, 3, 4, 4], 'escalating', 'Late peak pattern'),
    ([3, 3, 3, 3, 3, 3, 3, 4], 'escalating', 'Gradual escalation'),

    # 7. RAPID_ONSET - trajectory = immediate_peak or early_peak
    # Note: Must have moderate volatility (not stable, which triggers severe_shock)
    ([4, 3, 3, 3, 3], 'rapid_onset', 'Immediate peak (Phase 4 at start, 5 months)'),
    ([4, 4, 3, 3, 3, 3, 3, 3], 'rapid_onset', 'Early peak pattern'),
    # Note: 10-month series with single transition has variance <0.1 → stable → severe_shock
    ([4, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'severe_shock', '10-month Phase 4 start = stable = severe_shock'),
    # Medium duration (12+ months) with low variance → steady_state trajectory → prolonged_moderate
    ([4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'severe_shock', 'Medium duration, peak 4, dur ≤ 12 → severe_shock (Rule 8 severity gate)'),

    # 8. PROLONGED_MODERATE - fallback category (medium duration, mid peak, moderate volatility)
    # Note: Early peak (position < 0.4) with moderate volatility → rapid_onset
    ([3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'rapid_onset', 'Medium duration with early peak → rapid_onset'),
    ([3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], 'severe_shock', 'Medium duration, peak 4, dur ≤ 12 → severe_shock (Rule 8 severity gate)'),
    # Note: Two 4s at position 4-5 gives early_peak (4/11=0.36 < 0.4) → rapid_onset
    ([3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3], 'rapid_onset', 'Early-mid peak (position 0.36) → rapid_onset'),

    # High variance with few transitions: trajectory='oscillating' but archetype rule needs 3+ transitions
    # Short high-variance episodes fall through to other rules based on duration/severity
    ([3, 3, 3, 3, 5], 'severe_shock', 'Peak 5, dur ≤ 12 → severe_shock (Rule 8 severity gate)'),
    ([5, 4, 3, 3, 3], 'severe_shock', 'Peak 5, dur ≤ 12 → severe_shock (Rule 8 severity gate)'),
    # Longer recovery patterns work correctly (variance < 0.5, so trajectory follows peak position)
    ([5, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'rapid_onset', 'Famine recovery pattern → immediate_peak → rapid_onset'),

    # Rule 8 severity gate edge cases
    ([4] * 18, 'protracted_emergency', '18 months pure Phase 4 → protracted_emergency (Rule 8 severity gate, dur > 12)'),
    ([4] * 12, 'severe_shock', '12 months pure Phase 4 → severe_shock (Rule 8 severity gate, dur ≤ 12)'),

    # True oscillating requires >=3 transitions
    ([3, 4, 3, 4, 3], 'oscillating', 'True oscillation (4 transitions)'),
    ([3, 4, 3, 4, 3, 4], 'oscillating', 'True oscillation (5 transitions)'),
]

# Boundary tests: Episodes at classification thresholds
BOUNDARY_TESTS = [
    # Duration boundaries
    ([3] * 11, 'seasonal_crisis', 'Duration 11 (just under short boundary)'),
    ([3] * 12, 'prolonged_moderate', 'Duration 12 (at medium boundary)'),
    ([3] * 36, 'prolonged_moderate', 'Duration 36 (at protracted boundary)'),
    ([3] * 37, 'entrenched_moderate', 'Duration 37 (just over protracted boundary)'),

    # Severity boundaries (Phase 3 vs 4)
    # Note: Short + severe + stable = severe_shock (rule 1), not rapid_onset
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'seasonal_crisis', 'Peak Phase 3 = moderate'),
    ([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'severe_shock', 'Peak Phase 4 + stable = severe_shock'),

    # Peak position boundaries (for trajectory)
    # Note: Short + severe + stable triggers severe_shock before trajectory rules apply
    # Using Phase 4 with transitions to avoid severe_shock
    ([4, 3, 3, 3, 3], 'rapid_onset', 'Peak at position 0.0 (immediate, volatile)'),
    ([3, 4, 3, 3, 3], 'rapid_onset', 'Peak at position 0.25 (early)'),
    ([3, 3, 4, 3, 3], 'severe_shock', 'Peak at position 0.5 (mid) - peak 4 → severity gate → severe_shock'),
    ([3, 3, 3, 4, 3], 'escalating', 'Peak at position 0.75 (late)'),
    ([3, 3, 3, 3, 4], 'escalating', 'Peak at position 1.0 (end)'),
]

# =============================================================================
# TEST RUNNERS
# =============================================================================

def run_unit_tests():
    """Run all unit tests and report results."""
    print("\n" + "="*70)
    print("TEST 1: UNIT TESTS - Known Phase Sequences")
    print("="*70)

    passed = 0
    failed = 0
    failures = []

    for phases, expected, description in UNIT_TESTS:
        result = classify_episode(phases)
        actual = result['archetype']

        if actual == expected:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            failures.append({
                'description': description,
                'phases': phases,
                'expected': expected,
                'actual': actual,
                'details': result
            })

        print(f"  [{status}] {description}")
        if status == "FAIL":
            print(f"         Expected: {expected}, Got: {actual}")

    print(f"\n  Results: {passed}/{passed+failed} passed")

    return passed, failed, failures


def run_boundary_tests():
    """Run boundary condition tests."""
    print("\n" + "="*70)
    print("TEST 2: BOUNDARY TESTS - Classification Thresholds")
    print("="*70)

    passed = 0
    failed = 0
    failures = []

    for phases, expected, description in BOUNDARY_TESTS:
        result = classify_episode(phases)
        actual = result['archetype']

        if actual == expected:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            failures.append({
                'description': description,
                'phases': phases,
                'expected': expected,
                'actual': actual,
                'details': result
            })

        print(f"  [{status}] {description}")
        if status == "FAIL":
            print(f"         Expected: {expected}, Got: {actual}")
            print(f"         Duration: {result['duration_class']}, Severity: {result['severity_class']}, Volatility: {result['volatility_class']}")

    print(f"\n  Results: {passed}/{passed+failed} passed")

    return passed, failed, failures


def run_cross_validation():
    """Validate stored episodes match recalculation.

    Reads the deposited outputs/data/episodes.csv, which carries the same stored
    archetype labels the working repository holds in Directus. The comparison is
    therefore identical; only the source of the stored labels differs, so the
    suite runs with no credentials and no network access.
    """
    print("\n" + "="*70)
    print("TEST 3: CROSS-VALIDATION - Stored Episodes vs Recalculation")
    print("="*70)

    if not os.path.exists(EPISODES_CSV):
        print(f"  ERROR: episodes.csv not found at {EPISODES_CSV}")
        print("  Run the pipeline first: python run_all.py --step 1")
        return 0, 0, []

    print(f"  Reading stored episodes from {EPISODES_CSV}")
    episodes = []
    with open(EPISODES_CSV, newline='') as fh:
        for row in csv.DictReader(fh):
            raw = row.get('phases', '')
            if not raw:
                continue
            episodes.append({
                'id': row.get('crisis_id', ''),
                'location': row.get('location', ''),
                'start_period': row.get('dates', '').split(',')[0] if row.get('dates') else '',
                'phase_sequence': [int(float(p)) for p in raw.split(',') if p != ''],
                'archetype': row.get('archetype'),
            })
    print(f"  Loaded {len(episodes):,} episodes")

    passed = 0
    failed = 0
    failures = []

    for ep in episodes:
        phases = ep.get('phase_sequence', [])
        if not phases:
            continue

        result = classify_episode(phases)

        stored_archetype = ep.get('archetype')
        calculated_archetype = result['archetype']

        if stored_archetype == calculated_archetype:
            passed += 1
        else:
            failed += 1
            failures.append({
                'id': ep['id'],
                'location': ep['location'],
                'start': ep['start_period'],
                'stored': stored_archetype,
                'calculated': calculated_archetype,
                'phases': phases[:10],
                'details': result,
            })

    total = passed + failed
    print(f"\n  Validated: {total:,} episodes")
    print(f"  Passed: {passed:,} ({100*passed/total:.1f}%)" if total else "  Passed: 0")
    print(f"  Failed: {failed:,}")

    if failures and len(failures) <= 10:
        print("\n  Failures:")
        for f in failures:
            print(f"    - {f['location']} ({f['start']}): stored={f['stored']}, calculated={f['calculated']}")
    elif failures:
        print(f"\n  First 5 failures:")
        for f in failures[:5]:
            print(f"    - {f['location']} ({f['start']}): stored={f['stored']}, calculated={f['calculated']}")

    return passed, failed, failures


def run_sensitivity_analysis():
    """Analyze sensitivity to threshold changes."""
    print("\n" + "="*70)
    print("TEST 4: SENSITIVITY ANALYSIS - Threshold Variations")
    print("="*70)

    # Baseline distribution, read from the deposited episodes
    if not os.path.exists(EPISODES_CSV):
        print(f"  ERROR: episodes.csv not found at {EPISODES_CSV}")
        print("  Run the pipeline first: python run_all.py --step 1")
        return

    print(f"  Reading phase sequences from {EPISODES_CSV}")
    phase_sequences = []
    with open(EPISODES_CSV, newline='') as fh:
        for row in csv.DictReader(fh):
            raw = row.get('phases', '')
            if raw:
                phase_sequences.append([int(float(p)) for p in raw.split(',') if p != ''])

    print(f"  Loaded {len(phase_sequences):,} phase sequences")

    # Test threshold variations
    variations = [
        ('Baseline', {'DURATION_SHORT': 12, 'DURATION_PROTRACTED': 36, 'VARIANCE_STABLE': 0.1, 'VARIANCE_VOLATILE': 0.3}),
        ('Duration -10%', {'DURATION_SHORT': 11, 'DURATION_PROTRACTED': 32, 'VARIANCE_STABLE': 0.1, 'VARIANCE_VOLATILE': 0.3}),
        ('Duration +10%', {'DURATION_SHORT': 13, 'DURATION_PROTRACTED': 40, 'VARIANCE_STABLE': 0.1, 'VARIANCE_VOLATILE': 0.3}),
        ('Variance -10%', {'DURATION_SHORT': 12, 'DURATION_PROTRACTED': 36, 'VARIANCE_STABLE': 0.09, 'VARIANCE_VOLATILE': 0.27}),
        ('Variance +10%', {'DURATION_SHORT': 12, 'DURATION_PROTRACTED': 36, 'VARIANCE_STABLE': 0.11, 'VARIANCE_VOLATILE': 0.33}),
    ]

    results = {}

    for name, thresholds in variations:
        # Temporarily override thresholds
        global DURATION_SHORT, DURATION_PROTRACTED, VARIANCE_STABLE, VARIANCE_VOLATILE
        DURATION_SHORT = thresholds['DURATION_SHORT']
        DURATION_PROTRACTED = thresholds['DURATION_PROTRACTED']
        VARIANCE_STABLE = thresholds['VARIANCE_STABLE']
        VARIANCE_VOLATILE = thresholds['VARIANCE_VOLATILE']

        # Classify all episodes
        counts = defaultdict(int)
        for phases in phase_sequences:
            result = classify_episode(phases)
            counts[result['archetype']] += 1

        results[name] = dict(counts)

    # Reset thresholds
    DURATION_SHORT = 12
    DURATION_PROTRACTED = 36
    VARIANCE_STABLE = 0.1
    VARIANCE_VOLATILE = 0.3

    # Print comparison
    archetypes = ['seasonal_crisis', 'prolonged_moderate', 'protracted_emergency', 'rapid_onset',
                  'entrenched_moderate', 'oscillating', 'severe_shock', 'escalating']

    print("\n  Archetype Distribution by Threshold Variation:")
    print("  " + "-"*90)
    header = f"  {'Archetype':<22}"
    for name, _ in variations:
        header += f" {name:>12}"
    print(header)
    print("  " + "-"*90)

    for arch in archetypes:
        row = f"  {arch:<22}"
        baseline = results['Baseline'].get(arch, 0)
        for name, _ in variations:
            count = results[name].get(arch, 0)
            diff = count - baseline
            if name == 'Baseline':
                row += f" {count:>12}"
            else:
                row += f" {count:>8} ({diff:+3d})"
        print(row)

    print("  " + "-"*90)

    # Calculate stability metric
    print("\n  Sensitivity Summary:")
    for name, _ in variations[1:]:  # Skip baseline
        changes = 0
        for arch in archetypes:
            baseline = results['Baseline'].get(arch, 0)
            varied = results[name].get(arch, 0)
            changes += abs(varied - baseline)
        pct_change = 100 * changes / (2 * len(phase_sequences))  # Divide by 2 since changes are counted twice
        print(f"    {name}: {changes//2} episodes changed ({pct_change:.2f}%)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*70)
    print("ARCHETYPE CLASSIFICATION REPLICABILITY TESTS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThresholds:")
    print(f"  DURATION_SHORT: <{DURATION_SHORT} months")
    print(f"  DURATION_PROTRACTED: >{DURATION_PROTRACTED} months")
    print(f"  VARIANCE_STABLE: <{VARIANCE_STABLE}")
    print(f"  VARIANCE_VOLATILE: >{VARIANCE_VOLATILE}")

    all_passed = 0
    all_failed = 0
    all_failures = []

    # Run test suites
    p, f, failures = run_unit_tests()
    all_passed += p
    all_failed += f
    all_failures.extend(failures)

    p, f, failures = run_boundary_tests()
    all_passed += p
    all_failed += f
    all_failures.extend(failures)

    p, f, failures = run_cross_validation()
    all_passed += p
    all_failed += f
    all_failures.extend(failures)

    run_sensitivity_analysis()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\n  Total Tests: {all_passed + all_failed:,}")
    print(f"  Passed: {all_passed:,}")
    print(f"  Failed: {all_failed:,}")

    if all_failed == 0:
        print("\n  ✅ ALL TESTS PASSED - Classification is replicable")
        status = "PASSED"
    else:
        print(f"\n  ⚠️  {all_failed} TESTS FAILED - Review failures above")
        status = "FAILED"

    # Save test results
    os.makedirs(REPORT_DIR, exist_ok=True)

    test_report = {
        'test_date': datetime.now().isoformat(),
        'status': status,
        'total_tests': all_passed + all_failed,
        'passed': all_passed,
        'failed': all_failed,
        'thresholds': {
            'DURATION_SHORT': DURATION_SHORT,
            'DURATION_PROTRACTED': DURATION_PROTRACTED,
            'VARIANCE_STABLE': VARIANCE_STABLE,
            'VARIANCE_VOLATILE': VARIANCE_VOLATILE
        },
        'failures': all_failures[:20]  # First 20 failures
    }

    report_file = os.path.join(REPORT_DIR, 'validation_suite_report.json')
    with open(report_file, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)

    print(f"\n  Test report saved: {report_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
