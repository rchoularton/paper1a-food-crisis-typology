#!/usr/bin/env python3
# @status:   maturing
# @process:  P6-audit-v24
# @paper:    paper1
"""
19_config_parity.py — Guard against constant drift between config.py and the pipeline
=====================================================================================

The Paper 1 pipeline is deliberately self-contained: `01_reference_pipeline.py`
hardcodes its constants rather than importing the project's
`scripts/analysis/config.py`. Nothing previously enforced that the two stayed
in agreement — which is exactly how the working-repo and capsule copies of the
pipeline drifted apart before the 2026-07-19 consolidation (v24 audit, finding
H2: a silent MEAN-aggregation fallthrough).

This step fails loudly if any shared constant diverges:

- Direct comparison: CRISIS_THRESHOLD, N_BOOTSTRAP, BOOTSTRAP_SEED,
  MAX_INTERPOLATION_GAP (== pipeline DEFAULT_INTERPOLATION_GAP).
- Behavioral probes: DURATION_SHORT, DURATION_PROTRACTED, VARIANCE_STABLE,
  TRANSITIONS_STABLE, TRANSITIONS_VOLATILE are literals inside
  `_classify_archetype`, so we classify synthetic episodes at boundaries
  DERIVED FROM the config values and assert the archetype flips exactly there.
  If either side moves, the probe fails.
- Source scan: VARIANCE_VOLATILE — in the primary classifier the `var > 0.3`
  disjunct is structurally redundant (its rule also requires
  `total_trans >= 3`), so no behavioral probe can reach it; we assert the
  literal still matches config in both classifier sources.
- Signature check: `_classify_archetype_with_thresholds` keyword defaults
  (duration_short, duration_long, variance_stable) must equal config.

config.py is read by AST (never imported/executed). When config.py is absent —
the self-contained Code Ocean deposit — there is nothing to compare and the
step reports that explicitly and passes.
"""

import ast
import importlib.util
import inspect
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(_HERE)
CONFIG_PATH = os.path.join(PACKAGE_ROOT, '..', '..', '..',
                           'scripts', 'analysis', 'config.py')

CONFIG_CONSTANTS = [
    'CRISIS_THRESHOLD', 'DURATION_SHORT', 'DURATION_PROTRACTED',
    'VARIANCE_STABLE', 'VARIANCE_VOLATILE', 'TRANSITIONS_STABLE',
    'TRANSITIONS_VOLATILE', 'N_BOOTSTRAP', 'BOOTSTRAP_SEED',
    'MAX_INTERPOLATION_GAP',
]


def read_config_constants(path):
    """Extract top-level numeric constant assignments from config.py via AST."""
    tree = ast.parse(open(path).read())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in CONFIG_CONSTANTS:
                try:
                    values[tgt.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        '_reference_pipeline_impl', os.path.join(_HERE, '01_reference_pipeline.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_row(dur, peak, var, phases):
    return pd.Series({'duration_months': dur, 'peak_phase': peak,
                      'phase_variance': var, 'phases': phases})


def main():
    print('=' * 70)
    print('  CONFIG PARITY CHECK (consolidation guard)')
    print('=' * 70)

    cfg_path = os.path.abspath(CONFIG_PATH)
    if not os.path.exists(cfg_path):
        print('  scripts/analysis/config.py not present — self-contained deposit,')
        print('  nothing to compare. PASS (standalone mode).')
        return 0

    cfg = read_config_constants(cfg_path)
    missing = [c for c in CONFIG_CONSTANTS if c not in cfg]
    if missing:
        print(f'  FAIL: constants not found in config.py: {missing}')
        return 1

    pipe = load_pipeline()
    classify = pipe._classify_archetype
    failures = []

    # --- Direct comparisons -------------------------------------------------
    direct = [
        ('CRISIS_THRESHOLD', pipe.CRISIS_THRESHOLD),
        ('N_BOOTSTRAP', pipe.N_BOOTSTRAP),
        ('BOOTSTRAP_SEED', pipe.BOOTSTRAP_SEED),
        ('MAX_INTERPOLATION_GAP', pipe.DEFAULT_INTERPOLATION_GAP),
    ]
    for name, pipeline_value in direct:
        ok = cfg[name] == pipeline_value
        print(f'  {name}: config={cfg[name]} pipeline={pipeline_value} '
              f'{"OK" if ok else "MISMATCH"}')
        if not ok:
            failures.append(name)

    # --- Behavioral probes (boundaries derived from config) -----------------
    DS, DP = cfg['DURATION_SHORT'], cfg['DURATION_PROTRACTED']
    VS = cfg['VARIANCE_STABLE']
    TS, TV = cfg['TRANSITIONS_STABLE'], cfg['TRANSITIONS_VOLATILE']

    probes = [
        # DURATION_SHORT: steady moderate flips seasonal -> prolonged at DS
        ('DURATION_SHORT',
         classify(make_row(DS - 1, 3, 0.0, [3] * (DS - 1))) == 'seasonal_crisis'
         and classify(make_row(DS, 3, 0.0, [3] * DS)) == 'prolonged_moderate'),
        # DURATION_PROTRACTED: steady moderate flips prolonged -> entrenched past DP
        ('DURATION_PROTRACTED',
         classify(make_row(DP, 3, 0.0, [3] * DP)) == 'prolonged_moderate'
         and classify(make_row(DP + 1, 3, 0.0, [3] * (DP + 1))) == 'entrenched_moderate'),
        # VARIANCE_STABLE: short severe flips severe_shock -> rapid_onset at VS
        ('VARIANCE_STABLE',
         classify(make_row(DS - 1, 4, VS - 0.01, [4] * (DS - 1))) == 'severe_shock'
         and classify(make_row(DS - 1, 4, VS, [4] * (DS - 1))) == 'rapid_onset'),
        # TRANSITIONS_STABLE: severe_shock tolerates TS transitions, not TS+1
        ('TRANSITIONS_STABLE',
         classify(make_row(3, 4, 0.05, [4, 4, 3])) == 'severe_shock'      # 1 transition
         and classify(make_row(3, 4, 0.05, [4, 3, 4])) != 'severe_shock'  # 2 transitions
         if TS == 1 else None),
        # TRANSITIONS_VOLATILE: oscillating requires >= TV transitions
        ('TRANSITIONS_VOLATILE',
         classify(make_row(4, 4, 0.35, [3, 4, 3, 4])) == 'oscillating'    # 3 transitions
         and classify(make_row(3, 4, 0.35, [3, 4, 3])) != 'oscillating'   # 2 transitions
         if TV == 3 else None),
    ]
    for name, ok in probes:
        if ok is None:
            print(f'  {name}: config value {cfg[name]} differs from the probe '
                  f'design point — treat as MISMATCH')
            failures.append(name)
            continue
        print(f'  {name}: boundary probe {"OK" if ok else "MISMATCH"}')
        if not ok:
            failures.append(name)

    # --- VARIANCE_VOLATILE: source scan (redundant disjunct in primary rule) --
    sources = (inspect.getsource(pipe._classify_archetype)
               + inspect.getsource(pipe._classify_archetype_with_thresholds))
    literal_hits = set(re.findall(r'var > ([0-9.]+)', sources))
    vv_ok = str(cfg['VARIANCE_VOLATILE']) in literal_hits
    print(f"  VARIANCE_VOLATILE: config={cfg['VARIANCE_VOLATILE']} "
          f"source literals={sorted(literal_hits)} {'OK' if vv_ok else 'MISMATCH'}")
    if not vv_ok:
        failures.append('VARIANCE_VOLATILE')

    # --- Threshold-variant keyword defaults ----------------------------------
    sig = inspect.signature(pipe._classify_archetype_with_thresholds)
    kw = {k: v.default for k, v in sig.parameters.items()
          if v.default is not inspect.Parameter.empty}
    for kwname, cfgname in [('duration_short', 'DURATION_SHORT'),
                            ('duration_long', 'DURATION_PROTRACTED'),
                            ('variance_stable', 'VARIANCE_STABLE')]:
        ok = kw.get(kwname) == cfg[cfgname]
        print(f'  _classify_archetype_with_thresholds {kwname}={kw.get(kwname)} '
              f'vs {cfgname}={cfg[cfgname]} {"OK" if ok else "MISMATCH"}')
        if not ok:
            failures.append(f'kwarg:{kwname}')

    if failures:
        print(f'\n  FAIL — constant drift detected: {failures}')
        return 1
    print('\n  All constants in parity. PASS.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
