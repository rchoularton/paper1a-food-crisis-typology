#!/usr/bin/env python3
# @status:   import
# @process:  shared
# @paper:    paper1
"""
reference_transition_analysis.py — import shim for the revision robustness scripts.
==================================================================================

The capsule's canonical pipeline lives in ``01_reference_pipeline.py``. Because a
module name cannot begin with a digit, that file cannot be imported by name. The
revision robustness/spatial scripts (``r1_*.py``) were written to import the
pipeline functions from a module called ``reference_transition_analysis`` — the
name the pipeline had in the working repository before it was renamed for the
capsule.

This shim loads ``01_reference_pipeline.py`` by file path and re-exports the
functions those scripts need, so the jackknife and spatial analyses run against
the *exact* same, frozen pipeline that produces the primary results (FEWS
priority, MAX aggregation, 12-month interpolation). Keeping one pipeline module
avoids any risk of the robustness code drifting from the primary analysis.
"""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_PATH = os.path.join(_HERE, "01_reference_pipeline.py")

_spec = importlib.util.spec_from_file_location("_reference_pipeline_impl", _PIPELINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # __name__ != "__main__" → pipeline main() does not run

# Re-export the functions the r1_* scripts import.
load_hfid = _mod.load_hfid
preprocess = _mod.preprocess
preprocess_admin2 = _mod.preprocess_admin2
interpolate = _mod.interpolate
compute_transitions = _mod.compute_transitions
compute_key_ratios = _mod.compute_key_ratios
