# Reproducibility Package: A Typology of Food Security Crises

**Paper:** A Typology of Food Security Crises — Eight Archetypes from 1,658 Episodes across 51 Countries

**Authors:** Richard Choularton

**Journal:** Nature Food

---

## Abstract

This repository contains the code and data needed to reproduce all analyses and figures in the paper. Starting from the raw Harmonized Food Insecurity Dataset (HFID v1.1.1), the pipeline detects 1,658 food security crisis episodes across 51 countries (2007–2024), classifies them into eight archetypes, computes monthly IPC phase transition matrices with bootstrap confidence intervals, and generates all publication figures.

## System Requirements

- **Python:** 3.9 or higher
- **RAM:** ~4 GB
- **Disk:** ~200 MB (including input data)
- **OS:** Any (tested on macOS 14, Ubuntu 22.04, Windows 11)
- **Dependencies:** pandas, numpy, scipy, matplotlib (see `requirements.txt`)

## Installation

### Option A: pip (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Optional: for Extended Data geographic map (step 14)
pip install -r requirements-maps.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate paper1a
```

## Quick Start

```bash
python run_all.py                   # Run all 14 steps
python run_all.py --paper-only      # Paper figures only (skips supplementary)
```

Expected runtime: **15–25 minutes** on a modern laptop (most time is spent on bootstrap confidence intervals in step 01).

### Run individual steps

```bash
python run_all.py --step 1          # Run only step 1
python run_all.py --analysis-only   # Steps 01-05 (data analysis)
python run_all.py --figures-only    # Steps 06-14 (figures, requires step 01 first)
python run_all.py --paper-only      # Steps 01-08, 12-14 (skip supplementary 09-11)
```

## Pipeline Overview

```
data/HFID_hv1.csv
       │
       ▼
┌──────────────────┐
│ 01  Reference    │──→ episodes.csv, transition matrices, sensitivity analysis
│     Pipeline     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 02  Transitions  │──→ archetype_transitions.csv
└──────┬───────────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│ 03  Gap Analysis │     │ 04  Gap Compress  │
└──────────────────┘     │     Robustness    │
                         └──────────────────┘
┌──────────────────┐
│ 05  HFID         │──→ FEWS/CH consistency analysis
│     Consistency  │
└──────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ Figures 06-14: Publication figures (PNG+PDF)  │
│   06 Figure 1: Archetype scatter             │
│   07 Figure 2: Alluvial transitions          │
│   08 Figure 3: Phase dynamics (6 panels)     │
│   09 Supplementary: Recovery degradation     │
│   10 Supplementary: Temporal trends          │
│   11 Supplementary: Conceptual framework     │
│   12 Extended Data Fig 1: Crisis staircase   │
│   13 Figure 4 + ED: Gap compression          │
│   14 Extended Data Fig 2: Geographic map     │
└──────────────────────────────────────────────┘
```

## Script Descriptions

| Step | Script | Description | Key Outputs |
|------|--------|-------------|-------------|
| 01 | `01_reference_pipeline.py` | Core analysis: HFID → episodes, transition matrices, bootstrap CIs, sensitivity (9 variants), admin2 check | `episodes.csv`, `full_transition_matrix.json`, `sensitivity_summary.csv` |
| 02 | `02_generate_transitions.py` | Episodes → archetype transition sequences between consecutive episodes at each location | `archetype_transitions.csv`, `location_summaries.csv` |
| 03 | `03_gap_analysis.py` | Analysis of inter-episode recovery gaps: duration, escalation risk, rapid cycling | `location_gap_patterns.csv`, `country_gap_patterns.csv` |
| 04 | `04_gap_compression_robustness.py` | Tests whether gap compression is real vs composition bias (consistent locations, within-location trends) | `within_location_gap_trends.csv` |
| 05 | `05_hfid_consistency.py` | FEWS NET vs CH/IPC classification agreement; IPC phase vs food security indicators | `hfid_consistency/` |
| 06 | `06_fig1_archetypes.py` | **Figure 1:** Scatter plot of 8 archetypes (duration × severity) with convex hulls | `Figure1_archetype_scatter.png/.pdf` |
| 07 | `07_fig2_alluvial.py` | **Figure 2:** Annual alluvial diagram showing archetype evolution (2016 cohort) | `Figure2_alluvial.png/.pdf` |
| 08 | `08_fig3_phase_dynamics.py` | **Figure 3:** 6-panel grid showing transition matrix, asymmetry, and duration effects | `Figure3_phase_dynamics.png/.pdf` |
| 09 | `09_fig4_recovery.py` | *Supplementary:* Recovery probability decay curves with exponential fit | `Figure4_recovery_degradation.png/.pdf` |
| 10 | `10_fig5_temporal.py` | *Supplementary:* Temporal trends in Phase 4+ rates and archetype distribution | `Figure5_temporal_trends.png/.pdf` |
| 11 | `11_fig6_framework.py` | *Supplementary:* Conceptual framework schematic (no data dependency) | `Figure6_conceptual_framework.png/.pdf` |
| 12 | `12_extdata_staircase.py` | **Extended Data Fig 1:** Crisis staircase alluvial diagram | `ExtData_crisis_staircase.png/.pdf` |
| 13 | `13_extdata_gap_compression.py` | **Figure 4 + Extended Data:** Gap compression dual-panel and time series | `ExtData_gap_compression.png/.pdf` |
| 14 | `14_extdata_gap_map.py` | **Extended Data Fig 2:** Geographic map of gap patterns (requires geopandas) | `ExtData_gap_compression_map.png/.pdf` |

## Output Manifest

### Key statistics (from step 01)
| Statistic | File | Field |
|-----------|------|-------|
| Total episodes (1,658) | `episodes.csv` | row count |
| 8 archetypes | `episodes.csv` | `archetype` column |
| Recovery ratio (4→3)/(3→4) | `full_transition_matrix.json` | `key_ratios.ratio_4to3_over_3to4` |
| Recovery probability decay (R²=0.93) | `phase3_crossover.json` | `decay_fit.r_squared` |
| 9-variant sensitivity | `sensitivity_summary.csv` | all columns |

### Paper Figure Correspondence

| Paper Figure | Code Step | Output File |
|---|---|---|
| Figure 1: Archetype scatter | Step 06 | `outputs/figures/Figure1_archetype_scatter.png` |
| Figure 2: Alluvial transitions | Step 07 | `outputs/figures/Figure2_alluvial.png` |
| Figure 3: Phase dynamics 6-panel | Step 08 | `outputs/figures/Figure3_phase_dynamics.png` |
| Figure 4: Gap compression dual-panel | Step 13 | `outputs/figures/ExtData_gap_compression.png` |
| Extended Data Fig 1: Crisis staircase | Step 12 | `outputs/figures/ExtData_crisis_staircase.png` |
| Extended Data Fig 2: Gap compression map | Step 14 | `outputs/figures/ExtData_gap_compression_map.png` |

### Supplementary figures (from steps 09–11)

These steps produce additional analysis not included in the final paper but available for reference:

| Output File | Description |
|---|---|
| `outputs/figures/Figure4_recovery_degradation.png` | Recovery probability decay curves |
| `outputs/figures/Figure5_temporal_trends.png` | Temporal trends in Phase 4+ rates |
| `outputs/figures/Figure6_conceptual_framework.png` | Conceptual framework schematic |

## Expected Runtime

| Step | Description | Approx. Time |
|------|-------------|-------------|
| 01 | Reference pipeline (with bootstrap) | 10–15 min |
| 02 | Transition sequences | <10 sec |
| 03 | Gap analysis | <10 sec |
| 04 | Gap compression robustness | <10 sec |
| 05 | HFID consistency | <30 sec |
| 06–14 | All figures | 1–2 min total |
| **Total** | | **15–25 min** |

## Data Sources

- **HFID v1.1.1:** Machefer et al. (2025). Harmonized Food Insecurity Dataset. CC-BY-4.0. https://doi.org/10.5281/zenodo.14593822
- See `data/README.md` for full provenance and integrity verification (SHA-256 hash).

## Citation

If you use this code, please cite both the paper and this repository:

```bibtex
@article{choularton2026typology,
  title={A Typology of Food Security Crises: Eight Archetypes from 1,658 Episodes across 51 Countries},
  author={Choularton, Richard},
  journal={Nature Food},
  year={2026}
}

@software{choularton2026typology_code,
  title={Reproducibility Package: A Typology of Food Security Crises},
  author={Choularton, Richard},
  year={2026},
  url={https://github.com/rchoularton/paper1a-food-crisis-typology}
}
```

## License

MIT License. See `LICENSE` for details.
