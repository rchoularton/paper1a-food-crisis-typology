# Reproducibility Package: A Typology of Food Security Crises

**Paper:** Reimagining Food Crises: Archetypes, Collapsing Recovery Gaps, Crisis Staircases, and Recovery Traps

**Authors:** Richard Choularton, Krishna Krishnamurthy, Jean Martin Bauer, Per Becker

**Journal:** Nature Food

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18954539.svg)](https://doi.org/10.5281/zenodo.18954539)

---

## Abstract

This repository contains the code and data needed to reproduce all analyses and figures in the paper. Starting from the raw Harmonized Food Insecurity Dataset (HFID v1.1.1), the pipeline detects 1,658 food security crisis episodes across 49 countries (2007-2024), classifies them into eight archetypes, computes monthly IPC phase transition matrices with bootstrap confidence intervals, and generates all publication figures.

## System Requirements

- **Python:** 3.9 or higher
- **RAM:** ~4 GB
- **Disk:** ~200 MB (including input data)
- **OS:** Any (tested on macOS 14, Ubuntu 22.04, Windows 11)
- **Dependencies:** pandas, numpy, scipy, matplotlib, openpyxl, geopandas (see `requirements.txt`)

## Installation

### Option A: pip (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate paper1a
```

### Option C: Code Ocean

This code is also available as an executable [Code Ocean compute capsule](https://codeocean.com/) (DOI will be assigned upon publication), which runs in the browser with no local installation required.

## Quick Start

```bash
python run_all.py                   # Run all 17 steps
```

Expected runtime: **20-30 minutes** on a modern laptop (most time is spent on bootstrap confidence intervals in step 01).

### Run individual steps

```bash
python run_all.py --step 1          # Run only step 1
python run_all.py --analysis-only   # Steps 01-10 (data analysis)
python run_all.py --figures-only    # Steps 11-17 (figures, requires analysis steps first)
```

## Pipeline Overview

```
data/HFID_hv1.csv
       |
       v
+------------------+
| 01  Reference    |-->  episodes.csv, transition matrices, sensitivity analysis
|     Pipeline     |
+------+-----------+
       |
       v
+------------------+
| 02  Transitions  |-->  archetype_transitions.csv
+------+-----------+
       |
       v
+------------------+     +------------------+
| 03  Gap Analysis |     | 04  Gap Compress  |
+------------------+     |     Robustness    |
                         +------------------+
+------------------+
| 05  HFID         |-->  FEWS/CH consistency analysis
|     Consistency  |
+------------------+
       |
       v
+----------------------------------------------+
| Revision robustness / spatial (steps 07-10)   |
|   07 R1.2 jackknife (Extended Data Table 6)   |
|   08 R1.4 national footprint (ED Fig 3a)      |
|   09 R1.4 neighbour co-escalation (ED Fig 3b) |
|   10 R3.7/R3.9 source + unit composition      |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
| Figures 11-17: Publication figures (PNG+PDF)  |
|   11 Figure 1: Archetype scatter             |
|   12 Figure 2: Alluvial transitions          |
|   13 Figure 3: Phase dynamics (6 panels)     |
|   14 Extended Data Fig 1: Crisis staircase   |
|   15 Figure 4: Gap compression               |
|   16 Extended Data Fig 2: Geographic map     |
|   17 Extended Data Fig 3: Spatial dynamics   |
+----------------------------------------------+
```

## Script Descriptions

| Step | Script | Description | Key Outputs |
|------|--------|-------------|-------------|
| 01 | `01_reference_pipeline.py --phase core` | Core pipeline: HFID → episodes, transition matrices, bootstrap CIs, admin2 check | `episodes.csv`, `full_transition_matrix.json` |
| 02 | `01_reference_pipeline.py --phase robustness` | Robustness: sensitivity analysis (9 variants) + verification | `sensitivity_summary.csv` |
| 03 | `02_generate_transitions.py` | Episodes → archetype transition sequences between consecutive episodes at each location | `archetype_transitions.csv`, `location_summaries.csv` |
| 04 | `03_gap_analysis.py` | Analysis of inter-episode recovery gaps: duration, escalation risk, rapid cycling | `location_gap_patterns.csv`, `country_gap_patterns.csv` |
| 05 | `04_gap_compression_robustness.py` | Tests whether gap compression is real vs composition bias (consistent locations, within-location trends) | `within_location_gap_trends.csv` |
| 06 | `05_hfid_consistency.py` | FEWS NET vs CH/IPC classification agreement; IPC phase vs food security indicators | `hfid_consistency/` |
| 07 | `r1_jackknife_robustness.py` | **R1.2:** leave-one-country-out jackknife of headline ratios + archetype-merge check (**Extended Data Table 6**) | `r1_jackknife_robustness.json`, `r1_jackknife_by_country.csv` |
| 08 | `r1_spatial_extent.py` | **R1.4:** national crisis footprint (admin1 areas at Phase 3+) by archetype over the episode life cycle (**ED Fig 3a**) | `r1_spatial_extent_by_archetype.json`, `r1_spatial_extent_by_episode.csv` |
| 09 | `r1_spatial_adjacency.py` | **R1.4:** neighbour co-escalation from within-country admin1 adjacency built off the HFID geometry (**ED Fig 3b**; requires geopandas) | `r1_admin1_adjacency.csv`, `r1_spatial_adjacency_by_archetype.json`, `r1_spatial_adjacency_by_episode.csv` |
| 10 | `r1_source_unit_composition.py` | **R3.7/R3.9:** FEWS-vs-IPC/CH source mix, carried-forward share, and admin1/admin2 unit composition | `r1_source_unit_composition.json` |
| 11 | `06_fig1_archetypes.py` | **Figure 1:** Scatter plot of 8 archetypes (duration × severity) with zone-band design | `Figure1_archetype_scatter.png/.pdf`, `SourceData_Fig1.xlsx` |
| 12 | `07_fig2_alluvial.py` | **Figure 2:** Annual alluvial diagram showing archetype evolution (2016 cohort) | `Figure2_combined_alluvial.png/.pdf`, `SourceData_Fig2.xlsx` |
| 13 | `08_fig3_phase_dynamics.py` | **Figure 3:** 6-panel grid showing transition matrix, asymmetry, and duration effects | `Figure3_recovery_asymmetry.png/.pdf`, `SourceData_Fig3.xlsx` |
| 14 | `12_extdata_staircase.py` | **Extended Data Fig 1:** Crisis staircase diagram | `Figure_crisis_staircase.png/.pdf`, `SourceData_EDFig1.xlsx` |
| 15 | `13_extdata_gap_compression.py` | **Figure 4:** Gap compression dual-panel and escalation risk | `Figure4_gap_compression.png/.pdf`, `SourceData_Fig4.xlsx` |
| 16 | `14_extdata_gap_map.py` | **Extended Data Fig 2:** Geographic map of gap patterns (requires geopandas) | `ExtDataFig_gap_compression_map.png/.pdf`, `SourceData_EDFig2.xlsx` |
| 17 | `15_extdata_spatial.py` | **Extended Data Fig 3:** Spatial dynamics by archetype — footprint (a) + neighbour co-escalation (b) | `ExtDataFig_spatial_dynamics.png/.pdf` |

## Output Manifest

### Key statistics (from step 01)
| Statistic | File | Field |
|-----------|------|-------|
| Total episodes (1,658) | `episodes.csv` | row count |
| 8 archetypes | `episodes.csv` | `archetype` column |
| Recovery ratio (4->3)/(3->4) | `full_transition_matrix.json` | `key_ratios.ratio_4to3_over_3to4` |
| Recovery probability decay (R²=0.96) | `phase3_crossover.json` | `decay_fit.r_squared` |
| 9-variant sensitivity | `sensitivity_summary.csv` | all columns |

### Paper Figure Correspondence

| Paper Figure | Code Step | Output File |
|---|---|---|
| Figure 1: Archetype scatter | Step 11 | `outputs/figures/Figure1_archetype_scatter.png` |
| Figure 2: Alluvial transitions | Step 12 | `outputs/figures/Figure2_combined_alluvial.png` |
| Figure 3: Phase dynamics 6-panel | Step 13 | `outputs/figures/Figure3_recovery_asymmetry.png` |
| Figure 4: Gap compression dual-panel | Step 15 | `outputs/figures/Figure4_gap_compression.png` |
| Extended Data Fig 1: Crisis staircase | Step 14 | `outputs/figures/Figure_crisis_staircase.png` |
| Extended Data Fig 2: Gap map | Step 16 | `outputs/figures/ExtDataFig_gap_compression_map.png` |
| Extended Data Fig 3: Spatial dynamics | Step 17 | `outputs/figures/ExtDataFig_spatial_dynamics.png` |
| Extended Data Table 6: Robustness (jackknife) | Step 07 | `outputs/data/r1_jackknife_robustness.json` |

### Source Data Files (Nature Food requirement)

Each figure script also generates an Excel file with the underlying data:

| File | Contents |
|---|---|
| `SourceData_Fig1.xlsx` | Episode-level archetype, duration, peak severity |
| `SourceData_Fig2.xlsx` | Transition matrix and annual cohort counts |
| `SourceData_Fig3.xlsx` | Phase transition probabilities and duration-conditioned data |
| `SourceData_Fig4.xlsx` | Yearly gap statistics and escalation risk by gap bin |
| `SourceData_EDFig1.xlsx` | Archetype-to-archetype transition counts |
| `SourceData_EDFig2.xlsx` | Location-level gap data with trajectory classification |

## Expected Runtime

| Step | Description | Approx. Time |
|------|-------------|-------------|
| 01-02 | Reference pipeline (core + robustness) | 10-15 min |
| 03 | Transition sequences | <10 sec |
| 04 | Gap analysis | <10 sec |
| 05 | Gap compression robustness | <10 sec |
| 06 | HFID consistency | <30 sec |
| 07 | Jackknife robustness (re-runs pipeline per country) | 3-6 min |
| 08-10 | Spatial extent, adjacency, source/unit composition | 1-3 min |
| 11-17 | All figures | 1-2 min total |
| **Total** | | **20-30 min** |

## Data Sources

- **HFID v1.1.1:** Machefer et al. (2025). Harmonized Food Insecurity Dataset. CC-BY-4.0. https://doi.org/10.5281/zenodo.15017473
- See `data/README.md` for full provenance and integrity verification (SHA-256 hash).

## Code Ocean

This code is also available as a [Code Ocean compute capsule](https://codeocean.com/) (DOI will be assigned upon publication) that can be run in the browser without any local installation. The capsule includes all code, data, and a pre-configured Python environment.

To reproduce results on Code Ocean:
1. Navigate to the capsule page
2. Click **Reproducible Run**
3. All outputs appear in the **Results** tab

## Citation

If you use this code, please cite both the paper and this repository:

```bibtex
@article{choularton2026typology,
  title={Reimagining Food Crises: Archetypes, Collapsing Recovery Gaps, Crisis Staircases, and Recovery Traps},
  author={Choularton, Richard and Krishnamurthy, Krishna and Bauer, Jean Martin and Becker, Per},
  journal={Nature Food},
  year={2026}
}

@software{choularton2026typology_code,
  title={paper1a-food-crisis-typology: Reproducibility package for Reimagining Food Crises},
  author={Choularton, Richard and Krishnamurthy, Krishna and Bauer, Jean Martin and Becker, Per},
  year={2026},
  url={https://github.com/rchoularton/paper1a-food-crisis-typology},
  doi={10.5281/zenodo.18954539}
}
```

## License

MIT License. See `LICENSE` for details.
