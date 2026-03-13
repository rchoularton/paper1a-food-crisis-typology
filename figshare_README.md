# Dataset: Processed data and figure source data for Reimagining Food Crises

**Paper:** Reimagining Food Crises: Archetypes, Collapsing Recovery Gaps, Crisis Staircases, and Recovery Traps

**Authors:** Richard Choularton (ORCID: 0000-0003-3275-2883), Krishna Krishnamurthy, Jean Martin Bauer, Per Becker

**License:** CC-BY-4.0

---

## Description

This dataset contains the processed analytical data and per-figure source data generated from the Harmonized Food Insecurity Dataset (HFID v1.1.1) using the analysis code deposited on GitHub/Zenodo. All files are produced deterministically by the reproducibility pipeline.

## Provenance

- **Source data:** HFID v1.1.1 (Machefer et al. 2025), DOI: 10.5281/zenodo.15017473
- **Analysis code:** https://github.com/rchoularton/crisis-archetype-analysis
- **Pipeline:** 12-step Python pipeline; see code repository README for full details
- **Reference configuration:** FEWS priority, MAX aggregation, 12-month interpolation, left-censored included

## File Inventory

### Analytical Data Files (from `outputs/data/`)

| File | Description | Rows (approx.) |
|------|-------------|-----------------|
| `episodes.csv` | All 1,658 classified crisis episodes with archetype, duration, peak phase, dates | 1,658 |
| `archetype_transitions.csv` | Sequential transitions between episodes at each location with gap duration | ~1,200 |
| `location_summaries.csv` | Per-location summary: episode count, archetype sequence, unique archetypes | ~600 |
| `full_transition_matrix.json` | Monthly IPC phase transition probabilities (5x5) with bootstrap CIs | - |
| `phase{1-5}_duration_conditioned.json` | Duration-conditioned transition probabilities per phase | - |
| `phase3_crossover.json` | Phase 3 recovery-escalation crossover analysis with decay fit | - |
| `sensitivity_summary.csv` | 9-variant sensitivity analysis results | 9 |
| `location_gap_patterns.csv` | Per-location gap duration statistics | ~400 |
| `country_gap_patterns.csv` | Per-country gap aggregates | ~45 |
| `within_location_gap_trends.csv` | Within-location gap compression trends | ~200 |
| `hfid_consistency/` | FEWS vs CH/IPC source agreement analysis | - |

### Figure Source Data Files (from `outputs/figures/`)

| File | Figure | Contents |
|------|--------|----------|
| `SourceData_Fig1.xlsx` | Figure 1: Archetype scatter | Episode-level data (archetype, duration, peak phase) and summary statistics |
| `SourceData_Fig2.xlsx` | Figure 2: Alluvial transitions | Archetype transition matrix and annual cohort counts (2011-2023) |
| `SourceData_Fig3.xlsx` | Figure 3: Phase dynamics | 5x5 transition probability matrix and duration-conditioned probabilities with CIs |
| `SourceData_Fig4.xlsx` | Figure 4: Gap compression | Yearly gap statistics (mean, median, IQR) and escalation risk by gap bin |
| `SourceData_EDFig1.xlsx` | ED Figure 1: Crisis staircase | Archetype-to-archetype transition counts |
| `SourceData_EDFig2.xlsx` | ED Figure 2: Gap map | Location-level gap data with trajectory classification |

## Key Variables

### episodes.csv

| Column | Description |
|--------|-------------|
| `location` | Admin1 location identifier (ISO3_ADMIN1) |
| `iso3` | ISO 3166-1 alpha-3 country code |
| `archetype` | One of 8 crisis archetypes (seasonal_crisis, prolonged_moderate, etc.) |
| `duration_months` | Episode duration in months |
| `peak_phase` | Maximum IPC phase reached during episode (3-5) |
| `dates` | Comma-separated list of year-month values in episode |
| `mean_phase` | Mean IPC phase during episode |

### archetype_transitions.csv

| Column | Description |
|--------|-------------|
| `location` | Admin1 location identifier |
| `from_archetype` | Archetype of episode N |
| `to_archetype` | Archetype of episode N+1 |
| `gap_months` | Months between end of episode N and start of N+1 |
| `severity_change` | Change in peak phase (positive = escalation) |
| `same_archetype` | Boolean: same archetype repeated |

## Reproducibility

All files can be regenerated from the HFID source data using:

```bash
git clone https://github.com/rchoularton/crisis-archetype-analysis
cd crisis-archetype-analysis
pip install -r requirements.txt
# Place HFID_hv1.csv in data/
python run_all.py
```

Expected runtime: 15-25 minutes on a modern laptop.
