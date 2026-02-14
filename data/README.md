# Data Sources

## HFID_hv1.csv — Harmonized Food Insecurity Dataset v1.1.1

**Citation:**
Machefer, M., Charpentier, A., Chotard, S., & Machefer, M. (2025).
Harmonized Food Insecurity Dataset (HFID) v1.1.1.
https://doi.org/10.5281/zenodo.14593822

**Description:**
Monthly subnational food security classifications from two complementary systems:
- FEWS NET (Famine Early Warning Systems Network)
- IPC/CH (Integrated Food Security Phase Classification / Cadre Harmonise)

**Key fields used in this analysis:**
| Field | Description |
|-------|-------------|
| `iso3` | ISO 3166-1 alpha-3 country code |
| `ADMIN1` | First administrative level name |
| `ADMIN2` | Second administrative level name |
| `region` | Geographic region |
| `year_month` | Observation month (YYYY-MM) |
| `ipc_phase_fews` | IPC phase from FEWS NET (1-5, 6=areas of concern) |
| `ipc_phase_ipcch` | IPC phase from IPC/CH analysis (1-5, 6=areas of concern) |
| `fcs_lit` | Food Consumption Score — proportion with insufficient consumption (0-1) |
| `rcsi_lit` | Reduced Coping Strategy Index — proportion in crisis coping (0-1) |

**Coverage:** 80 countries, June 2007 – May 2024
**Records:** 311,838 location-months
**File size:** 37 MB
**SHA-256:** `0145a721827a34a354322d828dff0f9525b3d2d22a72b19c4285820e417a0084`

**License:** CC-BY-4.0

**Original download:** https://zenodo.org/records/14593822

---

## admin1_centroids.csv — Admin1 Region Centroids

Pre-computed centroids (latitude, longitude) for admin1 regions in the HFID dataset.
Used by the optional geographic map figure (step 14).

Derived from HFID spatial geometries. Contains columns:
- `location` — Location identifier (ISO3_ADMIN1)
- `iso3` — Country code
- `admin1` — Admin1 name
- `latitude` — Centroid latitude
- `longitude` — Centroid longitude
