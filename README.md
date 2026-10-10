# Reservoir regulation rotates river thermal response geometry across a contiguous-U.S. benchmark

Analysis archive for a contiguous-U.S. benchmark of how river thermal peaks track atmospheric heat across three hydrologic settings: minimally disturbed reference streams, gauges above reservoirs, and tailwater gauges downstream of reservoirs.

Repository: https://github.com/yuelangmanle/river-peak-coupling

The analysis treats each reach as a three-coordinate response geometry: annual peak coupling, transmission of percentile-defined heatwave days, and observed warm burden. The repository contains the analysis pipeline, tests, figures, manuscript files, and the derived tables needed to inspect the reported results. A common-scale gauge-level diagnostic (`33_response_geometry.py`) tests whether peak sensitivity and event transmission separate on one relative scale. A flow-aware sensitivity (`34_discharge_sensitivity.py`) adds peak-window discharge for 2,841 gauge-years from 202 sites and retains the negative tailwater contrast. A storage-state panel (`35_storage_state_panel.py`) links 310 tailwater years from 18 non-Shasta reservoirs to antecedent storage; the estimated peak shift is −0.334 °C per within-reservoir SD. The manuscript frames storage memory, withdrawal configuration, inflow and discharge as a mechanism hierarchy for the next observation stage.

## Repository layout

- `code/pipeline/`: analysis scripts and the single-entry reproduction driver.
- `data/samples/`: compact sample and derived tables used by the audit and figure steps.
- `data/raw/us_states.json`: small geographic lookup used by the plotting pipeline.
- `figures/`: vector and raster versions of the manuscript figures.
- `paper/`: manuscript, supplementary material, JoH cover letter and highlights, CRediT author statement, formal competing-interests declaration, and the submission DOCX.
- `tests/`: unit tests for site classification and heatwave rules.
- `REPRODUCTION.md`: input requirements and the recommended run order.
- `paper/supplementary_material.md`: audit tables, the common-scale diagnostic (S9), mechanism bridge (S10), flow-aware sensitivity (S11), and storage-state panel (S12).
- `RELEASE_MANIFEST.sha256`: SHA256 checksums for the tracked release files.

## Journal-preparation files

The `paper/` directory includes editable upload materials prepared for Journal of Hydrology: `Cover_Letter_JoH.docx`, `Highlights.docx`, `CRediT_author_statement.docx`, and the clean single-statement `Declaration_of_Competing_Interests.docx`. Draft declaration files are kept outside the submission directory.

## Quick checks

From the repository root:

```bash
python3 -m pytest -q tests
python3 code/pipeline/reproduce.py --audit-only
```

The audit-only mode checks the archived canonical results against the manuscript without downloading the large source datasets. It is the recommended first check for a fresh checkout.

## Full analysis

The full pipeline requires Python 3.9 or later and the packages listed in `requirements.txt`. It also requires the source data described in `REPRODUCTION.md`, including daily water-temperature records, Daymet air-temperature inputs, GAGES-II attributes, reservoir operations, and river-network files. These source files are not redistributed here because of size and source-specific access conditions.

```bash
python3 code/pipeline/reproduce.py
```

To run the optional network-routing verification, add `--with-routing`. Use `--quick` to skip figure regeneration while checking numerical outputs.

## Data and code notes

The released tables under `data/samples/` are compact analysis inputs or derived outputs, not a replacement for the complete source archives. File names and table schemas are kept consistent with the scripts so that each reported number can be traced to a script and an intermediate table.

The primary analysis uses a fixed three-tier design and a gauge/year eligibility rule described in `paper/manuscript.md` and `paper/supplementary_material.md`. The canonical result files under `data/samples/analysis/` are retained for auditability. The response-geometry, flow-aware and storage-state outputs (`response_geometry.json`, `response_geometry_gauge.csv`, `discharge_peak_window.csv`, `discharge_sensitivity.json`, `storage_peak_panel.csv`, and `storage_state_panel.json`) are standalone derived diagnostics included in the reproduction chain after the core panel steps.

## Citation

Please cite the associated article and this repository when reusing the code or derived tables. Author metadata are provided in `CITATION.cff`.
