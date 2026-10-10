# Reproduction guide

This guide describes the inputs, execution order, and expected outputs for the river-peak coupling analysis.

## Environment

Use Python 3.9 or later. Install the listed dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The pipeline uses pandas, NumPy, SciPy, statsmodels, Matplotlib, GeoPandas, Shapely, PyProj, PyYAML, DBFRead, requests, and pytest. Some routing and source-data preparation steps may require GDAL-compatible GeoPandas support.

## First-pass validation

Run the tests and the manuscript audit before attempting a full rebuild:

```bash
python3 -m pytest -q tests
python3 code/pipeline/reproduce.py --audit-only
```

The audit reads the canonical result tables in `data/samples/analysis/` and checks the numerical anchors used in the manuscript. It does not require the large daily time series or river-network archives. The response-geometry diagnostic is reproduced from the released annual panel and writes `response_geometry.json` and `response_geometry_gauge.csv`. The flow-aware sensitivity uses the released `discharge_peak_window.csv` table and writes `discharge_sensitivity.json`; it does not download daily NWIS data during an audit run. The storage-state panel uses the released `storage_peak_panel.csv` table and writes `storage_state_panel.json`; raw ResOpsUS series are required only when that compact table is rebuilt. The mechanism bridge is documented in Supplementary Tables S10 and S12 and is not presented as a completed national causal mediation model.

## Full input set

The full pipeline expects the following project-relative inputs:

- `data/cache/daily_water.parquet` and `data/cache/daily_air.parquet`;
- GAGES-II gauge attributes and drainage-area files;
- reservoir attributes and operations time series;
- Shasta case-study inputs;
- NLDI routing tables, when network verification is enabled;
- the North American HydroRIVERS extract for the independent routing check.

The exact filenames and preflight checks are defined in `code/pipeline/reproduce.py`. Place source files at those paths or set `REPRO_BASE` to a project root with the same layout.

## Run order

```bash
python3 code/pipeline/reproduce.py
```

The driver executes unit tests, panel construction, canonical statistics, the Shasta case study, the multi-reservoir analysis, mechanism and sensitivity blocks, routing sensitivity, figure generation, the common-scale response-geometry diagnostic, the flow-aware discharge sensitivity, the storage-state panel, and the final manuscript audit. Steps that need external routing data are skipped unless `--with-routing` is supplied.

Useful variants:

```bash
python3 code/pipeline/reproduce.py --quick
python3 code/pipeline/reproduce.py --with-routing
```

## Outputs

Numerical outputs and intermediate tables are written under `data/samples/analysis/`. Figures are written under `figures/`. The manuscript audit compares the regenerated canonical values with the claims in `paper/manuscript.md`.
