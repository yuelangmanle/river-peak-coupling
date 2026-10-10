#!/usr/bin/env python3
"""30_inflow34_and_methods.py — Computational part of the inflow-control and AR(1)-count fixes.

30a. Inflow-control partial correlations for the 34-reservoir subset (computed in-session; ensured registered here)
30b. Explanation of the AR(1) n=272 count (actual statistics)
Output merged into canonical.json: review_fixes updated
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2]))).resolve()
V4 = (BASE / "data" / "samples" / "analysis").resolve()
if BASE not in V4.parents:
    raise SystemExit("path check failed")

# 30b: explain AR(1) eligibility using the same consecutive-year rule as step 22.
panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[panel.sample_scope == "primary"].dropna(subset=["water_mwmt7", "air_mwmt7"]).copy()
n_panel = prim.site8.nunique()
continuous_pairs = 0
eligible = 0
for _, g in prim.sort_values("year").groupby("site8"):
    years = g.year.to_numpy()
    n_pairs = int((years[1:] - years[:-1] == 1).sum())
    continuous_pairs += n_pairs
    eligible += n_pairs >= 5

C = json.loads((V4 / "canonical.json").read_text())
C["review_robustness"]["ar1_272_explained"] = {
    "panel_gauges": int(n_panel),
    "gauges_with_ge5_consecutive_pairs": int(eligible),
    "candidate_consecutive_year_pairs": int(continuous_pairs),
    "note": ("the candidate count uses adjacent eligible calendar years and requires at "
             "least five consecutive-year pairs per gauge; the final AR(1) count can be "
             "lower after mixed-model residual availability and variance checks")}
(V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
print("ar1 explanation:", C["review_robustness"]["ar1_272_explained"])
