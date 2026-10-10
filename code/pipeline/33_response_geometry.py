#!/usr/bin/env python3
"""33_response_geometry.py — common-scale test of peak/event separation.

The primary models use different response families for annual peaks and
percentile-defined heatwave days.  This companion diagnostic puts both on the
same gauge-level slope scale, then compares the tailwater/reference ratios
with a tier-stratified gauge bootstrap.  It is a formal separation test for
the response geometry, not a replacement for the mixed model or NB-GEE.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
PANEL = V4 / "annual_panel.parquet"
OUT_JSON = V4 / "response_geometry.json"
OUT_CSV = V4 / "response_geometry_gauge.csv"

MIN_YEARS = 8
N_BOOT = 5000
SEED = 20261010


def slope(frame, x, y):
    """Return an OLS slope when the gauge has a usable annual series."""
    g = frame.dropna(subset=[x, y])
    if len(g) < MIN_YEARS or g[x].nunique() < 2:
        return np.nan, int(len(g))
    return float(stats.linregress(g[x], g[y]).slope), int(len(g))


panel = pd.read_parquet(PANEL)
panel = panel[(panel.sample_scope == "primary") &
              panel.tier.isin(["reference", "upstream", "tailwater"])].copy()
panel["site8"] = panel.site.astype(str).str.zfill(8)

rows = []
for (site, tier), g in panel.groupby(["site8", "tier"], sort=True):
    peak, n_peak = slope(g, "air_mwmt7", "water_mwmt7")
    event, n_event = slope(g, "air_hw", "water_hw")
    rows.append({"site": site, "tier": tier, "peak_slope": peak,
                 "event_slope": event, "n_peak": n_peak,
                 "n_event": n_event})
G = pd.DataFrame(rows)
G.to_csv(OUT_CSV, index=False)

def tier_mean(metric, tier):
    return float(G.loc[G.tier == tier, metric].dropna().mean())

peak_ref = tier_mean("peak_slope", "reference")
peak_tw = tier_mean("peak_slope", "tailwater")
event_ref = tier_mean("event_slope", "reference")
event_tw = tier_mean("event_slope", "tailwater")
peak_ratio = peak_tw / peak_ref
event_ratio = event_tw / event_ref

rng = np.random.default_rng(SEED)
boot = np.empty((N_BOOT, 3), dtype=float)
ref = G[G.tier == "reference"]
tw = G[G.tier == "tailwater"]
for i in range(N_BOOT):
    rb = ref.sample(len(ref), replace=True,
                    random_state=int(rng.integers(0, 2**32 - 1)))
    tb = tw.sample(len(tw), replace=True,
                   random_state=int(rng.integers(0, 2**32 - 1)))
    pr = tb.peak_slope.mean() / rb.peak_slope.mean()
    er = tb.event_slope.mean() / rb.event_slope.mean()
    boot[i] = (pr, er, er - pr)

separation = float(event_ratio - peak_ratio)
ci = np.percentile(boot[:, 2], [2.5, 97.5])
peak_ci = np.percentile(boot[:, 0], [2.5, 97.5])
event_ci = np.percentile(boot[:, 1], [2.5, 97.5])
result = {
    "estimand": "tailwater/reference ratio difference on common gauge-level slope scale",
    "n_gauges": {
        "reference_peak": int(ref.peak_slope.notna().sum()),
        "tailwater_peak": int(tw.peak_slope.notna().sum()),
        "reference_event": int(ref.event_slope.notna().sum()),
        "tailwater_event": int(tw.event_slope.notna().sum()),
    },
    "means": {
        "peak_reference": round(peak_ref, 6),
        "peak_tailwater": round(peak_tw, 6),
        "event_reference": round(event_ref, 6),
        "event_tailwater": round(event_tw, 6),
    },
    "ratios": {
        "peak_tailwater_reference": round(peak_ratio, 6),
        "peak_tailwater_reference_ci95": [round(float(peak_ci[0]), 6), round(float(peak_ci[1]), 6)],
        "event_tailwater_reference": round(event_ratio, 6),
        "event_tailwater_reference_ci95": [round(float(event_ci[0]), 6), round(float(event_ci[1]), 6)],
        "event_minus_peak": round(separation, 6),
        "event_minus_peak_ci95": [round(float(ci[0]), 6), round(float(ci[1]), 6)],
    },
    "bootstrap": {"n_rep": N_BOOT, "seed": SEED,
                   "unit": "gauge within tier",
                   "p_event_minus_peak_gt_0": round(float(np.mean(boot[:, 2] > 0)), 6)},
    "interpretation": (
        "Peak coupling is attenuated below reservoirs while the common-scale "
        "heatwave-day slope is not attenuated. This diagnostic formalizes the "
        "response-geometry separation; the primary NB-GEE remains the inferential "
        "model for percentile-defined heatwave transmission."
    ),
}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, indent=2, ensure_ascii=False))
