#!/usr/bin/env python3
"""Flow-aware sensitivity for the annual peak-coupling contrast.

The primary benchmark does not require a complete discharge record at every
temperature gauge.  This script uses the released USGS 00060 daily-discharge
window table to test whether the tier contrast persists when log10 discharge
over the seven-day water-temperature peak window is added to the same mixed
model used for the primary beta comparison.

The released table is an auditable aggregation of public NWIS daily values;
the script does not silently download new data during an audit run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/samples/analysis/annual_panel.parquet"
FLOW = ROOT / "data/samples/analysis/discharge_peak_window.csv"
OUT = ROOT / "data/samples/analysis/discharge_sensitivity.json"


def linear_combo(result, weights: dict[str, float]) -> tuple[float, float, float, float]:
    """Return estimate, standard error and 95% Wald interval."""
    names = list(result.fe_params.index)
    w = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    cov = result.cov_params().loc[names, names].to_numpy()
    estimate = float(w @ result.fe_params.to_numpy())
    se = float(np.sqrt(max(0.0, w @ cov @ w)))
    return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se


def fit_model(df: pd.DataFrame, with_flow: bool):
    formula = "water_mwmt7 ~ air_c10*C(tier)"
    if with_flow:
        formula += " + log_q_c*C(tier)"
    model = smf.mixedlm(formula, df, groups=df["site"], re_formula="~air_c10")
    return model.fit(method="lbfgs", reml=False, maxiter=2000, disp=False), formula


def summarize(result, formula: str, df: pd.DataFrame, with_flow: bool) -> dict:
    # air_c10 is air temperature centered and divided by ten, so slopes below
    # are converted back to degrees C water per degree C air.
    ref_w = {"air_c10": 0.1}
    tw_w = {"air_c10": 0.1, "air_c10:C(tier)[T.tailwater]": 0.1}
    ref = linear_combo(result, ref_w)
    tw = linear_combo(result, tw_w)
    diff_w = {k: tw_w.get(k, 0.0) - ref_w.get(k, 0.0) for k in set(ref_w) | set(tw_w)}
    diff = linear_combo(result, diff_w)
    return {
        "formula": formula,
        "with_flow": with_flow,
        "n_rows": int(len(df)),
        "n_sites": int(df["site"].nunique()),
        "n_sites_by_tier": {k: int(v) for k, v in df.groupby("tier")["site"].nunique().items()},
        "reference_slope_degC_per_degC": round(ref[0], 6),
        "reference_slope_ci95": [round(ref[2], 6), round(ref[3], 6)],
        "tailwater_slope_degC_per_degC": round(tw[0], 6),
        "tailwater_slope_ci95": [round(tw[2], 6), round(tw[3], 6)],
        "tailwater_minus_reference_degC_per_degC": round(diff[0], 6),
        "tailwater_minus_reference_ci95": [round(diff[2], 6), round(diff[3], 6)],
        "tailwater_minus_reference_p": round(float(result.pvalues.get("air_c10:C(tier)[T.tailwater]", np.nan)), 8),
        "converged": bool(result.converged),
    }


def main() -> None:
    panel = pd.read_parquet(PANEL)
    flow = pd.read_csv(FLOW, dtype={"site": str}, parse_dates=["peak_date"])
    panel["site"] = panel["site"].astype(str)
    flow["site"] = flow["site"].astype(str)
    df = panel.merge(flow[["site", "year", "q_peak7_cfs", "q_valid_days"]], on=["site", "year"], how="inner")
    df = df[(df["sample_scope"] == "primary") & df["water_mwmt7"].notna() & df["air_mwmt7"].notna()]
    df = df[df["q_valid_days"] >= 5].copy()
    if df.empty:
        raise RuntimeError("No primary gauge-years have complete discharge windows")
    df["air_c10"] = (df["air_mwmt7"] - df["air_mwmt7"].mean()) / 10.0
    df["log_q_c"] = np.log10(df["q_peak7_cfs"].clip(lower=1e-6))
    primary, primary_formula = fit_model(df, with_flow=False)
    flow_model, flow_formula = fit_model(df, with_flow=True)
    result = {
        "source": {
            "provider": "USGS NWIS daily values",
            "parameter": "00060",
            "statistic": "00003 (mean)",
            "window": "trailing seven days ending on the annual AMWT7 end date",
            "minimum_valid_days": 5,
            "retrieval_date": "2026-10-10",
        },
        "sample": {
            "n_rows": int(len(df)),
            "n_sites": int(df["site"].nunique()),
            "n_sites_by_tier": {k: int(v) for k, v in df.groupby("tier")["site"].nunique().items()},
            "rows_by_tier": {k: int(v) for k, v in df["tier"].value_counts().items()},
        },
        "models": {
            "q_complete_unadjusted": summarize(primary, primary_formula, df, False),
            "flow_adjusted": summarize(flow_model, flow_formula, df, True),
        },
        "interpretation": (
            "The flow-aware model is a sensitivity analysis on the subset with an observed peak-window discharge. "
            "It tests persistence of the tier contrast and is not interpreted as a causal mediation estimate."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
