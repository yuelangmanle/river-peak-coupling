#!/usr/bin/env python3
"""Post-hoc national storage-state panel for tailwater annual peaks.

The compact released table stores annual peak water and air temperature with the
mean ResOpsUS storage over [T-30, T-1].  Raw ResOpsUS series are not
redistributed; --raw-root rebuilds the compact table when those external inputs
are available.  The analysis uses one longest-record primary tailwater gauge per
reservoir, dams with at least 10 complete years, within-reservoir centered air
temperature, within-reservoir standardized storage, reservoir fixed effects, and
reservoir-clustered standard errors.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import antecedent_window_mask

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
PANEL_FILE = V4 / "storage_peak_panel.csv"
OUTPUT_FILE = V4 / "storage_state_panel.json"
SEED = 20261010


def rebuild_compact_table(raw_root):
    panel = pd.read_parquet(V4 / "annual_panel.parquet")
    panel["site8"] = panel.site.astype(str).str.zfill(8)
    primary = panel[
        (panel.sample_scope == "primary")
        & (panel.tier == "tailwater")
        & panel.dam_id.notna()
    ].copy()
    records = []
    for dam_id, candidates in primary.groupby("dam_id"):
        source = Path(raw_root) / f"ResOpsUS_{dam_id}.csv"
        if not source.is_file():
            continue
        storage = pd.read_csv(source, parse_dates=["date"]).replace("NA", np.nan)
        if "storage" not in storage:
            continue
        storage["storage"] = pd.to_numeric(storage.storage, errors="coerce")
        storage = storage.dropna(subset=["storage"])[["date", "storage"]]
        site_counts = candidates.groupby("site8").water_mwmt7.count()
        site = site_counts.idxmax()
        rows = candidates[candidates.site8 == site]
        for row in rows.itertuples(index=False):
            if pd.isna(row.water_peak_doy):
                continue
            peak = pd.Timestamp(int(row.year), 1, 1) + pd.Timedelta(
                days=int(row.water_peak_doy) - 1
            )
            window = storage[antecedent_window_mask(storage.date, peak, days=30)]
            if len(window) < 20:
                continue
            if any(pd.isna(x) for x in (row.water_mwmt7, row.air_mwmt7)):
                continue
            records.append(
                {
                    "dam_id": str(dam_id),
                    "site": str(site),
                    "year": int(row.year),
                    "water_peak": float(row.water_mwmt7),
                    "air_peak": float(row.air_mwmt7),
                    "antecedent_storage_30d": float(window.storage.mean()),
                    "valid_storage_days": int(len(window)),
                    "shasta": str(dam_id) == "132",
                }
            )
    compact = pd.DataFrame(records)
    counts = compact.groupby("dam_id").year.transform("count")
    compact = compact[counts >= 10].copy()
    compact.to_csv(PANEL_FILE, index=False)
    return compact


def prepare(df, group_col="dam_id"):
    out = df.copy()
    means = out.groupby(group_col)[["air_peak", "antecedent_storage_30d", "year"]].transform("mean")
    sds = out.groupby(group_col)["antecedent_storage_30d"].transform("std")
    out["air_c"] = out.air_peak - means.air_peak
    out["storage_z"] = (out.antecedent_storage_30d - means.antecedent_storage_30d) / sds
    out["year_c"] = out.year - means.year
    return out


def fit_clustered(df, with_year=False):
    formula = "water_peak ~ C(dam_id) + air_c * storage_z"
    if with_year:
        formula += " + year_c"
    model = smf.ols(formula, data=df).fit()
    return model.get_robustcov_results(
        cov_type="cluster", groups=df.dam_id, use_correction=True
    )


def coefficient(result, name):
    idx = list(result.model.exog_names).index(name)
    return {
        "coef": round(float(result.params[idx]), 6),
        "se": round(float(result.bse[idx]), 6),
        "ci95": [round(float(x), 6) for x in result.conf_int()[idx]],
        "p": round(float(result.pvalues[idx]), 8),
    }


def bootstrap_coefficients(df, n_boot=5000):
    rng = np.random.default_rng(SEED)
    dams = np.array(sorted(df.dam_id.unique()))
    values = {"storage_z": [], "air_c:storage_z": []}
    for _ in range(n_boot):
        selected = rng.choice(dams, size=len(dams), replace=True)
        pieces = []
        for copy_id, dam in enumerate(selected):
            part = df[df.dam_id == dam].copy()
            part["bootstrap_cluster"] = f"{copy_id:03d}_{dam}"
            pieces.append(part)
        sample = pd.concat(pieces, ignore_index=True)
        sample = prepare(sample, group_col="bootstrap_cluster")
        try:
            result = fit_clustered(
                sample.drop(columns="dam_id").rename(columns={"bootstrap_cluster": "dam_id"})
            )
            for term in values:
                values[term].append(coefficient(result, term)["coef"])
        except (np.linalg.LinAlgError, ValueError):
            continue
    return {
        "n_successful": len(values["storage_z"]),
        "storage_main": {
            "median": round(float(np.median(values["storage_z"])), 6),
            "ci95": [round(float(np.quantile(values["storage_z"], 0.025)), 6),
                     round(float(np.quantile(values["storage_z"], 0.975)), 6)],
            "share_negative": round(float(np.mean(np.asarray(values["storage_z"]) < 0)), 4),
        },
        "air_x_storage": {
            "median": round(float(np.median(values["air_c:storage_z"])), 6),
            "ci95": [round(float(np.quantile(values["air_c:storage_z"], 0.025)), 6),
                     round(float(np.quantile(values["air_c:storage_z"], 0.975)), 6)],
            "share_negative": round(float(np.mean(np.asarray(values["air_c:storage_z"]) < 0)), 4),
        },
    }


def add_shasta_year(compact):
    """Add the separately archived Shasta annual records as one additional cluster."""
    shasta = pd.read_csv(V4 / "shasta_annual.csv")
    out = shasta[["year", "water_mwmt7", "air_mwmt7", "stor_peak30"]].copy()
    out.columns = ["year", "water_peak", "air_peak", "antecedent_storage_30d"]
    out.insert(0, "site", "shasta_case")
    out.insert(0, "dam_id", "132")
    out["valid_storage_days"] = 30
    out["shasta"] = True
    return pd.concat([compact, out], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory containing ResOpsUS_<dam_id>.csv; rebuilds the compact table.",
    )
    args = parser.parse_args()
    raw_root = args.raw_root or BASE / "data/raw/resopsus_ts"
    if raw_root.is_dir():
        compact = rebuild_compact_table(raw_root)
    elif PANEL_FILE.is_file():
        compact = pd.read_csv(PANEL_FILE, dtype={"dam_id": str, "site": str})
    else:
        raise SystemExit(
            "storage_peak_panel.csv is absent; provide --raw-root to rebuild it from ResOpsUS."
        )

    compact = prepare(compact)
    non_shasta = compact[~compact.shasta].copy()
    with_shasta = prepare(add_shasta_year(compact))
    primary = fit_clustered(non_shasta)
    year_control = fit_clustered(non_shasta, with_year=True)
    shasta_model = fit_clustered(with_shasta)

    loo = []
    for dam in sorted(non_shasta.dam_id.unique()):
        reduced = non_shasta[non_shasta.dam_id != dam]
        result = fit_clustered(reduced)
        loo.append(coefficient(result, "air_c:storage_z")["coef"])

    result = {
        "status": "post hoc",
        "design": (
            "one longest-record primary tailwater gauge per reservoir; >=10 complete years; "
            "within-reservoir standardized storage; reservoir fixed effects; clustered SE"
        ),
        "primary_non_shasta": {
            "n_years": int(len(non_shasta)),
            "n_reservoirs": int(non_shasta.dam_id.nunique()),
            "median_years_per_reservoir": float(non_shasta.groupby("dam_id").year.count().median()),
            "storage_main": coefficient(primary, "storage_z"),
            "air_peak": coefficient(primary, "air_c"),
            "air_x_storage": coefficient(primary, "air_c:storage_z"),
        },
        "year_control": {
            "n_years": int(len(non_shasta)),
            "storage_main": coefficient(year_control, "storage_z"),
            "air_x_storage": coefficient(year_control, "air_c:storage_z"),
        },
        "include_shasta": {
            "n_years": int(len(with_shasta)),
            "n_reservoirs": int(with_shasta.dam_id.nunique()),
            "storage_main": coefficient(shasta_model, "storage_z"),
            "air_x_storage": coefficient(shasta_model, "air_c:storage_z"),
        },
        "leave_one_reservoir_out_interaction": {
            "range": [round(float(min(loo)), 6), round(float(max(loo)), 6)],
            "share_negative": round(float(np.mean(np.asarray(loo) < 0)), 4),
        },
        "reservoir_bootstrap": bootstrap_coefficients(non_shasta),
    }
    OUTPUT_FILE.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    canonical_path = V4 / "canonical.json"
    canonical = json.loads(canonical_path.read_text())
    canonical["storage_state_panel"] = result
    canonical_path.write_text(json.dumps(canonical, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
