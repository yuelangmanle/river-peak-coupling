#!/usr/bin/env python3
"""16_mechanism.py — Mechanism diagnostics in three parts (all data in hand).

M1 Driver substitution quantified: per dam, R^2(annual peak water temp ~ antecedent storage) vs R^2(annual peak water temp ~ annual peak air temp)
M2 Downstream distance profile: descriptive beta summaries in primary and sensitivity bins
M3 Alternative-mechanism diagnostic: compare the attenuation implied by a pure
   first-order thermal-inertia filter with the observed peak lag. This is a
   diagnostic, not a mechanism-identification test. Includes a +2 degC
   projection conditioned on the observed coupling.
Outputs: mechanism.json + merged into canonical.json
"""
import os
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from common import antecedent_window_mask
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
RAW_ROOT = Path(os.environ.get("RESOPSUS_RAW_ROOT", str(BASE / "data/raw/resopsus_ts")))
R = {}

# ---------- M1: driver substitution per dam ----------
panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[
    (panel.sample_scope == "primary")
    & (panel.tier == "tailwater")
    & panel.dam_id.notna()
].copy()
da = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
da = da.drop(columns=[c for c in ["outflow_mean_m3s", "outflow_cv", "release_summer_frac",
                                  "storage_range_frac"] if c in da.columns])
dam_meta = da.drop_duplicates("dam_id")

rows = []
for dam, gg in prim.groupby("dam_id"):
    f = RAW_ROOT / f"ResOpsUS_{dam}.csv"
    if not f.exists():
        continue
    st = pd.read_csv(f, parse_dates=["date"]).replace("NA", np.nan)
    if "storage" not in st:
        continue
    st["storage"] = pd.to_numeric(st["storage"], errors="coerce")
    st = st.dropna(subset=["storage"])[["date", "storage"]]
    counts = gg.groupby("site8").water_mwmt7.count()
    site = counts.idxmax()
    g = gg[gg.site8 == site]
    recs = []
    for _, r in g.iterrows():
        if pd.isna(r.water_peak_doy):
            continue
        peak = pd.Timestamp(int(r.year), 1, 1) + pd.Timedelta(days=int(r.water_peak_doy) - 1)
        win = st[antecedent_window_mask(st.date, peak, days=30)]
        if len(win) < 20 or pd.isna(r.water_mwmt7) or pd.isna(r.air_mwmt7):
            continue
        recs.append((float(r.water_mwmt7), float(r.air_mwmt7), float(win.storage.mean())))
    Y = pd.DataFrame(recs, columns=["y", "a", "s"])
    if len(Y) < 10:
        continue
    r2_s = float(stats.pearsonr(Y.y, Y.s)[0] ** 2)
    r2_a = float(stats.pearsonr(Y.y, Y.a)[0] ** 2)
    meta = dam_meta[dam_meta.dam_id == dam]
    rows.append(dict(dam_id=str(dam), dam_name=str(meta.dam_name.iloc[0]) if len(meta) else "",
                     site=site, n_years=int(len(Y)), r2_storage=round(r2_s, 3),
                     r2_air=round(r2_a, 3), storage_wins=bool(r2_s > r2_a)))

D = pd.DataFrame(rows)
D.to_csv(V4 / "driver_substitution.csv", index=False)
wd = stats.wilcoxon(D.r2_storage, D.r2_air, alternative="greater")
R["driver_substitution"] = {
    "n_dams": int(len(D)),
    "median_r2_storage": round(float(D.r2_storage.median()), 3),
    "median_r2_air": round(float(D.r2_air.median()), 3),
    "share_storage_explains_more": round(float(D.storage_wins.mean()), 3),
    "wilcoxon_greater_p": round(float(wd.pvalue), 5),
}
sh = D[D.dam_id == "132"]
R["driver_substitution"]["shasta_r2_storage_vs_air"] = (
    [float(sh.r2_storage.iloc[0]), float(sh.r2_air.iloc[0])] if len(sh) else None)

# ---------- M2: descriptive downstream distance profile ----------
L = pd.read_csv(V4 / "ladder.csv", dtype={"site_no": str})
L["site_no"] = L.site_no.str.zfill(8)
tw = L[(L.group == "tailwater") & L.dist_km.notna() & (L.dist_km > 0)
       & (L.dist_km <= 60)].dropna(subset=["beta"]).copy()
tw["sample_scope"] = tw.sample_scope.fillna("primary").replace({"band": "sensitivity"})
distance_bins = [
    ("primary_0_10km", "0-10 km (primary)", 0, 10, "primary"),
    ("primary_10_30km", "10-30 km (primary)", 10, 30, "primary"),
    ("sensitivity_30_60km", "30-60 km (sensitivity)", 30, 60, "sensitivity"),
]
rng = np.random.default_rng(5)
bin_summary = {}
for key, label, lower, upper, scope in distance_bins:
    sub = tw[(tw.sample_scope == scope) & (tw.dist_km > lower) & (tw.dist_km <= upper)]
    vals = sub.beta.to_numpy(dtype=float)
    cluster_id = sub.dam_id.astype("string").fillna("site:" + sub.site_no.astype(str))
    clusters = [g.beta.to_numpy(dtype=float) for _, g in sub.assign(_cluster=cluster_id).groupby("_cluster")]
    boots = np.array([
        np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]).mean()
        for _ in range(2000)
    ]) if clusters else np.array([])
    bin_summary[key] = {
        "label": label, "sample_scope": scope,
        "distance_bounds_km": [lower, upper], "lower_bound_inclusive": False,
        "upper_bound_inclusive": True,
        "n": int(len(vals)), "n_reservoirs": int(len(clusters)),
        "mean_beta": round(float(np.mean(vals)), 3) if len(vals) else None,
        "bootstrap_ci95": [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])]
        if len(boots) else None,
        "interval_method": "reservoir-cluster percentile bootstrap",
    }
R["recovery"] = {
    "n_primary_gauges": int(((tw.sample_scope == "primary") & (tw.dist_km <= 30)).sum()),
    "n_sensitivity_gauges": int(((tw.sample_scope == "sensitivity")
                                 & (tw.dist_km > 30) & (tw.dist_km <= 60)).sum()),
    "distance_bins": bin_summary,
    "note": ("Gauge-level distance-bin means and percentile bootstrap intervals are descriptive; "
             "30-60 km gauges are a sensitivity band and are kept separate from the primary 0-30 km profile."),
}

# ---------- M3: inertia falsification ----------
C = json.load(open(V4 / "canonical.json"))
beta_ref = C["two_stage_primary"]["mean"]["reference"]
beta_tw = C["two_stage_primary"]["mean"]["tailwater"]
a_tw = beta_tw / beta_ref
omega = 2 * np.pi / 365.0
tau_days = float(np.sqrt(1 / a_tw**2 - 1) / omega)
lag_pred = float(np.arctan(omega * tau_days) / omega)
obs_lag = C.get("peak_timing", {}).get("tailwater", {}).get("mean_offset_d", None)
att_pct = round((1 - a_tw) * 100)
R["inertia_diagnostic"] = {
    "amplitude_ratio_tw_over_ref": round(a_tw, 3),
    "implied_tau_days": round(tau_days, 0),
    "predicted_lag_days_if_inertia_filter": round(lag_pred, 0),
    "observed_mean_lag_days": obs_lag,
    "conclusion": (f"a pure first-order filter calibrated to {att_pct}% attenuation implies "
                   f"a ~{round(lag_pred)}-day phase lag, compared with an observed mean "
                   f"offset of ~{obs_lag} days; this comparison is diagnostic and does not "
                   "identify or rule out mechanisms"),
}

# ---------- Projection (+2 degC, conditioned on observed coupling) ----------
E = C["exposure_sensitivity"]
R["projection_plus2C"] = {
    "note": "under the observed coupling, +2 degC in annual peak air temperature",
    "peak_warming_degC": {"reference": round(2 * beta_ref, 2),
                          "tailwater": round(2 * beta_tw, 2)},
    "threshold_days_added": {"reference": round(2 * E["reference"]["mean_unweighted"], 1),
                             "tailwater": round(2 * E["tailwater"]["mean_unweighted"], 1),
                             "hot_matched_reference": 12.2},
}

(V4 / "mechanism.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
C["mechanism"] = R
(V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1))
print(D.sort_values("r2_storage", ascending=False).head(8).to_string(index=False))
