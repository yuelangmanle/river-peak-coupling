#!/usr/bin/env python3
"""15_multidam.py — Mechanism replication: dam-by-dam antecedent storage coupling x depth moderation (cross-dam generalization of the Shasta protocol).

Design (release-corrected):
  - For each dam, use the primary tailwater station with the longest record; annual peak day = water_peak_doy (current panel)
  - Antecedent storage = ResOpsUS daily storage mean over the 30 days before the peak (>=20 valid days in window)
  - >=10 years per dam (water peak + air peak + storage all available); z-standardized within dam
  - Per dam: y ~ A + S standardized coefficients => partial correlations r_S|A and r_A|S
  - Across dams: Fisher-z weighted pooling (weights 1/(n-3)); moderation = WLS z_rS ~ depth (controlling log capacity);
    symmetry check: cross-dam correlation of z_rS with z_rA (driver substitution)
  - Gate (pre-specified): deep-reservoir group (depth >= 20 m) share with |r_S| > 0.5 >= 60% => replication holds;
    else depth slope z > 1.64 => PARTIAL; else NULL (measurement-gap framing still stands)
Outputs: multidam_results.csv + multidam_block.json + merged into canonical.json
"""
import os
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from common import antecedent_window_mask

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
RAW_ROOT = Path(os.environ.get("RESOPSUS_RAW_ROOT", str(BASE / "data/raw/resopsus_ts")))

panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[
    (panel.sample_scope == "primary")
    & (panel.tier == "tailwater")
    & panel.dam_id.notna()
].copy()
da = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
da["depth_m"] = pd.to_numeric(da.depth_m, errors="coerce")
da.loc[da.depth_m < 0, "depth_m"] = np.nan
da["grand_cap_mcm"] = pd.to_numeric(da.grand_cap_mcm, errors="coerce")
da = da.drop(columns=[c for c in ["outflow_mean_m3s", "outflow_cv", "release_summer_frac",
                                  "storage_range_frac"] if c in da.columns])
ops = pd.read_csv(BASE / "data/samples/reservoir_ops_features.csv", dtype={"dam_id": str})
dam_meta = da.merge(ops[["dam_id", "release_summer_frac", "storage_range_frac", "outflow_cv"]],
                    on="dam_id", how="left").drop_duplicates("dam_id")

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
    rrows = []
    for _, r in g.iterrows():
        if pd.isna(r.water_peak_doy):
            continue
        peak = pd.Timestamp(int(r.year), 1, 1) + pd.Timedelta(days=int(r.water_peak_doy) - 1)
        # Keep this protocol aligned with the Shasta analysis and its stated
        # antecedent interval: [T-30, T-1], excluding the peak day itself.
        win = st[antecedent_window_mask(st.date, peak, days=30)]
        if len(win) < 20 or pd.isna(r.water_mwmt7) or pd.isna(r.air_mwmt7):
            continue
        rrows.append(dict(year=int(r.year), y=float(r.water_mwmt7),
                          a=float(r.air_mwmt7), s=float(win.storage.mean())))
    Y = pd.DataFrame(rrows)
    if len(Y) < 10 or Y[["y", "a", "s"]].std().min() == 0:
        continue
    Z = (Y[["y", "a", "s"]] - Y[["y", "a", "s"]].mean()) / Y[["y", "a", "s"]].std()
    X = np.column_stack([np.ones(len(Z)), Z.a, Z.s])
    beta = np.linalg.lstsq(X, Z.y, rcond=None)[0]
    resid = Z.y - X @ beta
    dfree = len(Z) - 3
    sigma2 = resid @ resid / dfree
    covb = sigma2 * np.linalg.inv(X.T @ X)
    tS = beta[2] / np.sqrt(covb[2, 2]); tA = beta[1] / np.sqrt(covb[1, 1])
    rS = float(np.sign(tS) * np.sqrt(tS**2 / (tS**2 + dfree)))
    rA = float(np.sign(tA) * np.sqrt(tA**2 / (tA**2 + dfree)))
    meta = dam_meta[dam_meta.dam_id == dam]
    rows.append(dict(
        dam_id=str(dam),
        dam_name=str(meta.dam_name.iloc[0]) if len(meta) else "",
        depth_m=float(meta.depth_m.iloc[0]) if len(meta) and pd.notna(meta.depth_m.iloc[0]) else np.nan,
        cap_mcm=float(meta.grand_cap_mcm.iloc[0]) if len(meta) and pd.notna(meta.grand_cap_mcm.iloc[0]) else np.nan,
        release_summer_frac=float(meta.release_summer_frac.iloc[0]) if len(meta) else np.nan,
        site=site, n_years=int(len(Y)),
        rS_partial=round(rS, 3), rA_partial=round(rA, 3),
        shasta=(str(dam) == "132")))

M = pd.DataFrame(rows).dropna(subset=["depth_m"])
M["zS"] = np.arctanh(M.rS_partial.clip(-0.999, 0.999))
M["zA"] = np.arctanh(M.rA_partial.clip(-0.999, 0.999))
M["w"] = 1 / (1 / (M.n_years - 3))
M.to_csv(V4 / "multidam_results.csv", index=False)

R = {"n_dams_total": int(len(M)),
     "median_years_per_dam": float(M.n_years.median()),
     "depth_range": [float(M.depth_m.min()), float(M.depth_m.max())]}
wmean = float((M.zS * M.w).sum() / M.w.sum())
i2 = None
if len(M) > 1:
    q = float(((M.zS - wmean) ** 2 * M.w).sum())
    c = M.w.sum() - (M.w ** 2).sum() / M.w.sum()
    i2 = round(max(0.0, (q - (len(M) - 1)) / q * 100), 1) if q > 0 else 0.0
R["pooled"] = {"weighted_mean_rS_partial": round(float(np.tanh(wmean)), 3),
               "heterogeneity_I2_pct": i2,
               "share_abs_rS_gt_0.5": round(float((M.rS_partial.abs() > 0.5).mean()), 3),
               "share_negative_rS": round(float((M.rS_partial < 0).mean()), 3)}
deep = M[M.depth_m >= 20]; shallow = M[M.depth_m < 20]
X = np.column_stack([np.ones(len(M)), M.depth_m.values])
Wd = np.diag(M.w.values)
b = np.linalg.solve(X.T @ Wd @ X, X.T @ Wd @ M.zS.values)
res = M.zS.values - X @ b
s2 = (res ** 2 * M.w.values).sum() / (len(M) - 2)
se_b = float(np.sqrt(np.linalg.inv(X.T @ Wd @ X)[1, 1] * s2))
Ml = M.dropna(subset=["cap_mcm"]).copy(); Ml["logcap"] = np.log(Ml.cap_mcm)
X3 = np.column_stack([np.ones(len(Ml)), Ml.depth_m.values, Ml.logcap.values])
Wd3 = np.diag(Ml.w.values)
b3 = np.linalg.solve(X3.T @ Wd3 @ X3, X3.T @ Wd3 @ Ml.zS.values)
res3 = Ml.zS.values - X3 @ b3
s23 = (res3 ** 2 * Ml.w.values).sum() / (len(Ml) - 3)
se_d3 = float(np.sqrt(np.linalg.inv(X3.T @ Wd3 @ X3)[1, 1] * s23))
wd_ = stats.mannwhitneyu(deep.rS_partial, shallow.rS_partial, alternative="less")
wr = stats.pearsonr(M.zS, M.zA)
R["depth_moderation"] = {
    "slope_zS_per_m": round(float(b[1]), 4),
    "p_depth": round(float(2 * (1 - stats.norm.cdf(abs(b[1] / se_b)))), 5),
    "slope_zS_per_m_control_logcap": round(float(b3[1]), 4),
    "p_depth_control_logcap": round(float(2 * (1 - stats.norm.cdf(abs(b3[1] / se_d3)))), 5),
    "deep_ge20m": {"n": int(len(deep)), "median_rS": round(float(deep.rS_partial.median()), 3),
                   "share_abs_gt_0.5": round(float((deep.rS_partial.abs() > 0.5).mean()), 3)},
    "shallow_lt20m": {"n": int(len(shallow)), "median_rS": round(float(shallow.rS_partial.median()), 3),
                      "share_abs_gt_0.5": round(float((shallow.rS_partial.abs() > 0.5).mean()), 3)},
    "mwu_deep_below_shallow_p": round(float(wd_.pvalue), 4)}
R["tradeoff"] = {"r_zS_zA_across_dams": round(float(wr[0]), 3), "p": round(float(wr[1]), 5)}
sh = M[M.shasta]
R["shasta_check"] = {"rS_partial": float(sh.rS_partial.iloc[0]) if len(sh) else None,
                     "note": "current bivariate r(peak-30d)=-0.749; partial controlling air expected weaker"}
gate_share = R["depth_moderation"]["deep_ge20m"]["share_abs_gt_0.5"]
if gate_share >= 0.6:
    verdict = "REPLICATION PASS"
elif b[1] / se_b > 1.64:
    verdict = "PARTIAL: depth gradient significant, share threshold not met"
else:
    verdict = "NULL: measurement-gap framing stands"
R["gate"] = {"deep_share_abs_rS_gt_0.5": gate_share, "verdict": verdict}

(V4 / "multidam_block.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
C = json.load(open(V4 / "canonical.json"))
C["multidam_replication"] = R
(V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1))
print(M.sort_values("depth_m")[["dam_name", "depth_m", "n_years", "rS_partial", "rA_partial"]].to_string(index=False))
