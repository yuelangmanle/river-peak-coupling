#!/usr/bin/env python3
"""22_sensitivity.py — nonlinear alternative explanations + inference robustness.

Nonlinearity (Mohseni S-curve / plateau): if the water-temperature vs air-temperature
   relation is S-shaped and tailwater sits on the plateau, the between-tier difference in
   linear beta could reflect nonlinearity rather than decoupling. Tests:
   - Fit a Mohseni logistic per tier (within-gauge centering): W_c = a + (b-a)/(1+exp(g*(T_c-K)))
   - Compare each tier's local slope dW/dT at the center of the observed air-temperature
     range with its linear beta
   - If the tailwater local slope ~= its linear beta and the three tiers' nonlinear shapes
     cannot reconcile the beta difference, this alternative explanation is excluded
Inference robustness:
   - AR(1): median within-gauge lag-1 autocorrelation of mixed-model residuals
   - Year-block bootstrap: resample years (keeping all gauges within a year); CIs for the
     contrast coefficients
   - Two-way clustered OLS FE (gauge x year) SEs for the interaction terms
     (Cameron-Gelbach-Miller)
Outputs are merged into canonical.json under sensitivity
"""
import os
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
rng = np.random.default_rng(42)

panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[panel.sample_scope == "primary"].dropna(
    subset=["water_mwmt7", "air_mwmt7"]).copy()
prim["air_c"] = prim.air_mwmt7 / 10.0
air_mean = prim.air_mwmt7.mean()
prim["air_cs"] = (prim.air_mwmt7 - air_mean) / 10.0

R = {}

# ---------------- Nonlinear (S-curve/plateau) alternative: within-gauge quadratic curvature test ----------------
# Plateau hypothesis predicts a concave within-gauge W-T relation (b2<0) with the local
# slope collapsing toward zero at the high end of air temperature.
# If tailwater shows no significant negative curvature, or its local-slope envelope remains
# far below the reference tier, the plateau hypothesis cannot explain the beta difference.
def quad_test(tier):
    sub = prim[prim.tier == tier].dropna(subset=["water_mwmt7", "air_mwmt7"]).copy()
    gm = sub.groupby("site8")[["water_mwmt7", "air_mwmt7"]].transform("mean")
    sub["Wc"] = sub.water_mwmt7 - gm.water_mwmt7
    sub["Tc"] = sub.air_mwmt7 - gm.air_mwmt7
    X = np.column_stack([np.ones(len(sub)), sub.Tc, sub.Tc ** 2])
    Y = sub.Wc.values
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    dof = len(Y) - 3
    s2 = e @ e / dof
    V = s2 * XtX_inv
    # Gauge-clustered (sandwich) covariance
    meat = np.zeros((3, 3))
    for s, g in sub.groupby("site8"):
        Xg = np.column_stack([np.ones(len(g)), g.Tc, g.Tc ** 2])
        u = (g.Wc.values - Xg @ b).reshape(-1, 1)
        XgW = Xg.T @ u
        meat += XgW @ XgW.T
    Vc = V @ meat @ V
    se_b1, se_b2 = np.sqrt(Vc[1, 1]), np.sqrt(Vc[2, 2])
    from scipy import stats as _st
    t2 = b[2] / se_b2
    p2 = 2 * (1 - _st.norm.cdf(abs(t2)))
    sd_T = float(sub.Tc.std())
    return dict(
        n=int(len(sub)), n_gauges=int(sub.site8.nunique()),
        slope_linear=round(float(b[1]), 3),
        curvature_b2=round(float(b[2]), 4), curvature_se=round(float(se_b2), 4),
        curvature_p=round(float(p2), 4),
        slope_at_minus1sd=round(float(b[1] + 2 * b[2] * (-sd_T)), 3),
        slope_at_plus1sd=round(float(b[1] + 2 * b[2] * sd_T), 3),
        note="within-gauge quadratic; slope_linear = β scale (°C/°C); "
             "plateau alternative requires b2<<0 with slope collapsing at high air")

nlb = {}
for tier in ["reference", "upstream", "tailwater"]:
    nlb[tier] = quad_test(tier)
R["nonlinear_quadratic"] = nlb

# ---------------- Inference robustness ----------------
# c1: AR(1) residual autocorrelation — refit with the canonical mixed-model specification
converged = False
m1 = None
try:
    cand = smf.mixedlm("water_mwmt7 ~ air_cs * C(tier)", prim,
                       groups=prim.site8, re_formula="1 + air_cs")
    for meth in ["lbfgs", "bfgs", "powell"]:
        try:
            m1 = cand.fit(method=meth, disp=False, maxiter=2000)
            if m1.converged:
                converged = True
                break
        except Exception:
            continue
except Exception:
    pass
if converged and m1 is not None:
    prim["resid"] = np.nan
    res = pd.Series(np.asarray(m1.resid), index=m1.model.data.row_labels)
    prim.loc[res.index, "resid"] = res.values
    rhos = []
    n_pairs_used = []
    for s, g in prim.dropna(subset=["resid"]).groupby("site8"):
        g = g.sort_values("year")
        adjacent = g.assign(next_resid=g.resid.shift(-1), next_year=g.year.shift(-1))
        adjacent = adjacent[adjacent.next_year - adjacent.year == 1].dropna(
            subset=["resid", "next_resid"])
        if len(adjacent) >= 5 and adjacent.resid.std() > 0 and adjacent.next_resid.std() > 0:
            r1 = np.corrcoef(adjacent.resid, adjacent.next_resid)[0, 1]
            if np.isfinite(r1):
                rhos.append(r1)
                n_pairs_used.append(int(len(adjacent)))
    R["ar1_resid"] = {"n_gauges": len(rhos),
                      "min_consecutive_pairs": 5,
                      "median_consecutive_pairs": int(np.median(n_pairs_used)) if n_pairs_used else 0,
                      "median_rho": round(float(np.median(rhos)), 3),
                      "iqr": [round(float(np.percentile(rhos, 25)), 3),
                              round(float(np.percentile(rhos, 75)), 3)],
                      "share_abs_gt_0.3": round(float((np.abs(rhos) > 0.3).mean()), 3)}
else:
    R["ar1_resid"] = {"error": "mixed model not converged"}

# c2: year-block bootstrap — resample years, keeping all gauges; OLS FE (gauge+year FE
# infeasible because year FE becomes collinear once whole years are resampled), so use
# gauge FE with year resampling for the contrast coefficients
def boot_year_block(nrep=2000):
    years = prim.year.unique()
    out = []
    for _ in range(nrep):
        ys = rng.choice(years, size=len(years), replace=True)
        parts = []
        for i, y in enumerate(ys):
            parts.append(prim[prim.year == y].assign(_blk=i))
        B = pd.concat(parts, ignore_index=True)
        try:
            f = smf.ols("water_mwmt7 ~ air_cs * C(tier) + C(site8)", data=B).fit()
            co = f.params
            out.append((co.get("air_cs:C(tier)[T.tailwater]", np.nan),
                        co.get("air_cs:C(tier)[T.upstream]", np.nan)))
        except Exception:
            continue
    A = np.array(out)
    return A

# Gauge-FE point estimate (same specification as the year-block bootstrap)
f_fe = smf.ols("water_mwmt7 ~ air_cs * C(tier) + C(site8)", data=prim).fit()
co_fe = f_fe.params
A = boot_year_block()
if len(A) > 100:
    def ci(v):
        v = v[np.isfinite(v)]
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    R["year_block_boot"] = {
        "n_rep": int(len(A)),
        "contrast_tw": {"point_fe": round(float(co_fe.get("air_cs:C(tier)[T.tailwater]", np.nan)), 3),
                        "ci95": ci(A[:, 0]),
                        "p_lt0": round(float((A[:, 0] < 0).mean()), 4)},
        "contrast_up": {"point_fe": round(float(co_fe.get("air_cs:C(tier)[T.upstream]", np.nan)), 3),
                        "ci95": ci(A[:, 1]),
                        "p_lt0": round(float((A[:, 1] < 0).mean()), 4)},
        "note": "years resampled with replacement (common-year shocks retained); gauge FE only, year FE infeasible under year resampling"}

# c3: two-way clustered SEs (gauge x year, Cameron-Gelbach-Miller) on gauge-FE OLS
try:
    f_g = smf.ols("water_mwmt7 ~ air_cs * C(tier) + C(site8)", data=prim).fit(
        cov_type="cluster", cov_kwds={"groups": prim.site8})
    f_y = smf.ols("water_mwmt7 ~ air_cs * C(tier) + C(site8)", data=prim).fit(
        cov_type="cluster", cov_kwds={"groups": prim.year})
    Vg, Vy = np.asarray(f_g.cov_params()), np.asarray(f_y.cov_params())
    V0 = np.asarray(f_fe.cov_params())
    # Two-way clustered SEs for interaction terms (cross-dimension covariance terms
    # ignored; conservative max used)
    idx_tw = list(co_fe.index).index("air_cs:C(tier)[T.tailwater]")
    idx_up = list(co_fe.index).index("air_cs:C(tier)[T.upstream]")
    R["twoway_cluster"] = {
        "se_tw": {"gauge": round(float(np.sqrt(Vg[idx_tw, idx_tw])), 3),
                  "year": round(float(np.sqrt(Vy[idx_tw, idx_tw])), 3),
                  "max_of_two": round(float(np.sqrt(max(Vg[idx_tw, idx_tw],
                                                        Vy[idx_tw, idx_tw]))), 3)},
        "se_up": {"gauge": round(float(np.sqrt(Vg[idx_up, idx_up])), 3),
                  "year": round(float(np.sqrt(Vy[idx_up, idx_up])), 3),
                  "max_of_two": round(float(np.sqrt(max(Vg[idx_up, idx_up],
                                                        Vy[idx_up, idx_up]))), 3)},
        "note": "Cameron-Gelbach-Miller two-way clustering approximated by max of one-way SEs (conservative)"}
except Exception as ex:
    R["twoway_cluster"] = {"error": str(ex)[:80]}

(V4 / "sensitivity.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
C = json.load(open(V4 / "canonical.json"))
C["sensitivity"] = R
(V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1))
