#!/usr/bin/env python3
"""32_area_matching.py — 流域面积配平分析（微信评审第一死穴）.

1. log(area) 协变量进混合模型
2. 面积卡尺配对参考-尾水
3. 面积分层衰减梯度
"""
import os
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2]))).resolve()
V4 = (BASE / "data" / "samples" / "analysis").resolve()
if BASE not in V4.parents:
    raise SystemExit("path check")

panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[panel.sample_scope == "primary"].dropna(subset=["water_mwmt7", "air_mwmt7"]).copy()
area = pd.read_csv(V4 / "site_drainage_area.csv", dtype={"site8": str})
area["site8"] = area.site8.str.zfill(8)
prim = prim.merge(area, on="site8", how="left")
prim = prim.dropna(subset=["area_km2"])
prim["log_area"] = np.log10(prim.area_km2)
area_mean = prim.log_area.mean()
prim["log_area_c"] = prim.log_area - area_mean

L = pd.read_csv(V4 / "ladder.csv", dtype={"site_no": str})
L["site_no"] = L.site_no.str.zfill(8)
Lp = L[L.sample_scope == "primary"].dropna(subset=["beta"])
Lp = Lp.merge(area, left_on="site_no", right_on="site8", how="left")

R = {}

# ---- 1. 混合模型 + log(area) 协变量 ----
prim["air_cs"] = (prim.air_mwmt7 - prim.air_mwmt7.mean()) / 10
m = None
for meth in ["lbfgs", "bfgs", "powell"]:
    try:
        cand = smf.mixedlm(
            "water_mwmt7 ~ air_cs * C(tier) + log_area_c", prim,
            groups=prim.site8, re_formula="1 + air_cs")
        r = cand.fit(method=meth, disp=False, maxiter=3000)
        if r.converged:
            m = r; break
    except Exception:
        continue
if m is not None and m.converged:
    co = m.params; V = np.asarray(m.cov_params())
    names = list(co.index)
    br = co["air_cs"] / 10
    idx_tw = names.index("air_cs:C(tier)[T.tailwater]")
    btw = (co["air_cs"] + co["air_cs:C(tier)[T.tailwater]"]) / 10
    se_tw = np.sqrt(V[idx_tw, idx_tw]) / 10
    p_tw = float(m.pvalues["air_cs:C(tier)[T.tailwater]"])
    la = co.get("log_area_c", None)
    la_p = float(m.pvalues["log_area_c"]) if la is not None else None
    R["area_covariate_mixed"] = {
        "n_gauges": int(prim.site8.nunique()), "n_obs": int(len(prim)),
        "beta_reference_perC": round(float(br), 3),
        "beta_tailwater_perC": round(float(btw), 3),
        "contrast_tw_perC": round(float(btw - br), 3),
        "contrast_se_perC": round(float(se_tw), 3),
        "contrast_p": p_tw,
        "log_area_coef_per_decade": round(float(la), 3) if la is not None else None,
        "log_area_p": la_p,
        "note": ("mixed model with centered log10(drainage area km2) as covariate; "
                 "log_area_c coefficient is per decade (10x) change in area")}

# ---- 2. 面积卡尺配对 ----
ref_l = Lp[Lp.group == "reference"].dropna(subset=["area_km2"])
tw_l = Lp[Lp.group == "tailwater"].dropna(subset=["area_km2"])
ref_l["log_a"] = np.log10(ref_l.area_km2)
tw_l["log_a"] = np.log10(tw_l.area_km2)

for cal in [0.3, 0.5, 1.0]:
    pairs = []
    used = set()
    for _, tr in tw_l.iterrows():
        cands = ref_l[~ref_l.site_no.isin(used) & ((ref_l.log_a - tr.log_a).abs() <= cal)]
        if cands.empty: continue
        best = cands.assign(d=(cands.log_a - tr.log_a).abs()).nsmallest(1, "d")
        used.add(best.site_no.iloc[0])
        pairs.append(dict(tw_beta=tr.beta, ref_beta=best.beta.iloc[0]))
    P = pd.DataFrame(pairs)
    if len(P) >= 5:
        diff = P.ref_beta - P.tw_beta
        mw = stats.wilcoxon(P.ref_beta, P.tw_beta, alternative="greater")
        R.setdefault("area_caliper_matching", {})[f"caliper_{cal}"] = {
            "n_pairs": len(P),
            "mean_ref": round(float(P.ref_beta.mean()), 3),
            "mean_tw": round(float(P.tw_beta.mean()), 3),
            "mean_deficit": round(float(diff.mean()), 3),
            "wilcoxon_p_one_sided": round(float(mw.pvalue), 4)}

# ---- 3. 面积分层梯度 ----
quartiles = np.percentile(Lp.dropna(subset=["area_km2"]).area_km2, [25, 50, 75])
strata = []
for qlo, qhi, lab in [(0, quartiles[0], "Q1"), (quartiles[0], quartiles[1], "Q2"),
                       (quartiles[1], quartiles[2], "Q3"), (quartiles[2], 1e9, "Q4")]:
    sl = Lp.dropna(subset=["area_km2"])
    sl = sl[(sl.area_km2 > qlo) & (sl.area_km2 <= qhi)]
    r_ = sl[sl.group == "reference"].beta
    t_ = sl[sl.group == "tailwater"].beta
    if len(r_) >= 5 and len(t_) >= 5:
        mw = stats.mannwhitneyu(t_, r_, alternative="less")
        strata.append(dict(stratum=lab, area_range=f"{qlo:.0f}-{qhi:.0f}" if qhi < 1e9 else f">{qlo:.0f}",
                           n_ref=len(r_), ref_beta=round(float(r_.mean()), 3),
                           n_tw=len(t_), tw_beta=round(float(t_.mean()), 3),
                           deficit_pct=round(100*(1-t_.mean()/r_.mean()), 1),
                           mwu_p=float(mw.pvalue)))
R["area_stratified_ladder"] = {"quartile_strata": strata,
    "note": "reference/tailwater β by drainage-area quartile; GAGES-II DRAIN_SQKM + NWIS drain_sqmi"}

# ---- 4. 尾水层内部: log(area) vs beta 的偏效应（控制 tier 后） ----
Lpa = Lp.dropna(subset=["area_km2"]).copy()
Lpa["log_a"] = np.log10(Lpa.area_km2)
Lpa["log_a_c"] = Lpa.log_a - Lpa.log_a.mean()
f = smf.ols("beta ~ C(group) + log_a_c", data=Lpa).fit(cov_type="cluster", cov_kwds={"groups": Lpa.site_no})
R["area_partial_effect"] = {
    "log_area_coef": round(float(f.params["log_a_c"]), 4),
    "log_area_p": round(float(f.pvalues["log_a_c"]), 4),
    "note": "OLS beta ~ tier + log_area_c, gauge-clustered; how much β changes per decade of area"}

# 落盘
cp = V4 / "canonical.json"
C = json.loads(cp.read_text())
C["review_robustness"]["area_matching"] = R
cp.write_text(json.dumps(C, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1))
