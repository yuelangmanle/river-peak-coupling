#!/usr/bin/env python3
"""23_routing_sensitivity.py — Routing-verification sensitivity analysis.

Verdict convention: verified = pass (COMID within the dam's downstream DM 35 km set) or
near (site coordinates within 2 km of the DM corridor; tributary-mouth/braided-channel tolerance).
Recomputed on the verified-only tailwater subsample:
  1. Two-stage site-level beta (>=8 qualifying years): per-tier means + MWU
  2. M1 mixed model (same specification as canonical)
Compared with the canonical full sample. If direction and significance hold, routing confounding cannot explain the main result.
Output merged into canonical.json: routing_verification
"""
import os
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"

panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[panel.sample_scope == "primary"].copy()

rv = pd.read_csv(V4 / "routing_verification.csv", dtype={"site_no": str, "dam_id": str})
rv["site8"] = rv.site_no.str.zfill(8)
bad_tw = set(rv[(rv.direction == "downstream") & (rv.verdict == "fail")].site8)
ok_tw = set(rv[(rv.direction == "downstream") & (rv.verdict.isin(["pass", "near"]))].site8)
bad_up = set(rv[(rv.direction == "upstream") & (rv.verdict == "fail")].site8)
ok_up = set(rv[(rv.direction == "upstream") & (rv.verdict.isin(["pass", "near"]))].site8)
unresolved_tw = set(rv[(rv.direction == "downstream") & (rv.verdict == "nocomid")].site8)
unresolved_up = set(rv[(rv.direction == "upstream") & (rv.verdict == "nocomid")].site8)

R = {"counts": {
    "tw_verified": len(ok_tw), "tw_fail": len(bad_tw), "tw_unresolved": len(unresolved_tw),
    "up_verified": len(ok_up), "up_fail": len(bad_up), "up_unresolved": len(unresolved_up)}}

# ---- Two-stage beta: verified-only ----
def twostage(P, min_years=8):
    rows = []
    for s, g in P.dropna(subset=["water_mwmt7", "air_mwmt7"]).groupby("site8"):
        if len(g) < min_years:
            continue
        x = g.air_mwmt7.values
        y = g.water_mwmt7.values
        if x.std() == 0 or y.std() == 0:
            continue
        b = np.corrcoef(x, y)[0, 1] * y.std() / x.std()
        rows.append(dict(site8=s, tier=g.tier.iloc[0], beta=b))
    return pd.DataFrame(rows)

prim_air = prim.dropna(subset=["water_mwmt7", "air_mwmt7"])
Bv = twostage(prim_air[(prim_air.tier != "tailwater") |
                       prim_air.site8.isin(ok_tw)])
Bv = Bv[~((Bv.tier == "upstream") & (Bv.site8.isin(bad_up)))]
t = {}
for tier in ["reference", "upstream", "tailwater"]:
    t[tier] = dict(n=int((Bv.tier == tier).sum()),
                   mean=round(float(Bv[Bv.tier == tier].beta.mean()), 3),
                   median=round(float(Bv[Bv.tier == tier].beta.median()), 3))
mw = stats.mannwhitneyu(Bv[Bv.tier == "tailwater"].beta,
                        Bv[Bv.tier == "reference"].beta, alternative="less")
R["twostage_verified_only"] = {"tiers": t, "tw_lt_ref_p": float(mw.pvalue),
                               "ratio_tw_ref": round(t["tailwater"]["mean"] / t["reference"]["mean"], 3)}

# ---- M1 verified-only ----
p1v = prim_air[(prim_air.tier != "tailwater") | (prim_air.site8.isin(ok_tw))].copy()
p1v = p1v[~((p1v.tier == "upstream") & (p1v.site8.isin(bad_up)))]
air_mean = p1v.air_mwmt7.mean()
p1v["air_cs"] = (p1v.air_mwmt7 - air_mean) / 10.0
m1 = None
try:
    cand = smf.mixedlm("water_mwmt7 ~ air_cs * C(tier)", p1v,
                       groups=p1v.site8, re_formula="1 + air_cs")
    for meth in ["lbfgs", "bfgs", "powell"]:
        try:
            m1 = cand.fit(method=meth, disp=False, maxiter=2000)
            if m1.converged:
                break
        except Exception:
            continue
except Exception:
    pass
if m1 is not None and m1.converged:
    co = m1.params
    V = np.asarray(m1.cov_params())
    idx = list(co.index).index("air_cs:C(tier)[T.tailwater]")
    se_i = float(np.sqrt(V[idx, idx]))
    p_tw = float(m1.pvalues["air_cs:C(tier)[T.tailwater]"])
    R["M1_verified_only"] = {
        "n_gauges": int(p1v.site8.nunique()), "n_obs": int(len(p1v)),
        "beta_reference": round(float(co["air_cs"]), 3),
        "beta_tailwater": round(float(co["air_cs"] + co["air_cs:C(tier)[T.tailwater]"]), 3),
        "contrast_tw_se": round(se_i, 3), "contrast_tw_p": p_tw}
else:
    R["M1_verified_only"] = {"error": "not converged"}

(V4 / "routing_sensitivity.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
C = json.load(open(V4 / "canonical.json"))
C["routing_verification"] = R
(V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1))
