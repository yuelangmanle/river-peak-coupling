#!/usr/bin/env python3
"""23b_both_sources_sensitivity.py — Intersection sensitivity across two routing sources.

Tailwater pairs that pass both independent routing sources — NLDI (pass/near) and HydroRIVERS (pass_downstream) —
form the highest-confidence "genuinely below the dam" subsample. Two-stage beta and M1 are recomputed on that subsample.
Output merged into canonical.json: routing_verification.both_sources_sensitivity
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

B = json.load(open(V4 / "routing_hyrorivers_block.json"))
both = set(B["both_sources_downstream_sites"])

prim_air = prim.dropna(subset=["water_mwmt7", "air_mwmt7"])
rows = []
for s, g in prim_air.groupby("site8"):
    if len(g) < 8:
        continue
    x, y = g.air_mwmt7.values, g.water_mwmt7.values
    if x.std() == 0 or y.std() == 0:
        continue
    r = np.corrcoef(x, y)[0, 1]
    rows.append(dict(site8=s, tier=g.tier.iloc[0], beta=r * y.std() / x.std()))
Bg = pd.DataFrame(rows)
Bb = Bg[(Bg.tier != "tailwater") | Bg.site8.isin(both)]
t = {}
for tier in ["reference", "upstream", "tailwater"]:
    t[tier] = dict(n=int((Bb.tier == tier).sum()),
                   mean=round(float(Bb[Bb.tier == tier].beta.mean()), 3))
mw = stats.mannwhitneyu(Bb[Bb.tier == "tailwater"].beta,
                        Bb[Bb.tier == "reference"].beta, alternative="less")

p1 = prim_air[(prim_air.tier != "tailwater") | prim_air.site8.isin(both)].copy()
p1["air_cs"] = (p1.air_mwmt7 - p1.air_mwmt7.mean()) / 10.0
m1 = None
try:
    cand = smf.mixedlm("water_mwmt7 ~ air_cs * C(tier)", p1,
                       groups=p1.site8, re_formula="1 + air_cs")
    for meth in ["lbfgs", "bfgs", "powell"]:
        try:
            m1 = cand.fit(method=meth, disp=False, maxiter=2000)
            if m1.converged:
                break
        except Exception:
            continue
except Exception:
    pass

out = {"n_both": len(both), "twostage": {"tiers": t, "tw_lt_ref_p": float(mw.pvalue)}}
if m1 is not None and m1.converged:
    co = m1.params
    V = np.asarray(m1.cov_params())
    idx = list(co.index).index("air_cs:C(tier)[T.tailwater]")
    out["M1"] = {
        "n_gauges": int(p1.site8.nunique()), "n_obs": int(len(p1)),
        "beta_reference": round(float(co["air_cs"]) / 10, 3),
        "beta_tailwater": round(float(co["air_cs"] + co["air_cs:C(tier)[T.tailwater]"]) / 10, 3),
        "contrast_se_beta": round(float(np.sqrt(V[idx, idx])) / 10, 3),
        "contrast_p": float(m1.pvalues["air_cs:C(tier)[T.tailwater]"])}
C = json.load(open(V4 / "canonical.json"))
C["routing_verification"]["both_sources_sensitivity"] = out
(V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1))
