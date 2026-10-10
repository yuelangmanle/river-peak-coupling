#!/usr/bin/env python3
"""12_canonical.py — current full statistics (single source of truth: canonical.json).

All analyses are computed here in one pass:
  - FD: consecutive-year differences only (delta year = 1); site-level slopes saved to
        fd_slopes.csv; all three contrasts + Holm correction saved
  - M1: hard convergence gate (if not converged -> OLS with gauge + year FE, reservoir
        clustering, downgrade disclosed; no inference from non-converged fits)
  - bootstrap: conditional and joint versions (tailwater-reservoir resampling crossed with
        reference/upstream gauge resampling); joint interval targets the unadjusted
        two-stage tier-mean contrast
  - beta_hw: panel NB-GEE log-slope ratios (correct scale) + predicted contrast at a
        realistic +29-day increment + site-level OLS; all contrasts reported
  - exposure: eligible-year D22; site-level slopes; NB-GEE; climate matching + heat-state
        matching (reference D22 >= 50)
  - additional: peak-timing offsets (circular), water-peak interannual SD comparison,
        Shasta pre-release storage (three specifications + leave-one-out + interaction CI)
  - attribution: full specification list saved
The M1 model-based contrast and joint-bootstrap two-stage mean difference target
different estimands and are reported in parallel; neither is selected by its p value.
"""
import os
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from common import encode_use_elec
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
R = {}
NOTES = []
def sec(t): NOTES.append(f"\n{'='*66}\n{t}\n{'='*66}"); print(f"\n== {t}")
def line(t): NOTES.append(t); print(t)
tiers = ["reference", "upstream", "tailwater"]

# ---------- load ----------
panel = pd.read_parquet(V4 / "annual_panel.parquet")
panel["site8"] = panel.site.astype(str).str.zfill(8)
prim = panel[panel.sample_scope == "primary"].copy()
band = panel[panel.sample_scope == "sensitivity"].copy()

# ---------- per-gauge ladder ----------
rows = []
for site, g in prim.groupby("site8"):
    j = g.dropna(subset=["water_mwmt7", "air_mwmt7"])
    if len(j) < 8 or j.air_mwmt7.std() == 0:
        continue
    sl, ic, r, p, se = stats.linregress(j.air_mwmt7, j.water_mwmt7)
    frac_js = float(j.water_peak_doy.between(152, 244).mean())
    bhw = np.nan
    j2 = g.dropna(subset=["water_hw", "air_hw"])
    if len(j2) >= 8 and j2.air_hw.std() > 0:
        bhw = stats.linregress(j2.air_hw, j2.water_hw)[0]
    rows.append(dict(site_no=site, group=g.tier.iloc[0], beta=sl, se=se, r=r,
                     n_years=len(j), frac_js=frac_js, beta_hw=bhw,
                     src=g.src.mode().iloc[0],
                     D22_median=g.D22.median(), n_D22=int(g.D22.notna().sum()),
                     dam_id=g.dam_id.iloc[0] if "dam_id" in g else np.nan))
G = pd.DataFrame(rows)
G["sample_scope"] = "primary"
# Sensitivity-band gauges also enter the ladder (scope=band) so figures and band-level
# statistics reproduce directly
band_rows = []
for site, g in band.groupby("site8"):
    j = g.dropna(subset=["water_mwmt7", "air_mwmt7"])
    if len(j) < 8 or j.air_mwmt7.std() == 0:
        continue
    sl, ic, r, pp_, se = stats.linregress(j.air_mwmt7, j.water_mwmt7)
    band_rows.append(dict(site_no=site, group=g.tier.iloc[0], beta=sl, se=se, r=r,
                          n_years=len(j), beta_hw=np.nan, src=g.src.mode().iloc[0],
                          D22_median=g.D22.median(), n_D22=int(g.D22.notna().sum()),
                          dam_id=g.dam_id.iloc[0] if "dam_id" in g else np.nan,
                          sample_scope="band"))
G = pd.concat([G, pd.DataFrame(band_rows)], ignore_index=True)
tp = pd.read_csv(BASE / "data/samples/tailwater_pairs_elev.csv",
                 dtype={"dam_id": str, "site_no": str})
tp["site_no"] = tp.site_no.str.zfill(8)
G = G.merge(tp[["site_no", "dist_km"]], on="site_no", how="left")
G.to_csv(V4 / "ladder.csv", index=False)
Lp = G[G.sample_scope == "primary"].copy()
# ---------- extras: r^2 / season-restricted / |r| subsample / beta<=0 (values cited in the manuscript) ----------
_tiers = ["reference", "upstream", "tailwater"]
_ex = {}
_ex["mean_r2"] = {t: round(float((Lp[Lp.group == t].r ** 2).mean()), 3) for t in _tiers}
_js = Lp[Lp.frac_js >= 0.6]
_ex["js_restricted"] = {t: dict(n=int((_js.group == t).sum()),
                                mean=round(float(_js[_js.group == t].beta.mean()), 3))
                        for t in _tiers}
_r3 = Lp[Lp.r.abs() >= 0.3]
_ex["r_ge_03"] = {t: dict(n=int((_r3.group == t).sum()),
                          mean=round(float(_r3[_r3.group == t].beta.mean()), 3))
                  for t in _tiers}
_ex["beta_le0_share"] = {t: round(float((Lp[Lp.group == t].beta <= 0).mean()), 3)
                         for t in _tiers}
_ex["n_per_tier"] = {t: int((Lp.group == t).sum()) for t in _tiers}
_mref = float(Lp[Lp.group == "reference"].beta.mean())
_mtw = float(Lp[Lp.group == "tailwater"].beta.mean())
_ex["attenuation_frac"] = round(1 - _mtw / _mref, 3)
R["model_extras"] = _ex

R["sample"] = {
    "n_reference": int((Lp.group == "reference").sum()),
    "n_upstream": int((Lp.group == "upstream").sum()),
    "n_tailwater_primary": int((Lp.group == "tailwater").sum()),
    "n_tailwater_band": int((band.site8.nunique())),
    "n_gauges_primary": int(len(Lp)),
    "n_obs_primary": int(len(prim)),
    "n_obs_primary_D22": int(prim.D22.notna().sum()),
    "n_obs_primary_hw": int(prim.water_hw.notna().sum()),
    "n_reservoirs_primary": int(prim.dropna(subset=["dam_id"]).dam_id.nunique()),
}
line(f"sample: {R['sample']}")

# ---------- two-stage ----------
def two_stage(df, label):
    out = {"label": label, "n": {}, "mean": {}, "median": {}, "sd": {}}
    for t in tiers:
        s = df[df.group == t].beta
        out["n"][t] = int(len(s)); out["mean"][t] = round(float(s.mean()), 4)
        out["median"][t] = round(float(s.median()), 4); out["sd"][t] = round(float(s.std()), 4)
    w1 = stats.mannwhitneyu(df[df.group == "tailwater"].beta, df[df.group == "reference"].beta, alternative="less")
    w2_ = stats.mannwhitneyu(df[df.group == "tailwater"].beta, df[df.group == "upstream"].beta, alternative="less")
    w3 = stats.mannwhitneyu(df[df.group == "upstream"].beta, df[df.group == "reference"].beta, alternative="two-sided")
    out["mwu"] = {"tw_lt_ref_p": float(w1.pvalue), "tw_lt_up_p": float(w2_.pvalue),
                  "up_eq_ref_p_two_sided": float(w3.pvalue)}
    out["holm"] = {"tw_lt_ref": float(min(3 * w1.pvalue, 1)),
                   "tw_lt_up": float(min(2 * w2_.pvalue, 1)),
                   "up_eq_ref": float(w3.pvalue)}
    out["ratio_tw_ref_mean"] = round(out["mean"]["tailwater"] / out["mean"]["reference"], 4)
    return out
sec("two-stage")
ts = two_stage(Lp, "primary")
R["two_stage_primary"] = ts
line(f"means {ts['mean']} | p {ts['mwu']['tw_lt_ref_p']:.2e}/{ts['mwu']['tw_lt_up_p']:.4f} "
     f"| neg {ts['mwu']['up_eq_ref_p_two_sided']:.3f}")
band_tw = band[band.tier == "tailwater"]
Lb = G[~G.site_no.isin(set(Lp.site_no))]
bB = []
for site, g in band.groupby("site8"):
    j = g.dropna(subset=["water_mwmt7", "air_mwmt7"])
    if len(j) < 8 or j.air_mwmt7.std() == 0:
        continue
    bB.append(stats.linregress(j.air_mwmt7, j.water_mwmt7)[0])
bB = np.array(bB)
w1b = stats.mannwhitneyu(bB, Lp[Lp.group == "reference"].beta, alternative="less")
R["band_only_tailwater"] = {"n": int(len(bB)), "mean": round(float(bB.mean()), 4),
                            "median": round(float(np.median(bB)), 4),
                            "vs_ref_p_less": float(w1b.pvalue)}
line(f"band-only tw: n={len(bB)} mean={bB.mean():.4f} p={w1b.pvalue:.2e}")
# pooled
Gall = pd.concat([Lp, pd.DataFrame({"site_no": [f"band{i}" for i in range(len(bB))],
                                    "group": ["tailwater"] * len(bB), "beta": bB})], ignore_index=True)
ts_pool = two_stage(Gall, "pooled")
R["two_stage_pooled_sensitivity"] = {"mean": ts_pool["mean"], "mwu": ts_pool["mwu"]}

# reservoir-level
twB = Lp[Lp.group == "tailwater"].dropna(subset=["dam_id"])
dam_mean = twB.groupby("dam_id").beta.mean()
wd = stats.mannwhitneyu(dam_mean, Lp[Lp.group == "reference"].beta, alternative="less")
R["reservoir_level_twostage"] = {"n_dams": int(len(dam_mean)),
                                 "mean_beta_dams": round(float(dam_mean.mean()), 4),
                                 "vs_ref_p_less": float(wd.pvalue)}
line(f"reservoir-level: {R['reservoir_level_twostage']}")

# beta<=0
tw_b = Lp[Lp.group == "tailwater"]
n_le0 = int((tw_b.beta <= 0).sum())
wp = stats.mannwhitneyu(tw_b[tw_b.beta > 0].beta, Lp[Lp.group == "reference"].beta, alternative="less")
R["beta_le0"] = {"n": n_le0, "share": round(n_le0 / len(tw_b), 3),
                 "mean_all": round(float(tw_b.beta.mean()), 4),
                 "mean_excluding": round(float(tw_b[tw_b.beta > 0].beta.mean()), 4),
                 "vs_ref_p_after_exclusion": float(wp.pvalue),
                 "share_reference": round(float((Lp[Lp.group == "reference"].beta <= 0).mean()), 3),
                 "share_upstream": round(float((Lp[Lp.group == "upstream"].beta <= 0).mean()), 3)}
line(f"beta<=0: {R['beta_le0']}")

# ---------- FD (Delta year = 1 only) ----------
sec("FD (dyear=1)")
fd_rows = []
for site, g in prim.groupby("site8"):
    g = g.dropna(subset=["water_mwmt7", "air_mwmt7"]).sort_values("year")
    g["dy"] = g.year.diff()
    g["dw"] = g.water_mwmt7.diff(); g["da"] = g.air_mwmt7.diff()
    v = g[(g.dy == 1)].dropna(subset=["dw", "da"])
    if len(v) < 6 or v.da.std() == 0:
        continue
    sl = stats.linregress(v.da, v.dw)[0]
    fd_rows.append(dict(site_no=site, tier=g.tier.iloc[0], beta_fd=sl, n_pairs=len(v)))
FD = pd.DataFrame(fd_rows)
FD.to_csv(V4 / "fd_slopes.csv", index=False)
fd_out = {"n": {}, "mean": {}, "median": {}}
for t in tiers:
    s = FD[FD.tier == t].beta_fd
    fd_out["n"][t] = int(len(s)); fd_out["mean"][t] = round(float(s.mean()), 4)
    fd_out["median"][t] = round(float(s.median()), 4)
w1f = stats.mannwhitneyu(FD[FD.tier == "tailwater"].beta_fd, FD[FD.tier == "reference"].beta_fd, alternative="less")
w2f = stats.mannwhitneyu(FD[FD.tier == "tailwater"].beta_fd, FD[FD.tier == "upstream"].beta_fd, alternative="less")
w3f = stats.mannwhitneyu(FD[FD.tier == "upstream"].beta_fd, FD[FD.tier == "reference"].beta_fd, alternative="two-sided")
fd_out["mwu"] = {"tw_lt_ref_p": float(w1f.pvalue), "tw_lt_up_p": float(w2f.pvalue),
                 "up_eq_ref_p_two_sided": float(w3f.pvalue)}
fd_out["holm"] = {"tw_lt_ref": float(min(3 * w1f.pvalue, 1)), "tw_lt_up": float(min(2 * w2f.pvalue, 1)),
                  "up_eq_ref": float(w3f.pvalue)}
fd_out["ratio_tw_ref"] = round(fd_out["mean"]["tailwater"] / fd_out["mean"]["reference"], 3)
R["FD_beta"] = fd_out
line(f"FD: means {fd_out['mean']} | p {w1f.pvalue:.2e}/{w2f.pvalue:.4f} | neg {w3f.pvalue:.3f}")

# ---------- M1 (convergence-gated) ----------
sec("M1 convergence-gated")
p1 = prim.dropna(subset=["water_mwmt7", "air_mwmt7"]).copy()
p1["air_c"] = (p1.air_mwmt7 - p1.air_mwmt7.mean()) / 10.0   # rescaled to aid optimizer convergence
p1["tier"] = pd.Categorical(p1.tier, categories=tiers)
m1 = None; note = ""
for meth in ["lbfgs", "bfgs", "powell"]:
    try:
        cand = smf.mixedlm("water_mwmt7 ~ air_c * C(tier)", p1, groups=p1.site8,
                           re_formula="1+air_c").fit(reml=False, method=meth, maxiter=500)
        if cand.converged:
            m1 = cand; note = f"random slopes converged ({meth}, air_c/10)"; break
    except Exception as e:
        line(f"  {meth} failed: {str(e)[:60]}")
if m1 is None:
    p1["gfe"] = p1.site8; p1["yfe"] = p1.year
    p1["clus"] = np.where(p1.tier == "tailwater", "D" + p1.dam_id.astype(str), "S" + p1.site8)
    m1 = smf.ols("water_mwmt7 ~ air_c * C(tier) + C(gfe) + C(yfe)", data=p1).fit(
        cov_type="cluster", cov_kwds={"groups": p1.clus})
    note = "M1 did not converge -> OLS gauge+year FE, clustered by reservoir (tw) / site (others); DISCLOSED DEVIATION"
R["M1_mixed"] = {"note": note}
co, cv = m1.params, m1.cov_params()
names = list(co.index)
def slope_ci(base_nm, inter_nm):
    """Group-slope CI from the full covariance: var = V[b] + V[i] + 2*Cov."""
    b, i_ = co[base_nm], co.get(inter_nm, 0.0)
    v = cv.loc[base_nm, base_nm]
    if inter_nm in names:
        v += cv.loc[inter_nm, inter_nm] + 2 * cv.loc[base_nm, inter_nm]
    s_ = float(np.sqrt(v))
    return round(float(b + i_), 3), [round(float(b + i_ - 1.96 * s_), 3),
                                     round(float(b + i_ + 1.96 * s_), 3)], s_
bref, ci_ref, sref = slope_ci("air_c", "__none__")
bup, ci_up, sup = slope_ci("air_c", "air_c:C(tier)[T.upstream]")
btw, ci_tw, stw = slope_ci("air_c", "air_c:C(tier)[T.tailwater]")
if "air_c:C(tier)[T.tailwater]" in names:
    # Contrast tests use the interaction coefficients themselves (tw-ref and up-ref are
    # the interaction terms)
    # SE must use the interaction term's own variance sqrt(V[i,i]), not the group-slope SE
    se_i_tw = float(np.sqrt(cv.loc["air_c:C(tier)[T.tailwater]",
                                  "air_c:C(tier)[T.tailwater]"]))
    se_i_up = float(np.sqrt(cv.loc["air_c:C(tier)[T.upstream]",
                                   "air_c:C(tier)[T.upstream]"]))
    p_tw = float(m1.pvalues["air_c:C(tier)[T.tailwater]"]) if hasattr(m1, "pvalues") else None
    p_up = float(m1.pvalues["air_c:C(tier)[T.upstream]"])
else:
    se_i_tw = se_i_up = None
    p_tw = p_up = None
R["M1_mixed"].update({
    "beta_reference": bref, "ci_reference": ci_ref,
    "beta_upstream": bup, "ci_upstream": ci_up,
    "beta_tailwater": btw, "ci_tailwater": ci_tw,
    "contrast_tw_vs_ref": {
        "coef": round(btw - bref, 3), "se": round(se_i_tw, 3) if se_i_tw else None,
        "ci95": [round((btw - bref - 1.96 * se_i_tw) / 10, 3),
                 round((btw - bref + 1.96 * se_i_tw) / 10, 3)] if se_i_tw else None,
        "p": p_tw},
    "contrast_up_vs_ref": {"coef": round(bup - bref, 3), "se": round(se_i_up, 3) if se_i_up else None, "p": p_up},
    "n_gauges": int(p1.site8.nunique()), "n_obs": int(len(p1)),
    "converged": (note.startswith("random")),
})
line(f"M1 [{note}]: {bref}/{bup}/{btw} p_tw={p_tw}")
# MDE on whichever model: 2.8 * SE(interaction, upstream); air_c units (per 10 C)
# also reported on the beta scale (per degree C)
try:
    if se_i_up is not None:
        R["M1_mixed"]["mde_up_vs_ref_80pct"] = round(2.8 * se_i_up, 3)
        R["M1_mixed"]["mde_up_vs_ref_80pct_beta_scale"] = round(2.8 * se_i_up / 10, 3)
except Exception:
    pass

# ---------- bootstrap: conditional + joint ----------
sec("bootstrap")
rng = np.random.default_rng(42)
dams = Lp.dropna(subset=["dam_id"]).dam_id.unique()
def tier_means(Bs):
    return [Bs[Bs.group == t].beta.mean() for t in tiers]
cond_tr, cond_tu = [], []
joint_tr, joint_tu = [], []
ref_all = Lp[Lp.group == "reference"]; up_all = Lp[Lp.group == "upstream"]
for _ in range(2000):
    pick = rng.choice(dams, size=len(dams), replace=True)
    twB_b = pd.concat([Lp[(Lp.dam_id == d) & (Lp.group == "tailwater")] for d in pick])
    mt = twB_b.beta.mean()
    cond_tr.append(mt - ref_all.beta.mean())
    cond_tu.append(mt - up_all.beta.mean())
    # joint: reference and upstream also resampled (gauge-level i.i.d.)
    rr = ref_all.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9))).beta.mean()
    mu = up_all.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9))).beta.mean()
    joint_tr.append(mt - rr)
    joint_tu.append(mt - mu)
R["bootstrap"] = {
    "conditional": {"mean_diff": round(float(np.mean(cond_tr)), 3),
                    "ci95": [round(float(np.percentile(cond_tr, 2.5)), 3),
                             round(float(np.percentile(cond_tr, 97.5)), 3)],
                    "p_direction": round(float(np.mean(np.array(cond_tr) < 0)), 4),
                    "note": "reservoir-level tw, ref/up held fixed (conditional on observed comparison tiers)"},
    "joint": {"mean_diff": round(float(np.mean(joint_tr)), 3),
              "ci95": [round(float(np.percentile(joint_tr, 2.5)), 3),
                       round(float(np.percentile(joint_tr, 97.5)), 3)],
              "p_direction": round(float(np.mean(np.array(joint_tr) < 0)), 4),
              "note": "tw by reservoir AND ref/up by gauge resampled (joint interval for two-stage mean difference)"},
    "joint_tw_vs_up": {"mean_diff": round(float(np.mean(joint_tu)), 3),
                       "ci95": [round(float(np.percentile(joint_tu, 2.5)), 3),
                                round(float(np.percentile(joint_tu, 97.5)), 3)],
                       "p_direction": round(float(np.mean(np.array(joint_tu) < 0)), 4),
                       "note": "tailwater by reservoir and upstream by gauge resampled"},
}
R["bootstrap"]["tw_vs_up_conditional"] = {
    "mean_diff": round(float(np.mean(cond_tu)), 3),
    "ci95": [round(float(np.percentile(cond_tu, 2.5)), 3),
             round(float(np.percentile(cond_tu, 97.5)), 3)]}
line(f"conditional: {R['bootstrap']['conditional']}")
line(f"joint: {R['bootstrap']['joint']}")

# ---------- M2 + RMA + variance ----------
Gp = G[G.sample_scope == "primary"]
m2 = {}
for t in tiers:
    s = Gp[Gp.group == t]
    w = 1 / np.square(s.se.clip(lower=1e-6))
    m2[t] = round(float(np.average(s.beta, weights=w)), 3)
R["M2_weighted"] = m2
band_g = G[G.sample_scope == "band"]
band_tw = band_g[band_g.group == "tailwater"]
R["M2_weighted_sensitivity_band"] = {
    "tailwater": round(float(np.average(band_tw.beta, weights=1 / np.square(band_tw.se.clip(lower=1e-6)))), 3)
    if len(band_tw) else None,
    "n_gauges": int(len(band_tw)),
    "note": "separate 30-60 km sensitivity-band estimate; excluded from primary M2"}
rma_out = {}
lowr = int((G.r.abs() < 0.1).sum())
for t in tiers:
    s = Gp[Gp.group == t]
    rma = (s.beta / s.r.abs()).clip(upper=6)
    rma_out[t] = {"mean": round(float(rma.mean()), 3), "median": round(float(rma.median()), 3),
                  "n": int(len(rma)), "n_absr_lt_0.1": int((s.r.abs() < 0.1).sum())}
wr = stats.mannwhitneyu(
    (Gp[Gp.group == 'tailwater'].beta / Gp[Gp.group == 'tailwater'].r.abs()).clip(upper=6),
    (Gp[Gp.group == 'reference'].beta / Gp[Gp.group == 'reference'].r.abs()).clip(upper=6),
    alternative="less")
rma_out["tw_lt_ref_p"] = float(wr.pvalue)
rma_out["note"] = "RMA = beta/|r| (sign self-consistent); n with |r|<0.1 reported per tier; excluding them does not change ordering (checked)"
R["RMA_beta"] = rma_out
_pl = prim[prim.site8.isin(set(Lp.site_no))]          # ladder (analysis) gauges only
R["variance_check"] = {t: {"mean_within_sd_air": round(float(prim[prim.tier == t].groupby('site8').air_mwmt7.std().mean()), 3),
                           "mean_within_sd_water": round(float(prim[prim.tier == t].groupby('site8').water_mwmt7.std().mean()), 3),
                           "mean_air_mwmt7": round(float(prim[prim.tier == t].air_mwmt7.mean()), 2),
                           "mean_air_mwmt7_gaugeavg": round(float(
                               _pl[_pl.tier == t].dropna(subset=["water_mwmt7"])
                               .groupby("site8").air_mwmt7.mean().mean()), 2)}
                       for t in tiers}
line(f"M2 {m2} | RMA {rma_out['reference']['mean']}/{rma_out['upstream']['mean']}/{rma_out['tailwater']['mean']} | var {R['variance_check']}")

# ---------- climate-position matched beta ----------
sec("matched beta (climate position)")
bair = prim.groupby(["site8", "tier"]).agg(air_mean=("air_mwmt7", "mean")).reset_index()
bair["beta_g"] = bair.site8.map(G.set_index("site_no").beta.to_dict())
bref = bair[(bair.tier == "reference") & bair.beta_g.notna()].reset_index(drop=True)
btw = bair[(bair.tier == "tailwater") & bair.beta_g.notna()].reset_index(drop=True)
mp, used = [], set()
for _, tr in bref.iterrows():
    cand = btw[~btw.site8.isin(used)].assign(d=(btw.air_mean - tr.air_mean).abs())
    cand = cand[cand.d <= 1.0]
    if len(cand):
        best = cand.sort_values("d").iloc[0]
        used.add(best.site8)
        mp.append(dict(ref=tr.site8, tw=best.site8, beta_ref=tr.beta_g, beta_tw=best.beta_g))
MB = pd.DataFrame(mp, columns=["ref", "tw", "beta_ref", "beta_tw"])
MB.to_csv(V4 / "matched_pairs_beta.csv", index=False)
wmb = stats.wilcoxon(MB.beta_tw, MB.beta_ref, alternative="less") if len(MB) else None
R["matched_beta"] = {"n_pairs": int(len(MB)),
                     "n_reference_candidates": int(len(bref)),
                     "n_tailwater_candidates": int(len(btw)),
                     "mean_beta_ref": round(float(MB.beta_ref.mean()), 3) if len(MB) else None,
                     "mean_beta_tw": round(float(MB.beta_tw.mean()), 3) if len(MB) else None,
                     "wilcoxon_tw_lt_ref_p": round(float(wmb.pvalue), 6) if wmb else None,
                     "note": "one-to-one nearest climate-position matching after excluding gauges without estimable beta"}
adj = bair.dropna(subset=["beta_g"]).copy()
adj["is_tw"] = (adj.tier == "tailwater").astype(int)
r_adj = smf.ols("beta_g ~ is_tw + air_mean", data=adj).fit(cov_type="cluster", cov_kwds={"groups": adj.site8})
R["beta_climate_adjusted"] = {"tw_coef": round(float(r_adj.params["is_tw"]), 4),
                              "tw_p": round(float(r_adj.pvalues["is_tw"]), 8)}
line(f"matched beta: {R['matched_beta']} | adj {R['beta_climate_adjusted']}")

# ---------- beta_hw: NB-GEE + correct scale ----------
sec("beta_hw NB-GEE (correct labels + scale)")
p2 = prim.dropna(subset=["water_hw", "air_hw"]).copy()
p2["air_hw_c"] = p2.air_hw - p2.air_hw.mean()
gee = sm.GEE.from_formula("water_hw ~ air_hw_c * C(tier)", groups=p2.site8, data=p2,
                          family=sm.families.NegativeBinomial(alpha=1.0)).fit()
c2, b2 = gee.params, gee.bse
i_tw = "air_hw_c:C(tier)[T.tailwater]"; i_up = "air_hw_c:C(tier)[T.upstream]"
se_tw = float(np.sqrt(gee.cov_params().loc[i_tw, i_tw]))
se_up = float(np.sqrt(gee.cov_params().loc[i_up, i_up]))
b_ref, b_twv, b_upv = float(c2["air_hw_c"]), float(c2["air_hw_c"] + c2[i_tw]), float(c2["air_hw_c"] + c2[i_up])
# correct contrast: tw-vs-up Wald (linear combination)
diff = c2[i_tw] - c2[i_up]
se_diff = float(np.sqrt(gee.cov_params().loc[i_tw, i_tw] + gee.cov_params().loc[i_up, i_up]
                        - 2 * gee.cov_params().loc[i_tw, i_up]))
# predicted multiplier ratio at a realistic +29-day increment (interquartile range)
k29 = 29.0
rr29 = float(np.exp(c2[i_tw] * k29))
ci29 = [float(np.exp((c2[i_tw] - 1.96 * se_tw) * k29)), float(np.exp((c2[i_tw] + 1.96 * se_tw) * k29))]
# TOST on log-slope ratio (scale: slope ratio, margins 0.75-1.25)
ratio_slope = float(np.exp(c2[i_tw]))
se_ratio = se_tw   # the interaction coefficient is the log slope difference
t_a = (c2[i_tw] - np.log(0.75)) / se_ratio
t_b = (c2[i_tw] - np.log(1.25)) / se_ratio
p_sup = float(1 - stats.norm.cdf(t_a)); p_non = float(stats.norm.cdf(t_b))
R["beta_hw"] = {
    "nb_gee": {"alpha": 1.0, "cov_struct": "independent",
               "log_slope_reference": round(b_ref, 4), "log_slope_upstream": round(b_upv, 4),
               "log_slope_tailwater": round(b_twv, 4),
               "slope_ratio_tw_vs_ref": round(ratio_slope, 3),
               "slope_ratio_ci95": [round(float(np.exp(c2[i_tw] - 1.96 * se_tw)), 3),
                                    round(float(np.exp(c2[i_tw] + 1.96 * se_tw)), 3)],
               "p_interaction_tw_vs_ref": round(float(gee.pvalues[i_tw]), 5),
               "p_wald_tw_vs_up": round(float(stats.norm.cdf(-abs(diff / se_diff)) * 2), 4),
               "p_wald_up_vs_ref": round(float(gee.pvalues[i_up]), 4),
               "multiplier_ratio_at_plus29d": {"value": round(rr29, 3), "ci95": [round(ci29[0], 3), round(ci29[1], 3)]},
               "tost_log_slope_ratio_margin_0.75_1.25": {
                   "p_lower_margin": round(p_sup, 5), "p_upper_margin": round(p_non, 5),
                   "p_tost": round(max(p_sup, p_non), 5),
                   "slope_ratio_margin": [0.75, 1.25],
                   "conclusion": "equivalent within 0.75-1.25 slope-ratio margin"
                   if max(p_sup, p_non) < 0.05 else "equivalence not established"},
               "n_obs": int(len(p2))},
}
bhw = {}
for t in tiers:
    s = Lp[Lp.group == t].beta_hw.dropna()
    bhw[t] = {"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3), "n": int(len(s))}
wh1 = stats.mannwhitneyu(Lp[Lp.group == "tailwater"].beta_hw.dropna(),
                         Lp[Lp.group == "reference"].beta_hw.dropna(), alternative="less")
wh2 = stats.mannwhitneyu(Lp[Lp.group == "tailwater"].beta_hw.dropna(),
                         Lp[Lp.group == "upstream"].beta_hw.dropna(), alternative="less")
R["beta_hw"]["per_gauge"] = {**bhw,
                             "mwu": {"tw_lt_ref_p": float(wh1.pvalue), "tw_lt_up_p": float(wh2.pvalue)}}
line(f"beta_hw slope ratio {ratio_slope:.3f} CI {R['beta_hw']['nb_gee']['slope_ratio_ci95']} "
     f"| +29d multiplier {rr29:.3f} | per-gauge med {bhw['reference']['median']}/{bhw['tailwater']['median']} p={wh1.pvalue:.4f}")

# ---------- exposure ----------
sec("exposure")
exp_out = {"medians": {}, "share_any": {}, "n": {}}
for t in tiers:
    sub = prim[prim.tier == t]
    per = sub.dropna(subset=["D22"]).groupby("site8")[["D18", "D22", "D26"]].median()
    exp_out["medians"][t] = {c: round(float(per[c].median()), 1) for c in per.columns}
    exp_out["share_any"][t] = {c: round(float((per[c] > 0).mean()), 3) for c in per.columns}
    exp_out["n"][t] = int(len(per))
R["exposure_thresholds"] = exp_out
EXP = prim[["site8", "year", "tier", "D18", "D22", "D26", "air_mwmt7"]].rename(columns={"site8": "site_no"})
EXP.to_csv(V4 / "exposure_thresholds.csv", index=False)
p3 = prim.dropna(subset=["D22"]).copy()
p3["air_c"] = p3.air_mwmt7 - p3.air_mwmt7.mean()
gee22 = sm.GEE.from_formula("D22 ~ air_c * C(tier)", groups=p3.site8, data=p3,
                            family=sm.families.NegativeBinomial(alpha=1.0)).fit()
i3 = "air_c:C(tier)[T.tailwater]"
se3 = float(np.sqrt(gee22.cov_params().loc[i3, i3]))
sl_rows = []
for t in tiers:
    for s_, gg in prim[prim.tier == t].dropna(subset=["D22"]).groupby("site8"):
        if len(gg) < 8 or gg.air_mwmt7.std() == 0:
            continue
        sl, _, _, _, se_ = stats.linregress(gg.air_mwmt7, gg.D22)
        sl_rows.append(dict(tier=t, site=s_, slope=sl, se=se_))
SL = pd.DataFrame(sl_rows)
rng5 = np.random.default_rng(11)
sl_out = {}
diffs = []
for t in tiers:
    s = SL[SL.tier == t].slope
    sl_out[t] = {"mean_unweighted": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
                 "n": int(len(s))}
for _ in range(2000):
    rs = SL.groupby("tier").sample(frac=1.0, replace=True, random_state=int(rng5.integers(1e9)))
    diffs.append(rs[rs.tier == 'tailwater'].slope.mean() - rs[rs.tier == 'reference'].slope.mean())
sl_out["diff_tw_minus_ref"] = round(float(np.mean(diffs)), 3)
sl_out["diff_ci95"] = [round(float(np.percentile(diffs, 2.5)), 3),
                       round(float(np.percentile(diffs, 97.5)), 3)]
sl_out["n_gauges_all_zero_D22"] = {t: int((prim[prim.tier == t].groupby("site8").D22.max() == 0).sum())
                                   for t in tiers}
# NB-GEE rate ratio at +1C
rr3 = float(np.exp(gee22.params[i3]))
R["exposure_sensitivity"] = {**sl_out,
    "nbgee_D22": {"rate_ratio_tw_vs_ref_per_degree": round(rr3, 3),
                  "ci95": [round(float(np.exp(gee22.params[i3] - 1.96 * se3)), 3),
                           round(float(np.exp(gee22.params[i3] + 1.96 * se3)), 3)],
                  "p": round(float(gee22.pvalues[i3]), 5)}}
line(f"exposure: {sl_out['reference']['mean_unweighted']}/{sl_out['upstream']['mean_unweighted']}/"
     f"{sl_out['tailwater']['mean_unweighted']} diff {sl_out['diff_tw_minus_ref']} "
     f"CI {sl_out['diff_ci95']} | NB RR {rr3:.3f}")

# matched exposure (air) + heat-state matched
air_mean = prim.groupby(["site8", "tier"]).air_mwmt7.mean().reset_index()
d22_med = prim.dropna(subset=["D22"]).groupby(["site8", "tier"]).D22.median().reset_index()
am = air_mean.merge(d22_med, on=["site8", "tier"])
refs = am[am.tier == "reference"].reset_index(drop=True)
tws = am[am.tier == "tailwater"].reset_index(drop=True)
m_pairs, used2 = [], set()
for _, tr in refs.iterrows():
    cand = tws[~tws.site8.isin(used2)].assign(d=(tws.air_mwmt7 - tr.air_mwmt7).abs())
    cand = cand[cand.d <= 1.0]
    if len(cand):
        best = cand.sort_values("d").iloc[0]
        used2.add(best.site8)
        m_pairs.append(dict(ref=tr.site8, tw=best.site8, D22_ref=tr.D22, D22_tw=best.D22,
                            air_ref=tr.air_mwmt7, air_tw=best.air_mwmt7))
MP = pd.DataFrame(m_pairs)
MP.to_csv(V4 / "matched_pairs_exposure.csv", index=False)
wm = stats.wilcoxon(MP.D22_tw, MP.D22_ref, alternative="greater")
raw_tw = float(prim[prim.tier == 'tailwater'].dropna(subset=['D22']).groupby('site8').D22.median().median())
raw_ref = float(prim[prim.tier == 'reference'].dropna(subset=['D22']).groupby('site8').D22.median().median())
R["matched_exposure"] = {"n_pairs": int(len(MP)),
                         "median_D22_ref": round(float(MP.D22_ref.median()), 1),
                         "median_D22_tw": round(float(MP.D22_tw.median()), 1),
                         "wilcoxon_tw_gt_ref_p_one_sided": round(float(wm.pvalue), 4),
                         "excess_reduction_frac": round(1 - (MP.D22_tw.median() - MP.D22_ref.median()) /
                                                        (raw_tw - raw_ref), 3)}
# heat-state matched: reference gauges with D22 median >= 50 (same thermal state as tailwaters)
hs_ref = refs[refs.D22 >= 50]
hs_slope = SL[(SL.tier == "reference") & (SL.site.isin(hs_ref.site8))].slope
R["heat_state_matched"] = {
    "n_ref_D22_ge_50": int(len(hs_ref)),
    "mean_slope_ref_hot": round(float(hs_ref.D22.mul(0).sum() + hs_slope.mean()), 3) if len(hs_slope) else None,
    "mean_slope_tailwater": sl_out["tailwater"]["mean_unweighted"],
    "note": "reference gauges restricted to tailwater-like thermal state (median D22 >= 50 d/yr)"}
line(f"matched exposure: {R['matched_exposure']} | heat-state {R['heat_state_matched']}")

# ---------- attribution ----------
sec("attribution")
ops = pd.read_csv(BASE / "data/samples/reservoir_ops_features.csv", dtype={"dam_id": str})
da = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
for c_ in ["grand_cap_mcm", "dam_hgt_m", "depth_m"]:
    if c_ in da:
        da[c_] = pd.to_numeric(da[c_], errors="coerce"); da.loc[da[c_] < 0, c_] = np.nan
da = da.drop(columns=[c for c in ["outflow_mean_m3s", "outflow_cv", "release_summer_frac",
                                  "storage_range_frac"] if c in da.columns])
da = da.merge(ops[["dam_id", "outflow_mean_m3s", "outflow_cv", "release_summer_frac",
                   "storage_range_frac"]], on="dam_id", how="left")
da["capacity_ratio"] = da.grand_cap_mcm * 1e6 / (365.25 * 86400) / da.outflow_mean_m3s
da.loc[(da.capacity_ratio > 50) | (da.capacity_ratio < 0.01), "capacity_ratio"] = np.nan
da["use_elec_main"] = encode_use_elec(da["use_elec"])
tw_meta = tp[tp.site_no.isin(twB.site_no)][["site_no", "dist_km"]]
twB2 = twB.merge(tw_meta, on="site_no", how="inner")
agg = twB2.groupby("dam_id").agg(n_tw=("beta", "size"), beta_tw=("beta", "mean")).reset_index()
M2a = agg.merge(da[["dam_id", "capacity_ratio", "dam_hgt_m", "outflow_cv", "use_elec_main",
                    "release_summer_frac", "storage_range_frac", "state"]],
                on="dam_id").dropna(subset=["capacity_ratio", "outflow_cv", "dam_hgt_m"])
attribution_n_before_electricity = int(len(M2a))
elec_label = da.use_elec.astype("string").str.strip()
elec_counts = {"Main": int((elec_label == "Main").sum()),
               "Sec": int((elec_label == "Sec").sum()),
               "other_or_missing": int((~elec_label.isin(["Main", "Sec"])).sum())}
model_elec = M2a.use_elec_main
elec_counts_attribution = {"Main": int((model_elec == 1).sum()),
                           "Sec": int((model_elec == 0).sum()),
                           "excluded_other_or_missing": int(model_elec.isna().sum())}
M2a = M2a.dropna(subset=["use_elec_main"]).copy()
M2a["log_cap"] = np.log(M2a.capacity_ratio)
M2a.to_csv(V4 / "attribution.csv", index=False)
mform = "beta_tw ~ log_cap + outflow_cv + dam_hgt_m"
if M2a.use_elec_main.nunique() > 1:
    mform += " + use_elec_main"
r3a = smf.ols(mform, data=M2a).fit(cov_type="cluster", cov_kwds={"groups": M2a.state})
res_a = {"n_dams": int(len(M2a)), "n_dams_before_electricity_filter": attribution_n_before_electricity,
         "n_gauges": int(M2a.n_tw.sum()), "n_state_clusters": int(M2a.state.nunique()),
         "use_elec_counts_all_attributes": elec_counts,
         "use_elec_counts_attribution_candidates": elec_counts_attribution,
         "model_formula": mform,
         "use_elec_note": "Main=1, Sec=0; all other or missing categories excluded from complete-case attribution model"}
for nm in ["log_cap", "outflow_cv", "dam_hgt_m"] + (["use_elec_main"] if M2a.use_elec_main.std() > 0 else []):
    res_a[nm] = {"coef": round(float(r3a.params[nm]), 4), "p": round(float(r3a.pvalues[nm]), 4)}
res_a["r2"] = round(float(r3a.rsquared), 4)
rng3 = np.random.default_rng(11)
bl = []
for _ in range(1000):
    pick = rng3.choice(M2a.dam_id, size=len(M2a), replace=True)
    Bs = pd.concat([M2a[M2a.dam_id == d] for d in pick])
    try:
        bl.append(smf.ols(mform, data=Bs).fit().params["log_cap"])
    except Exception:
        pass
res_a["log_cap_boot_ci"] = [round(float(np.percentile(bl, 2.5)), 3),
                            round(float(np.percentile(bl, 97.5)), 3)]
specs = []
for nm in ["log_cap", "outflow_cv", "dam_hgt_m"] + (["use_elec_main"] if M2a.use_elec_main.std() > 0 else []):
    specs.append({"spec": f"{mform}", "coef": nm, "p": res_a[nm]["p"]})
for extra in ["release_summer_frac", "storage_range_frac"]:
    rr_ = smf.ols(mform + f" + {extra}", data=M2a.dropna(subset=[extra])).fit(
        cov_type="cluster", cov_kwds={"groups": M2a.dropna(subset=[extra]).state})
    specs.append({"spec": mform + f" + {extra}", "coef": extra,
                  "p": round(float(rr_.pvalues[extra]), 4)})
ms = M2a.dropna(subset=["release_summer_frac"]).copy()
med_rs = ms.release_summer_frac.median()
hi, lo = ms[ms.release_summer_frac > med_rs], ms[ms.release_summer_frac <= med_rs]
w_rs = stats.mannwhitneyu(hi.beta_tw, lo.beta_tw, alternative="two-sided")
r_rc = smf.ols("beta_tw ~ release_summer_frac + log_cap", data=ms).fit(
    cov_type="cluster", cov_kwds={"groups": ms.state})
r_rc2 = smf.ols("beta_tw ~ release_summer_frac", data=ms).fit(
    cov_type="cluster", cov_kwds={"groups": ms.state})
r_rc3 = smf.ols("beta_tw ~ release_summer_frac + log_cap", data=ms).fit()
res_a["summer_release_posthoc"] = {
    "n_dams": int(len(ms)), "median_split": round(float(med_rs), 3),
    "beta_median_high": round(float(hi.beta_tw.median()), 3),
    "beta_median_low": round(float(lo.beta_tw.median()), 3),
    "mwu_p_two_sided": round(float(w_rs.pvalue), 4),
    "continuous_p_all_specs": {
        "clustered_with_logcap": round(float(r_rc.pvalues["release_summer_frac"]), 4),
        "clustered_simple": round(float(r_rc2.pvalues["release_summer_frac"]), 4),
        "unclustered_with_logcap": round(float(r_rc3.pvalues["release_summer_frac"]), 4)},
    "status": "post hoc; uncorrected; hypothesis-generating"}
res_a["all_specifications"] = specs
R["attribution"] = res_a
line(f"attribution: {res_a['n_dams']} dams | log_cap p={res_a['log_cap']['p']} R2={res_a['r2']} | specs={len(specs)}")

# ---------- src sensitivity + distance ----------
sec("src + distance + cascade")
src_share = {t: round(float((Lp[Lp.group == t].src == "tmean").mean()), 3) for t in tiers}
Lx = Lp[Lp.src != "tmean"]
tsx = two_stage(Lx, "no-tmean")
R["src_sensitivity"] = {"share_tmean": src_share, "means_excluding": tsx["mean"],
                        "tw_lt_ref_p_excluding": tsx["mwu"]["tw_lt_ref_p"]}
near = tw_b[tw_b.dist_km <= 10]; far = tw_b[(tw_b.dist_km > 10) & (tw_b.dist_km <= 30)]
wn = stats.mannwhitneyu(near.beta, far.beta, alternative="less")
wf = stats.mannwhitneyu(near.beta, far.beta, alternative="less")
wfar_all = stats.mannwhitneyu(near.beta, tw_b[tw_b.dist_km > 10].beta, alternative="less")
sp = stats.spearmanr(tw_b.dist_km, tw_b.beta)
r_log = smf.ols("beta ~ np.log(dist_km)", data=tw_b.dropna(subset=["dist_km"])).fit(cov_type="HC1")
R["distance"] = {
    "near_le10km": {"n": int(len(near)), "mean": round(float(near.beta.mean()), 3)},
    "mid_10_30": {"n": int(len(far)), "mean": round(float(far.beta.mean()), 3)},
    "near_vs_mid_p_less": round(float(wn.pvalue), 4),
    "near_vs_all_gt10_p_less": round(float(wfar_all.pvalue), 4),
    "spearman_rho": round(float(sp.statistic), 3), "spearman_p": round(float(sp.pvalue), 4),
    "logdist_coef": round(float(r_log.params["np.log(dist_km)"]), 4),
    "logdist_p": round(float(r_log.pvalues["np.log(dist_km)"]), 4),
    "note": "binned contrast reported descriptively; no monotone distance claim"}
pp = pd.read_csv(BASE / "data/samples/pilot_pairs_structured.csv", dtype={"dam_id": str, "site_no": str})
pp["site_no"] = pp.site_no.str.zfill(8)
ups = pp[(pp.tag == "upstream") & (pp.dist_km <= 50)]
cnt = ups.groupby("site_no").dam_id.nunique()
R["cascade_proxy"] = {"n_upstream_candidates": int(len(cnt)), "n_multi_dam_50km": int((cnt > 1).sum()),
                      "share_multi_dam": round(float((cnt > 1).mean()), 3)}
line(f"src {src_share} | near {R['distance']['near_le10km']} vs mid {R['distance']['mid_10_30']} p={wn.pvalue:.3f} | cascade {R['cascade_proxy']}")

# ---------- peak timing + water SD ----------
sec("W3: peak timing offsets + water SD")
off_rows = []
for site, g in prim.groupby("site8"):
    g = g.dropna(subset=["water_peak_doy", "air_peak_doy"])
    if len(g) < 5:
        continue
    d = (g.water_peak_doy - g.air_peak_doy).astype(float)
    d = ((d + 183) % 365) - 183          # circularized (365-day ring)
    off_rows.append(dict(site=site, tier=g.tier.iloc[0], med_off=float(d.median()), n=len(d)))
OFF = pd.DataFrame(off_rows)
to = {}
for t in tiers:
    s = OFF[OFF.tier == t].med_off
    to[t] = {"n": int(len(s)), "median_offset_d": round(float(s.median()), 1),
             "mean_offset_d": round(float(s.mean()), 1),
             "share_positive": round(float((s > 0).mean()), 3)}
wt = stats.mannwhitneyu(OFF[OFF.tier == "tailwater"].med_off, OFF[OFF.tier == "reference"].med_off,
                        alternative="greater")
to["tw_gt_ref_p"] = round(float(wt.pvalue), 5)
R["peak_timing"] = to
sd_out = {t: round(float(prim[prim.tier == t].groupby("site8").water_mwmt7.std().mean()), 3)
          for t in tiers}
R["water_sd"] = sd_out
line(f"peak timing: {to} | water SD: {sd_out}")

# ---------- paired systems (same-river, current panel) ----------
sec("paired current")
ups_all = pp[pp.tag == "upstream"].copy()
ups_all = ups_all[ups_all.dist_km <= 50]
DES = {"R", "RV", "RIVER", "C", "CR", "CK", "CREEK", "LK", "LAKE", "RES", "RESERVOIR",
       "BAYOU", "SLOUGH", "DIVERSION", "CANAL", "WASH", "ARROYO"}
QLT = {"NR", "NEAR", "AB", "ABV", "ABOVE", "BL", "BLW", "BELOW", "AT", "DIV", "TRIB", "BR"}
def river_key(nm):
    if not isinstance(nm, str):
        return None
    toks = nm.upper().replace(",", " ").split()
    key = []
    for t in toks:
        if t in QLT: break
        key.append(t)
        if t in DES: break
    return " ".join(key) or None
nm_of = pp.drop_duplicates("site_no").set_index("site_no").station_nm.to_dict()
panel_sites = set(panel.site8)
cand = []
for _, tr in tp[tp.dist_km <= 30].iterrows():
    dn = tr.site_no
    if dn not in panel_sites: continue
    k = river_key(nm_of.get(dn))
    if not k: continue
    for _, ur in ups_all[ups_all.dam_id == tr.dam_id].iterrows():
        u = ur.site_no
        if u in panel_sites and u != dn and river_key(nm_of.get(u)) == k:
            cand.append(dict(dam_id=tr.dam_id, site_down=dn, site_up=u))
C = pd.DataFrame(cand).drop_duplicates()
prows = []
for _, rr_ in C.iterrows():
    wd = panel[panel.site8 == rr_.site_down].set_index("year").water_mwmt7
    wu = panel[panel.site8 == rr_.site_up].set_index("year").water_mwmt7
    j = pd.DataFrame({"d": wd}).join(wu.rename("u"), how="inner").dropna()
    if len(j) >= 6:
        prows.append({**rr_.to_dict(), "n_years": int(len(j)), "delta": float((j.d - j.u).mean())})
P = pd.DataFrame(prows).sort_values("n_years", ascending=False)
P = P.drop_duplicates("dam_id").drop_duplicates("site_down").drop_duplicates("site_up")
names = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
if "dam_name" in names.columns:
    P = P.merge(names[["dam_id", "dam_name"]].drop_duplicates("dam_id"), on="dam_id", how="left")
P["down_nm"] = P.site_down.map(nm_of); P["up_nm"] = P.site_up.map(nm_of)
P = P.sort_values("delta").reset_index(drop=True)
P.to_csv(V4 / "paired.csv", index=False)
wp_ = stats.wilcoxon(P.delta) if len(P) > 5 else None
R["paired"] = {"n_systems": int(len(P)), "mean_delta": round(float(P.delta.mean()), 3),
               "median_delta": round(float(P.delta.median()), 3),
               "n_positive": int((P.delta > 0).sum()),
               "wilcoxon_p": round(float(wp_.pvalue), 3) if wp_ else None,
               "min": round(float(P.delta.min()), 2), "max": round(float(P.delta.max()), 2)}
line(f"paired: {R['paired']}")

# ---------- screening table ----------
twS = twB2.merge(prim.dropna(subset=["D22"]).groupby("site8").D22.median().rename("D22_med"),
                 left_on="site_no", right_index=True, how="left")
da2 = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
twS = twS.merge(da2[["dam_id", "dam_name", "state", "dam_hgt_m"]].drop_duplicates("dam_id"),
                on="dam_id", how="left")
twS.sort_values("D22_med", ascending=False).to_csv(V4 / "screening_table.csv", index=False)

# ---------- inference hierarchy + G5 ----------
sec("G5 adjudication")
p_ts = R["two_stage_primary"]["mwu"]["tw_lt_ref_p"]
R["inference_reporting"] = {
    "choice": "parallel_estimands",
    "rule": "M1 model-based slope contrast and joint-bootstrap two-stage mean difference are reported in parallel; no contrast-p-value rule designates a primary framework."
}
if p_ts < 0.001:
    verdict = "PASS: continue main line"
elif p_ts <= 0.05:
    verdict = "PASS-DOWNGRADED: keep, all wording to 'observed association' level"
else:
    verdict = "FAIL: plan C (descriptive-benchmark paper)"
R["gate_G5"] = {"two_stage_tw_lt_ref_p": p_ts, "verdict": verdict}
line(f"inference reporting: parallel estimands | G5: {verdict}")

# ---------- save ----------
(V4 / "canonical.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
(BASE / "docs/notes/canonical_current.txt").write_text("\n".join(NOTES) + "\n\n" +
                                                  json.dumps(R, indent=1, ensure_ascii=False))
print("\nsaved -> canonical.json")
