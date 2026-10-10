#!/usr/bin/env python3
"""14_figures.py — current six-figure set (Okabe-Ito palette; all numbers from canonical_current / current data files)."""
import os
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
FIG = BASE / "figures"
R = json.load(open(V4 / "canonical.json"))
SH = json.load(open(V4 / "shasta_numbers.json"))
TIER_C = {"reference": "#009E73", "upstream": "#E69F00", "tailwater": "#0072B2"}
TIER_LABEL = {"reference": "Reference", "upstream": "Above-reservoir", "tailwater": "Tailwater"}
order = ["reference", "upstream", "tailwater"]
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42})

def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("saved", name)

L = pd.read_csv(V4 / "ladder.csv", dtype={"site_no": str})
L["site_no"] = L.site_no.str.zfill(8)
panel = pd.read_parquet(V4 / "annual_panel.parquet")
prim = panel[panel.sample_scope == "primary"].copy()
prim["site8"] = prim.site.astype(str).str.zfill(8)

# ---------- Fig 1 ----------
states = gpd.read_file(BASE / "data/raw/us_states.json")
conus_names = {"Washington","Oregon","California","Nevada","Idaho","Montana","Wyoming","Utah",
    "Colorado","Arizona","New Mexico","Texas","Oklahoma","Kansas","Nebraska","South Dakota",
    "North Dakota","Minnesota","Iowa","Missouri","Arkansas","Louisiana","Mississippi","Alabama",
    "Tennessee","Kentucky","Georgia","Florida","South Carolina","North Carolina","Virginia",
    "West Virginia","Ohio","Indiana","Illinois","Wisconsin","Michigan","Pennsylvania","New York",
    "Vermont","New Hampshire","Maine","Massachusetts","Rhode Island","Connecticut","New Jersey",
    "Delaware","Maryland","District of Columbia"}
conus = states[states.name.isin(conus_names)]
dams = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
dams["lat"] = pd.to_numeric(dams.lat, errors="coerce"); dams["lon"] = pd.to_numeric(dams.lon, errors="coerce")
dams = dams.dropna(subset=["lat","lon"])
dams = dams[(dams.lat > 23) & (dams.lat < 50) & (dams.lon > -125) & (dams.lon < -66)]
coords = pd.read_csv(V4 / "gauge_coords.csv", dtype={"site_no": str})
Lc = L.merge(coords, on="site_no", how="left")
band_sites = set(L.loc[L.sample_scope == "band", "site_no"])

fig = plt.figure(figsize=(7.2, 3.55))
gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1], wspace=0.25)
fig.subplots_adjust(bottom=0.28)
ax = fig.add_subplot(gs[0])
conus.boundary.plot(ax=ax, color="0.78", linewidth=0.5)
ax.scatter(dams.lon, dams.lat, marker="+", s=3, c="0.6", linewidths=0.4, zorder=2,
           label=f"ResOpsUS reservoirs (n={len(dams)})")
for t in order:
    sub = Lc[Lc.group == t]
    p = sub[~sub.site_no.isin(band_sites)]
    ax.scatter(p.lon, p.lat, s=9, c=TIER_C[t], zorder=3,
               label=f"{TIER_LABEL[t]} (n={len(p)})")
b = Lc[Lc.site_no.isin(band_sites)]
ax.scatter(b.lon, b.lat, s=10, facecolors="none", edgecolors=TIER_C["tailwater"],
           linewidths=0.7, zorder=3, label=f"Tailwater 30\u201360 km band (n={len(b)})")
ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(loc="upper left", bbox_to_anchor=(0.01, -0.20), frameon=False,
          handletextpad=0.25, borderaxespad=0.1, fontsize=6.5, ncol=2)
ax.set_title("(a)", loc="left", fontsize=9)

lad_sites = set(L.loc[L.sample_scope == "primary", "site_no"])
prim_f = prim[prim.site8.isin(lad_sites) & prim.water_mwmt7.notna()]
air_mean = prim_f.groupby(["site8", "tier"]).air_mwmt7.mean().reset_index()
ax2 = fig.add_subplot(gs[1])
data_b = [air_mean[air_mean.tier == t].air_mwmt7 for t in order]
bp = ax2.boxplot(data_b, tick_labels=["Reference", "Above-\nreservoir", "Tailwater"],
                 widths=0.55, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", lw=1.2))
for patch, t in zip(bp["boxes"], order):
    patch.set_facecolor(TIER_C[t]); patch.set_alpha(0.55); patch.set_edgecolor("0.3")
for i, t in enumerate(order):
    m = air_mean[air_mean.tier == t].air_mwmt7.mean()
    ax2.scatter(i + 1, m, marker="D", s=18, color="black", zorder=5)
    ax2.annotate(f"{m:.1f}", (i + 1.2, m), fontsize=7, va="center")
ax2.set_ylim(21, 42)
ax2.set_ylabel("Gauge-mean annual peak air temperature (°C)")
ax2.set_title("(b)", loc="left", fontsize=9)
save(fig, "fig1_design")

# ---------- Fig 2 (coupling distribution and peak timing) ----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.2))
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.92, wspace=0.52)
rng = np.random.default_rng(3)
for i, t in enumerate(order):
    s = L[(L.group == t) & (L.sample_scope == "primary")].beta.values
    parts = axa.violinplot([s], positions=[i], widths=0.75, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(TIER_C[t]); pc.set_alpha(0.45); pc.set_edgecolor("none")
    x = rng.uniform(i - 0.16, i + 0.16, len(s))
    axa.scatter(x, s, s=4, c="black", alpha=0.35, linewidths=0, zorder=3)
    med, mean = np.median(s), np.mean(s)
    axa.hlines(med, i - 0.3, i + 0.3, color="crimson", lw=1.6, zorder=4)
    axa.scatter(i, mean, marker="D", s=22, c="white", edgecolors="black", lw=0.8, zorder=5)
axa.axhline(1.0, color="0.45", ls="--", lw=0.9)
axa.annotate("\u03b2 = 1 (unit-slope reference)", (0.02, 1.08), xycoords=("axes fraction", "data"),
             fontsize=6.8, color="0.35")
axa.set_ylim(-0.65, 1.72)
axa.set_xticks(range(3))
axa.set_xticklabels([f"{TIER_LABEL[t]}\n(n={int(((L.group==t)&(L.sample_scope=='primary')).sum())})" for t in order])
axa.set_ylabel("Air\u2013water peak coupling \u03b2")
axa.set_title("(a)", loc="left", fontsize=9)

OFF = []
for site, g in prim.groupby("site8"):
    g = g.dropna(subset=["water_peak_doy", "air_peak_doy"])
    if len(g) < 8: continue
    d = (g.water_peak_doy - g.air_peak_doy).astype(float)
    OFF.append(dict(tier=g.tier.iloc[0], med=float((((d + 183) % 365) - 183).median())))
OFF = pd.DataFrame(OFF)
PT = {t: R["peak_timing"][t] for t in order}
data_c = [OFF[OFF.tier == t].med for t in order]
bpc = axb.boxplot(data_c, tick_labels=["Reference", "Above-\nreservoir", "Tailwater"],
                  widths=0.55, patch_artist=True, showfliers=False,
                  medianprops=dict(color="black", lw=1.2))
for patch, t in zip(bpc["boxes"], order):
    patch.set_facecolor(TIER_C[t]); patch.set_alpha(0.55); patch.set_edgecolor("0.3")
for i, t in enumerate(order):
    m = OFF[OFF.tier == t].med.mean()
    m = 0.0 if abs(float(m)) < 0.05 else float(m)
    axb.scatter(i + 1, m, marker="D", s=18, color="black", zorder=5)
    axb.annotate(f"{m:.1f} d", (i + 1.18, m), fontsize=7, va="center",
                 bbox=dict(fc="white", ec="none", pad=0.2))
meds = " / ".join(
    f"{(0.0 if abs(float(PT[t]['median_offset_d'])) < 0.05 else float(PT[t]['median_offset_d'])):.1f}"
    for t in order
)
axb.annotate(f"medians {meds} d", (0.98, 0.16), xycoords="axes fraction",
             ha="right", fontsize=6.2, color="0.35")
axb.axhline(0, color="0.6", lw=0.7, ls=":")
axb.set_ylabel("Water-minus-air peak date offset (days)")
axb.set_title("(b)", loc="left", fontsize=9)
axb.annotate("diamonds: tier means", (0.98, 0.08), xycoords="axes fraction",
             ha="right", fontsize=6.2, color="0.4")
axb.set_ylim(-13.5, 22)
save(fig, "fig2_ladder")

# ---------- Fig 3 ----------
M1 = R["M1_mixed"]; TS = R["two_stage_primary"]; FD = R["FD_beta"]; M2 = R["M2_weighted"]
rng = np.random.default_rng(42)
dams_u = L.dropna(subset=["dam_id"]).dam_id.unique()
ci = {}
for t in order:
    bs = []
    if t == "tailwater":
        for _ in range(2000):
            pick = rng.choice(dams_u, size=len(dams_u), replace=True)
            bs.append(pd.concat([L[(L.dam_id == d) & (L.group == t)] for d in pick]).beta.mean())
    else:
        s = L[L.group == t].beta.values
        for _ in range(2000):
            bs.append(s[rng.integers(0, len(s), len(s))].mean())
    ci[t] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
FDdf = pd.read_csv(V4 / "fd_slopes.csv")
fd_se = {t: float(FDdf[FDdf.tier == t].beta_fd.std() / np.sqrt((FDdf.tier == t).sum())) for t in order}
frameworks = [
    ("Mixed model", {t: (M1[f"beta_{t}"] / 10, M1[f"ci_{t}"][0] / 10, M1[f"ci_{t}"][1] / 10) for t in order}),
    ("Two-stage mean", {t: (TS["mean"][t], ci[t][0], ci[t][1]) for t in order}),
    ("First difference", {t: (FD["mean"][t], FD["mean"][t] - 1.96 * fd_se[t],
                              FD["mean"][t] + 1.96 * fd_se[t]) for t in order}),
    ("Precision-weighted", {t: (M2[t], np.nan, np.nan) for t in order}),
]
fig, ax = plt.subplots(figsize=(7.0, 3.0))
y_base = {fw: i * 1.0 for i, (fw, _) in enumerate(frameworks)}
off = {"reference": 0.24, "upstream": 0.0, "tailwater": -0.24}
for fw, est in frameworks:
    for t in order:
        v, lo_, hi_ = est[t]
        y = y_base[fw] + off[t]
        if np.isnan(lo_):
            ax.scatter(v, y, s=20, color=TIER_C[t], zorder=3)
        else:
            ax.hlines(y, lo_, hi_, color=TIER_C[t], lw=1.4, zorder=2)
            ax.scatter(v, y, s=22, color=TIER_C[t], zorder=3)
ax.axvline(0, color="0.6", lw=0.8)
ax.set_yticks([y_base[fw] for fw, _ in frameworks])
ax.set_yticklabels([fw for fw, _ in frameworks])
ax.invert_yaxis()
ax.set_xlabel("Tier slope \u03b2 (95% CI where available)")
handles = [Line2D([], [], color=TIER_C[t], marker="o", lw=1.4, label=TIER_LABEL[t]) for t in order]
ax.legend(handles=handles, loc="lower right", frameon=False)
save(fig, "fig3_frameworks")

# ---------- Fig 4 ----------
P = pd.read_csv(V4 / "paired.csv", dtype={"site_down": str, "site_up": str})
P["site_down"] = P.site_down.str.zfill(8); P["site_up"] = P.site_up.str.zfill(8)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
med_of = panel.groupby(panel.site.astype(str).str.zfill(8)).water_mwmt7.median()
lab_rows = []
for _, r in P.iterrows():
    c = "#C44E52" if r.delta > 0 else "#0072B2"
    u, d = med_of.get(r.site_up, np.nan), med_of.get(r.site_down, np.nan)
    ax.plot([0, 1], [u, d], color=c, lw=1.3, alpha=0.85)
    ax.scatter([0, 1], [u, d], s=18, color=c, zorder=3)
    lab_rows.append({"dam": r.dam_name, "delta": r.delta, "y": d, "c": c, "n": r.n_years})
lab_rows = sorted(lab_rows, key=lambda z: z["y"])
for i in range(1, len(lab_rows)):
    if lab_rows[i]["y"] - lab_rows[i - 1]["y"] < 1.15:
        lab_rows[i]["y"] = lab_rows[i - 1]["y"] + 1.15
for lr in lab_rows:
    ax.annotate(f"{lr['dam']} ({lr['delta']:+.1f}, {lr['n']} yr)", (1.02, lr["y"]),
                fontsize=6.3, color=lr["c"], va="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Upstream of dam", "Downstream (tailwater)"])
ax.set_xlim(-0.15, 1.85)
ax.set_ylabel("Mean annual AMWT7 (\u00b0C)")
save(fig, "fig4_paired")

# ---------- Fig 5 ----------
EX = R["exposure_thresholds"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
x = np.arange(3); w = 0.26
for k, (thr, c) in enumerate([("D18", "#9ecae1"), ("D22", "#4292c6"), ("D26", "#08519c")]):
    vals = [EX["medians"][t][thr] for t in order]
    axa.bar(x + (k - 1) * w, vals, width=w * 0.92, color=c, label=f">{thr[1:]} \u00b0C",
            edgecolor="white", lw=0.4)
    for xi, v in zip(x + (k - 1) * w, vals):
        axa.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=6)
axa.set_xticks(x)
axa.set_xticklabels([f"{TIER_LABEL[t]}\n(n={EX['n'][t]})" for t in order])
axa.set_ylabel("Median days per year above threshold")
axa.legend(frameon=False, loc="upper left")
axa.set_title("(a)", loc="left", fontsize=9)

rng5 = np.random.default_rng(11)
def boot_slope(tier, sites=None):
    sl = []
    sub = pm0 if False else prim
    sub = sub[sub.tier == tier].dropna(subset=["D22"])
    if sites is not None:
        sub = sub[sub.site8.isin(sites)]
    for site, gg in sub.groupby("site8"):
        if len(gg) < 8 or gg.air_mwmt7.std() == 0: continue
        sl.append(stats.linregress(gg.air_mwmt7, gg.D22)[0])
    sl = np.array(sl)
    boots = [sl[rng5.integers(0, len(sl), len(sl))].mean() for _ in range(2000)]
    lo_, hi_ = np.percentile(boots, [2.5, 97.5])
    return sl.mean(), lo_, hi_
for i, t in enumerate(order):
    m, lo_, hi_ = boot_slope(t)
    axb.errorbar(i, m, yerr=[[m - lo_], [hi_ - m]], fmt="o", ms=6, color=TIER_C[t],
                 capsize=3, lw=1.3)
    axb.annotate(f"{m:.1f}", (i + 0.08, m), fontsize=7.5, va="center")
hot = prim[prim.tier == "reference"].dropna(subset=["D22"]).groupby("site8").D22.median()
hot = set(hot[hot >= 50].index)
m, lo_, hi_ = boot_slope("reference", hot)
axb.errorbar(3.0, m, yerr=[[m - lo_], [hi_ - m]], fmt="o", ms=7, color=TIER_C["reference"],
             capsize=3, lw=1.3, mfc="white")
axb.annotate(f"{m:.1f}", (3.08, m), fontsize=7.5, va="center")
axb.axhline(0, color="0.7", lw=0.7)
axb.set_xticks([0, 1, 2, 3])
axb.set_xticklabels(["Reference", "Above-\nreservoir", "Tailwater", "Reference\n(hot-matched)"])
axb.set_ylabel("Marginal exposure sensitivity\n(additional days > 22 \u00b0C per \u00b0C)")
axb.set_title("(b)", loc="left", fontsize=9)
save(fig, "fig5_exposure")

# ---------- Fig 6 ----------
S = pd.read_csv(V4 / "shasta_annual.csv")
S["wpk"] = pd.to_datetime(S.water_peak)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
may = S[S.wpk.dt.month == 5]
axa.plot(S.year, S.air_mwmt7, "-s", color="#E69F00", ms=3, lw=1.0, label="Air peaks (Daymet)")
axa.plot(S.year, S.water_mwmt7, "-o", color="#0072B2", ms=3.5, lw=1.3, label="Water peaks (tailwater)")
axa.scatter(may.year, may.water_mwmt7, s=30, color="#D55E00", zorder=6)
axa.set_ylabel("Annual peak temperature (AMWT7, \u00b0C)")
axa.set_ylim(8, 50)
axa.set_xticks([2006, 2010, 2014, 2018, 2022])
axa.legend(loc="center right", frameon=False, fontsize=6.8)
axa.set_title("(a)", loc="left", fontsize=9)
axa.annotate("orange: May peaks (cold-release years)", (0.98, 0.02),
             xycoords="axes fraction", ha="right", fontsize=6.3, color="0.4")

pv = SH["storage_variants"]["peak_minus_30d_mean"]
S2 = S.dropna(subset=["stor_peak30"])
axb.scatter(S2.stor_peak30, S2.water_mwmt7, s=22,
            c=["#D55E00" if y in set(may.year) else "#0072B2" for y in S2.year], zorder=3)
xx = np.linspace(S2.stor_peak30.min() - 0.05, S2.stor_peak30.max() + 0.1, 50)
bb = np.polyfit(S2.stor_peak30, S2.water_mwmt7, 1)
axb.plot(xx, np.polyval(bb, xx), color="black", lw=1.1)
for _, r in S2.iterrows():
    if r.year in (2014, 2017):
        axb.annotate(str(int(r.year)), (r.stor_peak30, r.water_mwmt7),
                     xytext=(5, -10 if r.water_mwmt7 > 15 else 5),
                     textcoords="offset points", fontsize=6.5, color="0.3")
axb.set_xlabel("Antecedent storage, 30 d (MAF)")
axb.set_ylabel("Tailwater AMWT7 (\u00b0C)")
axb.set_title("(b)", loc="left", fontsize=9)
axb.annotate(f"r = {pv['r_water_storage']:.2f}\nslope = {bb[0]:.1f} \u00b0C per MAF",
             (0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", fc="0.94", ec="0.8"))
save(fig, "fig6_shasta")

# ---------- Fig 7 (multidam forest, Fisher-z CIs) ----------
M = pd.read_csv(V4/'multidam_results.csv', dtype={'dam_id':str}).copy()
M['z'] = np.arctanh(M.rS_partial.clip(-0.999, 0.999))
M['zse'] = 1/np.sqrt(M.n_years-3)
M['lo'] = np.tanh(M.z - 1.96*M.zse); M['hi'] = np.tanh(M.z + 1.96*M.zse)
w = 1/(1/(M.n_years-3))
zw = float((M.z*w).sum()/w.sum())
q = float(((M.z-zw)**2*w).sum()); c = w.sum()-(w**2).sum()/w.sum()
tau2 = max(0.0, (q-(len(M)-1))/c)
se_re = float(np.sqrt(1/w.sum() + tau2/len(M)))
pm, pmlo, pmhi = float(np.tanh(zw)), float(np.tanh(zw-1.96*se_re)), float(np.tanh(zw+1.96*se_re))
M = M.sort_values('rS_partial')
fig, ax = plt.subplots(figsize=(6.4, 7.2))
y = np.arange(len(M))
colors = ['#0072B2' if v<0 else '#D55E00' for v in M.rS_partial]
ax.errorbar(M.rS_partial, y, xerr=[M.rS_partial-M.lo, M.hi-M.rS_partial], fmt='none',
            ecolor='0.6', elinewidth=0.8, capsize=1.5, zorder=2)
ax.scatter(M.rS_partial, y, s=22, c=colors, zorder=3)
ax.axvline(0, color='0.5', lw=0.9)
ax.axvline(pm, color='crimson', lw=1.2, ls='--')
ax.axvspan(pmlo, pmhi, color='crimson', alpha=0.12, zorder=1)
ax.set_ylim(-1, len(M) + 1.8)
ax.annotate(f'pooled {pm:+.2f} (random-effects Fisher-z 95% CI {pmlo:+.2f} to {pmhi:+.2f})',
            (0.02, 1.015), xycoords='axes fraction', fontsize=7, color='crimson', va='bottom', clip_on=False)
ax.set_yticks(y)
ax.set_yticklabels([f"{n} ({d:.0f} m)" for n, d in zip(M.dam_name, M.depth_m)], fontsize=6.5)
ax.set_xlabel('Partial correlation: tailwater peak vs antecedent storage\n(controlling air peak; Fisher-z 95% CIs; label: reservoir mean depth)')
ax.set_xlim(-1.05, 1.05)
save(fig, "fig7_multidam")

# ---------- Fig 8 (driver substitution + downstream re-coupling) ----------
Dd = pd.read_csv(V4/'driver_substitution.csv')
twr = L[(L.group == "tailwater") & L.dist_km.notna() & (L.dist_km > 0) & (L.dist_km <= 60)].dropna(subset=["beta"])
twr["sample_scope"] = twr.sample_scope.fillna("primary").replace({"band": "sensitivity"})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.2))
fig.subplots_adjust(left=0.19, right=0.99, bottom=0.20, top=0.92, wspace=0.42)
mx = max(Dd.r2_storage.max(), Dd.r2_air.max()) * 1.05
axa.plot([0, mx], [0, mx], color="0.6", lw=0.9, ls=":")
axa.scatter(Dd.r2_air, Dd.r2_storage, s=26, c="#0072B2", alpha=0.8, zorder=3)
axa.set_xlabel("Bivariate Pearson r\u00b2: tailwater peak vs air peak")
axa.set_ylabel("Pearson r\u00b2: peak vs storage")
axa.set_title("(a)", loc="left", fontsize=9)
n_above = int((Dd.r2_storage > Dd.r2_air).sum())
axa.annotate(f"{n_above} of {len(Dd)} reservoirs above the 1:1 line",
             (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=6.8, color="0.25",
             bbox=dict(fc="white", ec="none", pad=0.2))

mec = R["mechanism"]["recovery"]
primary = twr[(twr.sample_scope == "primary") & (twr.dist_km <= 30)]
sensitivity = twr[(twr.sample_scope == "sensitivity") & (twr.dist_km > 30)]
axb.scatter(primary.dist_km, primary.beta, s=12, c=TIER_C["tailwater"], alpha=0.32,
            linewidths=0, label="Primary gauges (0-30 km)")
axb.scatter(sensitivity.dist_km, sensitivity.beta, s=15, facecolors="none",
            edgecolors="0.35", alpha=0.7, linewidths=0.7,
            label="Sensitivity band (30-60 km)")
for key, xpos in [("primary_0_10km", 5), ("primary_10_30km", 20),
                  ("sensitivity_30_60km", 45)]:
    b = mec["distance_bins"][key]
    if b["n"] == 0:
        continue
    lo, hi = b["bootstrap_ci95"]
    mean_beta = b["mean_beta"]
    axb.errorbar(xpos, mean_beta, yerr=[[max(mean_beta - lo, 0)], [max(hi - mean_beta, 0)]],
                 fmt="D", ms=5, color="black" if b["sample_scope"] == "primary" else "0.35",
                 capsize=2.5, lw=1.1, zorder=5)
    dx = -18 if key == "sensitivity_30_60km" else 0
    axb.annotate(f"n={b['n']}\nβ={mean_beta:.2f}", (xpos, hi),
                 xytext=(dx, 6), textcoords="offset points", ha="center", va="bottom", fontsize=6.3,
                 bbox=dict(fc="white", ec="none", pad=0.2))
axb.axhline(R["two_stage_primary"]["mean"]["reference"], color=TIER_C["reference"],
            lw=1.1, ls="--")
axb.annotate(f"reference mean {R['two_stage_primary']['mean']['reference']:.3f}",
             (59, R["two_stage_primary"]["mean"]["reference"] + 0.02), ha="right",
             fontsize=6.8, color=TIER_C["reference"],
             bbox=dict(fc="white", ec="none", pad=0.2))
axb.legend(frameon=False, fontsize=6.2, loc="lower right")
axb.set_xlabel("Distance below reservoir (km)")
axb.set_ylabel("Air\u2013water peak coupling \u03b2")
axb.set_title("(b)", loc="left", fontsize=9)
save(fig, "fig8_mechanism")
print("all figures done")
