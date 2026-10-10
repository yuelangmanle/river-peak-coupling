#!/usr/bin/env python3
"""13_shasta.py — Shasta v2: antecedent-storage timing fix + leave-one-out + interaction CIs + peak-date table.

Fix: previous explained annual peaks that can occur before 9/30 using 9/30 storage (e.g., a 5/19 peak falls in May).
This version:
  - Strict 7-day window MWMT7 (min_periods=7); peak = window end date
  - Three storage definitions: peak-30d mean (30 days before peak), 6/30 storage, 9/30 storage (comparison)
  - Per definition: bivariate r/slope/CI; joint model with air (HC1); air x storage interaction and CI
  - Leave-one-out: range of r
  - Full peak-date table; lag-1 autocorrelation reported descriptively
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
RAW = BASE / "data/raw/shasta"
V4 = BASE / "data/samples/analysis"

frames = []
for y in range(2006, 2025):
    tabs = pd.read_html(RAW / f"cbr_{y}.html")
    t = next(t for t in tabs if any("KWK" in str(c) for c in t.columns))
    t.columns = [str(c) for c in t.columns]
    t = t[t["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
    frames.append(t)
D = pd.concat(frames, ignore_index=True)
D["date"] = pd.to_datetime(D["Date"])
D = D.sort_values("date").drop_duplicates("date")
D = D[(D.date.dt.year >= 2006) & (D.date.dt.year <= 2024)]
D["stor"] = pd.to_numeric(D["Shasta Storage (MAF)"], errors="coerce")
D["wtemp_c"] = (pd.to_numeric(D["KWK DAT WTemp °F"], errors="coerce") - 32) * 5 / 9
D["year"] = D.date.dt.year

recs = []
for p in sorted(RAW.glob("daymet_11370500_*.csv")):
    lines = p.read_text().replace("\r\n", "\n").split("\n")
    hi = next(i for i, l in enumerate(lines) if l.startswith("year,yday"))
    for l in lines[hi + 1:]:
        pt = l.split(",")
        try:
            recs.append((int(pt[0]), int(pt[1]), float(pt[2])))
        except (ValueError, IndexError):
            continue
A = pd.DataFrame(recs, columns=["year", "yday", "air_tmax"])
A["date"] = pd.to_datetime(A.year.astype(str), format="%Y") + pd.to_timedelta(A.yday - 1, unit="D")

rows = []
peak_dates = {}
for y in sorted(D.year.unique()):
    gw = D[(D.year == y)].dropna(subset=["wtemp_c"]).sort_values("date")
    if len(gw) < 100:
        continue
    sw = pd.Series(gw.wtemp_c.values, index=pd.DatetimeIndex(gw.date))
    roll = sw.rolling("7D", min_periods=7).mean()
    pk = roll.idxmax()
    ga = A[(A.year == y)].sort_values("date")
    sa = pd.Series(ga.air_tmax.values, index=pd.DatetimeIndex(ga.date))
    rolla = sa.rolling("7D", min_periods=7).mean()
    pka = rolla.idxmax()
    d30 = gw[(gw.date >= pk - pd.Timedelta(days=30)) & (gw.date < pk)]
    d30_inclusive = gw[(gw.date >= pk - pd.Timedelta(days=30)) & (gw.date <= pk)]
    jun30 = gw[gw.date <= pd.Timestamp(f"{y}-06-30")]
    sep30 = gw[gw.date <= pd.Timestamp(f"{y}-09-30")]
    rows.append(dict(
        year=int(y),
        water_mwmt7=float(roll.max()), water_peak=str(pk.date()),
        air_mwmt7=float(rolla.max()), air_peak=str(pka.date()),
        stor_peak30=float(d30.stor.mean()) if len(d30) else np.nan,
        stor_peak30_inclusive=float(d30_inclusive.stor.mean()) if len(d30_inclusive) else np.nan,
        stor_jun30=float(jun30.stor.iloc[-1]) if len(jun30) else np.nan,
        stor_sep30=float(sep30.stor.iloc[-1]) if len(sep30) else np.nan,
        coverage=int(len(gw)),
    ))
S = pd.DataFrame(rows)
S.to_csv(V4 / "shasta_annual.csv", index=False)

R = {"n_years": int(len(S)), "period": "2006-2024 (calendar years)",
     "site": "Sacramento River below Keswick Dam (KWK; 13 km below Shasta Dam)",
     "metric": "MWMT7 strict 7-day window (min 7 obs); peak date = window end",
     "peak_dates": S[["year", "water_peak", "air_peak", "water_mwmt7"]].to_dict("records"),
     "n_peaks_before_sep30": int((pd.to_datetime(S.water_peak).to_numpy()
                                   < pd.to_datetime(S.year.astype(str) + "-09-30").to_numpy()).sum()),
     "n_peaks_may": int((pd.DatetimeIndex(S.water_peak).month == 5).sum())}

out = {}
for variant, col in [("peak_minus_30d_mean", "stor_peak30"), ("jun30", "stor_jun30"),
                     ("sep30_comparison_only", "stor_sep30")]:
    s_ = S.dropna(subset=[col, "water_mwmt7", "air_mwmt7"])
    r_w = stats.pearsonr(s_.water_mwmt7, s_[col])
    m_b = smf.ols(f"water_mwmt7 ~ {col}", data=s_).fit(cov_type="HC1")
    m_j = smf.ols(f"water_mwmt7 ~ air_mwmt7 + {col}", data=s_).fit(cov_type="HC1")
    r_as = stats.pearsonr(s_.air_mwmt7, s_[col])
    out[variant] = {
        "n": int(len(s_)),
        "r_water_storage": round(float(r_w[0]), 3), "p_water_storage": float(f"{r_w[1]:.2e}"),
        "slope_per_MAF_HC1": round(float(m_b.params[col]), 3),
        "slope_ci95_HC1": [round(float(m_b.conf_int().loc[col, 0]), 3),
                           round(float(m_b.conf_int().loc[col, 1]), 3)],
        "r2_bivariate": round(float(m_b.rsquared), 3),
        "joint_air_p_HC1": round(float(m_j.pvalues["air_mwmt7"]), 4),
        "air_partial_r_HC1": round(float(np.sign(m_j.params["air_mwmt7"]) * np.sqrt(
            m_j.tvalues["air_mwmt7"] ** 2 / (m_j.tvalues["air_mwmt7"] ** 2
                                             + int(m_j.df_resid)))), 3),
        "joint_storage_p_HC1": float(f"{m_j.pvalues[col]:.2e}"),
        "r_air_storage": round(float(r_as[0]), 3), "p_air_storage": round(float(r_as[1]), 3),
    }
    if variant != "sep30_comparison_only":
        m_i = smf.ols(f"water_mwmt7 ~ air_mwmt7 * {col}", data=s_).fit(cov_type="HC1")
        iname = f"air_mwmt7:{col}"
        out[variant]["interaction_coef_per_MAF_per_degC"] = round(float(m_i.params[iname]), 4)
        out[variant]["interaction_p_HC1"] = round(float(m_i.pvalues[iname]), 4)
        out[variant]["interaction_ci95"] = [round(float(m_i.conf_int().loc[iname, 0]), 4),
                                            round(float(m_i.conf_int().loc[iname, 1]), 4)]
        loo = []
        for drop in range(len(s_)):
            s2 = s_.drop(s_.index[drop])
            loo.append(stats.pearsonr(s2.water_mwmt7, s2[col])[0])
        out[variant]["leave_one_out_r_range"] = [round(float(np.min(loo)), 3),
                                                 round(float(np.max(loo)), 3)]
R["storage_variants"] = out
R["autocorrelation"] = {}
s_ = S.dropna(subset=["stor_peak30", "water_mwmt7", "air_mwmt7"])
mb = smf.ols("water_mwmt7 ~ stor_peak30", data=s_).fit()
R["autocorrelation"] = {"lag1_residual": round(float(mb.resid.autocorr(1)), 3),
                        "note": "reported descriptively; no conservativeness claim"}
(V4 / "shasta_numbers.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1)[:3500])
print("\nsaved -> shasta_numbers.json / shasta_annual.csv")
