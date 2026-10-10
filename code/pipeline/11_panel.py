#!/usr/bin/env python3
"""11_panel.py — current annual panel reconstruction (four metric revisions).

Revisions:
  F1 Strict 7-day window: rolling('7D', min_periods=7), same rule for water and air
  F2 Heatwaves: day-of-year thresholds by direct lookup (eliminates off-by-one)
      + marineHeatWaves event rules (group excess-day indices, split groups when the
      gap exceeds 2 days, count as an event only if the group span >= 5 days,
      event days = full span including 1-day interior gaps, excluding trailing
      non-exceedance days); missing days = non-exceedance
  F3 Annual coverage eligibility: D18/D22/D26 and annual hw values assigned only
      when valid days / days in year >= 0.90
  F4 Peak day-of-year: day-of-year of the window end of the annual water/air peak
      (used in the peak-timing-offset analysis)
All other conventions unchanged from previous: quality gate (>=5 complete summers incl.
>=3 consecutive, >=100 valid days/year), >=8 overlapping years,
primary = dist <= 30 km, sensitivity band 30-60 km.
"""
import os
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import classify_pair_sites

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"

HW = dict(pctl=90, window=5, min_n=30, baseline=(1996, 2021), gap=2, minrun=5)
COV_MIN = 0.90

def clim_doy(dates):
    """366-day climatological calendar: leap years use true day-of-year (Feb 29 = 60);
    non-leap years Mar 1..Dec 31 use doy+1. Aligns thresholds across years and
    eliminates leap-year DOY misalignment. Slot 60 occurs only in leap years."""
    doy = np.asarray(pd.DatetimeIndex(dates).dayofyear, int)
    leap = np.asarray(pd.DatetimeIndex(dates).is_leap_year, bool)
    return np.where(leap | (doy <= 59), doy, doy + 1)

def hw_thresholds(base_doys, base_t):
    """90th-percentile thresholds within the baseline years; returns a length-367
    array where index k is the threshold for day k (1..366). No off-by-one."""
    thr = np.full(367, np.nan)
    for dd in range(1, 367):
        w = (base_doys >= dd - HW["window"]) & (base_doys <= dd + HW["window"])
        if w.sum() >= HW["min_n"]:
            thr[dd] = np.percentile(base_t[w], HW["pctl"])
    return pd.Series(thr).interpolate(limit_direction="both").values

def hw_days(doys, t, thr):
    """marineHeatWaves event rules; returns a daily boolean array."""
    ex = t > thr[doys]                      # direct index lookup, no off-by-one
    idx = np.where(ex)[0]
    hw = np.zeros(len(t), bool)
    if len(idx) == 0:
        return hw
    splits = np.where(np.diff(idx) > HW["gap"])[0]   # split when gap > 2 days (only 1-day interior gaps merge)
    for g in np.split(idx, splits + 1):
        if g[-1] - g[0] + 1 >= HW["minrun"]:
            hw[g[0]:g[-1] + 1] = True
    return hw

def annual_metrics(dates, t, label):
    """Single-station daily series -> annual table: MWMT7 (strict 7-day), peak
    day-of-year, heatwave days (revised rules), coverage fraction.
    Heatwaves use the full calendar (missing = NaN = not above threshold), so row
    indices can no longer bridge events."""
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "t": np.asarray(t, float)}).dropna()
    df = df.sort_values("date").drop_duplicates("date")
    df["year"] = df.date.dt.year
    df["cdoy"] = clim_doy(df.date)
    base = df[(df.year >= HW["baseline"][0]) & (df.year <= HW["baseline"][1])]
    if len(base) >= 365:
        thr = hw_thresholds(base.cdoy.values, base.t.values)
    else:
        thr = None
    out = []
    for y, g in df.groupby("year"):
        if len(g) < 100:
            continue
        ser = pd.Series(g.t.values, index=g.date)
        # Full calendar: missing days are NaN -> not above threshold -> events end naturally
        # Convention: events are truncated at year boundaries (independent per year);
        # NH summer peaks are unaffected
        full_idx = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        ser_full = ser.reindex(full_idx)
        roll = ser_full.rolling("7D", min_periods=7).mean()
        peak_end = roll.idxmax()
        cov_frac = ser_full.notna().sum() / len(full_idx)
        if thr is not None:
            full_cdoy = clim_doy(full_idx)
            full_t = ser_full.values
            # NaN -> -999: missing days never exceed the threshold
            full_t_f = np.where(np.isnan(full_t), -999.0, full_t)
            hw = hw_days(full_cdoy, full_t_f, thr)
            n_hw = int(hw.sum())
        else:
            n_hw = np.nan
        out.append(dict(year=int(y), mwmt7=float(roll.max()),
                        peak_doy=int(peak_end.dayofyear), hw=n_hw,
                        cov_frac=float(cov_frac), src=label))
    return pd.DataFrame(out)

def main():
    elig = pd.read_csv(V4 / "gauge_eligibility.csv", dtype={"site_no": str})
    keep = set(elig[elig.keep].site_no.str.zfill(8))
    tp = pd.read_csv(BASE / "data/samples/tailwater_pairs_elev.csv",
                     dtype={"dam_id": str, "site_no": str})
    pair_sites = classify_pair_sites(tp)
    pair_sites = pair_sites[pair_sites.tier != "excluded"]
    pair_sites = pair_sites[pair_sites.site_no.isin(keep)]
    tiers = dict(zip(pair_sites.site_no, pair_sites.tier))
    scopes = dict(zip(pair_sites.site_no, pair_sites.sample_scope))
    dist = dict(zip(pair_sites.site_no, pair_sites.dist_km))
    damof = dict(zip(pair_sites.site_no, pair_sites.dam_id))
    pp = pd.read_csv(BASE / "data/samples/pilot_pairs_structured.csv",
                     dtype={"site_no": str})
    pp["site_no"] = pp.site_no.str.zfill(8)
    for s in pp[pp.tag == "upstream"].site_no.unique():
        if s in keep:
            tiers.setdefault(s, "upstream")
    ref = set(x.strip().zfill(8) for x in
              (BASE / "data/samples/gages2_ref_with_temp.txt").read_text().split())
    for s in ref:
        if s in keep:
            tiers.setdefault(s, "reference")

    water = pd.read_parquet(BASE / "data/cache/daily_water.parquet")
    water = water[water.site.isin(keep)].copy()
    print(f"water rows (keep sites): {len(water):,}")

    frames = []
    for site, g in water.groupby("site"):
        m = annual_metrics(g.date.values, g.t.values, g.src.mode().iloc[0])
        m["site"] = site
        frames.append(m)
    W = pd.concat(frames, ignore_index=True)
    print(f"water annual rows: {len(W):,}")

    air = pd.read_parquet(BASE / "data/cache/daily_air.parquet")
    air = air[air.site.isin(keep)].copy()
    air["date"] = pd.to_datetime(air.year.astype(str), format="%Y") + \
                  pd.to_timedelta(air.yday - 1, unit="D")
    aframes = []
    for site, g in air.groupby("site"):
        m = annual_metrics(g.date.values, g.t.values, "daymet")
        m["site"] = site
        aframes.append(m)
    A = pd.concat(aframes, ignore_index=True).rename(
        columns={"mwmt7": "air_mwmt7", "hw": "air_hw", "peak_doy": "air_peak_doy",
                 "cov": "air_cov", "src": "air_src"})
    print(f"air annual rows: {len(A):,}")

    M = W.merge(A[["site", "year", "air_mwmt7", "air_hw", "air_peak_doy"]],
                on=["site", "year"], how="inner")
    # D18/D22/D26 computed directly from daily values (missing days not counted;
    # filtered by coverage eligibility)
    dw = water[["site", "date", "t"]].copy()
    dw["year"] = dw.date.dt.year
    cov_map = W.set_index(["site", "year"])["cov_frac"]
    for thr_c in [18, 22, 26]:
        cnt = dw[dw.t > thr_c].groupby(["site", "year"]).size().rename(f"D{thr_c}")
        M = M.merge(cnt, left_on=["site", "year"], right_index=True, how="left")
        M[f"D{thr_c}"] = M[f"D{thr_c}"].fillna(0).astype(int)
        covv = M.set_index(["site", "year"]).index.map(cov_map)
        M.loc[covv < COV_MIN, f"D{thr_c}"] = np.nan
    # F3: for years with coverage < 90%, annual hw is likewise withheld; MWMT7 and
    # peak_doy as well (uniform coverage rule)
    M.loc[M.cov_frac < COV_MIN, "hw"] = np.nan
    M.loc[M.cov_frac < COV_MIN, "mwmt7"] = np.nan
    M.loc[M.cov_frac < COV_MIN, "peak_doy"] = np.nan

    M["tier"] = M.site.map(tiers)
    M = M.dropna(subset=["tier"])
    M["dam_id"] = M.site.map(damof)
    M["dist_km"] = M.site.map(dist)
    M["sample_scope"] = M.site.map(scopes).fillna("primary")
    nyr = M.groupby("site").size().rename("n_overlap")
    M = M.merge(nyr, on="site")
    M = M[M.n_overlap >= 8]

    # Quality gate: >=5 complete summers (>=90% valid), including >=3 consecutive
    summer = water[water.date.dt.month.isin([6, 7, 8])]
    ok = {}
    for (s, y), sub in summer.groupby(["site", "year"]):
        d = sub.drop_duplicates("date")
        if d.t.notna().sum() >= 0.9 * 92:
            ok.setdefault(s, set()).add(int(y))
    def passes(s):
        ys = sorted(ok.get(s, set()))
        if len(ys) < 5:
            return False
        run = mx = 1
        for a, b in zip(ys, ys[1:]):
            run = run + 1 if b == a + 1 else 1
            mx = max(mx, run)
        return mx >= 3
    M = M[M.site.map(passes)]

    M = M.rename(columns={"mwmt7": "water_mwmt7", "hw": "water_hw", "peak_doy": "water_peak_doy",
                          "cov_frac": "year_cov_frac"})
    cols = ["site", "year", "water_mwmt7", "water_peak_doy", "D18", "D22", "D26",
            "water_hw", "src", "air_mwmt7", "air_hw", "air_peak_doy",
            "tier", "dam_id", "dist_km", "sample_scope", "n_overlap", "year_cov_frac"]
    M = M[cols].reset_index(drop=True)
    M.to_parquet(V4 / "annual_panel.parquet", index=False)

    prim = M[M.sample_scope == "primary"]
    print("\n=== current panel ===")
    print("rows:", len(M), "| primary:", len(prim), "| gauges:", M.site.nunique(),
          "| primary gauges:", prim.site.nunique())
    print("gauge counts by tier x scope:")
    print(M.groupby(["tier", "sample_scope"]).site.nunique().to_string())
    print("\nprimary-sample D22 available site-years:", int(prim.D22.notna().sum()), "/", len(prim),
          f"({prim.D22.notna().mean():.1%})")
    print("primary-sample hw available site-years:", int(prim.water_hw.notna().sum()), "/", len(prim))
    print("previous vs current panel change check (primary sample):")
    old = pd.read_parquet(V4 / "annual_panel_previous.parquet")
    old = old[old.sample_scope == "primary"]
    mm = prim.merge(old[["site", "year", "water_mwmt7"]].rename(
        columns={"water_mwmt7": "old"}), on=["site", "year"])
    d = (mm.water_mwmt7 - mm.old).abs()
    print(f"  paired site-years {len(mm)}; |Δ|>0.05: {(d > 0.05).sum()} ({(d > 0.05).mean():.1%}); "
          f"median |Δ|={d.median():.3f}")
    print("\nwater_hw distribution (primary sample, available years):")
    print(prim.water_hw.describe().round(1).to_string())

if __name__ == "__main__":
    main()
