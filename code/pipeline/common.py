#!/usr/bin/env python3
"""Shared utilities: config loading, DV/Daymet parsing, group definitions."""
import os
import glob, os, re, json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))

def cfg():
    with open(BASE / "code" / "pipeline" / "config.yaml") as f:
        return yaml.safe_load(f)

def load_gages2_ref():
    p = BASE / "data/samples/gages2_ref_with_temp.txt"
    return set(x.strip().zfill(8) for x in p.read_text().split() if x.strip())

def classify_pair_sites(pairs):
    """Assign pair-table gauges using the frozen distance, elevation, and direction rules."""
    p = pairs.copy()
    p["site_no"] = p.site_no.astype(str).str.zfill(8)
    p["dist_km"] = pd.to_numeric(p.dist_km, errors="coerce")
    p["tag"] = p.tag.fillna("").astype(str).str.strip().str.lower()
    p["elev_judge"] = p.elev_judge.fillna("").astype(str).str.strip().str.lower()
    rows = []
    for site, g in p.groupby("site_no", sort=False):
        primary = g[(g.dist_km >= 0) & (g.dist_km <= 30)
                    & g.tag.isin(["downstream", "unknown"])
                    & (g.elev_judge == "below")]
        upstream = g[(g.dist_km >= 0) & (g.dist_km <= 50) & (g.tag == "upstream")]
        band = g[(g.dist_km > 30) & (g.dist_km <= 60) & (g.tag != "upstream")]
        if not primary.empty:
            chosen = primary.sort_values("dist_km").iloc[0]
            tier, scope = "tailwater", "primary"
        elif not upstream.empty:
            chosen = upstream.sort_values("dist_km").iloc[0]
            tier, scope = "upstream", "primary"
        elif not band.empty:
            chosen = band.sort_values("dist_km").iloc[0]
            tier, scope = "tailwater", "sensitivity"
        else:
            chosen = g.sort_values("dist_km", na_position="last").iloc[0]
            tier, scope = "excluded", "excluded"
        rows.append({"site_no": site, "tier": tier, "sample_scope": scope,
                     "dam_id": chosen.dam_id, "dist_km": chosen.dist_km})
    return pd.DataFrame(rows)

def antecedent_window_mask(dates, peak, days=30):
    """Select the exact antecedent interval [peak-days, peak), excluding peak day."""
    dates = pd.to_datetime(dates)
    peak = pd.Timestamp(peak)
    return (dates >= peak - pd.Timedelta(days=days)) & (dates < peak)

def encode_use_elec(values):
    """Apply the preregistered Main=1, Sec=0 encoding; all other values remain missing."""
    normalized = values.astype("string").str.strip()
    return normalized.map({"Main": 1.0, "Sec": 0.0}).astype(float)

def load_groups():
    """Return dict: site -> tier (reference/upstream/tailwater), based on pair tables + the reference-site list."""
    SMP = BASE / "data/samples"
    ref = load_gages2_ref()
    st = pd.read_csv(SMP / "pilot_pairs_structured.csv", dtype={"dam_id": str, "site_no": str})
    st["site_no"] = st.site_no.str.zfill(8)
    tw = pd.read_csv(SMP / "tailwater_pairs_elev.csv", dtype={"dam_id": str, "site_no": str})
    assigned = classify_pair_sites(tw)
    tiers = dict(zip(assigned[assigned.tier != "excluded"].site_no,
                     assigned[assigned.tier != "excluded"].tier))
    for s in st[st.tag == "upstream"].site_no.unique():
        tiers.setdefault(s, "upstream")
    for s in ref:
        tiers.setdefault(s, "reference")
    return tiers

def load_dam_of_tailwater():
    SMP = BASE / "data/samples"
    tw = pd.read_csv(SMP / "tailwater_pairs_elev.csv", dtype={"dam_id": str, "site_no": str})
    assigned = classify_pair_sites(tw)
    assigned = assigned[assigned.tier == "tailwater"].sort_values("dist_km")
    return dict(zip(assigned.site_no, assigned.dam_id))

def parse_dv_water(cfg):
    """Parse all DV rdb files -> DataFrame[site, date, tmax, tmean] (long table)."""
    raw = BASE / cfg["paths"]["raw"]
    paths = []
    for d in cfg["data"]["dv_dirs"]:
        paths += sorted(glob.glob(str(raw / d / "*.rdb")))
    VALCOL = re.compile(r"^(\d+)_00010_(00001|00003)$")
    BAD = {"", "Ice", "Bkw", "Eqp", "Ssn", "Prb", "Rat"}
    frames = []
    for path in paths:
        header = None
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split("\t")
                if cols[0] == "agency_cd":
                    header = cols; h = {c: i for i, c in enumerate(cols)}; continue
                if header is None or cols == header or cols[0] == "5s" or len(cols) != len(header):
                    continue
                site, date = cols[h["site_no"]], cols[h["datetime"]]
                if not re.match(r"^\d{4}-", date):
                    continue
                tmax = tmean = np.nan
                for cname, cidx in h.items():
                    m = VALCOL.match(cname)
                    if not m: continue
                    val = cols[cidx].strip()
                    if val in BAD or not val.replace(".", "", 1).replace("-", "", 1).isdigit():
                        continue
                    v = float(val)
                    if v < -5 or v > 45: continue
                    if m.group(2) == "00001": tmax = v
                    else: tmean = v
                if not (np.isnan(tmax) and np.isnan(tmean)):
                    rows.append((site, date, tmax, tmean))
        if rows:
            frames.append(pd.DataFrame(rows, columns=["site", "date", "tmax", "tmean"]))
    df = pd.concat(frames, ignore_index=True)
    df["site"] = df["site"].str.zfill(8)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df[(df.year >= cfg["data"]["water_start"]) & (df.year <= cfg["data"]["water_end"])]
    # Harmonize water temperature: daily maximum preferred, daily mean as fallback; flag the source
    df["t"] = df["tmax"].fillna(df["tmean"])
    df["src"] = np.where(df["tmax"].notna(), "tmax", "tmean")
    # P0-1 fix: deduplicate by site+date (each site-date must be unique)
    n_before = len(df)
    df = df.sort_values(["site", "date", "src"]).drop_duplicates(subset=["site", "date"], keep="first")
    df = df.sort_values(["site", "date"]).reset_index(drop=True)
    if n_before != len(df):
        print(f"  [dedup] removed {n_before - len(df):,} duplicate site-date rows ({n_before:,} -> {len(df):,})")
    return df[["site", "date", "year", "t", "src", "tmax", "tmean"]]

def load_daymet_air():
    """Daymet chunked CSVs -> DataFrame[site, year, yday, tmax]."""
    out = {}
    for p in sorted(glob.glob(str(BASE / "data/raw/daymet_station/daymet_*_*.csv"))):
        site = os.path.basename(p).replace(".csv", "").split("_")[1]
        lines = open(p, newline=None).read().replace("\r\n", "\n").strip().split("\n")
        hi = next((i for i, l in enumerate(lines) if l.startswith("year,yday")), None)
        if hi is None: continue
        recs = out.setdefault(site, [])
        for l in lines[hi + 1:]:
            pt = l.split(",")
            try: y, yd, t = int(pt[0]), int(pt[1]), float(pt[2])
            except (ValueError, IndexError): continue
            if -50 < t < 55: recs.append((y, yd, t))
    frames = []
    for site, recs in out.items():
        if len(recs) < 3650: continue
        df = pd.DataFrame(recs, columns=["year", "yday", "t"])
        df["site"] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def air_annual_and_heatwave(df_air, cfg):
    """Annual air MWMT7 and heatwave day counts (with gap tolerance). Returns [site, year, air_mwmt7, air_hw]."""
    hw_cfg = cfg["heatwave"]; y0, y1 = hw_cfg["baseline"]
    out = []
    for site, df in df_air.groupby("site"):
        df = df.sort_values(["year", "yday"]).reset_index(drop=True)
        thr = np.full(367, np.nan)
        for dd in range(1, 367):
            w = df["yday"].between(dd - hw_cfg["window"], dd + hw_cfg["window"])
            if w.sum() >= 30:
                thr[dd] = np.percentile(df["t"][w], hw_cfg["pctl"])
        thr = pd.Series(thr).interpolate(limit_direction="both").values
        ex = df["t"].values > thr[df["yday"].values - 1]
        years = df["year"].values
        # heatwave days with tolerance: up to `gap` non-exceedance days do not interrupt an event
        gap = hw_cfg["gap_tolerance"]; minrun = hw_cfg["min_run"]
        hw_mask = np.zeros(len(ex), bool)
        i, n = 0, len(ex)
        while i < n:
            if ex[i]:
                j = i; miss = 0
                while j < n:
                    if ex[j]: j += 1; miss = 0
                    elif miss < gap: j += 1; miss += 1
                    else: break
                dur = j - i
                if dur >= minrun: hw_mask[i:j] = True
                i = j
            else:
                i += 1
        hw_days = pd.Series(hw_mask.astype(int)).groupby(years).sum()
        ann = df.groupby("year")["t"].apply(lambda x: x.rolling(7).mean().max() if len(x) >= 7 else np.nan)
        for y in ann.index:
            out.append((site, int(y), ann[y], int(hw_days.get(y, 0))))
    return pd.DataFrame(out, columns=["site", "year", "air_mwmt7", "air_hw"])

def summer_quality_gates(water_df, cfg):
    """Return per-site quality gates: the set of complete summer years."""
    q = cfg["quality"]
    months = tuple(q["summer_months"])
    ok_years = defaultdict(set)
    g = water_df[water_df.date.dt.month.isin(months)]
    for (site, y), sub in g.groupby(["site", "year"]):
        present = sub.dropna(subset=["t"]).shape[0]
        if present >= q["summer_complete_frac"] * q["summer_days"]:
            ok_years[site].add(int(y))
    return ok_years

from collections import defaultdict
