#!/usr/bin/env python3
"""25_upstream_verify_hyrorivers.py — upstream-tier routing verification (independent HydroRIVERS cross-check).

Background: in the NLDI verification, 31/55 upstream pairs were unresolved
(position lookup failed) and 19 failed. This script instead uses the local
HydroRIVERS v10 (North America) product as a fully independent data source and algorithm:
  - Dam -> river segment: GDW reservoir database (GRAND_ID -> HYRIV_ID)
  - Gauge -> river segment: snap gauge coordinates to the nearest segment (1 km tolerance)
  - Upstream test: walk downstream from the gauge segment along the NEXT_DOWN chain;
    hitting the dam segment means the gauge lies upstream of the dam
  - Downstream re-verification: walk downstream from the dam segment along the NEXT_DOWN
    chain; hitting the gauge segment means the gauge lies downstream of the dam
Distance caps: upstream 150 km (river-network distance; the pairing criterion is a
50 km radial distance), downstream 60 km.
Outputs: routing_verification_hyrorivers.csv + merged into canonical.json
"""
import os
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
HR_LOCAL = BASE / "data/raw/hyrorivers_na/HydroRIVERS_v10_na.shp"
HR_TMP = Path("/tmp/hyoriv_na/HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na.shp")
HR_PROJECT = BASE.parent.parent / "data/raw/hyrorivers_na/HydroRIVERS_v10_na.shp"
HR_SHIP = HR_LOCAL if HR_LOCAL.exists() else (HR_PROJECT if HR_PROJECT.exists() else HR_TMP)
GDW = BASE / "data/raw/gdw/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp"
if not GDW.exists() and (BASE.parent.parent / "data/raw/gdw/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp").exists():
    GDW = BASE.parent.parent / "data/raw/gdw/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp"

CAP_UP_KM = 150.0
CAP_DN_KM = 60.0
SNAP_TOL_KM = 1.0
DISTANCE_CRS = "EPSG:5070"  # CONUS Albers, metres


def main():
    t0 = time.time()
    # ---- River network (keep only necessary columns) ----
    riv = gpd.read_file(HR_SHIP, columns=["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM", "geometry"])
    if riv.crs is None:
        raise ValueError("HydroRIVERS source CRS is missing; refusing degree-based distance calculations")
    source_crs = CRS.from_user_input(riv.crs)
    if not source_crs.is_geographic:
        raise ValueError(f"Expected a geographic HydroRIVERS source CRS, got {source_crs}")
    riv = riv.to_crs(DISTANCE_CRS)
    if not CRS.from_user_input(riv.crs).axis_info[0].unit_name.lower().startswith("met"):
        raise ValueError(f"Distance CRS must use metres, got {riv.crs}")
    print(f"HydroRIVERS NA: {len(riv):,} segments ({time.time()-t0:.0f}s)")
    nxt = dict(zip(riv.HYRIV_ID.astype("int64"), riv.NEXT_DOWN.astype("int64")))
    ln = dict(zip(riv.HYRIV_ID.astype("int64"), riv.LENGTH_KM.astype(float)))

    # ---- Dam -> HYRIV_ID (GDW) ----
    gdw = gpd.read_file(GDW, columns=["GRAND_ID", "HYRIV_ID"])
    dam_riv = dict(zip(gdw.GRAND_ID.astype("int64"), gdw.HYRIV_ID.astype("int64")))
    print(f"GDW dams: {len(dam_riv):,}")

    # ---- Gauge coordinates ----
    coords = json.load(open(V4 / "nwis_coords.json"))

    def snap(lon, lat):
        """Gauge coordinates -> (HYRIV_ID, snap distance in km). Uses sindex nearest-neighbor."""
        pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(DISTANCE_CRS)
        try:
            idx = list(riv.sindex.nearest(pt.geometry, max_distance=SNAP_TOL_KM * 1000.0)[1])
        except Exception:
            return None, np.nan
        if not idx:
            return None, np.nan
        # Keep the candidate with the smallest actual distance
        best, bd = None, 1e9
        for i in idx:
            d = riv.geometry.iloc[i].distance(pt.geometry.iloc[0]) / 1000.0
            if d < bd:
                bd, best = d, i
        if bd > SNAP_TOL_KM:
            return None, bd
        return int(riv.HYRIV_ID.iloc[best]), bd

    def walk_down(start_id, cap_km, targets):
        """Walk downstream from start_id along the NEXT_DOWN chain; return (river distance) on hitting any target."""
        seen = set()
        cur = start_id
        dist = 0.0
        while cur in nxt and cur not in seen and cur != 0:
            if cur in targets:
                return dist
            seen.add(cur)
            dist += ln.get(cur, 0.0)
            if dist > cap_km:
                return None
            cur = nxt[cur]
        return None

    # ---- Pairing tables ----
    tp = pd.read_csv(BASE / "data/samples/tailwater_pairs_elev.csv", dtype=str)
    pp = pd.read_csv(BASE / "data/samples/pilot_pairs_structured.csv", dtype=str)
    up = pp[pp.tag == "upstream"].drop_duplicates("site_no")

    rows = []
    # Upstream gauges: gauge -> downstream walk -> dam
    cache_snap = {}
    for _, r in up.iterrows():
        s = str(r.site_no).zfill(8)
        d = int(r.dam_id)
        c = coords.get(s)
        if c is None or d not in dam_riv:
            rows.append(dict(dam_id=r.dam_id, site_no=s, direction="upstream",
                             verdict="unresolved", note="no coords or dam riv"))
            continue
        key = (round(c["lon"], 4), round(c["lat"], 4))
        if key not in cache_snap:
            cache_snap[key] = snap(c["lon"], c["lat"])
        seg, sd = cache_snap[key]
        if seg is None:
            rows.append(dict(dam_id=r.dam_id, site_no=s, direction="upstream",
                             verdict="unsnap", snap_km=round(float(sd), 3),
                             note="no HydroRIVERS segment within 1km"))
            continue
        hit = walk_down(seg, CAP_UP_KM, {dam_riv[d]})
        v = "pass_upstream" if hit is not None else "fail"
        rows.append(dict(dam_id=r.dam_id, site_no=s, direction="upstream", verdict=v,
                         snap_km=round(float(sd), 3),
                         river_km=round(float(hit), 1) if hit is not None else None))

    # Tailwater gauges: dam -> downstream walk -> gauge (independent re-verification of NLDI)
    tw_sites = {}
    for _, r in tp.iterrows():
        s = str(r.site_no).zfill(8)
        tw_sites.setdefault(s, int(r.dam_id))
    for s, d in tw_sites.items():
        c = coords.get(s)
        if c is None or d not in dam_riv:
            rows.append(dict(dam_id=str(d), site_no=s, direction="downstream",
                             verdict="unresolved", note="no coords or dam riv"))
            continue
        key = (round(c["lon"], 4), round(c["lat"], 4))
        if key not in cache_snap:
            cache_snap[key] = snap(c["lon"], c["lat"])
        seg, sd = cache_snap[key]
        if seg is None:
            rows.append(dict(dam_id=str(d), site_no=s, direction="downstream",
                             verdict="unsnap", snap_km=round(float(sd), 3)))
            continue
        hit = walk_down(dam_riv[d], CAP_DN_KM, {seg})
        v = "pass_downstream" if hit is not None else "fail"
        rows.append(dict(dam_id=str(d), site_no=s, direction="downstream", verdict=v,
                         snap_km=round(float(sd), 3),
                         river_km=round(float(hit), 1) if hit is not None else None))

    RV = pd.DataFrame(rows)
    RV.to_csv(V4 / "routing_verification_hyrorivers.csv", index=False)
    summ = RV.groupby(["direction", "verdict"]).size().to_dict()
    print(RV.groupby(["direction", "verdict"]).size().to_string())

    up_ver = RV[(RV.direction == "upstream") & (RV.verdict == "pass_upstream")].site_no.nunique()
    up_fail = RV[(RV.direction == "upstream") & (RV.verdict == "fail")].site_no.nunique()
    up_uns = RV[(RV.direction == "upstream") & (RV.verdict.isin(["unsnap", "unresolved"]))].site_no.nunique()
    dn_ver = RV[(RV.direction == "downstream") & (RV.verdict == "pass_downstream")].site_no.nunique()
    dn_fail = RV[(RV.direction == "downstream") & (RV.verdict == "fail")].site_no.nunique()
    dn_uns = RV[(RV.direction == "downstream") & (RV.verdict.isin(["unsnap", "unresolved"]))].site_no.nunique()
    # Two-source agreement (NLDI pass/near ∩ HydroRIVERS pass)
    nldi = pd.read_csv(V4 / "routing_verification.csv", dtype={"site_no": str})
    nldi["site8"] = nldi.site_no.str.zfill(8)
    hr = RV.copy()
    hr["site8"] = hr.site_no.str.zfill(8)
    dn = hr[hr.direction == "downstream"]
    nl = nldi[nldi.direction == "downstream"]
    m = dn.merge(nl[["site8", "verdict"]], on="site8", suffixes=("_hr", "_nldi"))
    agree = int(((m.verdict_hr == "pass_downstream") ==
                 (m.verdict_nldi.isin(["pass", "near"]))).sum())
    R = {
        "source": "HydroRIVERS v10 NA (independent of NHDPlus/NLDI)",
        "caps_km": {"upstream": CAP_UP_KM, "downstream": CAP_DN_KM, "snap_tol": SNAP_TOL_KM},
        "counts": {
            "upstream": {"verified": int(up_ver), "fail": int(up_fail),
                         "unsnap_or_unresolved": int(up_uns), "n_pairs": int(len(up))},
            "downstream": {"verified": int(dn_ver), "fail": int(dn_fail),
                           "unsnap_or_unresolved": int(dn_uns), "n_pairs": int(len(tw_sites))}},
        "nldi_agreement_downstream": {
            "n_compared": int(len(m)),
            "agree": agree,
            "rate": round(agree / len(m), 3) if len(m) else None},
        "groupby": {f"{k[0]}|{k[1]}": int(v) for k, v in summ.items()},
    }
    # Two-source intersection (downstream: NLDI pass/near ∩ HydroRIVERS pass)
    both = m[(m.verdict_hr == "pass_downstream") & (m.verdict_nldi.isin(["pass", "near"]))]
    R["both_sources_downstream_sites"] = sorted(both.site8.unique().tolist())
    R["n_both_sources_downstream"] = int(both.site8.nunique())
    (V4 / "routing_hyrorivers_block.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    C = json.load(open(V4 / "canonical.json"))
    C["routing_verification"]["hyrorivers_crosscheck"] = R
    (V4 / "canonical.json").write_text(json.dumps(C, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
