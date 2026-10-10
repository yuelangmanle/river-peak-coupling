#!/usr/bin/env python3
"""21_routing_verify.py — River-network routing verification.

Real network routing via USGS NLDI (NHDPlus):
  1. Site coordinates: batch dec_lat/dec_long from the NWIS site service (incl. 10-digit site numbers)
  2. Dam coordinates -> COMID (position lookup)
  3. Navigate the dam COMID 35 km along DM (downstream main), collecting downstream flowlines
  4. Tailwater gauge coordinates -> COMID; check membership in the downstream set => same river, below the dam
  5. Upstream gauges treated symmetrically: UM (upstream main) 55 km
Verdicts: pass = gauge COMID is in the navigated set;
      near = COMID not in the set but site coordinates within 2 km of any navigated flowline (tributary-mouth proximity tolerance);
      fail = otherwise; nocomid = lookup failed
Also records NWIS huc8 and drainage area (backup consistency checks).
Output: data/samples/analysis/routing_verification.csv
"""
import os
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
V4 = BASE / "data/samples/analysis"
NLDI = "https://api.water.usgs.gov/nldi/linked-data/comid"
S = requests.Session()
S.headers["User-Agent"] = "research-routing-verify/1.0"

_cache_f = V4 / "nldi_cache.json"
cache = json.loads(_cache_f.read_text()) if _cache_f.exists() else {}

def lookup_comid(lon, lat):
    k = f"P{lon:.4f},{lat:.4f}"
    if k in cache:
        return cache[k]
    try:
        r = S.get(f"{NLDI}/position", params={"f": "json", "coords": f"POINT({lon} {lat})"},
                  timeout=30)
        fs = r.json().get("features", [])
        v = int(fs[0]["properties"]["comid"]) if fs else None
    except Exception:
        v = None
    cache[k] = v
    return v

def _nav(comid, dist, code):
    k = f"{code}{comid},{dist}"
    if k in cache:
        return cache[k]
    try:
        r = S.get(f"{NLDI}/{comid}/navigation/{code}/flowlines",
                  params={"distance": dist}, timeout=90)
        feats = r.json().get("features", [])
        ids = [int(f["properties"].get("nhdplus_comid") or f["properties"]["comid"])
               for f in feats]
        geoms = [f["geometry"]["coordinates"] for f in feats]
        v = [ids, geoms]
    except Exception:
        v = [[], []]
    cache[k] = v
    return v

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def min_dist_to_polylines(lon, lat, geoms):
    best = 1e9
    for line in geoms or []:
        pts = []
        for c in line:
            pts.extend(c) if isinstance(c[0], list) else pts.append(c)
        if not pts:
            continue
        arr = np.array(pts)
        d = haversine_km(lon, lat, arr[:, 0], arr[:, 1]).min()
        best = min(best, float(d))
    return best

def fetch_nwis_coords(sites):
    """Batch NWIS site info: dec_lat/dec_long/huc8/drain_area. Persistent cache."""
    cf = V4 / "nwis_coords.json"
    out = json.loads(cf.read_text()) if cf.exists() else {}
    sites = [s for s in sites if s not in out]
    for i in range(0, len(sites), 100):
        batch = [s for s in sites[i:i + 100]]
        for attempt in range(5):
            try:
                r = S.get("https://waterservices.usgs.gov/nwis/site/",
                          params={"format": "rdb", "sites": ",".join(batch),
                                  "siteOutput": "expanded", "siteStatus": "all"},
                          timeout=90)
                got = 0
                for ln in r.text.splitlines():
                    if ln.startswith("USGS\t"):
                        p = ln.split("\t")
                        sid = p[1]
                        try:
                            lat, lon = float(p[6]), float(p[7])
                        except (ValueError, IndexError):
                            continue
                        huc = p[23] if len(p) > 23 else ""
                        try:
                            da = float(p[29]) if len(p) > 29 and p[29] else np.nan
                        except ValueError:
                            da = np.nan
                        out[sid] = dict(lon=lon, lat=lat, huc8=huc[:8], drain_sqmi=da)
                        got += 1
                if got >= len(batch) * 0.5:
                    break
            except Exception:
                pass
            time.sleep(3)
        cf.write_text(json.dumps(out))
        time.sleep(1)
        print(f"  nwis batch {i // 100 + 1}/{(len(sites) - 1) // 100 + 1} (cum {len(out)})")
    return out

def main():
    tp = pd.read_csv(BASE / "data/samples/tailwater_pairs_elev.csv", dtype=str)
    pp = pd.read_csv(BASE / "data/samples/pilot_pairs_structured.csv", dtype=str)
    up = pp[pp.tag == "upstream"].drop_duplicates("site_no")
    da = pd.read_csv(BASE / "data/samples/dam_attributes.csv", dtype={"dam_id": str})
    dlon = dict(zip(da.dam_id, pd.to_numeric(da.lon)))
    dlat = dict(zip(da.dam_id, pd.to_numeric(da.lat)))

    sites = set(tp.site_no.str.zfill(8)) | set(up.site_no.str.zfill(8))
    print(f"pairs={len(tp)} dams={tp.dam_id.nunique()} upstream={len(up)} sites={len(sites)}")
    coords = fetch_nwis_coords(sites)
    print(f"coords fetched: {len(coords)}/{len(sites)}")

    rows = []
    dams = tp.dam_id.unique()
    dam_comid = {}
    for i, d in enumerate(dams):
        dam_comid[d] = lookup_comid(dlon.get(d), dlat.get(d))
        time.sleep(0.05)
        if (i + 1) % 25 == 0:
            print(f"  dam lookup {i + 1}/{len(dams)}")
            _cache_f.write_text(json.dumps(cache))
    dn_ids, dn_geoms = {}, {}
    for i, d in enumerate(dams):
        c = dam_comid.get(d)
        dn_ids[d], dn_geoms[d] = _nav(c, 35, "DM") if c else ([], [])
        time.sleep(0.05)
        if (i + 1) % 25 == 0:
            print(f"  nav {i + 1}/{len(dams)}")
            _cache_f.write_text(json.dumps(cache))

    def judge(r, direction, ids, geoms):
        s = str(r.site_no).zfill(8)
        d = r.dam_id
        cinfo = coords.get(s)
        if cinfo is None:
            rows.append(dict(dam_id=d, site_no=s, direction=direction, verdict="nosite"))
            return
        lon, lat = cinfo["lon"], cinfo["lat"]
        gc = lookup_comid(lon, lat)
        time.sleep(0.05)
        if gc is None or (not ids and not geoms):
            v = "nocomid"
        elif gc in ids:
            v = "pass"
        else:
            md = min_dist_to_polylines(lon, lat, geoms)
            v = "near" if md < 2.0 else "fail"
        rows.append(dict(dam_id=d, site_no=s, direction=direction, verdict=v,
                         gauge_comid=gc, dam_comid=dam_comid.get(d),
                         huc8=cinfo["huc8"], drain_sqmi=cinfo["drain_sqmi"]))

    for _, r in tp.iterrows():
        judge(r, "downstream", dn_ids.get(r.dam_id, []), dn_geoms.get(r.dam_id, []))

    up_ids, up_geoms = {}, {}
    for d in up.dam_id.unique():
        c = dam_comid.get(d)
        up_ids[d], up_geoms[d] = _nav(c, 55, "UM") if c else ([], [])
        time.sleep(0.05)
    for _, r in up.iterrows():
        judge(r, "upstream", up_ids.get(r.dam_id, []), up_geoms.get(r.dam_id, []))

    RV = pd.DataFrame(rows)
    RV.to_csv(V4 / "routing_verification.csv", index=False)
    _cache_f.write_text(json.dumps(cache))
    print(RV.groupby(["direction", "verdict"]).size().to_string())
    print("saved -> routing_verification.csv")

if __name__ == "__main__":
    main()
