#!/usr/bin/env python3
"""29_r3m1_upstream_fix.py — Upstream verified-only rule: three registered variants.

The upstream tier of script 23 used an asymmetric rule (tailwater required NLDI pass/near,
upstream only excluded NLDI fail), so the registered n=24 exceeded the two-source
verification upper bound. This script recomputes with the correct rule and registers
the three rule variants.
"""
import os
import json
from pathlib import Path
import pandas as pd
from scipy import stats

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2]))).resolve()
V4 = (BASE / "data" / "samples" / "analysis").resolve()
if BASE not in V4.parents:
    raise SystemExit("path check failed")

lad = pd.read_csv(V4 / "ladder.csv", dtype={"site_no": str})
lad["site_no"] = lad.site_no.str.zfill(8)
Lp = lad[lad.sample_scope == "primary"].dropna(subset=["beta"])
hr = pd.read_csv(V4 / "routing_verification_hyrorivers.csv", dtype={"site_no": str})
hr["site8"] = hr.site_no.str.zfill(8)
ok_hr_up = set(hr[(hr.direction == "upstream") & (hr.verdict == "pass_upstream")].site8)
up_hr = Lp[(Lp.group == "upstream") & Lp.site_no.isin(ok_hr_up)]
ref = Lp[Lp.group == "reference"]
mw = stats.mannwhitneyu(up_hr.beta, ref.beta, alternative="two-sided")
print(f"HydroRIVERS-verified upstream: n={len(up_hr)} mean beta={up_hr.beta.mean():.3f} "
      f"vs ref {ref.beta.mean():.3f} two-sided p={mw.pvalue:.3f}")

reg_path = V4 / "canonical.json"
C = json.loads(reg_path.read_text())
rv = C["routing_verification"]
rv["verified_only_rule_note"] = {
    "issue_found": ("the registered twostage_verified_only upstream tier (n=24) used an "
                    "asymmetric rule: tailwater required NLDI pass/near, upstream only "
                    "excluded NLDI fail"),
    "upstream_rule_variants": {
        "nldi_verified_only": {"n": 5, "note": "too few for a two-stage reanalysis"},
        "hyrorivers_verified_only": {
            "n": int(len(up_hr)), "mean_beta": round(float(up_hr.beta.mean()), 3),
            "vs_ref_two_sided_p": round(float(mw.pvalue), 3)},
        "exclude_nldi_fail_original": {"n": 24, "status": "DEPRECATED asymmetric rule"}},
    "tailwater_rule": "NLDI pass/near (70 pairs -> 24 gauges with >=8 eligible years)"}
reg_path.write_text(json.dumps(C, indent=1, ensure_ascii=False))
print("registry updated")
