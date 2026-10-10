#!/usr/bin/env python3
"""reproduce.py — single-entry reproduction script (in-process execution, no shell).

Usage:
  python3 reproduce.py                 # offline full chain: unit tests -> panel -> statistics ->
                                       # Shasta -> multidam -> mechanism -> sensitivity -> routing
                                       # sensitivity (using persisted routing tables) -> figures ->
                                       # forensic audit
  python3 reproduce.py --with-routing  # additionally runs 21_routing_verify.py (needs network: NLDI + NWIS)
  python3 reproduce.py --quick         # skip figures (numbers only)
  python3 reproduce.py --audit-only    # check the archived manuscript against saved canonical results

Steps and dependencies:
  00 test_hw_unit_robustness        heatwave boundary unit tests        offline
  01 11_panel_current              annual panel construction           offline (reads data/cache + data/samples)
  02 12_canonical_current          full statistics -> canonical        offline
  03 13_shasta_current             Shasta case study                   offline
  04 15_multidam              34-reservoir replication            offline
  05 16_mechanism             mechanism block                     offline
  06 22_sensitivity      nonlinearity + robust inference     offline
  07 23_routing_sensitivity   routing sensitivity (reads persisted tables) offline
  08 14_figures_current            8 figures                           offline (skipped with --quick)
  09 33_response_geometry     common-scale peak/event separation   offline
  10 21_routing_verify        river-network routing verification  network (only with --with-routing)
  11 35_storage_state_panel   post-hoc storage-state panel       offline
  12 24_forensic_audit_robustness   forensic audit (manuscript vs canonical) offline

Environment: Python 3.9+; numpy pandas scipy statsmodels geopandas pyarrow matplotlib;
Data dependencies: data/cache/daily_{water,air}.parquet, data/samples/*.csv, data/raw/(gages2, resopsus_ts).
BASE path: defaults to the parent of this script's directory (repo root); override with the REPRO_BASE env var.
"""
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("REPRO_BASE", str(HERE.parents[1])))

# (name, flag, script filename) — all static literals
STEPS = [
    ("Unit tests: pipeline rules", False, "pytest"),
    ("Annual panel 11_panel_current", False, "11_panel.py"),
    ("Full statistics 12_canonical_current", False, "12_canonical.py"),
    ("Shasta 13_shasta_current", False, "13_shasta.py"),
    ("Multidam replication 15_multidam", False, "15_multidam.py"),
    ("Mechanism block 16_mechanism", False, "16_mechanism.py"),
    ("Sensitivity 22_sensitivity", False, "22_sensitivity.py"),
    ("Robustness additions 27_robustness_checks", False, "27_robustness_checks.py"),
    ("Post-review robustness 28_s12_fixes", False, "28_s12_fixes.py"),
    ("Routing verification 21_routing_verify [network]", "routing", "21_routing_verify.py"),
    ("Routing sensitivity 23_routing_sensitivity", False, "23_routing_sensitivity.py"),
    ("HydroRIVERS re-verification 25 [requires data/raw/hyrorivers_na]", False, "25_upstream_verify_hyrorivers.py"),
    ("Both-sources intersection sensitivity 23b", False, "23b_both_sources_sensitivity.py"),
    ("Upstream rule fix 29_r3m1", False, "29_r3m1_upstream_fix.py"),
    ("Final additions 30 [ar1/registered]", False, "30_inflow34_and_methods.py"),
    ("Area matching 32", False, "32_area_matching.py"),
    ("Figures 14_figures", "quick", "14_figures.py"),
    ("Response geometry 33", False, "33_response_geometry.py"),
    ("Flow-aware discharge sensitivity 34", False, "34_discharge_sensitivity.py"),
    ("Storage-state panel 35", False, "35_storage_state_panel.py"),
    ("Forensic audit 24_forensic_audit", False, "24_forensic_audit.py"),
]


def run_step(script_name):
    """Run a pipeline script in-process (runpy, equivalent to `python <script>`; no subprocess, no shell)."""
    if script_name == "pytest":
        try:
            import pytest
        except ImportError:
            print("pytest is required for the pipeline test step")
            return 1
        return int(pytest.main(["-q", str(BASE / "tests")]))
    path = str(HERE / script_name)
    try:
        runpy.run_path(path, run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


def main():
    argv = set(sys.argv[1:])
    with_routing = "--with-routing" in argv
    quick = "--quick" in argv
    audit_only = "--audit-only" in argv
    if audit_only:
        required = [
            BASE / "data/samples/analysis/canonical.json",
            BASE / "data/samples/analysis/shasta_numbers.json",
            BASE / "paper/manuscript.md",
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            print("Audit-only preflight failed; missing files:")
            for path in missing:
                print(f"  - {path}")
            sys.exit(2)
        os.chdir(str(BASE))
        rc = run_step("24_forensic_audit.py")
        if rc:
            sys.exit(rc)
        return

    required = [
        "data/cache/daily_water.parquet",
        "data/cache/daily_air.parquet",
        "data/samples/tailwater_pairs_elev.csv",
        "data/samples/pilot_pairs_structured.csv",
        "data/samples/gages2_ref_with_temp.txt",
        "data/samples/dam_attributes.csv",
        "data/samples/reservoir_ops_features.csv",
        "data/samples/analysis/gauge_eligibility.csv",
        "data/samples/analysis/gauge_drainage.csv",
        "data/samples/analysis/site_drainage_area.csv",
        "data/samples/analysis/nwis_coords.json",
        "data/samples/analysis/routing_verification.csv",
        "data/raw/gages2/gagesII_9322_point_shapefile/gagesII_9322_sept30_2011.dbf",
        "data/raw/gdw/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp",
        "data/raw/hyrorivers_na/HydroRIVERS_v10_na.shp",
        "data/raw/shasta/cbr_2006.html",
        "data/raw/shasta/cbr_2024.html",
        "data/raw/shasta/daymet_11370500_0.csv",
    ]
    missing = [BASE / rel for rel in required if not (BASE / rel).is_file()]
    if not (BASE / "data/raw/resopsus_ts").is_dir():
        missing.append(BASE / "data/raw/resopsus_ts/")
    if missing:
        print("Full-reproduction preflight failed; required inputs are missing:")
        for path in missing:
            print(f"  - {path}")
        print("Use --audit-only to verify included canonical results without rebuilding them.")
        sys.exit(2)

    os.chdir(str(BASE))
    failures = []
    t0 = time.time()
    for name, flag, script in STEPS:
        if flag == "quick" and quick:
            print(f"[skip] {name} (--quick)")
            continue
        if flag == "routing" and not with_routing:
            print(f"[skip] {name} (offline by default; pass --with-routing to enable)")
            continue
        t = time.time()
        print(f"[run ] {name} ...", flush=True)
        rc = run_step(script)
        dt = time.time() - t
        if rc == 0:
            print(f"[ ok ] {name} ({dt:.0f}s)")
        else:
            print(f"[FAIL] {name} (exit {rc}, {dt:.0f}s)")
            failures.append(name)
            if flag is False:
                print("\ncore step failed, aborting reproduction chain:", name)
                break
    total = time.time() - t0
    print(f"\n=== reproduce finished ({total / 60:.1f} min) ===")
    if failures:
        print("failed steps:", ", ".join(failures))
        sys.exit(1)
    print("All steps succeeded. canonical.json and the manuscript forensic audit have been regenerated.")


if __name__ == "__main__":
    main()
