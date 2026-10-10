#!/usr/bin/env python3
"""Audit manuscript claims against the regenerated current canonical results."""
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
C = json.loads((BASE / "data/samples/analysis/canonical.json").read_text())
S = json.loads((BASE / "data/samples/analysis/shasta_numbers.json").read_text())
SV = json.loads((BASE / "data/samples/analysis/sensitivity.json").read_text())
MS = (BASE / "paper/manuscript.md").read_text()
ok = True

abstract_text = MS.split("## Abstract", 1)[1].split("**Keywords**", 1)[0]
abstract_words = len(abstract_text.split())
print("== Submission structure ==")
print(f"  {'PASS' if abstract_words <= 250 else 'FAIL'}  abstract word count: {abstract_words} (target 250)")
if abstract_words > 250:
    ok = False
abstract_first_person = re.search(r"\b(?:we|our)\b", abstract_text, flags=re.I) is not None
print(f"  {'PASS' if not abstract_first_person else 'FAIL'}  abstract avoids first-person construction")
if abstract_first_person:
    ok = False
for label, pattern in (
    ("attribution table path", r"data/samples/analysis/pred_reservoir_features_tailwater\.csv"),
    ("observed warm-burden coordinate", r"B,\s+an\s+observed\s+coordinate"),
):
    passed = re.search(pattern, MS, flags=re.I) is not None
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    if not passed:
        ok = False
figure_code = (BASE / "code/pipeline/14_figures.py").read_text()
figure_label_ok = (
    "ordnance" not in figure_code.lower()
    and "reference mean" in figure_code
    and ":.3f" in figure_code.split("reference mean", 1)[1][:200]
)
print(f"  {'PASS' if figure_label_ok else 'FAIL'}  Fig. 8b reference-mean label is typo-free and three-decimal")
if not figure_label_ok:
    ok = False
body_text = MS.split("## Data availability", 1)[0]
style_terms = {
    "rather than": len(re.findall(r"\brather than\b", body_text, flags=re.I)),
    "decisive": len(re.findall(r"\bdecisive\b", body_text, flags=re.I)),
    "testable": len(re.findall(r"\btestable\b", body_text, flags=re.I)),
    "falsifiable": len(re.findall(r"\bfalsifiable\b", body_text, flags=re.I)),
    "frozen": len(re.findall(r"\bfrozen\b", body_text, flags=re.I)),
    "canonical": len(re.findall(r"\bcanonical\b", body_text, flags=re.I)),
}
style_ok = style_terms["rather than"] <= 5 and all(
    style_terms[k] == 0 for k in ("decisive", "testable", "falsifiable", "frozen", "canonical")
)
print(f"  {'PASS' if style_ok else 'FAIL'}  LLM style guard: {style_terms}")
if not style_ok:
    ok = False
for label, pattern in (
    ("no 'at the heart of the problem'", r"at the heart of the problem"),
    ("no 'reframes'", r"\breframes\b"),
    ("US spelling modelling/modeling consistent", r"\bmodelling\b"),
):
    found = re.search(pattern, body_text, flags=re.I) is not None
    print(f"  {'FAIL' if found else 'PASS'}  {label}")
    if found:
        ok = False
cover_path = BASE / "paper/Cover_Letter_JoH.txt"
cover_text = cover_path.read_text()
cover_style_ok = not re.search(r"\brather than\b|\btestable\b|\bestimands\b", cover_text, flags=re.I)
print(f"  {'PASS' if cover_style_ok else 'FAIL'}  cover letter matches manuscript style guard")
if not cover_style_ok:
    ok = False
draft_present = any((BASE / "paper" / name).exists() for name in (
    "Declaration_of_Competing_Interests_draft.docx",
    "Declaration_of_competing_interests_draft.txt",
))
print(f"  {'FAIL' if draft_present else 'PASS'}  draft declarations absent from paper/")
if draft_present:
    ok = False

def check_value(name, observed, expected, tol=0.001):
    global ok
    passed = observed is not None and abs(observed - expected) <= tol
    if not passed:
        ok = False
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: canonical={observed}, expected={expected}")


def require_text(name, *patterns):
    global ok
    passed = any(re.search(pattern, MS, flags=re.I | re.S) for pattern in patterns)
    if not passed:
        ok = False
    print(f"  {'PASS' if passed else 'FAIL'}  manuscript includes {name}")


def forbid_text(name, *patterns):
    global ok
    found = [pattern for pattern in patterns if re.search(pattern, MS, flags=re.I | re.S)]
    passed = not found
    if not passed:
        ok = False
    print(f"  {'PASS' if passed else 'FAIL'}  manuscript avoids {name}")
    for pattern in found:
        print(f"         matched: {pattern}")


print("== Primary national comparison ==")
ts = C["two_stage_primary"]
for tier, expected_n, expected_beta in (
    ("reference", 117, 0.483),
    ("upstream", 36, 0.460),
    ("tailwater", 60, 0.283),
):
    check_value(f"two-stage {tier} n", ts["n"][tier], expected_n, tol=0)
    check_value(f"two-stage {tier} mean beta", ts["mean"][tier], expected_beta)
check_value("two-stage tailwater/reference ratio", ts["ratio_tw_ref_mean"], 0.585)
check_value("two-stage tailwater < reference p", ts["mwu"]["tw_lt_ref_p"], 1.0979e-5, tol=1e-8)
require_text(
    "the 117/36/60 sample",
    r"117\s+reference.{0,100}36\s+(?:above-reservoir|upstream).{0,100}60\s+tailwater",
    r"117\s*/\s*36\s*/\s*60",
)
for value in ("0.483", "0.460", "0.283"):
    require_text(f"two-stage mean {value}", rf"\b{value}\b")

print("== Mixed model ==")
m1 = C["M1_mixed"]
for tier, expected in (("reference", 0.453), ("upstream", 0.416), ("tailwater", 0.310)):
    check_value(f"mixed-model {tier} beta per C", m1[f"beta_{tier}"] / 10, expected)
ct = m1["contrast_tw_vs_ref"]
check_value("mixed-model tailwater contrast per C", ct["coef"] / 10, -0.143)
check_value("mixed-model tailwater contrast p", ct["p"], 0.000328, tol=1e-6)
check_value("mixed-model gauge count", m1["n_gauges"], 251, tol=0)
check_value("mixed-model observation count", m1["n_obs"], 3676, tol=0)
require_text("mixed-model reference beta 0.453", r"\b0\.453\b")
require_text("mixed-model tailwater beta 0.310", r"\b0\.310\b")
require_text("mixed-model contrast −0.143", r"(?:−|-)0\.143")
require_text("mixed-model p = 0.00033", r"p\s*=\s*0\.000(?:328|33)")

print("== Heatwave response ==")
g = C["beta_hw"]["nb_gee"]
check_value("heatwave slope ratio", g["slope_ratio_tw_vs_ref"], 1.003)
check_value("heatwave interaction p", g["p_interaction_tw_vs_ref"], 0.390, tol=0.001)
check_value("+29-day multiplier ratio", g["multiplier_ratio_at_plus29d"]["value"], 1.085)
require_text("heatwave slope ratio 1.003", r"\b1\.003\b")
require_text("heatwave interaction p = 0.390", r"p\s*=\s*0\.390")
for old in (r"\b1\.006\b", r"\b1\.20\b", r"p\s*=\s*0\.038"):
    forbid_text("superseded heatwave estimate", old)

print("== Signal, climate and area robustness ==")
rf = C["review_robustness"]
snr = rf["snr_gradient"]["ladder_by_snr_threshold"]
check_value("|r| >= 0.4 attenuation (%)", snr[2]["attenuation_pct"], 22.6, tol=0.1)
check_value("|r| >= 0.4 high-signal subset p", snr[2]["mwu_p"], 0.00345, tol=0.0001)
check_value("long-record high-signal subset p", snr[-1]["mwu_p"], 0.2243, tol=0.001)
area = rf["area_matching"]["area_covariate_mixed"]
check_value("area-adjusted tailwater contrast", area["contrast_tw_perC"], -0.155)
check_value("area-adjusted tailwater p", area["contrast_p"], 0.0003603, tol=1e-6)
require_text("high-signal attenuation about 22.6%", r"22\.6%|22\.6\s*per\s*cent")
for old in (r"smallest high-signal subset.{0,100}9%", r"\b9%\s*\(p\s*=\s*0\.30\)"):
    forbid_text("superseded high-signal estimate", old)

print("== Shasta storage case ==")
sw = rf["shasta_water_balance"]
window_key = next(
    (key for key in sw["window_sensitivity"] if key.startswith("[T-30, T-1")),
    None,
)
check_value(
    "strict antecedent-window Shasta r",
    sw["window_sensitivity"].get(window_key, {}).get("r"),
    -0.748,
)
check_value("storage partial r controlling inflow", sw["inflow_controlled"]["partial_r_storage_given_inflow"], -0.724)
check_value("Shasta inflow-control years", sw["inflow_controlled"]["n_years_with_inflow"], 13, tol=0)
check_value("Shasta storage variance eta squared", sw["eta2_season_grouping"]["storage"], 0.496)
require_text("Shasta strict antecedent correlation −0.748", r"(?:−|-)?0\.748")
require_text("Shasta inflow-adjusted storage partial correlation −0.724", r"(?:−|-)?0\.724")
for old in (
    r"storage signal largely disappeared",
    r"partial correlation at\s*(?:−|-)?0\.72\b",
    r"including the peak day.{0,60}main window",
):
    forbid_text("stale Shasta window or inflow interpretation", old)

print("== Attribute attribution ==")
at = C["attribution"]
check_value("attribution reservoirs", at["n_dams"], 10, tol=0)
check_value("attribution gauges", at["n_gauges"], 21, tol=0)
check_value("attribution state clusters", at["n_state_clusters"], 7, tol=0)
check_value("dam-height p", at["dam_hgt_m"]["p"], 0.9734, tol=0.001)
check_value("attribution model R2 (%)", at["r2"] * 100, 49.75, tol=0.1)
storage_p = C["review_checks"]["attribution_review_fixes"]["storage_range_detail"]["p_clustered"]
check_value("storage-range clustered p", storage_p, 0.8939, tol=0.001)
require_text(
    "attribution sample 10 reservoirs / 21 gauges / 7 state clusters",
    r"10\s+reservoirs.{0,150}21\s+(?:tailwater\s+)?gauges.{0,150}(?:7|seven)\s+(?:state\s+)?clusters",
    r"10\s*/\s*21\s*/\s*7",
)
require_text("attribution variables are not significant", r"(?:all|none of the).{0,120}(?:not significant|null|no detectable association)")
for old in (
    r"66\s+tailwater gauges on 38 reservoirs",
    r"38\s+reservoirs.{0,120}66\s+tailwater gauges",
    r"storage.range.{0,100}p\s*=\s*0\.045",
    r"survives multiplicity correction",
):
    forbid_text("superseded attribution result", old)

print("== Multi-reservoir evidence ==")
md = C["multidam_replication"]
check_value("multi-reservoir n", md["n_dams_total"], 18, tol=0)
check_value("multi-reservoir pooled partial r", md["pooled"]["weighted_mean_rS_partial"], -0.194)
check_value("multi-reservoir negative-sign share", md["pooled"]["share_negative_rS"], 12 / 18, tol=0.01)
check_value("multi-reservoir sign-test p", rf["multidam_sign_and_dl"]["exact_binomial_p_one_sided"], 0.11894, tol=0.001)
check_value("multi-reservoir heterogeneity I2 (%)", md["pooled"]["heterogeneity_I2_pct"], 73.5, tol=0.1)
require_text("multi-reservoir sample 12/18", r"12\s+(?:of|/)\s*18")
require_text("multi-reservoir sign test p = 0.119", r"p\s*=\s*0\.119")
for old in (r"13\s+(?:of|/)\s*19", r"p\s*=\s*0\.084"):
    forbid_text("superseded non-tailwater multi-reservoir site claim", old)

print("== Storage-state panel ==")
sp = C["storage_state_panel"]["primary_non_shasta"]
check_value("storage-panel years", sp["n_years"], 310, tol=0)
check_value("storage-panel reservoirs", sp["n_reservoirs"], 18, tol=0)
check_value("storage-panel effect", sp["storage_main"]["coef"], -0.334107)
check_value("storage-panel p", sp["storage_main"]["p"], 0.03580188, tol=0.0001)
check_value("year-controlled storage effect", C["storage_state_panel"]["year_control"]["storage_main"]["coef"], -0.329841)
check_value("Shasta-inclusive storage effect", C["storage_state_panel"]["include_shasta"]["storage_main"]["coef"], -0.375253)
require_text("storage-panel sample 310/18", r"310.{0,80}18.{0,80}non-Shasta|18.{0,80}non-Shasta.{0,100}310")
require_text("storage-panel effect -0.334", r"0\.334\s*°C")
require_text("storage interaction unresolved", r"interaction.{0,160}(?:unresolved|small)")
require_text("Champion Creek exclusion reason", r"Champion\s+Creek.{0,220}six\s+storage-complete\s+years")
require_text("storage eligibility fixed before coefficient inspection", r"gates.{0,260}fixed\s+before\s+panel\s+coefficients")
require_text("storage operational endogeneity acknowledged", r"Storage\s+and\s+release\s+decisions.{0,180}outlet\s+temperature")
for old in (
    r"26\s+of\s+34 reservoirs",
    r"34 reservoirs.{0,180}(?:significant|sign imbalance).{0,100}0\.0015",
    r"p\s*=\s*0\.0015",
    r"34\s+reservoirs with 10 to 32",
):
    forbid_text("superseded multi-reservoir significance claim", old)

print("== Distance profile and mechanism limits ==")
me = C["mechanism"]
recovery = me["recovery"]
legacy_fit_keys = {"L_km", "L_ci95", "beta_inf", "beta_inf_ci95"}
check_value("canonical fitted-length fields removed", len(legacy_fit_keys.intersection(recovery)), 0, tol=0)
bins = recovery.get("distance_bins", {})
check_value("descriptive distance bins", len(bins), 3, tol=0)
for key, expected_n, expected_mean in (
    ("primary_0_10km", 19, 0.223),
    ("primary_10_30km", 41, 0.310),
    ("sensitivity_30_60km", 26, 0.392),
):
    b = bins.get(key, {})
    check_value(f"{key} n", b.get("n"), expected_n, tol=0)
    check_value(f"{key} mean beta", b.get("mean_beta"), expected_mean)
    check_value(f"{key} bootstrap interval available", len(b.get("bootstrap_ci95", [])), 2, tol=0)
d = C["distance"]
check_value("near/mid distance comparison p", d["near_vs_mid_p_less"], 0.1868, tol=0.001)
check_value("distance monotone-trend p", d["spearman_p"], 0.9697, tol=0.001)
require_text("downstream distance profile reported", r"downstream\s+profile|distance.{0,80}profile")
require_text("distance bins 0.223 and 0.310", r"0\.223.{0,180}0\.310|0\.310.{0,180}0\.223")
require_text("distance uncertainty deferred to supplement", r"interval estimates are reported")
for old in (
    r"apparent\s+L\s*=\s*3\.3\s*km",
    r"fitted\s+re-?coupling\s+length",
    r"L_km|L_ci95",
    r"recovers? to levels statistically indistinguishable",
    r"recovery to levels statistically indistinguishable",
    r"disturbed reach of order\s*10\s*km",
    r"re-coupling length",
):
    forbid_text("unsupported fitted recovery-length or distance-recovery claim", old)

print("== Dependence and HUC4 inference ==")
sv = SV
hc = rf["huc_cluster_sensitivity"]
check_value("HUC4 units", hc["n_huc4"], 46, tol=0)
check_value("HUC4 clustered contrast", hc["contrast_coef_huc"], -0.1251)
check_value("HUC4 clustered SE", hc["contrast_se_huc"], 0.0682)
check_value("HUC4 clustered p", hc["contrast_p_huc"], 0.07331, tol=0.001)
check_value("AR(1) median rho", sv["ar1_resid"]["median_rho"], 0.058)
yb = sv["year_block_boot"]["contrast_tw"]
check_value("year-block bootstrap contrast", yb["point_fe"] / 10, -0.126)
require_text("HUC4 clustered p = 0.073", r"HUC4.{0,300}p\s*=\s*0\.073|p\s*=\s*0\.073.{0,300}HUC4")
require_text("HUC4 result described as inconclusive", r"HUC4.{0,300}(?:not significant|inconclusive|does not reach|does not retain|uncertain|less precise|spatially clustered estimate)|(?:not significant|inconclusive|less precise).{0,300}HUC4")
for old in (r"survives it\s*\(t\s*=\s*3\.1,\s*p\s*=\s*0\.003", r"p\s*=\s*0\.003 on 55 df"):
    forbid_text("superseded HUC4 significance claim", old)

print("== Obsolete headline values ==")
for old in (
    r"across\s+115 reference,\s*32 above-reservoir and\s*99 tailwater",
    r"46% in the two-stage comparison",
    r"39% in the mixed model",
    r"34\s+further reservoirs",
    r"a\s+58-day lag",
    r"phase lag of about\s*58\s*days",
    r"n\s*=\s*37.{0,80}0\.18",
    r"Five same-river paired systems",
):
    forbid_text("obsolete sample or effect-size claim", old)

print()
print("AUDIT " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
