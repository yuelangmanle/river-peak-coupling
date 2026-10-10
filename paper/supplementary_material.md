# Supplementary Material

## Supplementary Table S1. Candidate-gauge eligibility and final tier assignment

The 353 rows in `gauge_eligibility.csv` are candidate gauges entering the eligibility audit before final tier reassignment; they are not 353 final analysis tiers. Preliminary labels were reference (147), above-reservoir (37) and tailwater (169). The USGS site-type screen removed 14 non-stream gauges (8 lake/pond, 4 spring, 1 estuary and 1 groundwater well), all carrying a preliminary tailwater label, leaving 339 stream-type candidates.

Applying the final distance, direction, elevation and precedence rules assigned 295 gauges: 150 reference, 41 above-reservoir, 71 primary tailwater (≤30 km) and 33 tailwater sensitivity-band (30–60 km). Of the 155 stream-type candidates carrying the preliminary tailwater label, 7 were reassigned to another tier by precedence rules (3 reference and 4 above-reservoir). The remaining 44 were not assigned to a final tier because their linked pair records did not satisfy the required below-reservoir elevation criterion: 37 were judged above and 7 had unresolved alternative elevations. Before the site-type screen, the corresponding candidate counts were 41 and 7; four non-stream gauges among them were removed at that earlier step. This reconciles the 339 stream-type candidates with the 295 final tier assignments. After the 90% annual-coverage rule and the ≥8 eligible-year gate, the per-gauge samples were 117 reference, 36 above-reservoir and 60 primary tailwater gauges, plus 26 gauges in the separate sensitivity band. Full candidate dispositions: `gauge_eligibility.csv` (353 rows).

## Supplementary Table S2. River-network routing verification

Every tailwater and above-reservoir gauge was snapped to the NHDPlus river network via the USGS Network-Linked Data Index (NLDI). A tailwater pair is *verified* when the gauge reach lies on the dam's downstream main-stem path (35 km along-network) by reach identity (`pass`) or within 2 km of that path (`near`); `fail` = more than 2 km off the path; `unresolved` = position lookup failed. Full table: `routing_verification.csv`.

| Direction | pass | near | fail | unresolved |
|---|---|---|---|---|
| Tailwater (downstream of dam) | 60 | 10 | 125 | 2 |
| Above-reservoir (upstream of dam) | 3 | 2 | 19 | 31 |

Verified-only analyses retain all reference gauges; a tailwater gauge enters only if its pair is NLDI-verified (pass/near). After the ≥8-year eligibility gate, 24 tailwater gauges remain: mean β = 0.175 versus 0.483 at reference (one-sided Mann–Whitney p = 5.22 × 10⁻⁶). The corresponding mixed model gives a per-degree contrast of approximately −0.230 (p = 2.11 × 10⁻⁵).

**Upstream verified-only rule correction.** An earlier registered version composed the upstream verified-only tier by excluding NLDI-fail gauges only (n = 24), an asymmetric rule that could exceed the number of verifiable upstream pairs. The corrected rule variants are: NLDI-verified upstream (5 gauges, too few for reanalysis) and HydroRIVERS-verified upstream (16 gauges with ≥8 eligible years, mean β = 0.469 versus reference 0.483, two-sided p = 0.559) — the verified upstream control behaves like reference. Registry key: `routing_verification.verified_only_rule_note`.

**Independent cross-check (HydroRIVERS v10 NA).** A chain-walk algorithm on the HydroRIVERS network — a different dataset (HydroSHEDS) and algorithm from the NHDPlus/NLDI check above — repeats the verification: a gauge is upstream-verified when the NEXT_DOWN chain from its snapped segment reaches the dam's segment (cap 150 km), and downstream-verified when the chain from the dam's segment reaches the gauge's segment (cap 60 km; snap tolerance 1 km).

| Direction | verified | fail | unsnap/unresolved |
|---|---|---|---|
| Tailwater (dam → gauge) | 69 | 117 | 11 |
| Above-reservoir (gauge → dam) | 18 | 34 | 3 |

After reprojecting HydroRIVERS to a CONUS metre-based CRS for point-to-line snapping, downstream verdicts agree between the two sources for 166 of 197 pairs (84.3%). Restricting to the 54 tailwater pairs verified by **both** sources leaves 17 tailwater gauges with ≥8 eligible years: mean β = 0.195 versus 0.483 at reference (two-stage p = 3.26 × 10⁻⁴); the mixed-model contrast is −0.215 per degree (p = 7.72 × 10⁻⁴). The deficit therefore persists under both routing checks, although the point estimate is not interpreted as a monotone function of verification stringency. The released HydroRIVERS script asserts a geographic source CRS and performs all distance thresholds after reprojection; its three distance-regression tests are included with the pipeline tests.

## Supplementary Table S3. Analysis-plan deviations and post hoc additions

| # | Type | Description |
|---|---|---|
| 1 | deviation | Planned nested reservoir random intercept did not converge. The gauge-level mixed model is reported with model-based uncertainty; the joint reservoir/gauge bootstrap reports an interval for the unadjusted two-stage mean contrast, a different estimand. Reference and upstream gauges are resampled by gauge, so the bootstrap does not remove all spatial dependence. The estimates are presented in parallel, not as independent replications. |
| 2 | deviation | Heatwave-day transmission used NB-GEE rather than the planned mixed model |
| 3 | deviation | Two planned attribution covariates dropped (encoding correction; insufficient variation) |
| 4 | deviation | Precision-weighted synthesis demoted from primary to robustness |
| 5 | post hoc | Distance-decay binning (≤10 km vs 10–30 km) |
| 6 | post hoc | RMA error-symmetric estimation |
| 7 | post hoc | Climate-position matching of β (49 pairs, ±1 °C) |
| 8 | post hoc | Reservoir-level two-stage comparison (30 dams after the coverage rule) |
| 9 | post hoc | Daily-mean exclusion test (0–5% of gauges by tier) |
| 10 | post hoc | Matched and multi-threshold exposure analyses |
| 11 | post hoc | Heatwave-rule correction and exploratory equivalence check using a 0.75–1.25 transmission-ratio margin; the margin was not prespecified and is not used as the primary inference |
| 12 | post hoc | Peak-timing offset analysis |
| 13 | post hoc | Thermal-state matching of exposure sensitivity |
| 14 | post hoc | Shasta case study (antecedent storage variants) |
| 15 | post hoc | Stream-type eligibility screen (14 gauges removed) |
| 16 | post hoc | 90% annual-coverage rule extended to AMWT7 and peak dates (previously exposure/heatwave only) |
| 17 | post hoc | The M1 convergence gate was added after the initial plan. The canonical adjudication rule also designated M1 as primary only when it converged and its contrast p value was <0.05. Because this significance-conditioned rule can affect which result receives emphasis, the manuscript reports the model-based contrast and the joint-bootstrap mean difference in parallel, with their distinct estimands, and does not treat either as confirmatory evidence selected by that gate. |
| 18 | post hoc | Heatwave event rule rebuilt on a complete daily calendar with a 366-day climatology calendar (leap alignment) and boundary unit tests |
| 19 | post hoc | River-network routing verification (NLDI/NHDPlus) and verified-pair sensitivity |
| 20 | post hoc | Within-gauge quadratic nonlinearity test (thermal-plateau alternative) |
| 21 | post hoc | Inference robustness: AR(1) residual diagnostic, year-block bootstrap, two-way clustered intervals |
| 22 | post hoc | Storage-state panel added after inspection of the mechanism evidence; the multi-reservoir site selector was corrected to require primary tailwater gauges. Champion Creek Dam is excluded because its eligible tailwater gauge has six storage-complete years, below the ten-year gate. |

The above-reservoir comparison is an auxiliary siting comparator. Its 80%-power minimum detectable slope contrast was 0.144 °C/°C, similar in magnitude to the mixed-model tailwater–reference contrast (0.143 °C/°C); the near-reference estimate therefore does not establish equivalence or rule out smaller upstream differences.

## Supplementary Table S4. Signal-to-noise and Shasta power checks

The signal-to-noise ladder is reported as a sensitivity analysis because filtering on |r| also selects on the coupling being studied. Across the full sample, attenuation was 41.5% (reference β = 0.483; tailwater β = 0.283; p = 1.10 × 10⁻⁵). The corresponding values were 30.9% (|r| ≥ 0.3; p = 9.41 × 10⁻⁵), 22.6% (|r| ≥ 0.4; p = 0.00345), 21.2% (|r| ≥ 0.5; p = 0.00640), 11.4% (|r| ≥ 0.3 and ≥15 years; p = 0.0787), and −1.6% (|r| ≥ 0.4 and ≥18 years; p = 0.224). These thresholds progressively reduce the sample and answer different questions from the primary comparison.

The Shasta storage association is based on 19 years. A noncentral-t calculation gives an approximately 0.598 partial-correlation magnitude as the 80%-power detection threshold; the observed air partial correlation was 0.30 (p = 0.21), so the case supports a storage association without resolving a small atmospheric partial contribution.

## Supplementary Table S5. Same-river paired systems

| Dam | Upstream gauge | Downstream gauge | Overlapping years | Δ AMWT7 (°C) |
|---|---|---|---|---|
| Don Pedro | 11276600 | 11289650 | 25 | −7.9 |
| James H. Turner | 11172945 | 11173575 | 7 | +0.6 |
| Lake Tapps Dike 2-B | 12098700 | 12100490 | 8 | +2.3 |

## Supplementary Table S6. Matching and coverage details

- Climate-position matching: ±1 °C caliper on gauge-mean annual peak air temperature; reference-driven greedy pairing; released tables `matched_pairs_beta.csv` (49 pairs) and `matched_pairs_exposure.csv` (57 pairs).
- Annual coverage: ≥90% of calendar days required for every annual metric (AMWT7, peak date, threshold-days, heatwave days). The regenerated annual panel contains 5,179 gauge-years: 4,546 in the primary scope and 633 in the 30–60 km sensitivity band. The primary mixed model uses 3,676 primary-scope gauge-years with complete AMWT7 and exposure metrics; the percentile heatwave GEE uses 3,666 gauge-years with complete heatwave metrics. These gauge-year samples are not restricted to the ≥8 eligible-year threshold used for per-gauge β summaries, so their 251 mixed-model gauges need not equal the 213-gauge primary per-gauge sample.
- Heatwave definition: Hobday et al. (2016) as implemented in the reference marine heatwave code, evaluated on a complete daily calendar with a 366-day climatology calendar for leap-year alignment (exceedance runs split at gaps >2 calendar days, spans ≥5 days retained, internal single-day gaps counted, trailing non-exceedance excluded). Boundary behaviour is covered by released unit tests.


## Supplementary Table S7. Peak-timing analysis

Peak dates are the end dates of the strict 7-day annual maximum water and air windows. For each gauge, the circular offset was computed as water-peak day minus air-peak day on a 365-day ring. The primary comparison used the same ≥8 eligible-year gate as the per-gauge β analysis, yielding 117 reference, 36 above-reservoir and 60 tailwater gauges. Mean gauge-level offsets were 0.9, 4.3 and 8.0 days, respectively; the tailwater–reference mean difference was 7.1 days (gauge bootstrap 95% CI 1.7–13.1 days; Welch two-sided p = 0.018). The corresponding Mann–Whitney comparison was less sensitive to the tied one-day medians (two-sided p = 0.134). Medians were 1 day in both reference and tailwater tiers. The timing result therefore reflects a longer-tailed downstream offset distribution rather than a uniform shift at every gauge. The timing comparison is complementary to the primary β estimand and does not redefine the heatwave-day outcome. The calculation is reproduced by `code/pipeline/35_peak_timing_analysis.py`.

## Supplementary Table S8. Inference robustness and secondary checks

The primary tier contrast was stable across alternative dependence structures, clustering choices and signal-to-noise restrictions. The following checks are secondary diagnostics; they are reported here to keep the main text focused on the hydrological interpretation of the benchmark.

| Check | Estimate or result |
|---|---|
| Within-gauge quadratic term | Tailwater −0.010 (p = 0.63); reference +0.002 (p = 0.77); no evidence of a broad thermal plateau. |
| HUC4-clustered contrast | −0.1251 °C/°C (SE 0.0682; p = 0.07331; 46 HUC4 units). |
| ≥8-year model | Reference β = 0.446; tailwater β = 0.302; p = 0.00056. |
| High-signal subset (|r| ≥ 0.4) | 22.6% attenuation; p = 0.00345. |
| High-signal, long-record subset (|r| ≥ 0.4; ≥18 years) | p = 0.224; the estimate is less precise after the joint record-length and signal filter. |
| Year-block resampling | The negative tailwater contrast was retained under blocks defined by calendar year. |
| Conservative gauge/year standard errors | The ordering was unchanged when the larger of gauge- and year-clustered standard errors was used. |
| Residual dependence | AR(1) and residual-autocorrelation diagnostics did not alter the direction of the primary contrast. |
| Equal-weight reservoir bootstrap | Giving each reservoir equal weight preserved the negative tailwater contrast. |

The full machine-readable outputs, including bootstrap draws and model-specific confidence intervals, are archived with the release tables and can be regenerated with `reproduce.py --audit-only`.

## Supplementary Table S9. Common-scale response-geometry diagnostic

This diagnostic formally compares peak sensitivity and percentile-defined event transmission on the same gauge-level relative scale. For each primary gauge, the analysis estimates (i) the slope of annual AMWT7 against annual peak air temperature and (ii) the slope of annual heatwave-day count against annual air heatwave-day count. The tailwater/reference ratio is calculated separately for each slope, and the separation statistic is the event-transmission ratio minus the peak-sensitivity ratio. The comparison is on relative ratios rather than raw response units, so it tests the geometry of tier contrasts rather than equating temperature and event-count slopes. Uncertainty uses 5,000 tier-stratified bootstrap resamples of gauges (seed = 20261010); within each bootstrap replicate, the reference and tailwater gauge samples are drawn once and both slope ratios and their difference are calculated from that same draw.

| Estimand | Reference | Tailwater | Tailwater/reference | 95% bootstrap CI | Gauge counts |
|---|---:|---:|---:|---|---:|
| Peak sensitivity slope (°C/°C) | 0.483119 | 0.282641 | 0.585033 | 0.424699–0.766292 | 117 / 60 |
| Event-transmission slope | 1.007225 | 1.142495 | 1.134299 | 0.886947–1.466581 | 116 / 60 |

The event-transmission slope uses 116 reference gauges because one reference gauge lacked a usable annual heatwave-count series; all 60 tailwater gauges remained eligible.
| Separation: event ratio − peak ratio | — | — | 0.549267 | 0.244136–0.893091 | common-scale diagnostic |

The separation statistic is positive in all 5,000 bootstrap draws. It is a formal relative-scale diagnostic rather than a replacement for the primary mixed model or negative-binomial GEE, because the two slopes summarize different response variables. The reproducible script is `code/pipeline/33_response_geometry.py`; machine-readable outputs are `data/samples/analysis/response_geometry.json` and `data/samples/analysis/response_geometry_gauge.csv`.

## Supplementary Table S10. State-dependent mechanism bridge

The national benchmark is the baseline stage of a state-dependent model in which antecedent storage or withdrawal state, discharge and release configuration can modify the intercept and, potentially, the atmospheric slope of a regulated reach. The released Shasta record supplies the strongest within-system state signal (storage–peak r = −0.748; storage partial r = −0.724 after conditioning on the atmospheric peak and available inflow), while the corrected 18-reservoir comparison supplies a heterogeneous cross-system test (pooled partial r = −0.194; I² = 73.5%). The storage-state panel provides the corresponding station-year intercept estimate. These estimates do not imply a universal storage coefficient or a resolved slope interaction.

| Mechanism term | Available evidence in this release | Predictive role | Next direct measurement |
|---|---|---|---|
| Atmospheric peak, \(A^{peak}\) | 213-gauge three-tier benchmark; baseline β | Sets the unregulated and regulated response geometry | Continued daily air and water records |
| Antecedent storage, \(S\) | Shasta, 18-reservoir comparison and 310-year storage panel | Shifts baseline peak; may modify β | Storage history aligned to each gauge-year |
| Inflow/discharge, \(Q\) | Shasta inflow-control subset; operation-level outflow summaries | Separates storage-state effects from flow-mediated exchange | Complete station-year discharge series |
| Release/withdrawal state | Interpreted from storage, timing and reservoir configuration | Links reservoir state to outlet thermal condition | Outlet temperature profiles and withdrawal depth |

The corresponding test model is given in Section 2.7 of the manuscript. It is a mechanism extension of the benchmark and does not alter the primary tier estimands.

## Supplementary Table S11. Flow-aware peak-coupling sensitivity

The flow-aware sensitivity links each annual AMWT7 value to the mean daily discharge over the seven days ending on the AMWT7 end date. Discharge is USGS NWIS parameter 00060 with statistic 00003 (mean daily value); windows with fewer than five valid days were excluded. The analysis uses 2,841 primary gauge-years from 202 gauges. The complete-flow subset is not a replacement for the primary national estimand because station-year discharge coverage is incomplete and the window is aligned to the thermal peak.

| Model | Reference slope | Tailwater slope | Tailwater − reference | 95% CI | p | N |
|---|---:|---:|---:|---|---:|---|
| Complete-flow subset, no Q term | 0.462031 | 0.304976 | −0.157055 | −0.243785 to −0.070325 | 0.0003863 | 2,841 / 202 |
| Add centered log10(Q_peak7) and tier interaction | 0.427173 | 0.253859 | −0.173314 | −0.260417 to −0.086211 | 0.00009622 | 2,841 / 202 |

The result is interpreted as persistence of the tier contrast under observed peak-window flow conditions, not as causal mediation or a completed universal flow-adjusted model. The reproducible script is `code/pipeline/34_discharge_sensitivity.py`; the compact input table is `data/samples/analysis/discharge_peak_window.csv`, and the machine-readable output is `data/samples/analysis/discharge_sensitivity.json`. Raw daily discharge is not redistributed.

## Supplementary Table S12. Post-hoc storage-state panel

The panel uses one longest-record primary tailwater gauge per reservoir, at least 10 complete station-years per reservoir, at least 20 valid storage days in [T−30, T−1], within-reservoir standardized storage, within-reservoir centered air temperature and reservoir fixed effects. The 20-day and ten-year gates follow the existing multi-reservoir protocol and were fixed before the panel coefficient was inspected. Champion Creek Dam is the only reservoir excluded by the ten-year rule: its eligible tailwater gauge supplies six storage-complete years. Standard errors are clustered by reservoir. The primary model excludes Shasta; the Shasta sensitivity adds the separately archived 19-year case record.

| Model | N years / reservoirs | Storage main effect (°C per within-reservoir SD) | 95% CI | p | Air × storage |
|---|---:|---:|---|---:|---:|
| Primary non-Shasta | 310 / 18 | −0.334 | −0.643 to −0.025 | 0.036 | 0.016 (p = 0.796) |
| Add centered year | 310 / 18 | −0.330 | −0.654 to −0.006 | 0.046 | 0.018 (p = 0.781) |
| Add Shasta case | 329 / 19 | −0.375 | −0.677 to −0.074 | 0.018 | 0.021 (p = 0.718) |

The reservoir-cluster bootstrap gave a storage median of −0.328 °C per SD (95% CI −0.610 to −0.079); 99.6% of successful bootstrap estimates were negative. Leave-one-reservoir-out runs were not used to summarize the storage coefficient; they showed that the air-by-storage interaction remained unresolved (range −0.028 to 0.042). The panel is exploratory, so interpretation emphasizes the fixed eligibility rule, clustered estimate, bootstrap interval, year control and Shasta sensitivity rather than a confirmatory p-value. The reproducible script is `code/pipeline/35_storage_state_panel.py`; the compact input is `data/samples/analysis/storage_peak_panel.csv`, and the machine-readable output is `data/samples/analysis/storage_state_panel.json`. Raw ResOpsUS time series are not redistributed.
