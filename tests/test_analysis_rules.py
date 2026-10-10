import pandas as pd
import common


def test_pair_classification_requires_elevation_and_respects_direction():
    pairs = pd.DataFrame([
        {"site_no": "00000001", "dam_id": "1", "tag": "downstream", "dist_km": 12, "elev_judge": "below"},
        {"site_no": "00000002", "dam_id": "2", "tag": "unknown", "dist_km": 18, "elev_judge": "below"},
        {"site_no": "00000003", "dam_id": "3", "tag": "upstream", "dist_km": 10, "elev_judge": "below"},
        {"site_no": "00000004", "dam_id": "4", "tag": "downstream", "dist_km": 12, "elev_judge": "above"},
        {"site_no": "00000005", "dam_id": "5", "tag": "unknown", "dist_km": 12, "elev_judge": "unknown_alt"},
        {"site_no": "00000006", "dam_id": "6", "tag": None, "dist_km": 45, "elev_judge": None},
        {"site_no": "00000007", "dam_id": "7", "tag": "downstream", "dist_km": 70, "elev_judge": "below"},
    ])

    classify_pair_sites = getattr(common, "classify_pair_sites", None)
    assert callable(classify_pair_sites), "pair classification rule is missing"
    result = classify_pair_sites(pairs).set_index("site_no")

    assert result.loc["00000001", ["tier", "sample_scope"]].tolist() == ["tailwater", "primary"]
    assert result.loc["00000002", ["tier", "sample_scope"]].tolist() == ["tailwater", "primary"]
    assert result.loc["00000003", ["tier", "sample_scope"]].tolist() == ["upstream", "primary"]
    assert result.loc["00000004", "tier"] == "excluded"
    assert result.loc["00000005", "tier"] == "excluded"
    assert result.loc["00000006", ["tier", "sample_scope"]].tolist() == ["tailwater", "sensitivity"]
    assert result.loc["00000007", "tier"] == "excluded"


def test_antecedent_window_is_exactly_thirty_days_before_peak():
    peak = pd.Timestamp("2020-08-31")
    dates = pd.date_range(peak - pd.Timedelta(days=31), peak, freq="D")

    antecedent_window_mask = getattr(common, "antecedent_window_mask", None)
    assert callable(antecedent_window_mask), "antecedent window rule is missing"
    mask = antecedent_window_mask(dates, peak, days=30)

    selected = dates[mask]
    assert len(selected) == 30
    assert selected[0] == peak - pd.Timedelta(days=30)
    assert selected[-1] == peak - pd.Timedelta(days=1)
    assert peak not in selected


def test_use_elec_encoding_preserves_unknown_and_missing_as_nan():
    values = pd.Series(["Main", "Sec", "Major", None, "other"])
    encode_use_elec = getattr(common, "encode_use_elec", None)
    assert callable(encode_use_elec), "preregistered electricity-use encoding is missing"
    encoded = encode_use_elec(values)
    assert encoded.iloc[:2].tolist() == [1.0, 0.0]
    assert encoded.iloc[2:].isna().all()
