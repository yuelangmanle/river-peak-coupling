"""Boundary tests for the annual heatwave detector used in the current panel."""
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


_mod = SourceFileLoader(
    "panel_current_mod", str(Path(__file__).parents[1] / "code" / "pipeline" / "11_panel.py")
).load_module()


def _flat_threshold(value=20.0):
    return np.full(367, value)


def test_five_consecutive_exceedance_days_form_event():
    dates = pd.date_range("2010-07-01", periods=5)
    event = _mod.hw_days(_mod.clim_doy(dates), np.full(5, 25.0), _flat_threshold())
    assert event.sum() == 5


def test_single_missing_day_is_included_in_merged_event_span():
    dates = pd.date_range("2010-07-01", periods=7)
    values = np.array([25, 25, np.nan, 25, 25, 25, 25], dtype=float)
    event = _mod.hw_days(_mod.clim_doy(dates), np.nan_to_num(values, nan=-999), _flat_threshold())
    assert event.sum() == 7


def test_three_day_gap_splits_subthreshold_runs():
    dates = pd.date_range("2010-07-01", periods=8)
    values = np.array([25, 25, -999, -999, -999, 25, 25, 20.0])
    event = _mod.hw_days(_mod.clim_doy(dates), values, _flat_threshold())
    assert event.sum() == 0


def test_two_missing_days_split_when_exceedance_spacing_is_three():
    dates = pd.date_range("2010-07-01", periods=10)
    values = np.array([25, 25, 25, -999, -999, 25, 25, 25, 25, 25])
    event = _mod.hw_days(_mod.clim_doy(dates), values, _flat_threshold())
    assert event.sum() == 5


def test_events_are_truncated_at_year_boundaries():
    december = pd.date_range("2010-12-29", periods=3)
    january = pd.date_range("2011-01-01", periods=4)
    first = _mod.hw_days(_mod.clim_doy(december), np.full(3, 25.0), _flat_threshold())
    second = _mod.hw_days(_mod.clim_doy(january), np.full(4, 25.0), _flat_threshold())
    assert first.sum() == second.sum() == 0


def test_leap_day_calendar_alignment():
    nonleap = _mod.clim_doy(pd.to_datetime(["2009-02-28", "2009-03-01", "2009-03-02"]))
    leap = _mod.clim_doy(pd.to_datetime(["2012-02-28", "2012-02-29", "2012-03-01"]))
    assert nonleap.tolist() == [59, 61, 62]
    assert leap.tolist() == [59, 60, 61]
    assert nonleap[1] == leap[2]


def test_trailing_nonexceedance_days_are_not_counted():
    dates = pd.date_range("2010-07-01", periods=8)
    values = np.array([25, 25, 25, 25, 25, 20, 20, 20])
    event = _mod.hw_days(_mod.clim_doy(dates), values, _flat_threshold())
    assert event.sum() == 5


def test_threshold_builder_interpolates_all_calendar_slots():
    dates = pd.date_range("1996-01-01", "2021-12-31", freq="D")
    seasonal = 10 + 12 * np.sin(2 * np.pi * (dates.dayofyear - 105) / 365.25)
    values = seasonal + np.random.default_rng(42).normal(0, 1.5, len(dates))
    threshold = _mod.hw_thresholds(_mod.clim_doy(dates), values)
    assert len(threshold) == 367
    assert np.isfinite(threshold).all()
    assert threshold[200] > threshold[20]
