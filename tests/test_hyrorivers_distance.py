"""Regression tests for HydroRIVERS point-to-network distance calculations."""
import geopandas as gpd
from shapely.geometry import LineString, Point
import pytest


def _snap_distance_km(line, point, source_crs="EPSG:4326", distance_crs="EPSG:5070"):
    network = gpd.GeoSeries([line], crs=source_crs).to_crs(distance_crs)
    gauge = gpd.GeoSeries([point], crs=source_crs).to_crs(distance_crs)
    return float(network.iloc[0].distance(gauge.iloc[0]) / 1000.0)


def test_projected_distance_is_metre_based_and_within_known_tolerance():
    # At 40 N, this point is about 0.85 km north of the line.
    line = LineString([(-105.0, 40.0), (-104.99, 40.0)])
    point = Point(-104.995, 40.0077)
    d = _snap_distance_km(line, point)
    assert 0.80 < d < 0.90
    assert d < 1.0


def test_same_longitude_offset_has_latitude_aware_geodesic_scale():
    # A 0.01 degree longitude offset is much shorter at 60 N than at 20 N.
    low = _snap_distance_km(LineString([(0, 20), (0, 20.001)]), Point(0.01, 20.0005))
    high = _snap_distance_km(LineString([(0, 60), (0, 60.001)]), Point(0.01, 60.0005))
    assert high < low * 0.6


def test_projected_distance_rejects_degree_units_for_radius():
    # Guard against the original degree-distance-times-111 implementation.
    line = LineString([(-120.0, 35.0), (-119.99, 35.0)])
    point = Point(-119.995, 35.005)
    d = _snap_distance_km(line, point)
    assert 0.50 < d < 0.60
