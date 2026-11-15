"""Unit tests for geospatial data validation, CRS standardization, and geochemical outlier audits."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.data.cleaning import summarize_geochemical_outliers, validate_and_standardize_geodata


def test_validate_and_standardize_geodata_filters_invalid_and_reprojects():
    pts = [
        Point(27.5, -12.5),  # Valid Copperbelt coordinate
        Point(27.5, -12.5),  # Duplicate coordinate
        Point(10.0, 50.0),   # Out of Zambia bounds (Europe)
        None,                # Null geometry
    ]
    gdf = gpd.GeoDataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "cu_ppm": [120.0, 140.0, 50.0, 80.0],
        },
        geometry=pts,
        crs="EPSG:4326",
    )

    cleaned, report = validate_and_standardize_geodata(gdf, target_crs="EPSG:32735")
    assert len(cleaned) == 1
    assert report["null_geometries_removed"] == 1
    assert report["out_of_zambia_bounds_removed"] == 1
    assert report["duplicate_coordinates_removed"] == 1
    assert report["is_projected_metric"] is True
    assert cleaned.crs.to_epsg() == 32735


def test_validate_raises_on_missing_crs():
    gdf = gpd.GeoDataFrame({"sample_id": ["S1"]}, geometry=[Point(27.5, -12.5)], crs=None)
    with pytest.raises(ValueError, match="no defined CRS"):
        validate_and_standardize_geodata(gdf)


def test_summarize_geochemical_outliers_detects_lognormal_anomalies():
    rng = np.random.default_rng(42)
    background = rng.lognormal(mean=3.0, sigma=0.3, size=200)
    anomalies = np.array([1500.0, 2400.0, 3200.0])
    df = pd.DataFrame({"cu_ppm": np.concatenate([background, anomalies])})

    summary = summarize_geochemical_outliers(df, ["cu_ppm"])
    assert len(summary) == 1
    assert summary.loc[0, "anomalous_samples_count"] >= 3
    assert summary.loc[0, "log10_mad_threshold_ppm"] > summary.loc[0, "median_ppm"]
