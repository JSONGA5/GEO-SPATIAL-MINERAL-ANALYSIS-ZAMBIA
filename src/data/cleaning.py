"""
Geospatial data quality, coordinate validation, and CRS standardization utilities.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_and_standardize_geodata(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:32735",
    lon_bounds: Tuple[float, float] = (21.5, 34.0),
    lat_bounds: Tuple[float, float] = (-18.5, -8.0),
    id_col: str = "sample_id",
) -> Tuple[gpd.GeoDataFrame, Dict[str, int | float | str]]:
    """
    Validate geometry integrity, bounding coordinates (Zambia national extent),
    duplicate records, missing values, and reproject to target metric CRS.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geospatial dataset.
    target_crs : str
        Target metric CRS (default EPSG:32735 for UTM Zone 35S).
    lon_bounds : Tuple[float, float]
        Valid longitude interval in decimal degrees for Zambia.
    lat_bounds : Tuple[float, float]
        Valid latitude interval in decimal degrees for Zambia.
    id_col : str
        Unique identifier column name.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, Dict[str, int | float | str]]
        Cleaned GeoDataFrame and a structured quality audit report.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no defined CRS. Refusing implicit CRS assumption.")

    initial_count = len(gdf)
    cleaned = gdf.copy()

    # 1. Geometry validity & null check
    null_geom_count = int(cleaned.geometry.isna().sum())
    invalid_geom_count = int((~cleaned.geometry.is_valid).sum())
    cleaned = cleaned[cleaned.geometry.notna() & cleaned.geometry.is_valid].copy()

    # 2. Coordinate bounds check in WGS84 (EPSG:4326)
    wgs84_gdf = cleaned.to_crs("EPSG:4326")
    lons = wgs84_gdf.geometry.x
    lats = wgs84_gdf.geometry.y
    in_bounds_mask = (
        (lons >= lon_bounds[0])
        & (lons <= lon_bounds[1])
        & (lats >= lat_bounds[0])
        & (lats <= lat_bounds[1])
    )
    out_of_bounds_count = int((~in_bounds_mask).sum())
    cleaned = cleaned.loc[in_bounds_mask].copy()

    # 3. Duplicate ID & spatial coordinate deduplication
    dup_id_count = int(cleaned.duplicated(subset=[id_col]).sum()) if id_col in cleaned.columns else 0
    if id_col in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[id_col], keep="first")

    dup_coord_count = int(cleaned.geometry.duplicated().sum())
    cleaned = cleaned.loc[~cleaned.geometry.duplicated()].copy()

    # 4. Standardize to metric CRS (UTM Zone 35S)
    if str(cleaned.crs) != target_crs:
        cleaned = cleaned.to_crs(target_crs)

    audit_report = {
        "initial_records": initial_count,
        "null_geometries_removed": null_geom_count,
        "invalid_geometries_removed": invalid_geom_count,
        "out_of_zambia_bounds_removed": out_of_bounds_count,
        "duplicate_ids_removed": dup_id_count,
        "duplicate_coordinates_removed": dup_coord_count,
        "final_valid_records": len(cleaned),
        "target_crs": str(cleaned.crs),
        "is_projected_metric": bool(cleaned.crs.is_projected),
    }
    logger.info("Geospatial QA audit complete: %s", audit_report)
    return cleaned, audit_report


def summarize_geochemical_outliers(
    df: pd.DataFrame, element_cols: List[str]
) -> pd.DataFrame:
    """
    Evaluate geochemical distributions in both raw and log-transformed space.
    In mineral exploration, upper-tail 'outliers' are often true mineralization anomalies
    (pathfinder halos) rather than data errors and must NOT be blindly winsorized or dropped.
    """
    rows = []
    for col in element_cols:
        series = df[col].dropna()
        log_s = np.log10(np.clip(series, a_min=1e-6, a_max=None))
        med = np.median(log_s)
        mad = np.median(np.abs(log_s - med))
        # Threshold: median + 2 * MAD in log10 space (standard exploration geochemistry threshold)
        log_thresh = med + 2.0 * (1.4826 * mad)
        ppm_thresh = 10.0 ** log_thresh
        n_anomalous = int((series > ppm_thresh).sum())
        rows.append(
            {
                "element": col,
                "count": len(series),
                "missing_pct": round(100.0 * df[col].isna().mean(), 2),
                "median_ppm": round(float(series.median()), 2),
                "iqr_ppm": round(float(series.quantile(0.75) - series.quantile(0.25)), 2),
                "max_ppm": round(float(series.max()), 2),
                "log10_mad_threshold_ppm": round(float(ppm_thresh), 2),
                "anomalous_samples_count": n_anomalous,
                "anomalous_samples_pct": round(100.0 * n_anomalous / len(series), 2),
            }
        )
    return pd.DataFrame(rows)
