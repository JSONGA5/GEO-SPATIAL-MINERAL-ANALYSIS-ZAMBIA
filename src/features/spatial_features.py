"""
Geospatial feature engineering and spatial autocorrelation diagnostics for
mineral prospectivity mapping in the Zambian Copperbelt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def compute_morans_i_knn(
    gdf: gpd.GeoDataFrame,
    variable_col: str,
    k_neighbors: int = 8,
) -> Dict[str, float]:
    """
    Compute global Moran's I statistic and permutation p-value using k-nearest
    spatial neighbors in projected metric coordinates (EPSG:32735).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Projected GeoDataFrame (meters).
    variable_col : str
        Continuous column to test for spatial autocorrelation.
    k_neighbors : int
        Number of nearest spatial neighbors for row-standardized weight matrix W.

    Returns
    -------
    Dict[str, float]
        Dictionary with Moran's I, expected I under null, z-score, and permutation p-value.
    """
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    values = gdf[variable_col].to_numpy(dtype=float)
    n = len(values)
    z = values - np.mean(values)
    denom = np.sum(z ** 2)
    if denom == 0:
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "p_value": 1.0}

    tree = cKDTree(coords)
    # Query k+1 because first match is the point itself (distance 0)
    _, idx = tree.query(coords, k=k_neighbors + 1)
    neighbor_idx = idx[:, 1:]

    # Row-standardized spatial lag: mean of k neighbors' deviations
    spatial_lag = np.mean(z[neighbor_idx], axis=1)
    morans_i = float(np.sum(z * spatial_lag) / denom)

    # Permutation test (199 permutations) for empirical p-value
    rng = np.random.default_rng(42)
    perm_i = np.empty(199, dtype=float)
    for p in range(199):
        z_perm = rng.permutation(z)
        lag_perm = np.mean(z_perm[neighbor_idx], axis=1)
        perm_i[p] = np.sum(z_perm * lag_perm) / denom

    expected_i = -1.0 / (n - 1)
    p_val = float((np.sum(np.abs(perm_i - expected_i) >= np.abs(morans_i - expected_i)) + 1) / 200.0)
    return {
        "morans_i": round(morans_i, 4),
        "expected_i": round(expected_i, 5),
        "perm_mean_i": round(float(np.mean(perm_i)), 4),
        "perm_std_i": round(float(np.std(perm_i)), 4),
        "p_value": round(p_val, 4),
    }


def sample_raster_at_points(
    gdf: gpd.GeoDataFrame,
    raster_path: str | Path,
    column_name: str,
) -> gpd.GeoDataFrame:
    """
    Sample a GeoTIFF raster at point geometries using Rasterio, automatically
    reprojecting query coordinates if the raster CRS differs from the GeoDataFrame CRS.
    """
    out_gdf = gdf.copy()
    with rasterio.open(raster_path) as src:
        pts_gdf = out_gdf.to_crs(src.crs) if src.crs and str(src.crs) != str(out_gdf.crs) else out_gdf
        coords = [(pt.x, pt.y) for pt in pts_gdf.geometry]
        sampled = [val[0] if len(val) > 0 else np.nan for val in src.sample(coords)]
        nodata = src.nodata
        if nodata is not None:
            sampled = [np.nan if v == nodata else float(v) for v in sampled]
        out_gdf[column_name] = sampled
    return out_gdf


def engineer_geospatial_features(
    samples_gdf: gpd.GeoDataFrame,
    faults_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Engineer physically grounded geological, geochemical, geophysical, and
    topographic predictors in UTM Zone 35S (EPSG:32735).

    Engineered Features & Scientific Rationale
    ------------------------------------------
    1. `fault_proximity_exp`: exp(-dist_to_fault_m / 5000). Models hydrothermal /
       basinal brine fluid pressure decay away from structural conduits.
    2. `fault_density_10km`: Count of fault segments within a 10 km structural
       damage zone buffer (captures structural intersections & dilation zones).
    3. `redox_trap_proximity_exp`: exp(-dist_to_redox_contact_m / 4000). Captures
       proximity to the Lower Roan reduced Ore Shale / oxidized arkose redox boundary.
    4. `structural_redox_interaction`: Product of `fault_proximity_exp` and
       `redox_trap_proximity_exp` (coincidence of fluid pathway + chemical trap).
    5. `log10_cu_ppm` & `log10_co_ppm`: Log-transformed trace element concentrations
       to stabilize variance and account for log-normal crustal dispersion.
    6. `cu_co_log_ratio`: log10(Cu / Co) ratio to distinguish stratiform Cu-Co
       mineralization halos from barren mafic background Co.
    7. `geochem_anomaly_index`: Standardized composite Z-score of log10(Cu) + log10(Co).
    8. `mag_vertical_gradient_proxy`: Local spatial deviation of magnetic anomaly
       from its 8-nearest-neighbor regional background (isolating shallow structures).
    9. `topographic_wetness_proxy`: log((1 + catchment_proxy) / tan(slope_rad)),
       controlling hydromorphic dispersion of Cu/Co in surface soils/streams.
    10. One-hot encoded lithology indicators (`lith_*`).
    """
    if not samples_gdf.crs or not samples_gdf.crs.is_projected:
        raise ValueError("samples_gdf must be in a projected metric CRS (e.g., EPSG:32735).")

    feat_gdf = samples_gdf.copy()

    # 1. Structural pathway features
    feat_gdf["fault_proximity_exp"] = np.exp(-feat_gdf["dist_to_fault_m"] / 5000.0)

    if faults_gdf is not None and len(faults_gdf) > 0:
        faults_proj = faults_gdf.to_crs(feat_gdf.crs)
        # Count how many fault segments lie within 10,000 m of each sample location
        buffers = feat_gdf.geometry.buffer(10_000.0)
        sindex = faults_proj.sindex
        counts = []
        for buf in buffers:
            possible_idx = list(sindex.intersection(buf.bounds))
            if not possible_idx:
                counts.append(0)
            else:
                sub = faults_proj.iloc[possible_idx]
                counts.append(int(sub.intersects(buf).sum()))
        feat_gdf["fault_density_10km"] = counts
    else:
        feat_gdf["fault_density_10km"] = (feat_gdf["dist_to_fault_m"] < 10_000.0).astype(int)

    # 2. Chemical/stratigraphic trap features
    feat_gdf["redox_trap_proximity_exp"] = np.exp(-feat_gdf["dist_to_redox_contact_m"] / 4000.0)
    feat_gdf["structural_redox_interaction"] = (
        feat_gdf["fault_proximity_exp"] * feat_gdf["redox_trap_proximity_exp"]
    )

    # 3. Geochemical transformation & pathfinder ratios
    feat_gdf["log10_cu_ppm"] = np.log10(np.clip(feat_gdf["cu_ppm"], 0.1, None))
    feat_gdf["log10_co_ppm"] = np.log10(np.clip(feat_gdf["co_ppm"], 0.1, None))
    feat_gdf["cu_co_log_ratio"] = feat_gdf["log10_cu_ppm"] - feat_gdf["log10_co_ppm"]

    cu_z = (feat_gdf["log10_cu_ppm"] - feat_gdf["log10_cu_ppm"].mean()) / feat_gdf["log10_cu_ppm"].std()
    co_z = (feat_gdf["log10_co_ppm"] - feat_gdf["log10_co_ppm"].mean()) / feat_gdf["log10_co_ppm"].std()
    feat_gdf["geochem_anomaly_index"] = 0.6 * cu_z + 0.4 * co_z

    # 4. Geophysical residual anomaly & local geochemical dispersion halo (k=8 neighbors)
    coords = np.column_stack([feat_gdf.geometry.x.values, feat_gdf.geometry.y.values])
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(9, len(feat_gdf)))
    neighbor_mag = feat_gdf["mag_anomaly_nt"].to_numpy()[idx[:, 1:]]
    feat_gdf["mag_residual_nt"] = feat_gdf["mag_anomaly_nt"] - np.mean(neighbor_mag, axis=1)
    neighbor_cu = feat_gdf["log10_cu_ppm"].to_numpy()[idx[:, 1:]]
    feat_gdf["local_cu_halo_log10"] = np.mean(neighbor_cu, axis=1)

    # 5. Topographic / Regolith dispersion proxy
    slope_rad = np.deg2rad(np.clip(feat_gdf["slope_deg"], 0.2, 85.0))
    feat_gdf["topographic_wetness_proxy"] = np.log(1000.0 / np.tan(slope_rad))

    # 6. Lithology one-hot encoding (drop Basement_Granite_Gneiss as reference category)
    lith_dummies = pd.get_dummies(feat_gdf["lithology"], prefix="lith", dtype=float)
    ref_col = "lith_Basement_Granite_Gneiss"
    if ref_col in lith_dummies.columns:
        lith_dummies = lith_dummies.drop(columns=[ref_col])
    feat_gdf = pd.concat([feat_gdf, lith_dummies], axis=1)

    feature_cols = [
        "dist_to_fault_m",
        "dist_to_redox_contact_m",
        "fault_proximity_exp",
        "fault_density_10km",
        "redox_trap_proximity_exp",
        "structural_redox_interaction",
        "log10_cu_ppm",
        "log10_co_ppm",
        "local_cu_halo_log10",
        "cu_co_log_ratio",
        "mag_anomaly_nt",
        "mag_residual_nt",
        "bouguer_gravity_mgal",
        "radiometric_k_th_ratio",
        "elevation_m",
        "slope_deg",
        "topographic_wetness_proxy",
    ] + list(lith_dummies.columns)

    logger.info("Engineered %d model features.", len(feature_cols))
    return feat_gdf, feature_cols


def compute_vif_table(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) to diagnose multicollinearity
    among candidate predictors prior to Logistic Regression modeling.
    """
    X = df[feature_cols].dropna().astype(float)
    X_const = X.assign(intercept=1.0)
    rows = []
    for i, col in enumerate(feature_cols):
        vif_val = variance_inflation_factor(X_const.values, i)
        rows.append({"feature": col, "VIF": round(float(vif_val), 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
