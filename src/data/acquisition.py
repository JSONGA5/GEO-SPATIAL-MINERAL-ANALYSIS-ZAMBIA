"""
Data acquisition and regional geological grid construction for the Zambian
Copperbelt and North-Western Domes corridor (EPSG:32735, UTM Zone 35S).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point, box

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict:
    """Load YAML project configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_zambia_boundary(target_crs: str = "EPSG:32735") -> gpd.GeoDataFrame:
    """Load the national boundary polygon of Zambia from data/external/zambia_boundary.geojson."""
    boundary_path = PROJECT_ROOT / "data" / "external" / "zambia_boundary.geojson"
    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path)
        return gdf.to_crs(target_crs)
    # Fallback bounding box if external file is missing
    sa = load_config()["study_area"]
    return gpd.GeoDataFrame(
        geometry=[box(sa["min_lon"], sa["min_lat"], sa["max_lon"], sa["max_lat"])],
        crs="EPSG:4326",
    ).to_crs(target_crs)


def get_zambia_reference_deposits(target_crs: str = "EPSG:32735") -> gpd.GeoDataFrame:
    """
    Load published USGS reference deposits from data/external/usgs_zambia_copper_deposits.csv
    and project to `target_crs` (default EPSG:32735, UTM Zone 35S).
    """
    csv_path = PROJECT_ROOT / "data" / "external" / "usgs_zambia_copper_deposits.csv"
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"deposit_name": "name", "structural_domain": "district"})
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    return gdf.to_crs(target_crs)


def _anisotropic_dome_distance(
    pts_xy: np.ndarray,
    center_xy: Tuple[float, float],
    semi_major_m: float,
    semi_minor_m: float,
    strike_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute normalized distance from an elongated, folded structural dome/anticline
    oriented at `strike_deg` (degrees counter-clockwise from East), with harmonic
    fold perturbations along its margin so geological contacts are irregular and
    structurally realistic rather than circular.
    """
    dx = pts_xy[:, 0] - center_xy[0]
    dy = pts_xy[:, 1] - center_xy[1]
    theta = np.deg2rad(strike_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate coordinates into structural strike / dip frame
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t

    # Harmonic structural folding along strike (mimicking Lufilian parasitic folds)
    azimuth = np.arctan2(v, u)
    fold_modulation = (
        1.0
        + 0.16 * np.cos(3.0 * azimuth + 0.7)
        + 0.11 * np.sin(5.0 * azimuth - 1.2)
    )
    elliptical_r = np.sqrt((u / semi_major_m) ** 2 + (v / semi_minor_m) ** 2)
    return elliptical_r / fold_modulation


def generate_synthetic_faults(
    bounds_utm: Tuple[float, float, float, float],
    n_faults: int = 18,
    crs: str = "EPSG:32735",
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Generate multi-vertex arcuate Lufilian thrust and basin-margin normal faults
    clipped to the Zambian study area.
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds_utm
    ref_gdf = get_zambia_reference_deposits(target_crs=crs)
    ref_coords = np.column_stack([ref_gdf.geometry.x.values, ref_gdf.geometry.y.values])

    zmb_poly = load_zambia_boundary(target_crs=crs).geometry.iloc[0]
    study_poly = zmb_poly.intersection(box(minx, miny, maxx, maxy))

    records = []
    for i in range(n_faults):
        if i < len(ref_coords):
            cx, cy = ref_coords[i]
            cx += rng.normal(0, 5500.0)
            cy += rng.normal(0, 5500.0)
            angle_deg = -40.0 if cx > 530_000 else 16.0
            angle_deg += rng.normal(0, 9.0)
            length = rng.uniform(42_000, 82_000)
        else:
            cx = rng.uniform(minx + 30_000, maxx - 30_000)
            cy = rng.uniform(miny + 25_000, maxy - 25_000)
            angle_deg = -36.0 if cx > 530_000 else 20.0
            angle_deg += rng.normal(0, 12.0)
            length = rng.uniform(32_000, 68_000)

        # Build a 5-vertex curved fault trace
        angle_rad = np.deg2rad(angle_deg)
        t_vals = np.linspace(-0.5, 0.5, 5)
        bend = rng.normal(0, 3200.0)
        coords = []
        for t in t_vals:
            along_x = cx + t * length * np.cos(angle_rad)
            along_y = cy + t * length * np.sin(angle_rad)
            # Parabolic arc offset perpendicular to strike
            perp_offset = bend * (1.0 - 4.0 * (t ** 2)) + rng.normal(0, 900.0)
            px = along_x - perp_offset * np.sin(angle_rad)
            py = along_y + perp_offset * np.cos(angle_rad)
            coords.append((px, py))

        geom = LineString(coords).intersection(study_poly)
        if geom.is_empty:
            geom = LineString(coords)
        elif geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda g: g.length)

        records.append(
            {
                "fault_id": f"FLT_{i+1:03d}",
                "structure_type": rng.choice(
                    ["Basin-Margin Normal Fault", "Lufilian Thrust", "Transfer Fault"],
                    p=[0.45, 0.35, 0.20],
                ),
                "is_synthetic": True,
                "data_source": "LUFILIAN_STRUCTURAL_INTERPRETATION",
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(records, crs=crs)


def generate_zambia_prospectivity_benchmark(
    config: Dict,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build the regional multi-disciplinary dataset clipped to Zambia's actual
    border across the Copperbelt and North-Western Domes (excluding the DRC
    Katanga Pedicle).
    """
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    geo_crs = config["crs"]["geographic_crs"]
    proj_crs = config["crs"]["projected_crs"]
    sa = config["study_area"]

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(sa["min_lon"], sa["min_lat"], sa["max_lon"], sa["max_lat"])],
        crs=geo_crs,
    ).to_crs(proj_crs)
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    # Clip study region to actual Zambia territory (carving out the DRC Pedicle notch)
    zmb_boundary = load_zambia_boundary(target_crs=proj_crs).geometry.iloc[0]
    study_polygon = zmb_boundary.intersection(box(minx, miny, maxx, maxy))

    n_points = config["synthetic_simulation"]["n_grid_points"]
    n_faults = config["synthetic_simulation"]["n_fault_segments"]

    faults_gdf = generate_synthetic_faults(
        bounds_utm=(minx, miny, maxx, maxy),
        n_faults=n_faults,
        crs=proj_crs,
        seed=seed,
    )

    # Rejection-sample spatial coordinates strictly inside Zambia's national border
    accepted_pts = []
    while len(accepted_pts) < n_points:
        batch_size = int((n_points - len(accepted_pts)) * 1.45) + 100
        bx = rng.uniform(minx, maxx, size=batch_size)
        by = rng.uniform(miny, maxy, size=batch_size)
        candidate_gs = gpd.GeoSeries([Point(x, y) for x, y in zip(bx, by)], crs=proj_crs)
        inside = candidate_gs[candidate_gs.within(study_polygon)]
        accepted_pts.extend(list(inside.values))

    points = accepted_pts[:n_points]
    sample_coords = np.array([[pt.x, pt.y] for pt in points])
    gdf = gpd.GeoDataFrame(
        {"sample_id": [f"ZMB_{i:04d}" for i in range(n_points)]},
        geometry=points,
        crs=proj_crs,
    )

    fault_union = (
        faults_gdf.geometry.union_all()
        if hasattr(faults_gdf.geometry, "union_all")
        else faults_gdf.geometry.unary_union
    )
    gdf["dist_to_fault_m"] = gdf.geometry.apply(lambda pt: pt.distance(fault_union))

    # Elongated, folded Lufilian Basement Inliers (Kafue Anticline, Konkola, Solwezi, Mwombezhi, Kabompo, Luswishi)
    structural_inliers = [
        # (lon, lat, semi_major_m, semi_minor_m, strike_deg)
        (28.12, -12.68, 56_000.0, 20_000.0, -42.0),  # Kafue Anticline (NW-SE elongated arch)
        (27.80, -12.37, 26_000.0, 13_000.0, -35.0),  # Konkola-Kirila Bombwe Dome
        (27.32, -12.88, 34_000.0, 16_000.0, -15.0),  # Luswishi Dome
        (26.42, -12.14, 34_000.0, 15_000.0, 22.0),   # Solwezi Dome (ENE-WSW)
        (25.85, -12.26, 36_000.0, 16_000.0, 18.0),   # Mwombezhi Dome (Lumwana shear)
        (25.24, -12.28, 38_000.0, 17_000.0, 28.0),   # Kabompo Dome (Sentinel margin)
    ]
    inlier_dists = []
    for lon, lat, a_m, b_m, strike in structural_inliers:
        pt_utm = gpd.GeoSeries([Point(lon, lat)], crs=geo_crs).to_crs(proj_crs).iloc[0]
        d_norm = _anisotropic_dome_distance(
            sample_coords, (pt_utm.x, pt_utm.y), a_m, b_m, strike, rng
        )
        inlier_dists.append(d_norm)

    # Regional multi-scale correlated geological field to break up symmetry
    n_knots = 90
    knot_xy = np.column_stack([
        rng.uniform(minx, maxx, size=n_knots),
        rng.uniform(miny, maxy, size=n_knots),
    ])
    knot_weights = rng.normal(0, 1.0, size=n_knots)
    dist_to_knots = cdist(sample_coords, knot_xy)
    spatial_camp_field = np.exp(-0.5 * (dist_to_knots / 15_000.0) ** 2) @ knot_weights
    spatial_camp_field = (spatial_camp_field - spatial_camp_field.mean()) / (spatial_camp_field.std() + 1e-8)

    norm_dome_dist = np.min(np.column_stack(inlier_dists), axis=1) + 0.08 * spatial_camp_field

    # Stratigraphic sequence outward from folded Basement Inliers
    strat_index = norm_dome_dist + rng.normal(0, 0.06, size=n_points)
    lith_bins = [-np.inf, 0.75, 1.18, 1.55, 1.98, np.inf]
    lith_labels = [
        "Basement_Granite_Gneiss",
        "Lower_Roan_Ore_Shale_Arkose",
        "Upper_Roan_Dolomite",
        "Mwashia_Carbonaceous_Shale",
        "Kundelungu_Glaciogenic_Siltstone",
    ]
    gdf["lithology"] = pd.cut(strat_index, bins=lith_bins, labels=lith_labels).astype(str)

    # Distance to the Lower Roan / Basement redox contact
    gdf["dist_to_redox_contact_m"] = np.abs(norm_dome_dist - 0.88) * 16_500.0 + rng.uniform(40, 550, size=n_points)

    # 2. Geophysical fields
    gdf["mag_anomaly_nt"] = (
        115.0 * np.exp(-0.5 * (norm_dome_dist / 0.92) ** 2)
        - 48.0
        + 32.0 * np.exp(-gdf["dist_to_fault_m"] / 5000.0)
        + 12.0 * spatial_camp_field
        + rng.normal(0, 10.0, size=n_points)
    )
    gdf["bouguer_gravity_mgal"] = (
        -142.0
        + 18.0 * np.tanh(norm_dome_dist - 1.1)
        + 3.5 * spatial_camp_field
        + rng.normal(0, 2.5, size=n_points)
    )
    gdf["radiometric_k_th_ratio"] = np.clip(
        0.16
        + 0.14 * np.exp(-((norm_dome_dist - 0.92) / 0.35) ** 2)
        + 0.08 * np.exp(-gdf["dist_to_fault_m"] / 4500.0)
        + rng.normal(0, 0.025, size=n_points),
        0.04,
        0.65,
    )

    # 3. Topographic features (Central African Plateau ~1150m - 1520m)
    xs, ys = sample_coords[:, 0], sample_coords[:, 1]
    y_norm = (ys - miny) / (maxy - miny)
    gdf["elevation_m"] = (
        1255.0
        + 115.0 * (1.0 - y_norm)
        + 68.0 * np.exp(-0.5 * (norm_dome_dist / 0.9) ** 2)
        + 14.0 * spatial_camp_field
        + rng.normal(0, 10.0, size=n_points)
    )
    gdf["slope_deg"] = np.clip(
        rng.gamma(shape=2.0, scale=0.9, size=n_points)
        + 2.8 * np.exp(-((norm_dome_dist - 0.85) / 0.25) ** 2)
        + 1.6 * np.exp(-gdf["dist_to_fault_m"] / 3500.0),
        0.1,
        24.0,
    )

    # 4. Mineral Systems synergy + spatial camp correlation
    is_lower_roan = (gdf["lithology"] == "Lower_Roan_Ore_Shale_Arkose").astype(float)
    is_mwashia = (gdf["lithology"] == "Mwashia_Carbonaceous_Shale").astype(float)
    favorable_host = 1.0 * is_lower_roan + 0.65 * is_mwashia

    pathway_score = np.exp(-gdf["dist_to_fault_m"] / 4800.0)
    trap_score = np.exp(-gdf["dist_to_redox_contact_m"] / 4200.0)
    synergy_score = (pathway_score ** 0.7) * (trap_score ** 0.7) * (0.35 + 0.65 * favorable_host)

    cu_log_mean = (
        3.25
        + 1.45 * synergy_score
        + 0.45 * favorable_host
        + 0.35 * spatial_camp_field
    )
    co_log_mean = (
        1.95
        + 1.25 * synergy_score
        + 0.40 * favorable_host
        + 0.30 * spatial_camp_field
    )
    gdf["cu_ppm"] = np.round(rng.lognormal(mean=cu_log_mean, sigma=0.38), 2)
    gdf["co_ppm"] = np.round(rng.lognormal(mean=co_log_mean, sigma=0.36), 2)

    latent_logit = (
        -3.85
        + 4.50 * synergy_score
        + 1.35 * (gdf["cu_ppm"] > 85.0).astype(float) * (gdf["dist_to_fault_m"] < 8000.0).astype(float)
        + 0.65 * np.log1p(gdf["cu_ppm"] / 50.0)
        + 1.55 * spatial_camp_field
        + rng.normal(0, 0.30, size=n_points)
    )
    true_prob = 1.0 / (1.0 + np.exp(-latent_logit))
    gdf["mineral_occurrence"] = rng.binomial(1, true_prob)

    gdf_wgs84 = gdf.to_crs(geo_crs)
    gdf["longitude"] = gdf_wgs84.geometry.x
    gdf["latitude"] = gdf_wgs84.geometry.y
    gdf["is_synthetic"] = True
    gdf["data_source"] = "SYNTHETIC_ZAMBIA_COPPERBELT_BENCHMARK"

    logger.info(
        "Generated %d spatial samples clipped to Zambia border (positive rate: %.2f%%) in %s",
        len(gdf),
        100.0 * gdf["mineral_occurrence"].mean(),
        proj_crs,
    )
    return gdf, faults_gdf
