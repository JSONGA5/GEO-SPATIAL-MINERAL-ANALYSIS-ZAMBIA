"""
Spatial Block Cross-Validation to prevent spatial autocorrelation data leakage
in mineral prospectivity mapping.
"""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


def assign_spatial_blocks(
    gdf: gpd.GeoDataFrame,
    block_size_m: float = 25_000.0,
) -> pd.Series:
    """
    Partition a projected metric GeoDataFrame (e.g., EPSG:32735) into regular
    2D square spatial blocks of side `block_size_m` meters.

    Why Spatial Blocking Matters
    ----------------------------
    Geological, geochemical, and geophysical fields exhibit positive spatial
    autocorrelation (Tobler's First Law: nearby samples share similar lithology,
    fault distance, and metal halos). Random train/test splitting places training
    points immediately adjacent to test points, leaking local spatial trends and
    inflating ROC-AUC and PR-AUC. Spatial blocking forces models to generalize to
    held-out geographic sub-regions.
    """
    if not gdf.crs or not gdf.crs.is_projected:
        raise ValueError("GeoDataFrame must have a projected metric CRS (meters) for spatial blocking.")

    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    minx, miny = xs.min(), ys.min()

    col_idx = np.floor((xs - minx) / block_size_m).astype(int)
    row_idx = np.floor((ys - miny) / block_size_m).astype(int)
    block_ids = pd.Series(
        [f"BLK_R{r:02d}_C{c:02d}" for r, c in zip(row_idx, col_idx)],
        index=gdf.index,
        name="spatial_block_id",
    )
    return block_ids


def get_spatial_block_splits(
    gdf: gpd.GeoDataFrame,
    target_col: str = "mineral_occurrence",
    block_size_m: float = 25_000.0,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pd.Series, list[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate Stratified Group K-Fold splits grouped by 2D spatial blocks so that:
    1. No spatial block is ever shared between train and test sets (`zero spatial leakage`).
    2. Positive mineral occurrence rates are balanced as evenly as possible across folds.

    Returns
    -------
    Tuple[pd.Series, list[Tuple[np.ndarray, np.ndarray]]]
        (spatial_block_ids, list of (train_idx, test_idx) arrays).
    """
    block_ids = assign_spatial_blocks(gdf, block_size_m=block_size_m)
    y = gdf[target_col].to_numpy()
    X_dummy = np.zeros((len(gdf), 1))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(X_dummy, y, groups=block_ids.to_numpy()))

    logger.info(
        "Created %d spatial folds across %d unique %.1f-km blocks.",
        n_splits,
        block_ids.nunique(),
        block_size_m / 1000.0,
    )
    return block_ids, splits
