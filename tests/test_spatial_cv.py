"""Unit tests for Spatial Block Cross-Validation ensuring zero spatial block leakage."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.models.spatial_cv import assign_spatial_blocks, get_spatial_block_splits


def test_spatial_block_splits_have_zero_block_overlap():
    rng = np.random.default_rng(10)
    xs = rng.uniform(400_000, 650_000, size=300)
    ys = rng.uniform(8_500_000, 8_700_000, size=300)
    y = rng.binomial(1, 0.15, size=300)

    gdf = gpd.GeoDataFrame(
        {"mineral_occurrence": y},
        geometry=[Point(x, y_) for x, y_ in zip(xs, ys)],
        crs="EPSG:32735",
    )

    block_ids, splits = get_spatial_block_splits(
        gdf, target_col="mineral_occurrence", block_size_m=25_000.0, n_splits=5, seed=42
    )
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        # Ensure index sets are disjoint
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        # Ensure spatial block IDs are strictly disjoint between train and test
        train_blocks = set(block_ids.iloc[train_idx])
        test_blocks = set(block_ids.iloc[test_idx])
        assert len(train_blocks.intersection(test_blocks)) == 0


def test_assign_spatial_blocks_rejects_unprojected_crs():
    gdf = gpd.GeoDataFrame(
        {"mineral_occurrence": [0, 1]},
        geometry=[Point(27.1, -12.5), Point(27.2, -12.6)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="projected metric CRS"):
        assign_spatial_blocks(gdf, block_size_m=25_000.0)
