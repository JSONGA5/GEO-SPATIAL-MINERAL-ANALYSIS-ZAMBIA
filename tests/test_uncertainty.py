"""Unit tests for spatial bootstrap uncertainty and target prioritization."""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from src.models.train_evaluate import build_candidate_models
from src.models.uncertainty import (
    classify_uncertainty_quadrants_and_targets,
    run_spatial_bootstrap_uncertainty,
)

def test_bootstrap_uncertainty_bounds_and_prioritization():
    rng = np.random.default_rng(7)
    n = 120
    xs = rng.uniform(500_000, 600_000, size=n)
    ys = rng.uniform(8_550_000, 8_650_000, size=n)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.uniform(0, 1, size=n)
    y = ((f1 + f2) > 0.8).astype(int)

    gdf = gpd.GeoDataFrame(
        {
            "sample_id": [f"S_{i}" for i in range(n)],
            "f1": f1,
            "structural_redox_interaction": f2,
            "mineral_occurrence": y,
            "spatial_block_id": [f"B_{i % 6}" for i in range(n)],
        },
        geometry=[Point(x, y_) for x, y_ in zip(xs, ys)],
        crs="EPSG:32735",
    )

    models = build_candidate_models(pos_weight=3.0, seed=42)
    unc_gdf = run_spatial_bootstrap_uncertainty(
        gdf,
        feature_cols=["f1", "structural_redox_interaction"],
        target_col="mineral_occurrence",
        base_pipeline=models["Logistic_Regression"],
        block_col="spatial_block_id",
        n_bootstraps=12,
        n_mc_samples=2,
        seed=42,
    )

    assert (unc_gdf["prob_mean"] >= 0.0).all() and (unc_gdf["prob_mean"] <= 1.0).all()
    assert (unc_gdf["prob_ci_upper"] >= unc_gdf["prob_ci_lower"]).all()
    assert (unc_gdf["prob_std"] >= 0.0).all()
    assert (unc_gdf["epistemic_std"] >= 0.0).all()
    assert (unc_gdf["aleatoric_std"] >= 0.0).all()
    assert (unc_gdf["coef_of_variation"] >= 0.0).all()

    prio_gdf = classify_uncertainty_quadrants_and_targets(unc_gdf)
    assert "uncertainty_regime" in prio_gdf.columns
    assert "exploration_priority" in prio_gdf.columns
    assert len(prio_gdf) == n


def test_bootstrap_rejects_missing_columns_and_invalid_target():
    gdf = gpd.GeoDataFrame(
        {
            "f1": [0.1, 0.2, 0.3],
            "mineral_occurrence": [0, 1, 2],
            "spatial_block_id": ["B1", "B2", "B3"],
        },
        geometry=[Point(500_000 + i, 8_600_000) for i in range(3)],
        crs="EPSG:32735",
    )

    try:
        run_spatial_bootstrap_uncertainty(
            gdf,
            feature_cols=["f1"],
            target_col="mineral_occurrence",
            base_pipeline=build_candidate_models(pos_weight=1.0)["Logistic_Regression"],
        )
        assert False, "Expected invalid binary target to raise"
    except ValueError as exc:
        assert "binary 0/1" in str(exc)


def test_priority_rule_distinguishes_uncertain_high_probability():
    gdf = gpd.GeoDataFrame(
        {
            "prob_mean": [0.60, 0.60, 0.40, 0.20],
            "prob_std": [0.01, 0.20, 0.10, 0.01],
            "prob_ci_upper": [0.65, 0.70, 0.45, 0.20],
            "structural_redox_interaction": [0.1, 0.1, 0.1, 0.1],
        },
        geometry=[Point(500_000 + i, 8_600_000) for i in range(4)],
        crs="EPSG:32735",
    )

    result = classify_uncertainty_quadrants_and_targets(
        gdf,
        std_median_quantile=0.55,
    )
    assert "Priority 1" in result.loc[0, "exploration_priority"]
    assert "Priority 2" in result.loc[1, "exploration_priority"]
    assert "Priority 2" in result.loc[2, "exploration_priority"]
