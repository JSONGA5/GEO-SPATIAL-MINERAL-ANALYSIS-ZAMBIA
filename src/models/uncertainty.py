"""
Uncertainty quantification (Spatial Block Bootstrap + Monte Carlo measurement noise)
and Exploration Target Prioritization for mineral prospectivity mapping.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def run_spatial_bootstrap_uncertainty(
    gdf: gpd.GeoDataFrame,
    feature_cols: List[str],
    target_col: str,
    base_pipeline: Pipeline,
    block_col: str = "spatial_block_id",
    n_bootstraps: int = 50,
    mc_noise_std: float = 0.05,
    n_mc_samples: int = 3,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """
    Estimate spatial prediction probability and epistemic + measurement uncertainty
    via Spatial Block Bootstrapping combined with Monte Carlo covariate perturbation.

    Scientific Formulation
    ----------------------
    1. **Spatial Cluster Bootstrap (Epistemic Model Uncertainty)**:
       In each iteration b = 1..B, we resample spatial blocks (`spatial_block_id`)
       with replacement. This mimics variability in regional survey coverage—if an
       area's high predicted probability depends solely on a single nearby cluster of
       training points, its bootstrap standard deviation will be large.
    2. **Monte Carlo Covariate Perturbation (Aleatoric / Measurement Uncertainty)**:
       During prediction in iteration b, continuous geophysical and geochemical
       inputs are perturbed by small Gaussian noise proportional to their standard
       deviation (`mc_noise_std`) to propagate laboratory assay and sensor noise.

    Outputs Added to GeoDataFrame
    -----------------------------
    - `prob_mean`: Posterior/ensemble mean prospectivity probability E[p(s)].
    - `prob_std`: Combined epistemic + measurement uncertainty SD[p(s)].
    - `epistemic_std`: Spatial bootstrap standard deviation.
    - `aleatoric_std`: Mean within-model Monte Carlo perturbation standard deviation.
    - `prob_ci_lower`: 5th percentile prediction interval bound.
    - `prob_ci_upper`: 95th percentile prediction interval bound.
    - `prob_ci_width`: 90% prediction interval width (p95 - p05).
    - `shannon_entropy`: Predictive entropy H(p) measuring total classification ambiguity.
    - `coef_of_variation`: Relative uncertainty SD[p(s)] / max(E[p(s)], epsilon).
    """
    if gdf.empty:
        raise ValueError("Cannot quantify uncertainty for an empty GeoDataFrame.")

    required_cols = set(feature_cols) | {target_col, block_col}
    missing_cols = sorted(required_cols.difference(gdf.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")
    if n_mc_samples < 1:
        raise ValueError("n_mc_samples must be at least 1.")
    if mc_noise_std < 0:
        raise ValueError("mc_noise_std must be non-negative.")

    rng = np.random.default_rng(seed)
    X_full = gdf[feature_cols].to_numpy(dtype=float)
    y_full = gdf[target_col].to_numpy(dtype=int)
    blocks = gdf[block_col].to_numpy()

    if set(np.unique(y_full)) != {0, 1}:
        raise ValueError("target_col must contain binary 0/1 values.")
    if pd.isna(blocks).any():
        raise ValueError("block_col cannot contain missing spatial block IDs.")
    if np.isinf(X_full).any():
        raise ValueError("Feature matrix contains infinite values.")

    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < 2:
        raise ValueError("At least two spatial blocks are required for bootstrap uncertainty.")

    feature_stds = np.nanstd(X_full, axis=0)
    is_binary = np.array([
        np.all(np.isin(np.unique(X_full[:, i][~np.isnan(X_full[:, i])]), [0.0, 1.0]))
        for i in range(X_full.shape[1])
    ])
    noise_scale = np.where(is_binary, 0.0, feature_stds * mc_noise_std)

    n_samples = len(gdf)
    epistemic_preds = np.zeros((n_bootstraps, n_samples), dtype=float)
    mc_preds = np.zeros((n_bootstraps, n_mc_samples, n_samples), dtype=float)

    for b in range(n_bootstraps):
        train_idx = np.array([], dtype=int)
        for _ in range(100):
            sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
            train_idx = np.concatenate([
                np.where(blocks == block_id)[0] for block_id in sampled_blocks
            ])
            if len(np.unique(y_full[train_idx])) == 2:
                break
        else:
            raise ValueError(
                "Spatial block bootstrap could not create a training sample containing both classes."
            )

        model_b = clone(base_pipeline)
        model_b.fit(X_full[train_idx], y_full[train_idx])

        epistemic_preds[b, :] = model_b.predict_proba(X_full)[:, 1]
        for m in range(n_mc_samples):
            X_perturbed = X_full + rng.normal(0.0, noise_scale, size=X_full.shape)
            mc_preds[b, m, :] = model_b.predict_proba(X_perturbed)[:, 1]

    out_gdf = gdf.copy()
    epistemic_std = np.std(
        epistemic_preds,
        axis=0,
        ddof=1 if n_bootstraps > 1 else 0,
    )
    aleatoric_std = np.mean(
        np.std(mc_preds, axis=1, ddof=1 if n_mc_samples > 1 else 0),
        axis=0,
    )
    prob_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)

    total_preds = mc_preds.reshape(n_bootstraps * n_mc_samples, n_samples)
    prob_mean = np.mean(total_preds, axis=0)
    ci_lower = np.percentile(total_preds, 5, axis=0)
    ci_upper = np.percentile(total_preds, 95, axis=0)

    # Clip to avoid log(0) in Shannon entropy calculation
    p_clip = np.clip(prob_mean, 1e-6, 1.0 - 1e-6)
    entropy = -(p_clip * np.log2(p_clip) + (1.0 - p_clip) * np.log2(1.0 - p_clip))

    out_gdf["prob_mean"] = np.round(prob_mean, 4)
    out_gdf["prob_std"] = np.round(prob_std, 4)
    out_gdf["epistemic_std"] = np.round(epistemic_std, 4)
    out_gdf["aleatoric_std"] = np.round(aleatoric_std, 4)
    out_gdf["prob_ci_lower"] = np.round(ci_lower, 4)
    out_gdf["prob_ci_upper"] = np.round(ci_upper, 4)
    out_gdf["prob_ci_width"] = np.round(ci_upper - ci_lower, 4)
    out_gdf["shannon_entropy"] = np.round(entropy, 4)
    out_gdf["coef_of_variation"] = np.round(
        prob_std / np.maximum(prob_mean, np.finfo(float).eps),
        4,
    )

    return out_gdf


def classify_uncertainty_quadrants_and_targets(
    gdf: gpd.GeoDataFrame,
    prob_threshold: float = 0.35,
    high_prob_threshold: float = 0.50,
    std_median_quantile: float = 0.55,
) -> gpd.GeoDataFrame:
    """
    Assign each location to:
    1. A 4-state Bivariate Prospectivity–Confidence Regime (`uncertainty_regime`):
       - 'High Prospectivity / High Confidence'
       - 'High Prospectivity / Low Confidence'
       - 'Low Prospectivity / High Confidence'
       - 'Low Prospectivity / Low Confidence'
    2. An Actionable Exploration Target Priority Tier (`exploration_priority`):
       - 'Priority 1: High Prospectivity + High Confidence'
       - 'Priority 2: High Prospectivity + Moderate Uncertainty'
       - 'Priority 3: Favorable Geological Signals + Data Gap'
       - 'Background: Low Prospectivity'

    Statistical Note on Bernoulli Variance
    --------------------------------------
    Because ensemble standard deviation SD[p(s)] naturally scales with p(1-p),
    comparing SD[p(s)] of high-probability points against near-zero background
    points would conflate signal magnitude with uncertainty. We therefore evaluate
    high-confidence vs. low-confidence conditioning on each regime (using the
    within-regime SD quantile and Coefficient of Variation CV = SD / Mean).
    """
    required_cols = {
        "prob_mean",
        "prob_std",
        "prob_ci_upper",
        "structural_redox_interaction",
    }
    missing_cols = sorted(required_cols.difference(gdf.columns))
    if missing_cols:
        raise ValueError(f"Missing required uncertainty columns: {missing_cols}")
    if not 0.0 < prob_threshold < 1.0:
        raise ValueError("prob_threshold must be between 0 and 1.")
    if not prob_threshold < high_prob_threshold < 1.0:
        raise ValueError(
            "high_prob_threshold must be greater than prob_threshold and less than 1."
        )
    if not 0.0 < std_median_quantile < 1.0:
        raise ValueError("std_median_quantile must be between 0 and 1.")

    out_gdf = gdf.copy()
    high_prob = out_gdf["prob_mean"] >= prob_threshold

    # Conditional SD cutoffs within high-prospectivity and low-prospectivity regimes
    if high_prob.sum() > 2:
        std_cut_high = float(out_gdf.loc[high_prob, "prob_std"].quantile(std_median_quantile))
    else:
        std_cut_high = float(out_gdf["prob_std"].quantile(std_median_quantile))

    if (~high_prob).sum() > 2:
        std_cut_low = float(out_gdf.loc[~high_prob, "prob_std"].quantile(0.75))
    else:
        std_cut_low = float(out_gdf["prob_std"].quantile(0.75))

    high_conf = np.where(high_prob, out_gdf["prob_std"] <= std_cut_high, out_gdf["prob_std"] <= std_cut_low)

    regimes = np.select(
        [
            high_prob & high_conf,
            high_prob & (~high_conf),
            (~high_prob) & high_conf,
            (~high_prob) & (~high_conf),
        ],
        [
            "High Prospectivity / High Confidence",
            "High Prospectivity / Low Confidence",
            "Low Prospectivity / High Confidence",
            "Low Prospectivity / Low Confidence",
        ],
        default="Low Prospectivity / High Confidence",
    )
    out_gdf["uncertainty_regime"] = regimes

    # Exploration Target Prioritization Rules
    p1_mask = (out_gdf["prob_mean"] >= high_prob_threshold) & high_conf
    p2_mask = (
        (~p1_mask)
        & (
            (
                (out_gdf["prob_mean"] >= prob_threshold)
                & (out_gdf["prob_mean"] < high_prob_threshold)
            )
            | (
                (out_gdf["prob_mean"] >= high_prob_threshold)
                & (~high_conf)
            )
        )
    )
    p3_mask = (
        (~p1_mask)
        & (~p2_mask)
        & (
            (out_gdf["prob_ci_upper"] >= prob_threshold)
            | (
                (out_gdf["structural_redox_interaction"] >= out_gdf["structural_redox_interaction"].quantile(0.85))
                & (~high_conf)
            )
        )
    )

    priorities = np.select(
        [p1_mask, p2_mask, p3_mask],
        [
            "Priority 1: High Prospectivity + High Confidence",
            "Priority 2: High Prospectivity + Moderate Uncertainty",
            "Priority 3: Favorable Geological Signals + Data Gap",
        ],
        default="Background: Low Prospectivity",
    )
    out_gdf["exploration_priority"] = priorities
    return out_gdf
