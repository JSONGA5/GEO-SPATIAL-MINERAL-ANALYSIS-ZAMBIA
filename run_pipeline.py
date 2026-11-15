"""
End-to-end execution script for the Zambian Copperbelt Mineral Prospectivity Mapping
portfolio project (Phases 1 through 8).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure saving

from src.data.acquisition import generate_zambia_prospectivity_benchmark, load_config
from src.data.cleaning import summarize_geochemical_outliers, validate_and_standardize_geodata
from src.features.spatial_features import (
    compute_morans_i_knn,
    compute_vif_table,
    engineer_geospatial_features,
)
from src.models.spatial_cv import get_spatial_block_splits
from src.models.train_evaluate import (
    evaluate_models_spatial_vs_random,
    extract_feature_importances,
)
from src.models.uncertainty import (
    classify_uncertainty_quadrants_and_targets,
    run_spatial_bootstrap_uncertainty,
)
from src.visualization.maps import (
    create_interactive_folium_target_map,
    plot_geochemical_and_geophysical_anomalies,
    plot_model_performance_and_importance,
    plot_prospectivity_uncertainty_and_targets,
    plot_study_area_and_geology,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mineral_prospects_pipeline")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "configs" / "config.yaml")
    processed_dir = project_root / config["data_paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2 & 3: Data Acquisition, Validation, and CRS Standardization
    raw_gdf, faults_gdf = generate_zambia_prospectivity_benchmark(config)
    clean_gdf, qa_report = validate_and_standardize_geodata(
        raw_gdf, target_crs=config["crs"]["projected_crs"]
    )
    logger.info("QA Report: %s", qa_report)

    geochem_outliers = summarize_geochemical_outliers(clean_gdf, ["cu_ppm", "co_ppm"])
    logger.info("Geochemical Outlier Audit:\n%s", geochem_outliers.to_string(index=False))

    # Phase 4 & 5: Spatial Autocorrelation & Feature Engineering
    feat_gdf, feature_cols = engineer_geospatial_features(clean_gdf, faults_gdf)
    moran_cu = compute_morans_i_knn(feat_gdf, "log10_cu_ppm", k_neighbors=8)
    moran_fault = compute_morans_i_knn(feat_gdf, "dist_to_fault_m", k_neighbors=8)
    logger.info("Moran's I (log10_cu_ppm): %s | (dist_to_fault_m): %s", moran_cu, moran_fault)

    modeling_features = [c for c in feature_cols if c != "cu_co_log_ratio"]
    vif_df = compute_vif_table(feat_gdf, modeling_features)
    logger.info("Top VIF Features:\n%s", vif_df.head(6).to_string(index=False))

    # Phase 6: Spatial Block CV & Multi-Model Comparison
    block_ids, spatial_splits = get_spatial_block_splits(
        feat_gdf,
        target_col="mineral_occurrence",
        block_size_m=config["validation"]["spatial_block_size_m"],
        n_splits=config["validation"]["n_splits"],
        seed=config["project"]["random_seed"],
    )
    feat_gdf["spatial_block_id"] = block_ids

    metrics_df, oof_spatial_preds, fitted_models = evaluate_models_spatial_vs_random(
        feat_gdf,
        feature_cols=modeling_features,
        target_col="mineral_occurrence",
        spatial_splits=spatial_splits,
        threshold=0.50,
        seed=config["project"]["random_seed"],
    )
    logger.info("Model Evaluation (Spatial Block CV vs Random CV):\n%s", metrics_df.to_string(index=False))

    importance_df = extract_feature_importances(
        fitted_models,
        feat_gdf,
        feature_cols=modeling_features,
        target_col="mineral_occurrence",
        seed=config["project"]["random_seed"],
    )
    logger.info("Top 8 Important Features:\n%s", importance_df.head(8).to_string(index=False))

    # Phase 7 & 8: Uncertainty Analysis & Exploration Target Prioritization
    unc_gdf = run_spatial_bootstrap_uncertainty(
        feat_gdf,
        feature_cols=modeling_features,
        target_col="mineral_occurrence",
        base_pipeline=fitted_models["XGBoost"],
        block_col="spatial_block_id",
        n_bootstraps=50,
        mc_noise_std=0.05,
        seed=config["project"]["random_seed"],
    )
    final_gdf = classify_uncertainty_quadrants_and_targets(unc_gdf)

    priority_counts = final_gdf["exploration_priority"].value_counts()
    logger.info("Exploration Priority Counts:\n%s", priority_counts.to_string())

    # Save tabular & spatial outputs
    metrics_df.to_csv(processed_dir / "model_comparison_metrics.csv", index=False)
    importance_df.to_csv(processed_dir / "feature_importances.csv", index=False)
    final_gdf.to_file(processed_dir / "zambia_prospectivity_predictions.gpkg", driver="GPKG")
    faults_gdf.to_file(processed_dir / "zambia_fault_lineaments.gpkg", driver="GPKG")

    # Generate all static and interactive visualizations
    plot_study_area_and_geology(
        final_gdf, faults_gdf, save_path=processed_dir / "fig01_study_area_and_geology.png"
    )
    plot_geochemical_and_geophysical_anomalies(
        final_gdf, save_path=processed_dir / "fig02_geochem_geophysics_anomalies.png"
    )
    plot_model_performance_and_importance(
        final_gdf["mineral_occurrence"].to_numpy(),
        oof_spatial_preds,
        metrics_df,
        importance_df,
        save_path=processed_dir / "fig03_model_performance_and_importance.png",
    )
    plot_prospectivity_uncertainty_and_targets(
        final_gdf, faults_gdf, save_path=processed_dir / "fig04_prospectivity_uncertainty_targets.png"
    )
    create_interactive_folium_target_map(
        final_gdf, faults_gdf, output_html_path=processed_dir / "zambia_exploration_targets_interactive.html"
    )

    # Copy publication figures to docs/figures/ for GitHub README rendering
    docs_fig_dir = project_root / "docs" / "figures"
    docs_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_name in [
        "fig01_study_area_and_geology.png",
        "fig02_geochem_geophysics_anomalies.png",
        "fig03_model_performance_and_importance.png",
        "fig04_prospectivity_uncertainty_targets.png",
    ]:
        shutil.copy2(processed_dir / fig_name, docs_fig_dir / fig_name)

    logger.info("Pipeline completed successfully. Outputs saved to %s and %s", processed_dir, docs_fig_dir)


if __name__ == "__main__":
    main()
