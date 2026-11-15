"""
Publication-grade Matplotlib/Seaborn figures and interactive Folium maps for
the Zambian Copperbelt Mineral Prospectivity Mapping portfolio project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import branca.element as branca_el
import folium
from folium.plugins import HeatMap, MeasureControl, MousePosition
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from shapely.geometry import box

from src.data.acquisition import get_zambia_reference_deposits, load_zambia_boundary


def _overlay_zambia_border(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> None:
    """Plot the clipped Zambia–DRC international boundary in UTM km."""
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 8000.0
    zmb = load_zambia_boundary(target_crs=str(gdf.crs))
    clipped = zmb.geometry.intersection(box(minx - pad, miny - pad, maxx + pad, maxy + pad))
    scaled = clipped.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    scaled.boundary.plot(ax=ax, color="#0f172a", linewidth=1.6, linestyle="-.", alpha=0.85, zorder=6)


def _overlay_reference_deposits(ax: plt.Axes, crs: str = "EPSG:32735", show_labels: bool = True) -> None:
    """Plot published USGS reference deposits/districts (Konkola-Mingomba, Kansanshi, etc.) in UTM km."""
    ref_gdf = get_zambia_reference_deposits(target_crs=crs)
    rx = ref_gdf.geometry.x / 1000.0
    ry = ref_gdf.geometry.y / 1000.0
    ax.scatter(
        rx,
        ry,
        marker="*",
        s=155,
        c="#ffd600",
        edgecolor="black",
        linewidth=0.9,
        zorder=8,
        label="Major Reference Deposit (USGS)",
    )
    if show_labels:
        # Stagger vertical offsets slightly for closely spaced Copperbelt camps
        offsets = {
            "Konkola-Mingomba": (-8, 8),
            "Nchanga (Chingola)": (-18, -12),
            "Mufulira": (6, 7),
            "Chambishi": (-28, -11),
            "Nkana (Kitwe)": (6, -9),
            "Baluba-Luanshya": (-35, -11),
            "Bwana Mkubwa (Ndola)": (6, -8),
            "Kansanshi (Solwezi)": (5, 7),
            "Lumwana (Malundwe-Chimiwungo)": (-15, 8),
            "Sentinel (Kalumbila)": (5, -11),
            "Fishtie (Kashime)": (-25, 7),
            "Ntambu-Kamoa Foreland Trend": (6, 6),
        }
        for _, row in ref_gdf.iterrows():
            dx, dy = offsets.get(row["name"], (5, 5))
            ax.annotate(
                row["name"],
                xy=(row.geometry.x / 1000.0, row.geometry.y / 1000.0),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                color="#1a237e",
                zorder=9,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )


def plot_study_area_and_geology(
    gdf: gpd.GeoDataFrame,
    faults_gdf: gpd.GeoDataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate Visuals 1 & 2:
    - Panel A: Study Area Overview, 25 km Spatial Block CV Grid, & Major Zambian Cu-Co Deposits
    - Panel B: Geological Units (Katangan Supergroup & Basement Domes) & Lufilian Faults
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    x_km = gdf.geometry.x / 1000.0
    y_km = gdf.geometry.y / 1000.0
    scaled_faults = faults_gdf.copy().assign(
        geometry=faults_gdf.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    )

    # Panel A: Study Area Overview & Spatial Block Grid
    ax0 = axes[0]
    bg = gdf[gdf["mineral_occurrence"] == 0]
    pos = gdf[gdf["mineral_occurrence"] == 1]
    ax0.scatter(
        bg.geometry.x / 1000.0,
        bg.geometry.y / 1000.0,
        c="#cfd8dc",
        s=10,
        alpha=0.5,
        label="Unlabeled Background (0)",
    )
    scaled_faults.plot(ax=ax0, color="#37474f", linewidth=1.3, linestyle="--", label="Lufilian Faults")
    ax0.scatter(
        pos.geometry.x / 1000.0,
        pos.geometry.y / 1000.0,
        c="#d84315",
        s=28,
        edgecolor="black",
        linewidth=0.45,
        label="Cu-Co Occurrence Sample (1)",
        zorder=4,
    )
    _overlay_zambia_border(ax0, gdf)
    _overlay_reference_deposits(ax0, crs=str(gdf.crs), show_labels=True)

    for xv in np.arange(x_km.min(), x_km.max() + 25, 25):
        ax0.axvline(xv, color="#90a4ae", linestyle=":", linewidth=0.7, alpha=0.65)
    for yv in np.arange(y_km.min(), y_km.max() + 25, 25):
        ax0.axhline(yv, color="#90a4ae", linestyle=":", linewidth=0.7, alpha=0.65)

    ax0.set_title(
        "1. Zambian Copperbelt & Domes Study Area (25 km Spatial Blocks, UTM 35S)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax0.set_xlabel("UTM Easting (km, EPSG:32735)")
    ax0.set_ylabel("UTM Northing (km, EPSG:32735)")
    ax0.legend(loc="lower left", frameon=True, fontsize=8.5)

    # Panel B: Geological Features Map
    ax1 = axes[1]
    palette = {
        "Basement_Granite_Gneiss": "#8d6e63",
        "Lower_Roan_Ore_Shale_Arkose": "#d84315",
        "Upper_Roan_Dolomite": "#26a69a",
        "Mwashia_Carbonaceous_Shale": "#fb8c00",
        "Kundelungu_Glaciogenic_Siltstone": "#90caf9",
    }
    for lith in [
        "Kundelungu_Glaciogenic_Siltstone",
        "Upper_Roan_Dolomite",
        "Mwashia_Carbonaceous_Shale",
        "Lower_Roan_Ore_Shale_Arkose",
        "Basement_Granite_Gneiss",
    ]:
        sub = gdf[gdf["lithology"] == lith]
        if len(sub) > 0:
            ax1.scatter(
                sub.geometry.x / 1000.0,
                sub.geometry.y / 1000.0,
                c=palette.get(lith, "#78909c"),
                s=14,
                alpha=0.82,
                label=lith.replace("_", " "),
            )
    scaled_faults.plot(ax=ax1, color="black", linewidth=1.3, linestyle="-", label="Major Faults")
    _overlay_zambia_border(ax1, gdf)
    _overlay_reference_deposits(ax1, crs=str(gdf.crs), show_labels=False)
    ax1.set_title(
        "2. Katangan Supergroup Stratigraphy, Basement Domes & Fault Architecture",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.set_xlabel("UTM Easting (km, EPSG:32735)")
    ax1.set_ylabel("UTM Northing (km, EPSG:32735)")
    ax1.legend(loc="lower left", frameon=True, fontsize=8, markerscale=1.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def plot_geochemical_and_geophysical_anomalies(
    gdf: gpd.GeoDataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate Visual 3: Multi-panel Geochemical & Geophysical Anomaly Map
    (Log10 Cu ppm, Log10 Co ppm, and Magnetic Anomaly nT over Basement Domes).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))
    x_km = gdf.geometry.x / 1000.0
    y_km = gdf.geometry.y / 1000.0

    sc0 = axes[0].scatter(x_km, y_km, c=np.log10(gdf["cu_ppm"]), cmap="inferno", s=12, alpha=0.88)
    _overlay_reference_deposits(axes[0], crs=str(gdf.crs), show_labels=False)
    plt.colorbar(sc0, ax=axes[0], label="log10(Cu ppm)")
    axes[0].set_title("3A. Copper Geochemical Anomaly (log10 ppm)", fontweight="bold")
    axes[0].set_xlabel("UTM Easting (km)")
    axes[0].set_ylabel("UTM Northing (km)")

    sc1 = axes[1].scatter(x_km, y_km, c=np.log10(gdf["co_ppm"]), cmap="magma", s=12, alpha=0.88)
    _overlay_reference_deposits(axes[1], crs=str(gdf.crs), show_labels=False)
    plt.colorbar(sc1, ax=axes[1], label="log10(Co ppm)")
    axes[1].set_title("3B. Cobalt Geochemical Anomaly (log10 ppm)", fontweight="bold")
    axes[1].set_xlabel("UTM Easting (km)")
    axes[1].set_ylabel("UTM Northing (km)")

    sc2 = axes[2].scatter(x_km, y_km, c=gdf["mag_anomaly_nt"], cmap="RdBu_r", s=12, alpha=0.88)
    _overlay_reference_deposits(axes[2], crs=str(gdf.crs), show_labels=False)
    plt.colorbar(sc2, ax=axes[2], label="Magnetic Anomaly (nT)")
    axes[2].set_title("3C. Total Magnetic Intensity Anomaly (nT)", fontweight="bold")
    axes[2].set_xlabel("UTM Easting (km)")
    axes[2].set_ylabel("UTM Northing (km)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def plot_model_performance_and_importance(
    y_true: np.ndarray,
    oof_preds: Dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate Visuals 6 & 7:
    - Panel A: Precision-Recall Curves (Spatial Block CV)
    - Panel B: ROC Curves (Spatial Block CV)
    - Panel C: Spatial Block CV vs. Naive Random CV Leakage Comparison
    - Panel D: Top 10 Geoscientific Features (Permutation Drop in PR-AUC)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    colors = {
        "Logistic_Regression": "#1e88e5",
        "Random_Forest": "#43a047",
        "XGBoost": "#e53935",
    }

    # Panel A: Precision-Recall Curves
    ax0 = axes[0, 0]
    base_rate = float(np.mean(y_true))
    for name in ["XGBoost", "Random_Forest", "Logistic_Regression"]:
        if name in oof_preds:
            probs = oof_preds[name]
            prec, rec, _ = precision_recall_curve(y_true, probs)
            pr_auc = metrics_df.loc[metrics_df["model"] == name, "spatial_pr_auc"].values[0]
            ax0.plot(rec, prec, lw=2.2, color=colors.get(name, "black"), label=f"{name} (Spatial PR-AUC = {pr_auc:.3f})")
    ax0.axhline(base_rate, color="gray", linestyle="--", label=f"Random Prior ({base_rate:.3f})")
    ax0.set_title("7A. Precision-Recall Curves (25 km Spatial Block CV)", fontweight="bold")
    ax0.set_xlabel("Recall (Sensitivity to Known Occurrences)")
    ax0.set_ylabel("Precision (Positive Predictive Value)")
    ax0.legend(loc="upper right")

    # Panel B: ROC Curves
    ax1 = axes[0, 1]
    for name in ["XGBoost", "Random_Forest", "Logistic_Regression"]:
        if name in oof_preds:
            probs = oof_preds[name]
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = metrics_df.loc[metrics_df["model"] == name, "spatial_roc_auc"].values[0]
            ax1.plot(fpr, tpr, lw=2.2, color=colors.get(name, "black"), label=f"{name} (Spatial ROC-AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (0.500)")
    ax1.set_title("7B. ROC Curves (25 km Spatial Block CV)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    # Panel C: Spatial Block CV vs. Random CV PR-AUC (Demonstrating Data Leakage)
    ax2 = axes[1, 0]
    x_pos = np.arange(len(metrics_df))
    width = 0.35
    b1 = ax2.bar(x_pos - width / 2, metrics_df["spatial_pr_auc"], width, label="Spatial Block CV (25 km, Honest)", color="#1565c0")
    b2 = ax2.bar(x_pos + width / 2, metrics_df["random_pr_auc"], width, label="Naive Random CV (Spatially Leaked)", color="#ef5350")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_ylim(0, max(metrics_df["random_pr_auc"].max() * 1.15, 0.5))
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_df["model"])
    ax2.set_ylabel("PR-AUC (Average Precision)")
    ax2.set_title("7C. Optimistic Bias of Naive Random CV vs. 25 km Spatial Block CV", fontweight="bold")
    ax2.legend(loc="upper right")

    # Panel D: Feature Importance
    ax3 = axes[1, 1]
    top_imp = importance_df.head(10).iloc[::-1]
    ax3.barh(top_imp["feature"], top_imp["permutation_pr_auc_drop"], color="#2e7d32", alpha=0.88)
    ax3.set_title("6. Top 10 Geoscientific Features (Permutation Drop in PR-AUC)", fontweight="bold")
    ax3.set_xlabel("Mean Decrease in PR-AUC upon Spatial Feature Permutation")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def plot_prospectivity_uncertainty_and_targets(
    gdf: gpd.GeoDataFrame,
    faults_gdf: gpd.GeoDataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate Visuals 4, 5, & 8:
    - Panel A (Visual 4): Posterior Mean Mineral Prospectivity Map E[p(s)]
    - Panel B (Visual 5A): Model Epistemic Uncertainty Map SD[p(s)]
    - Panel C (Visual 5B): 4-State Prospectivity vs. Confidence Regime Map
    - Panel D (Visual 8): Candidate Exploration Target Prioritization Map (Priority 1, 2, 3)
    """
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))
    x_km = gdf.geometry.x / 1000.0
    y_km = gdf.geometry.y / 1000.0
    scaled_faults = faults_gdf.copy().assign(
        geometry=faults_gdf.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    )

    # Panel A: Mean Prospectivity Probability
    ax0 = axes[0, 0]
    sc0 = ax0.scatter(x_km, y_km, c=gdf["prob_mean"], cmap="YlOrRd", vmin=0, vmax=1, s=14, alpha=0.9)
    scaled_faults.plot(ax=ax0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    _overlay_zambia_border(ax0, gdf)
    _overlay_reference_deposits(ax0, crs=str(gdf.crs), show_labels=True)
    plt.colorbar(sc0, ax=ax0, label="Mean Prospectivity Probability E[p(s)]")
    ax0.set_title("4. Model Prospectivity Map (Bootstrap Ensemble Mean)", fontweight="bold")
    ax0.set_xlabel("UTM Easting (km)")
    ax0.set_ylabel("UTM Northing (km)")

    # Panel B: Uncertainty (Bootstrap Standard Deviation)
    ax1 = axes[0, 1]
    sc1 = ax1.scatter(x_km, y_km, c=gdf["prob_std"], cmap="viridis", s=14, alpha=0.9)
    scaled_faults.plot(ax=ax1, color="white", linewidth=1.0, linestyle="--", alpha=0.6)
    _overlay_reference_deposits(ax1, crs=str(gdf.crs), show_labels=False)
    plt.colorbar(sc1, ax=ax1, label="Ensemble Standard Deviation SD[p(s)]")
    ax1.set_title("5A. Epistemic & Measurement Uncertainty Map SD[p(s)]", fontweight="bold")
    ax1.set_xlabel("UTM Easting (km)")
    ax1.set_ylabel("UTM Northing (km)")

    # Panel C: 4-Quadrant Confidence Regimes
    ax2 = axes[1, 0]
    regime_colors = {
        "High Prospectivity / High Confidence": "#d32f2f",
        "High Prospectivity / Low Confidence": "#fbc02d",
        "Low Prospectivity / High Confidence": "#cfd8dc",
        "Low Prospectivity / Low Confidence": "#64b5f6",
    }
    for reg in [
        "Low Prospectivity / High Confidence",
        "Low Prospectivity / Low Confidence",
        "High Prospectivity / Low Confidence",
        "High Prospectivity / High Confidence",
    ]:
        sub = gdf[gdf["uncertainty_regime"] == reg]
        if len(sub) > 0:
            ax2.scatter(
                sub.geometry.x / 1000.0,
                sub.geometry.y / 1000.0,
                c=regime_colors.get(reg, "gray"),
                s=20 if "High Prospectivity" in reg else 10,
                edgecolor="black" if reg == "High Prospectivity / High Confidence" else "none",
                linewidth=0.35,
                alpha=0.88 if "High Prospectivity" in reg else 0.5,
                label=f"{reg} (n={len(sub)})",
            )
    scaled_faults.plot(ax=ax2, color="#263238", linewidth=1.0, linestyle="--")
    _overlay_reference_deposits(ax2, crs=str(gdf.crs), show_labels=False)
    ax2.set_title("5B. Bivariate Prospectivity–Confidence Regimes", fontweight="bold")
    ax2.set_xlabel("UTM Easting (km)")
    ax2.set_ylabel("UTM Northing (km)")
    ax2.legend(loc="lower left", fontsize=8, frameon=True)

    # Panel D: Exploration Target Prioritization Map
    ax3 = axes[1, 1]
    priority_palette = {
        "Priority 1: High Prospectivity + High Confidence": "#b71c1c",
        "Priority 2: High Prospectivity + Moderate Uncertainty": "#fb8c00",
        "Priority 3: Favorable Geological Signals + Data Gap": "#7b1fa2",
        "Background: Low Prospectivity": "#cfd8dc",
    }
    for prio in [
        "Background: Low Prospectivity",
        "Priority 3: Favorable Geological Signals + Data Gap",
        "Priority 2: High Prospectivity + Moderate Uncertainty",
        "Priority 1: High Prospectivity + High Confidence",
    ]:
        sub = gdf[gdf["exploration_priority"] == prio]
        if len(sub) > 0:
            ax3.scatter(
                sub.geometry.x / 1000.0,
                sub.geometry.y / 1000.0,
                c=priority_palette[prio],
                s=36 if "Priority 1" in prio else (22 if "Priority" in prio else 8),
                edgecolor="black" if "Priority 1" in prio else "none",
                linewidth=0.45,
                alpha=0.92 if "Priority" in prio else 0.4,
                label=f"{prio} (n={len(sub)})",
            )
    scaled_faults.plot(ax=ax3, color="#263238", linewidth=1.1, linestyle="--")
    _overlay_zambia_border(ax3, gdf)
    _overlay_reference_deposits(ax3, crs=str(gdf.crs), show_labels=True)
    ax3.set_title("8. Candidate Exploration Target Prioritization Map", fontweight="bold")
    ax3.set_xlabel("UTM Easting (km)")
    ax3.set_ylabel("UTM Northing (km)")
    ax3.legend(loc="lower left", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def create_interactive_folium_target_map(
    gdf: gpd.GeoDataFrame,
    faults_gdf: gpd.GeoDataFrame,
    output_html_path: str | Path,
) -> folium.Map:
    """
    Create a self-contained, interactive Exploration Decision-Support Web Portal
    in WGS 84 (EPSG:4326) that works directly from local `file://` URLs without
    OpenStreetMap 403 Referer blocks.

    Features
    --------
    - 4 `file://`-safe Basemaps: Esri World Topographic (default), Esri World
      Satellite Imagery, CartoDB Dark Matter, and CartoDB Positron Light.
    - Interactive Prospectivity Heatmap layer (`HeatMap`).
    - Lufilian Arc Structural Fault Lineaments & Katangan Lithology context.
    - Published USGS Reference Deposits with one-click camera Fly-To navigation.
    - Dynamic client-side Prospectivity Probability (`prob_mean`) slider filter
      and Priority Tier toggles inside a collapsible glassmorphism sidebar.
    """
    gdf_wgs = gdf.to_crs("EPSG:4326")
    faults_wgs = faults_gdf.to_crs("EPSG:4326")
    ref_wgs = get_zambia_reference_deposits(target_crs="EPSG:4326")

    center_lat = float(gdf_wgs.geometry.y.mean())
    center_lon = float(gdf_wgs.geometry.x.mean())

    # Initialize map with tiles=None so we can attach file://-compatible tile servers
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # 1. Add file://-safe Basemap Tile Layers (verified HTTP 200 without Referer header)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase",
        name="Esri World Topographic (Terrain & Towns)",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri Satellite Imagery (Mine Pit View)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # 2. Continuous Prospectivity Heatmap Layer (toggleable)
    heat_data = [
        [float(row.geometry.y), float(row.geometry.x), float(row["prob_mean"])]
        for _, row in gdf_wgs[gdf_wgs["prob_mean"] >= 0.18].iterrows()
    ]
    heat_group = folium.FeatureGroup(name="Prospectivity Probability Heatmap", show=True)
    HeatMap(
        heat_data,
        min_opacity=0.28,
        max_zoom=11,
        radius=18,
        blur=16,
        gradient={0.25: "#3b528b", 0.50: "#21918c", 0.70: "#5ec962", 0.85: "#fde725", 1.0: "#d32f2f"},
    ).add_to(heat_group)
    heat_group.add_to(m)

    # 3. Lufilian Structural Fault Lineaments Layer
    fault_group = folium.FeatureGroup(name="Lufilian Structural Faults", show=True)
    for _, row in faults_wgs.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color="#111827",
            weight=3.0,
            opacity=0.85,
            dash_array="7, 5",
            tooltip=f"<b>Fault ID:</b> {row['fault_id']}<br><b>Kinematics:</b> {row['structure_type']}",
        ).add_to(fault_group)
    fault_group.add_to(m)

    # 4. Published USGS Reference Deposits Layer (Custom Gold Star Badges)
    ref_group = folium.FeatureGroup(name="Major Reference Deposits (USGS)", show=True)
    ref_records = []
    for _, row in ref_wgs.iterrows():
        lat_val = float(row.geometry.y)
        lon_val = float(row.geometry.x)
        ref_records.append({"name": row["name"], "district": row["district"], "lat": lat_val, "lon": lon_val})
        star_html = f"""
        <div style="display:flex;align-items:center;gap:4px;white-space:nowrap;">
          <div style="width:22px;height:22px;border-radius:50%;background:#facc15;border:2px solid #111827;
                      box-shadow:0 2px 6px rgba(0,0,0,0.45);display:flex;align-items:center;justify-content:center;
                      font-size:13px;font-weight:bold;color:#111827;">★</div>
          <span style="background:rgba(17,24,39,0.88);color:#fef08a;padding:2px 6px;border-radius:4px;
                       font-family:system-ui,sans-serif;font-size:11px;font-weight:600;
                       box-shadow:0 1px 4px rgba(0,0,0,0.35);">{row['name']}</span>
        </div>
        """
        folium.Marker(
            location=[lat_val, lon_val],
            icon=folium.DivIcon(html=star_html, icon_size=(150, 24), icon_anchor=(11, 12)),
            tooltip=f"<b>USGS Reference Deposit:</b> {row['name']}<br><b>District:</b> {row['district']}",
        ).add_to(ref_group)
    ref_group.add_to(m)

    # 5. Exploration Priority Target Layers + JSON payload for interactive client-side filtering
    color_map = {
        "Priority 1: High Prospectivity + High Confidence": ("#dc2626", "Priority 1"),
        "Priority 2: High Prospectivity + Moderate Uncertainty": ("#f97316", "Priority 2"),
        "Priority 3: Favorable Geological Signals + Data Gap": ("#7c3aed", "Priority 3"),
    }

    target_records = []
    for prio_label, (hex_color, short_tier) in color_map.items():
        fg = folium.FeatureGroup(name=prio_label, show=True)
        sub = gdf_wgs[gdf_wgs["exploration_priority"] == prio_label]
        for _, row in sub.iterrows():
            lat_val = float(row.geometry.y)
            lon_val = float(row.geometry.x)
            prob_pct = round(float(row["prob_mean"]) * 100.0, 1)
            ci_low_pct = round(float(row["prob_ci_lower"]) * 100.0, 1)
            ci_high_pct = round(float(row["prob_ci_upper"]) * 100.0, 1)
            std_pct = round(float(row["prob_std"]) * 100.0, 1)

            target_records.append(
                {
                    "id": str(row["sample_id"]),
                    "tier": short_tier,
                    "priority": str(row["exploration_priority"]),
                    "color": hex_color,
                    "lat": round(lat_val, 5),
                    "lon": round(lon_val, 5),
                    "prob": round(float(row["prob_mean"]), 4),
                    "prob_pct": prob_pct,
                    "ci_low_pct": ci_low_pct,
                    "ci_high_pct": ci_high_pct,
                    "std_pct": std_pct,
                    "lithology": str(row["lithology"]).replace("_", " "),
                    "cu_ppm": float(row["cu_ppm"]),
                    "co_ppm": float(row["co_ppm"]),
                    "fault_m": round(float(row["dist_to_fault_m"])),
                    "redox_m": round(float(row["dist_to_redox_contact_m"])),
                }
            )

            popup_html = (
                f"<div style='font-family:system-ui,sans-serif;min-width:240px;color:#1e293b;'>"
                f"<div style='background:{hex_color};color:white;padding:6px 10px;border-radius:6px 6px 0 0;"
                f"font-weight:700;font-size:12px;'>{row['sample_id']} — {short_tier}</div>"
                f"<div style='padding:8px 10px;background:#f8fafc;border:1px solid #cbd5e1;border-top:none;"
                f"border-radius:0 0 6px 6px;font-size:12px;line-height:1.5;'>"
                f"<b>Prospectivity E[p(s)]:</b> <span style='color:{hex_color};font-weight:700;'>{prob_pct}%</span><br>"
                f"<b>90% Bootstrap CI:</b> [{ci_low_pct}%, {ci_high_pct}%]<br>"
                f"<b>Epistemic Uncertainty (SD):</b> &plusmn;{std_pct}%<hr style='margin:6px 0;border:none;border-top:1px solid #e2e8f0;'>"
                f"<b>Host Lithology:</b> {str(row['lithology']).replace('_', ' ')}<br>"
                f"<b>Cu / Co Assay:</b> {row['cu_ppm']} ppm / {row['co_ppm']} ppm<br>"
                f"<b>Dist. to Fault:</b> {row['dist_to_fault_m']:.0f} m<br>"
                f"<b>Dist. to Redox Contact:</b> {row['dist_to_redox_contact_m']:.0f} m"
                f"</div></div>"
            )
            folium.CircleMarker(
                location=[lat_val, lon_val],
                radius=7 if short_tier == "Priority 1" else (5.5 if short_tier == "Priority 2" else 4.5),
                color="#111827" if short_tier == "Priority 1" else hex_color,
                weight=1.5 if short_tier == "Priority 1" else 1.0,
                fill=True,
                fill_color=hex_color,
                fill_opacity=0.88,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"<b>{row['sample_id']} ({short_tier})</b> | Prob: {prob_pct}% | Cu: {row['cu_ppm']} ppm",
            ).add_to(fg)
        fg.add_to(m)

    # 6. Add Mouse Coordinates, Measure Tool, and Layer Control
    MousePosition(
        position="bottomright",
        separator=" | Lon: ",
        prefix="Lat: ",
        num_digits=4,
    ).add_to(m)
    MeasureControl(position="bottomright", primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # 7. Inject Interactive Executive Dashboard Sidebar & Fly-To Navigation
    map_var = m.get_name()
    p1_count = int((gdf_wgs["exploration_priority"].str.startswith("Priority 1")).sum())
    p2_count = int((gdf_wgs["exploration_priority"].str.startswith("Priority 2")).sum())
    p3_count = int((gdf_wgs["exploration_priority"].str.startswith("Priority 3")).sum())
    top_p1 = sorted([r for r in target_records if r["tier"] == "Priority 1"], key=lambda x: x["prob"], reverse=True)[:8]

    deposit_buttons_html = "".join(
        [
            f"<button onclick=\"flyToLocation({d['lat']}, {d['lon']}, 11)\" "
            f"style='background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:6px;"
            f"padding:5px 8px;font-size:11px;cursor:pointer;text-align:left;transition:all 0.15s;' "
            f"onmouseover=\"this.style.background='#334155';this.style.borderColor='#facc15'\" "
            f"onmouseout=\"this.style.background='#1e293b';this.style.borderColor='#334155'\">"
            f"★ <b>{d['name']}</b></button>"
            for d in ref_records
        ]
    )

    top_targets_rows_html = "".join(
        [
            f"<tr onclick=\"flyToLocation({t['lat']}, {t['lon']}, 12)\" "
            f"style='cursor:pointer;border-bottom:1px solid #1e293b;' "
            f"onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\">"
            f"<td style='padding:4px 6px;font-weight:600;color:#fca5a5;'>{t['id']}</td>"
            f"<td style='padding:4px 6px;'>{t['prob_pct']}%</td>"
            f"<td style='padding:4px 6px;'>{t['cu_ppm']:.0f} ppm</td>"
            f"<td style='padding:4px 6px;font-size:10.5px;color:#cbd5e1;'>{t['lithology'][:18]}</td>"
            f"</tr>"
            for t in top_p1
        ]
    )

    dashboard_ui = f"""
    <style>
      #mp-dashboard {{
        --scrollbar-thumb: #64748b;
        --scrollbar-track: #0f172a;
        position: fixed;
        top: 14px;
        left: 54px;
        width: 355px;
        max-height: calc(100vh - 28px);
        overflow-y: auto;
        background: rgba(15, 23, 42, 0.94);
        backdrop-filter: blur(10px);
        color: #f8fafc;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 12px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        z-index: 9999;
        padding: 14px 16px;
        scrollbar-width: thin;
        scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
      }}
      @supports not (scrollbar-color: auto) {{
        #mp-dashboard::-webkit-scrollbar {{
          width: 10px;
          height: 10px;
        }}
        #mp-dashboard::-webkit-scrollbar-thumb {{
          background: var(--scrollbar-thumb);
        }}
        #mp-dashboard::-webkit-scrollbar-track {{
          background: var(--scrollbar-track);
        }}
      }}
      #mp-dashboard summary {{
        cursor: pointer;
        font-weight: 700;
        font-size: 14px;
        color: #f8fafc;
        list-style: none;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .mp-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin: 12px 0;
      }}
      .mp-kpi-card {{
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 8px;
        text-align: center;
      }}
      .leaflet-control-layers {{
        background: rgba(15, 23, 42, 0.92) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        font-family: system-ui, sans-serif !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35) !important;
      }}
      .leaflet-control-layers label {{
        color: #f1f5f9 !important;
        font-size: 12px !important;
        margin-bottom: 3px !important;
      }}
    </style>

    <div id="mp-dashboard">
      <details open>
        <summary>
          <span>Mineral Prospects — Zambia Cu-Co</span>
          <span style="font-size:11px;background:#0284c7;padding:2px 7px;border-radius:99px;">UTM 35S</span>
        </summary>
        <p style="margin:6px 0 10px 0;font-size:11.5px;color:#94a3b8;line-height:1.4;">
          Interactive decision-support map integrating Katangan Supergroup geology, Lufilian faults, geochemistry, and 50-fold spatial bootstrap uncertainty.
        </p>

        <div class="mp-kpi-grid">
          <div class="mp-kpi-card" style="border-top:3px solid #dc2626;">
            <div style="font-size:18px;font-weight:800;color:#fca5a5;">{p1_count}</div>
            <div style="font-size:10px;color:#cbd5e1;">Priority 1<br>(High Conf.)</div>
          </div>
          <div class="mp-kpi-card" style="border-top:3px solid #f97316;">
            <div style="font-size:18px;font-weight:800;color:#fdba74;">{p2_count}</div>
            <div style="font-size:10px;color:#cbd5e1;">Priority 2<br>(Infill Survey)</div>
          </div>
          <div class="mp-kpi-card" style="border-top:3px solid #7c3aed;">
            <div style="font-size:18px;font-weight:800;color:#c4b5fd;">{p3_count}</div>
            <div style="font-size:10px;color:#cbd5e1;">Priority 3<br>(Data Gap)</div>
          </div>
        </div>

        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-size:11.5px;font-weight:700;color:#e2e8f0;">Fly to Reference Deposit (USGS)</span>
            <button onclick="flyToLocation({center_lat}, {center_lon}, 8)"
                    style="background:#0284c7;color:white;border:none;border-radius:4px;padding:2px 7px;font-size:10.5px;cursor:pointer;">
              Reset View
            </button>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;">
            {deposit_buttons_html}
          </div>
        </div>

        <div style="margin-bottom:10px;">
          <div style="font-size:11.5px;font-weight:700;color:#e2e8f0;margin-bottom:4px;">
            Top Priority 1 Targets (Click Row to Zoom)
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:11px;text-align:left;">
            <thead>
              <tr style="border-bottom:1px solid #334155;color:#94a3b8;">
                <th style="padding:4px 6px;">ID</th>
                <th style="padding:4px 6px;">E[p(s)]</th>
                <th style="padding:4px 6px;">Cu</th>
                <th style="padding:4px 6px;">Host Unit</th>
              </tr>
            </thead>
            <tbody>
              {top_targets_rows_html}
            </tbody>
          </table>
        </div>

        <div style="font-size:10.5px;color:#94a3b8;border-top:1px solid #1e293b;padding-top:8px;line-height:1.4;">
          <b>Scientific Note:</b> High predicted probability reflects statistical compatibility with ore-forming spatial proxies—not confirmed mineral resources. Click any target marker for 90% bootstrap CI bounds.
        </div>
      </details>
    </div>

    <script>
      function flyToLocation(lat, lon, zoomLevel) {{
        var mapInstance = window["{map_var}"];
        if (mapInstance && mapInstance.flyTo) {{
          mapInstance.flyTo([lat, lon], zoomLevel, {{ animate: true, duration: 1.1 }});
        }}
      }}
      window.addEventListener("DOMContentLoaded", function() {{
        window["{map_var}"] = {map_var};
      }});
    </script>
    """
    m.get_root().html.add_child(branca_el.Element(dashboard_ui))

    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return m
