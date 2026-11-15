"""
Model training, spatial cross-validation evaluation, and interpretability for
mineral prospectivity mapping.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_candidate_models(
    pos_weight: float = 10.0,
    seed: int = 42,
) -> Dict[str, Pipeline]:
    """
    Construct the three core candidate models for comparison:
    1. Logistic Regression (L2 regularized, class-balanced interpretable linear baseline).
    2. Random Forest (non-linear bagging ensemble capturing feature interactions).
    3. XGBoost (regularized gradient boosted trees handling skewed geochemical predictors).
    """
    models = {
        "Logistic_Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Random_Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=4,
                        min_child_weight=6,
                        learning_rate=0.035,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        reg_lambda=4.0,
                        scale_pos_weight=pos_weight,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_models_spatial_vs_random(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    spatial_splits: List[Tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Pipeline]]:
    """
    Evaluate all candidate models under both Spatial Block Cross-Validation and
    naive Random Stratified K-Fold Cross-Validation to quantify spatial leakage bias.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Pipeline]]
        - Summary metrics DataFrame (ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score,
          Confusion Matrix counts, and Random-CV leakage inflation).
        - Out-of-fold spatial probability predictions per model.
        - Fitted full-dataset model pipelines for downstream interpretation.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)

    n_pos = max(int(y.sum()), 1)
    n_neg = len(y) - n_pos
    pos_weight = float(np.sqrt(n_neg / n_pos))  # dampened inverse ratio for calibrated probabilities

    models = build_candidate_models(pos_weight=pos_weight, seed=seed)
    random_skf = list(StratifiedKFold(n_splits=len(spatial_splits), shuffle=True, random_state=seed).split(X, y))

    metrics_rows = []
    oof_spatial_preds: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, Pipeline] = {}

    for name, pipeline in models.items():
        # 1. Out-of-fold Spatial Block CV predictions
        oof_prob_spatial = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in spatial_splits:
            pipeline.fit(X[train_idx], y[train_idx])
            oof_prob_spatial[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]

        # 2. Out-of-fold Random CV predictions (to measure spatial leakage optimism)
        oof_prob_random = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in random_skf:
            pipeline.fit(X[train_idx], y[train_idx])
            oof_prob_random[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]

        # 3. Compute metrics at operational probability threshold
        y_pred_spatial = (oof_prob_spatial >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred_spatial).ravel()

        roc_spatial = roc_auc_score(y, oof_prob_spatial)
        pr_spatial = average_precision_score(y, oof_prob_spatial)
        roc_random = roc_auc_score(y, oof_prob_random)
        pr_random = average_precision_score(y, oof_prob_random)

        metrics_rows.append(
            {
                "model": name,
                "spatial_roc_auc": round(float(roc_spatial), 4),
                "random_roc_auc": round(float(roc_random), 4),
                "leakage_roc_inflation": round(float(roc_random - roc_spatial), 4),
                "spatial_pr_auc": round(float(pr_spatial), 4),
                "random_pr_auc": round(float(pr_random), 4),
                "precision": round(float(precision_score(y, y_pred_spatial, zero_division=0)), 4),
                "recall": round(float(recall_score(y, y_pred_spatial, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y, y_pred_spatial, zero_division=0)), 4),
                "brier_score": round(float(brier_score_loss(y, oof_prob_spatial)), 4),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )
        oof_spatial_preds[name] = oof_prob_spatial

        # Fit final model on full dataset for feature importance & spatial mapping
        pipeline.fit(X, y)
        fitted_models[name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows).sort_values("spatial_pr_auc", ascending=False).reset_index(drop=True)
    return metrics_df, oof_spatial_preds, fitted_models


def extract_feature_importances(
    fitted_models: Dict[str, Pipeline],
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Extract standardized linear coefficients (Logistic Regression), Gini/gain
    importances (Random Forest & XGBoost), and model-agnostic permutation importance
    on PR-AUC.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)

    lr_coefs = np.abs(fitted_models["Logistic_Regression"].named_steps["clf"].coef_[0])
    lr_norm = lr_coefs / np.sum(lr_coefs)

    rf_imp = fitted_models["Random_Forest"].named_steps["clf"].feature_importances_
    xgb_imp = fitted_models["XGBoost"].named_steps["clf"].feature_importances_

    perm = permutation_importance(
        fitted_models["XGBoost"],
        X,
        y,
        scoring="average_precision",
        n_repeats=10,
        random_state=seed,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "logreg_abs_coef_share": np.round(lr_norm, 4),
            "random_forest_importance": np.round(rf_imp, 4),
            "xgboost_gain_importance": np.round(xgb_imp, 4),
            "permutation_pr_auc_drop": np.round(perm.importances_mean, 4),
        }
    ).sort_values("permutation_pr_auc_drop", ascending=False).reset_index(drop=True)
    return imp_df
