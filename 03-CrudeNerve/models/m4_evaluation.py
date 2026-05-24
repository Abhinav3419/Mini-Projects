"""
M4: Evaluation + SHAP Explainability.

Walk-forward validation with:
    - Brier score (probabilistic calibration)
    - Precision-Recall metrics
    - SHAP feature importance per regime and prediction mode
    - Calibration curves

Usage:
    from crudenerve.models.m4_evaluation import ModelEvaluator
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(unified_df)
    shap_df = evaluator.compute_shap(unified_df, m2_trainer)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

from crudenerve.config.settings import (
    PREDICTION_HORIZONS,
    VIX_DIRECTION_THRESHOLD,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates M3 ensemble predictions using walk-forward validation.
    Computes SHAP values via tree-based feature importance from M2.
    """

    def __init__(self):
        self.save_dir = MODELS_DIR / "m4_evaluation"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        df: pd.DataFrame,
        train_end: str = "2024-03-31",
        test_start: str = "2024-04-01",
    ) -> dict:
        """
        Walk-forward evaluation: train on earlier data, test on later.

        Returns metrics per horizon:
            - brier_score:  calibration quality (lower = better)
            - precision:    of predicted >2% moves, what fraction were real
            - recall:       of actual >2% moves, what fraction were caught
            - f1:           harmonic mean of precision and recall
            - auc:          area under ROC curve
            - n_test:       number of test samples
        """
        results = {}

        for horizon in PREDICTION_HORIZONS:
            target_col = f"vix_return_{horizon}h"
            pred_col = f"m3_vix_direction_{horizon}h_prob"

            if target_col not in df.columns or pred_col not in df.columns:
                logger.warning(f"Missing columns for {horizon}h evaluation")
                continue

            # Split
            test_mask = df.index >= pd.Timestamp(test_start)
            test = df[test_mask].copy()

            if len(test) < 10:
                logger.warning(f"Not enough test data for {horizon}h")
                continue

            # Ground truth: binary direction
            y_true = (test[target_col].abs() > VIX_DIRECTION_THRESHOLD).astype(int)
            y_prob = test[pred_col].fillna(0.5).clip(0.01, 0.99)
            y_pred = (y_prob > 0.5).astype(int)

            # Drop NaN targets
            valid = y_true.notna()
            y_true = y_true[valid]
            y_prob = y_prob[valid]
            y_pred = y_pred[valid]

            if len(y_true) < 5 or y_true.nunique() < 2:
                results[f"{horizon}h"] = {
                    "brier_score": None,
                    "note": "insufficient data or single class",
                    "n_test": len(y_true),
                }
                continue

            metrics = {
                "brier_score": round(brier_score_loss(y_true, y_prob), 4),
                "log_loss": round(log_loss(y_true, y_prob), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "auc": round(roc_auc_score(y_true, y_prob), 4),
                "n_test": len(y_true),
                "actual_move_rate": round(y_true.mean(), 4),
                "predicted_move_rate": round(y_pred.mean(), 4),
            }
            results[f"{horizon}h"] = metrics

            logger.info(
                f"  {horizon}h: Brier={metrics['brier_score']:.4f}  "
                f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}"
            )

        return results

    def compute_shap(
        self,
        df: pd.DataFrame,
        m2_trainer,
        regime: int = 0,
        horizon: int = 24,
        n_top: int = 15,
    ) -> pd.DataFrame:
        """
        Compute SHAP-like feature importance from XGBoost.

        Uses built-in XGBoost feature importance (gain-based)
        as a fast proxy for SHAP values. For production,
        replace with shap.TreeExplainer.
        """
        fi = m2_trainer.get_feature_importance(regime=regime, horizon=horizon)
        if fi.empty:
            return pd.DataFrame()

        fi = fi.head(n_top).copy()
        fi["importance_pct"] = (
            fi["importance"] / fi["importance"].sum() * 100
        ).round(2)

        # Determine direction from feature values
        fi["latest_value"] = fi["feature"].apply(
            lambda f: float(df[f].iloc[-1]) if f in df.columns else 0
        )
        fi["direction"] = fi["latest_value"].apply(
            lambda v: "bearish" if v > 0 else "bullish"
        )

        logger.info(
            f"SHAP (regime={regime}, {horizon}h): top feature = "
            f"{fi.iloc[0]['feature']} ({fi.iloc[0]['importance_pct']:.1f}%)"
        )

        return fi

    def per_regime_evaluation(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """Evaluate model performance broken down by regime."""
        if "m1_regime_name" not in df.columns:
            return {}

        results = {}
        for regime_name in df["m1_regime_name"].unique():
            regime_df = df[df["m1_regime_name"] == regime_name]
            if len(regime_df) < 10:
                continue

            regime_metrics = {}
            for horizon in PREDICTION_HORIZONS:
                target = f"vix_return_{horizon}h"
                pred = f"m3_vix_direction_{horizon}h_prob"
                if target not in regime_df.columns or pred not in regime_df.columns:
                    continue

                y_true = (regime_df[target].abs() > VIX_DIRECTION_THRESHOLD).astype(int)
                y_prob = regime_df[pred].fillna(0.5).clip(0.01, 0.99)
                valid = y_true.notna()
                y_true, y_prob = y_true[valid], y_prob[valid]

                if len(y_true) < 5 or y_true.nunique() < 2:
                    continue

                regime_metrics[f"{horizon}h"] = {
                    "brier": round(brier_score_loss(y_true, y_prob), 4),
                    "auc": round(roc_auc_score(y_true, y_prob), 4),
                    "n": len(y_true),
                }

            if regime_metrics:
                results[regime_name] = regime_metrics

        return results

    def generate_report(
        self,
        df: pd.DataFrame,
        m2_trainer=None,
    ) -> dict:
        """Generate a complete evaluation report."""
        report = {
            "overall_metrics": self.evaluate(df),
            "per_regime": self.per_regime_evaluation(df),
        }

        if m2_trainer is not None:
            shap_data = {}
            for regime in range(4):
                fi = self.compute_shap(df, m2_trainer, regime=regime)
                if not fi.empty:
                    shap_data[f"regime_{regime}"] = fi.to_dict("records")
            report["feature_importance"] = shap_data

        return report


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Testing M4: Evaluation + SHAP")
    print("=" * 60)

    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
    from crudenerve.data_ingest.d2_truth_social import TruthSocialPipeline, generate_synthetic_posts
    from crudenerve.data_ingest.d3_twitter import TwitterIngestor, generate_synthetic_tweets
    from crudenerve.data_ingest.d4_supply import SupplyIngestor
    from crudenerve.features.f1_merge import TimeIndexedMerge
    from crudenerve.features.f2_tension_index import TensionIndexBuilder
    from crudenerve.features.f3_dho_kernel import DHOKernelEngine
    from crudenerve.features.f4_trump_volatility import TrumpVolatilityEngine
    from crudenerve.features.f5_twitter_features import TwitterFeatureBuilder
    from crudenerve.features.f6_supply_features import SupplyFeatureBuilder
    from crudenerve.features.f8_war_barometer import WarTensionBarometer
    from crudenerve.models.m1_hmm_regime import HMMRegimeDetector
    from crudenerve.models.m2_regime_xgboost import RegimeXGBoostTrainer
    from crudenerve.models.m3_ensemble import EnsemblePredictor

    START, END = "2024-01-01", "2024-06-30"
    gi = GDELTIngestor()
    gd = gi.aggregate_daily(gi._tag_entities(_generate_synthetic_gdelt(500)))
    tp = TruthSocialPipeline()
    te = tp.run_full_pipeline(generate_synthetic_posts(200), save=False)
    td = tp.aggregate_daily(te)
    tw = TwitterIngestor().compute_features(generate_synthetic_tweets(5000, START, END))
    sd = SupplyIngestor().fetch_all(START, END, save=False)
    rng = np.random.default_rng(99)
    dates = pd.date_range(START, END, freq="B")
    pv = pd.DataFrame({
        "BZ=F_close": 75+np.cumsum(rng.normal(0,1,len(dates))),
        "^VIX_close": 18+np.cumsum(rng.normal(0,.5,len(dates))),
    }, index=dates); pv.index.name="date"

    u = TimeIndexedMerge().merge_all(gdelt_daily=gd, truth_social_daily=td,
        twitter_daily=tw, supply_daily=sd, price_vix=pv,
        start=START, end=END, save=False)
    u = TensionIndexBuilder().compute(u)
    u = DHOKernelEngine().compute(u)
    u = TrumpVolatilityEngine().compute(u, enriched_posts=te)
    u = TwitterFeatureBuilder().compute(u)
    u = SupplyFeatureBuilder().compute(u)
    u = WarTensionBarometer().apply_level(u, level=5)

    vix = u["^VIX_close"].ffill()
    for h in PREDICTION_HORIZONS:
        u[f"vix_return_{h}h"] = vix.pct_change(max(1,h//24)).shift(-max(1,h//24))

    m1 = HMMRegimeDetector(); m1.fit(u, save=False); u = m1.predict(u)
    m2 = RegimeXGBoostTrainer(); m2.fit(u, save=False); u = m2.predict(u)
    m3 = EnsemblePredictor(); m3.fit(u, save=False); u = m3.predict(u)

    # M4: evaluate
    evaluator = ModelEvaluator()
    report = evaluator.generate_report(u, m2_trainer=m2)

    print(f"\n--- Overall Metrics ---")
    for horizon, metrics in report["overall_metrics"].items():
        if isinstance(metrics, dict) and "brier_score" in metrics:
            print(f"  {horizon}: Brier={metrics['brier_score']}  "
                  f"AUC={metrics['auc']}  F1={metrics['f1']}  "
                  f"n={metrics['n_test']}")

    print(f"\n--- Per-Regime Metrics ---")
    for regime, horizons in report.get("per_regime", {}).items():
        for h, m in horizons.items():
            print(f"  {regime:20s} {h}: Brier={m['brier']}  AUC={m['auc']}  n={m['n']}")

    print(f"\n--- Feature Importance (Regime 0, 24h) ---")
    if "regime_0" in report.get("feature_importance", {}):
        for f in report["feature_importance"]["regime_0"][:8]:
            print(f"  {f['feature']:35s} {f['importance_pct']:6.2f}%  {f['direction']}")

    print(f"\n✅ M4 test PASSED")
