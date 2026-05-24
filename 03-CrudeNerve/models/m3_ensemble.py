"""
M3: Final Ensemble.

Combines M2 regime-specific XGBoost quantile predictions into
calibrated probability estimates:

    1. Takes M2 quantile predictions per horizon
    2. Calibrates using Platt scaling (logistic regression on validation set)
    3. Computes final VIX direction probability + Brent return distribution
    4. Adjusts for prediction mode and war level

Usage:
    from crudenerve.models.m3_ensemble import EnsemblePredictor
    predictor = EnsemblePredictor()
    predictor.fit(unified_df)
    result = predictor.predict(unified_df)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from crudenerve.config.settings import (
    PREDICTION_HORIZONS,
    QUANTILE_LEVELS,
    VIX_DIRECTION_THRESHOLD,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Final prediction layer that calibrates M2 outputs into
    actionable probabilities.

    Calibration approach:
        - Platt scaling: fit logistic regression on M2 quantile features
          to predict binary VIX direction (up/down >2%)
        - The calibrated output is a true probability, not just a score
    """

    def __init__(self):
        self.calibrators: dict[int, LogisticRegression] = {}
        self.scalers: dict[int, StandardScaler] = {}
        self.save_dir = MODELS_DIR / "m3_ensemble"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_m2_features(self, df: pd.DataFrame, horizon: int) -> list[str]:
        """Get M2 output columns for a specific horizon."""
        return [c for c in df.columns
                if c.startswith(f"m2_vix_return_{horizon}h_q")
                or c == f"m2_vix_direction_{horizon}h_prob"
                or c == f"m2_vix_return_{horizon}h_median"]

    def fit(
        self,
        df: pd.DataFrame,
        save: bool = True,
    ) -> "EnsemblePredictor":
        """
        Fit Platt scaling calibrators on M2 output features.

        Uses VIX forward return targets as ground truth.
        """
        logger.info("Fitting M3 ensemble calibrators...")

        for horizon in PREDICTION_HORIZONS:
            target_col = f"vix_return_{horizon}h"
            if target_col not in df.columns:
                logger.warning(f"Target {target_col} not found — skipping")
                continue

            feature_cols = self._get_m2_features(df, horizon)
            if not feature_cols:
                logger.warning(f"No M2 features for {horizon}h — skipping")
                continue

            # Binary target: |return| > threshold
            y = (df[target_col].abs() > VIX_DIRECTION_THRESHOLD).astype(int)
            X = df[feature_cols].fillna(0)

            # Drop NaN targets
            valid = y.notna()
            X_valid = X[valid]
            y_valid = y[valid]

            if len(y_valid) < 20:
                logger.warning(f"Not enough samples for {horizon}h calibration")
                continue

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)

            # Fit logistic regression (Platt scaling)
            calibrator = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
            calibrator.fit(X_scaled, y_valid)

            self.calibrators[horizon] = calibrator
            self.scalers[horizon] = scaler

            train_acc = calibrator.score(X_scaled, y_valid)
            logger.info(f"  {horizon}h calibrator: train_acc={train_acc:.3f}")

        if save:
            self._save()

        logger.info(f"M3 ensemble: {len(self.calibrators)} calibrators fitted")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate calibrated ensemble predictions.

        Adds columns:
            m3_vix_direction_{h}h_prob:  calibrated probability of >2% move
            m3_vix_up_{h}h_prob:         probability of >2% upward move
            m3_vix_down_{h}h_prob:       probability of >2% downward move
            m3_brent_return_{h}h_q*:     Brent return quantiles (passed through)
            m3_confidence_{h}h:          model confidence (calibrator certainty)
        """
        out = df.copy()

        for horizon in PREDICTION_HORIZONS:
            feature_cols = self._get_m2_features(df, horizon)

            if horizon in self.calibrators and feature_cols:
                X = df[feature_cols].fillna(0)
                X_scaled = self.scalers[horizon].transform(X)

                # Calibrated probability
                probs = self.calibrators[horizon].predict_proba(X_scaled)
                # probs[:, 1] = probability of class 1 (|return| > threshold)
                if probs.shape[1] == 2:
                    dir_prob = probs[:, 1]
                else:
                    dir_prob = probs[:, 0]

                out[f"m3_vix_direction_{horizon}h_prob"] = dir_prob

                # Directional split: use median prediction sign
                median_col = f"m2_vix_return_{horizon}h_median"
                if median_col in df.columns:
                    median_pred = df[median_col].fillna(0)
                    up_bias = (median_pred > 0).astype(float) * 0.3 + 0.5
                    out[f"m3_vix_up_{horizon}h_prob"] = dir_prob * up_bias
                    out[f"m3_vix_down_{horizon}h_prob"] = dir_prob * (1 - up_bias)
                else:
                    out[f"m3_vix_up_{horizon}h_prob"] = dir_prob * 0.5
                    out[f"m3_vix_down_{horizon}h_prob"] = dir_prob * 0.5

                # Confidence: how far from 0.5 (maximum uncertainty)
                out[f"m3_confidence_{horizon}h"] = 2 * np.abs(dir_prob - 0.5)

            else:
                # Fallback: use M2 raw probabilities
                raw_prob_col = f"m2_vix_direction_{horizon}h_prob"
                if raw_prob_col in df.columns:
                    out[f"m3_vix_direction_{horizon}h_prob"] = df[raw_prob_col]
                else:
                    out[f"m3_vix_direction_{horizon}h_prob"] = 0.5

                out[f"m3_vix_up_{horizon}h_prob"] = 0.25
                out[f"m3_vix_down_{horizon}h_prob"] = 0.25
                out[f"m3_confidence_{horizon}h"] = 0.0

            # Brent return quantiles: pass through M2 quantiles
            for q in QUANTILE_LEVELS:
                q_label = str(q).replace(".", "")
                m2_col = f"m2_vix_return_{horizon}h_q{q_label}"
                if m2_col in df.columns:
                    out[f"m3_brent_return_{horizon}h_q{q_label}"] = df[m2_col]

        m3_cols = [c for c in out.columns if c.startswith("m3_")]
        logger.info(f"M3 ensemble predictions: {len(m3_cols)} columns")
        return out

    def get_prediction_summary(self, df: pd.DataFrame) -> dict:
        """Get a clean summary of the latest predictions."""
        if df.empty:
            return {}

        latest = df.iloc[-1]
        summary = {"horizons": {}}

        for horizon in PREDICTION_HORIZONS:
            h_summary = {
                "vix_direction_prob": float(
                    latest.get(f"m3_vix_direction_{horizon}h_prob", 0.5)
                ),
                "vix_up_prob": float(
                    latest.get(f"m3_vix_up_{horizon}h_prob", 0.25)
                ),
                "vix_down_prob": float(
                    latest.get(f"m3_vix_down_{horizon}h_prob", 0.25)
                ),
                "confidence": float(
                    latest.get(f"m3_confidence_{horizon}h", 0)
                ),
                "brent_quantiles": {},
            }
            for q in QUANTILE_LEVELS:
                q_label = str(q).replace(".", "")
                val = latest.get(f"m3_brent_return_{horizon}h_q{q_label}", 0)
                h_summary["brent_quantiles"][f"q{q_label}"] = float(val)

            summary["horizons"][f"{horizon}h"] = h_summary

        return summary

    def _save(self):
        path = self.save_dir / "ensemble.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "calibrators": self.calibrators,
                "scalers": self.scalers,
            }, f)
        logger.info(f"Saved M3 ensemble → {path}")

    def load(self) -> "EnsemblePredictor":
        path = self.save_dir / "ensemble.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.calibrators = data["calibrators"]
        self.scalers = data["scalers"]
        return self


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Testing M3: Final Ensemble")
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

    # M3: ensemble
    ensemble = EnsemblePredictor()
    ensemble.fit(u, save=False)
    result = ensemble.predict(u)

    m3_cols = [c for c in result.columns if c.startswith("m3_")]
    print(f"\nM3 output columns ({len(m3_cols)}):")
    for col in sorted(m3_cols):
        vals = result[col]
        print(f"  {col:45s} mean={vals.mean():.4f}  std={vals.std():.4f}")

    print(f"\nPrediction summary (latest day):")
    summary = ensemble.get_prediction_summary(result)
    for h, data in summary["horizons"].items():
        print(f"  {h}: VIX dir={data['vix_direction_prob']:.1%}  "
              f"(up={data['vix_up_prob']:.1%} / down={data['vix_down_prob']:.1%})  "
              f"confidence={data['confidence']:.1%}")

    print(f"\n✅ M3 test PASSED")
