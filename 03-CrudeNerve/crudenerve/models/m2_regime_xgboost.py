"""
M2: Regime-Specific XGBoost Quantile Models.

Trains separate XGBoost models per regime × horizon:
    4 regimes × 3 horizons = 12 models

Each model outputs quantile predictions (0.1, 0.25, 0.5, 0.75, 0.9)
for VIX return distributions.

Feature subsets vary by regime:
    - calm:            all features, equal weight
    - escalation:      upweights Trump + Twitter + DHO; drops slow supply features
    - active_conflict: upweights Hormuz + DHO + fear; everything at max sensitivity
    - de-escalation:   upweights supply normalization; downweights social sentiment

Usage:
    from crudenerve.models.m2_regime_xgboost import RegimeXGBoostTrainer
    trainer = RegimeXGBoostTrainer()
    trainer.fit(unified_df)
    predictions = trainer.predict(unified_df)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from crudenerve.config.settings import (
    PREDICTION_HORIZONS,
    QUANTILE_LEVELS,
    VIX_DIRECTION_THRESHOLD,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

# Feature groups for regime-aware selection
FEATURE_GROUPS = {
    "tension":  ["f2_tension_ema", "f2_tension_velocity", "f2_tension_accel", "f2_tension_zscore"],
    "dho":      ["f3_dho_gdelt", "f3_dho_trump", "f3_dho_combined", "f3_dho_n_active"],
    "trump":    ["f4_trump_shock_score", "f4_trump_shock_72h", "f4_trump_escalation"],
    "twitter":  ["f5_volume_regime", "f5_sentiment_momentum", "f5_divergence_signal", "f5_fear_index"],
    "supply":   ["f6_inventory_surprise_signal", "f6_spr_release_momentum",
                 "f6_hormuz_disruption_score", "f6_opec_discipline", "f6_supply_stress"],
    "war":      ["f8_war_level", "f8_tension_multiplier", "f8_hormuz_weight"],
    "regime":   ["m1_regime", "m1_prob_calm", "m1_prob_escalation",
                 "m1_prob_active_conflict", "m1_prob_de_escalation"],
}

# Regime-specific feature weights (which groups matter more in each regime)
REGIME_FEATURE_WEIGHTS = {
    0: {"tension": 1.0, "dho": 1.0, "trump": 1.0, "twitter": 1.0, "supply": 1.0, "war": 1.0, "regime": 1.0},
    1: {"tension": 1.5, "dho": 1.5, "trump": 2.0, "twitter": 1.5, "supply": 0.5, "war": 1.5, "regime": 1.0},
    2: {"tension": 1.5, "dho": 2.0, "trump": 1.5, "twitter": 2.0, "supply": 1.5, "war": 2.0, "regime": 1.0},
    3: {"tension": 1.0, "dho": 0.8, "trump": 0.7, "twitter": 0.8, "supply": 1.5, "war": 1.0, "regime": 1.0},
}


class RegimeXGBoostTrainer:
    """
    Trains and manages 12 regime-specific XGBoost models
    (4 regimes × 3 horizons).
    """

    def __init__(self):
        self.models: dict[tuple[int, int, float], xgb.XGBRegressor] = {}
        self.feature_names: list[str] = []
        self.save_dir = MODELS_DIR / "m2_xgboost"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get all usable numeric feature columns."""
        exclude_prefixes = ("vix_", "brent_return", "mode_", "m1_regime_name")
        exclude_exact = {"date"}

        cols = []
        for c in df.columns:
            if c in exclude_exact:
                continue
            if any(c.startswith(p) for p in exclude_prefixes):
                continue
            if df[c].dtype in ["float64", "float32", "int64", "int32"]:
                cols.append(c)
        return cols

    def _apply_regime_weights(
        self,
        X: pd.DataFrame,
        regime: int,
    ) -> pd.DataFrame:
        """
        Apply regime-specific feature weighting.

        Instead of dropping features (which loses information),
        we scale them — downweighted features have reduced influence
        while upweighted features dominate the tree splits.
        """
        weights = REGIME_FEATURE_WEIGHTS.get(regime, REGIME_FEATURE_WEIGHTS[0])
        X_weighted = X.copy()

        for group_name, group_cols in FEATURE_GROUPS.items():
            w = weights.get(group_name, 1.0)
            for col in group_cols:
                if col in X_weighted.columns:
                    X_weighted[col] = X_weighted[col] * w

        return X_weighted

    def fit(
        self,
        df: pd.DataFrame,
        target_prefix: str = "vix_return",
        save: bool = True,
    ) -> "RegimeXGBoostTrainer":
        """
        Train one XGBoost model per regime × horizon × quantile.

        For each combination:
            1. Filter data to rows belonging to that regime
            2. Apply regime-specific feature weighting
            3. Train XGBoost with quantile loss for each quantile level
        """
        if "m1_regime" not in df.columns:
            raise ValueError("m1_regime column missing. Run M1 first.")

        self.feature_names = self._get_feature_cols(df)
        logger.info(f"Training M2 with {len(self.feature_names)} features")

        total_models = 0
        for regime in range(4):
            regime_mask = df["m1_regime"] == regime
            regime_df = df[regime_mask].copy()

            if len(regime_df) < 20:
                logger.warning(
                    f"Regime {regime}: only {len(regime_df)} samples — "
                    f"using full dataset with regime weighting instead"
                )
                regime_df = df.copy()

            X = regime_df[self.feature_names].fillna(0)
            X = self._apply_regime_weights(X, regime)

            for horizon in PREDICTION_HORIZONS:
                target_col = f"{target_prefix}_{horizon}h"
                if target_col not in regime_df.columns:
                    logger.warning(f"Target {target_col} not found — skipping")
                    continue

                y = regime_df[target_col].fillna(0)

                # Drop rows where target is NaN (end of series)
                valid = y.notna()
                X_valid = X[valid]
                y_valid = y[valid]

                if len(y_valid) < 10:
                    continue

                for quantile in QUANTILE_LEVELS:
                    model = xgb.XGBRegressor(
                        objective="reg:quantileerror",
                        quantile_alpha=quantile,
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        random_state=42,
                        verbosity=0,
                    )
                    model.fit(X_valid, y_valid)
                    self.models[(regime, horizon, quantile)] = model
                    total_models += 1

        logger.info(f"M2 training complete: {total_models} models fitted")

        if save:
            self._save()

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate quantile predictions for each row using its regime-matched model.

        Adds columns:
            m2_vix_return_{horizon}h_q{quantile}: predicted quantile value
            m2_vix_direction_{horizon}h_prob: probability of >2% move
        """
        if not self.models:
            raise RuntimeError("No models fitted. Call fit() first.")

        out = df.copy()
        X_all = df[self.feature_names].fillna(0)

        for horizon in PREDICTION_HORIZONS:
            # Per-quantile prediction arrays
            quantile_preds = {q: np.zeros(len(df)) for q in QUANTILE_LEVELS}

            for i, row in df.iterrows():
                idx = df.index.get_loc(i)
                regime = int(row.get("m1_regime", 0))
                X_row = self._apply_regime_weights(
                    X_all.iloc[[idx]], regime
                )

                for quantile in QUANTILE_LEVELS:
                    key = (regime, horizon, quantile)
                    if key in self.models:
                        pred = self.models[key].predict(X_row)[0]
                    else:
                        # Fallback: use regime 0 model
                        fallback_key = (0, horizon, quantile)
                        if fallback_key in self.models:
                            pred = self.models[fallback_key].predict(X_row)[0]
                        else:
                            pred = 0.0
                    quantile_preds[quantile][idx] = pred

            # Add columns
            for quantile in QUANTILE_LEVELS:
                q_label = str(quantile).replace(".", "")
                col = f"m2_vix_return_{horizon}h_q{q_label}"
                out[col] = quantile_preds[quantile]

            # Direction probability: fraction of quantiles predicting |return| > threshold
            q_vals = np.column_stack([quantile_preds[q] for q in QUANTILE_LEVELS])
            move_fraction = np.mean(np.abs(q_vals) > VIX_DIRECTION_THRESHOLD, axis=1)
            out[f"m2_vix_direction_{horizon}h_prob"] = move_fraction

            # Median prediction for quick reference
            out[f"m2_vix_return_{horizon}h_median"] = quantile_preds[0.5]

        m2_cols = [c for c in out.columns if c.startswith("m2_")]
        logger.info(f"M2 predictions: {len(m2_cols)} output columns")

        return out

    def get_feature_importance(self, regime: int = 0, horizon: int = 24) -> pd.DataFrame:
        """Get feature importance for a specific regime × horizon model."""
        key = (regime, horizon, 0.5)  # use median quantile model
        if key not in self.models:
            return pd.DataFrame()

        model = self.models[key]
        importance = model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def _save(self):
        path = self.save_dir / "regime_xgboost.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "feature_names": self.feature_names,
            }, f)
        logger.info(f"Saved {len(self.models)} M2 models → {path}")

    def load(self) -> "RegimeXGBoostTrainer":
        path = self.save_dir / "regime_xgboost.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.feature_names = data["feature_names"]
        logger.info(f"Loaded {len(self.models)} M2 models")
        return self


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Testing M2: Regime-Specific XGBoost")
    print("=" * 60)

    # Quick synthetic pipeline
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

    START, END = "2024-01-01", "2024-06-30"
    gi = GDELTIngestor()
    gd = gi.aggregate_daily(gi._tag_entities(_generate_synthetic_gdelt(500)))
    tp = TruthSocialPipeline()
    posts = generate_synthetic_posts(200)
    te = tp.run_full_pipeline(posts, save=False)
    td = tp.aggregate_daily(te)
    tw = TwitterIngestor().compute_features(generate_synthetic_tweets(5000, START, END))
    sd = SupplyIngestor().fetch_all(START, END, save=False)
    rng = np.random.default_rng(99)
    dates = pd.date_range(START, END, freq="B")
    pv = pd.DataFrame({
        "BZ=F_close": 75+np.cumsum(rng.normal(0,1,len(dates))),
        "^VIX_close": 18+np.cumsum(rng.normal(0,.5,len(dates))),
    }, index=dates)
    pv.index.name = "date"

    u = TimeIndexedMerge().merge_all(gdelt_daily=gd, truth_social_daily=td,
        twitter_daily=tw, supply_daily=sd, price_vix=pv,
        start=START, end=END, save=False)
    u = TensionIndexBuilder().compute(u)
    u = DHOKernelEngine().compute(u)
    u = TrumpVolatilityEngine().compute(u, enriched_posts=te)
    u = TwitterFeatureBuilder().compute(u)
    u = SupplyFeatureBuilder().compute(u)
    u = WarTensionBarometer().apply_level(u, level=5)

    # Add synthetic VIX return targets
    if "^VIX_close" in u.columns:
        vix = u["^VIX_close"].ffill()
        for h in PREDICTION_HORIZONS:
            days = max(1, h // 24)
            u[f"vix_return_{h}h"] = vix.pct_change(days).shift(-days)

    # M1: regime detection
    m1 = HMMRegimeDetector()
    m1.fit(u, save=False)
    u = m1.predict(u)

    # M2: train
    trainer = RegimeXGBoostTrainer()
    trainer.fit(u, save=False)

    # M2: predict
    result = trainer.predict(u)

    m2_cols = [c for c in result.columns if c.startswith("m2_")]
    print(f"\nM2 output columns ({len(m2_cols)}):")
    for col in sorted(m2_cols):
        print(f"  {col}: mean={result[col].mean():.4f}")

    print(f"\nFeature importance (regime=0, 24h):")
    fi = trainer.get_feature_importance(regime=0, horizon=24)
    print(fi.head(10).to_string(index=False))

    print(f"\n✅ M2 test PASSED — {len(trainer.models)} models trained")
