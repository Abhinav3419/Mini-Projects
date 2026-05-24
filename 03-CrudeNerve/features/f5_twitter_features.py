"""
F5: Twitter signal features.

Adds derived features from D3 Twitter data:
  - Volume spike regime (normal/elevated/extreme)
  - Sentiment momentum (trending positive/negative vs flat)
  - Elite-retail divergence signal (when elites disagree with retail)
  - Cross-platform echo detection (Twitter response to Truth Social)

Usage:
    from crudenerve.features.f5_twitter_features import TwitterFeatureBuilder
    builder = TwitterFeatureBuilder()
    df = builder.compute(unified_df)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TwitterFeatureBuilder:
    """
    Enriches raw Twitter features (from D3) with derived signals.

    The raw D3 features are volume spike ratio, sentiment velocity,
    and elite/retail split. This module adds regime classification,
    momentum signals, and cross-platform interaction.
    """

    def __init__(
        self,
        spike_threshold_elevated: float = 1.5,
        spike_threshold_extreme: float = 3.0,
        divergence_threshold: float = 0.3,
    ):
        self.spike_elevated = spike_threshold_elevated
        self.spike_extreme = spike_threshold_extreme
        self.divergence_threshold = divergence_threshold

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Twitter-derived features.

        Adds columns:
            - f5_volume_regime:        0=normal, 1=elevated, 2=extreme
            - f5_sentiment_momentum:   3-day rolling sentiment direction
            - f5_divergence_signal:     elite-retail gap exceeds threshold
            - f5_echo_intensity:        cross-platform signal (if ts data present)
            - f5_fear_index:            composite Twitter fear signal
        """
        out = df.copy()

        # ── Volume regime classification ─────────────────────────────────
        spike = out.get("tw_volume_spike_ratio", pd.Series(1.0, index=out.index)).fillna(1.0)
        out["f5_volume_regime"] = np.where(
            spike >= self.spike_extreme, 2,
            np.where(spike >= self.spike_elevated, 1, 0)
        )

        # ── Sentiment momentum ───────────────────────────────────────────
        velocity = out.get("tw_sentiment_velocity", pd.Series(0, index=out.index)).fillna(0)
        # 3-day rolling mean of velocity (smoothed direction)
        out["f5_sentiment_momentum"] = velocity.rolling(3, min_periods=1).mean()
        # Binary momentum: 1 = improving, -1 = deteriorating
        out["f5_sentiment_direction"] = np.sign(out["f5_sentiment_momentum"])

        # ── Elite-retail divergence signal ───────────────────────────────
        divergence = out.get(
            "tw_elite_retail_divergence",
            pd.Series(0, index=out.index),
        ).fillna(0)
        out["f5_divergence_signal"] = (
            divergence.abs() > self.divergence_threshold
        ).astype(int)
        # Direction: +1 = elites more positive (complacent?), -1 = elites more negative (warning)
        out["f5_divergence_direction"] = np.where(
            out["f5_divergence_signal"] == 1,
            np.sign(divergence),
            0,
        )

        # ── Cross-platform echo ──────────────────────────────────────────
        # When Trump posts on Truth Social, does Twitter volume spike
        # on the same day or next day?
        ts_relevant = out.get("ts_relevant_count", pd.Series(0, index=out.index)).fillna(0)
        ts_severity = out.get("ts_max_severity", pd.Series(0, index=out.index)).fillna(0)

        # Echo = Twitter volume spike on days with high-severity Truth Social posts
        out["f5_echo_intensity"] = (
            spike * (ts_severity / 5.0)
        ).fillna(0)

        # ── Composite Twitter fear index ─────────────────────────────────
        # Combines: volume spike (unusual attention), negative sentiment
        # momentum, and elite pessimism into a single fear signal.
        sentiment_level = out.get(
            "tw_sentiment_level", pd.Series(0, index=out.index)
        ).fillna(0)

        # Invert sentiment (negative = more fear)
        fear_sentiment = (-sentiment_level).clip(0, 1)
        fear_volume = (spike - 1).clip(0, 5) / 5.0
        fear_elite = (-out.get("tw_elite_sentiment", pd.Series(0, index=out.index)).fillna(0)).clip(0, 1)

        out["f5_fear_index"] = (
            fear_volume * 0.4
            + fear_sentiment * 0.35
            + fear_elite * 0.25
        )

        f5_cols = [c for c in out.columns if c.startswith("f5_")]
        logger.info(f"Twitter features computed: {len(f5_cols)} columns")
        return out


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing F5: Twitter Signal Features\n")

    from crudenerve.data_ingest.d3_twitter import TwitterIngestor, generate_synthetic_tweets

    raw = generate_synthetic_tweets(5000, "2024-01-01", "2024-06-30")
    ingestor = TwitterIngestor()
    tw_daily = ingestor.compute_features(raw)

    builder = TwitterFeatureBuilder()
    result = builder.compute(tw_daily)

    f5_cols = [c for c in result.columns if c.startswith("f5_")]
    print(f"F5 columns: {f5_cols}")
    for col in f5_cols:
        vals = result[col]
        if vals.dtype in ["float64", "int64"]:
            print(f"  {col:30s} mean={vals.mean():.3f}  max={vals.max():.3f}")

    print(f"\nVolume regime distribution:")
    print(result["f5_volume_regime"].value_counts().sort_index())
    print(f"\nFear index stats:")
    print(result["f5_fear_index"].describe())
