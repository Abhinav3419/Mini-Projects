"""
D3: Twitter/X sentiment ingestion.

Three feature types, not naive positive/negative:
  1. Volume spike ratio: current tweet volume vs 7-day rolling average
  2. Sentiment velocity: rate of change in sentiment, not the level
  3. Elite signal: accounts >100K followers weighted separately from retail

Usage:
    from crudenerve.data_ingest.d3_twitter import TwitterIngestor
    ingestor = TwitterIngestor()
    df = ingestor.fetch_and_process(start="2024-01-01", end="2024-06-30")
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from crudenerve.config.settings import TwitterConfig, RAW_DIR

logger = logging.getLogger(__name__)


class TwitterIngestor:
    """
    Fetches oil-related tweet data from X API v2 and computes
    volume, velocity, and elite vs retail sentiment features.
    """

    def __init__(self, config: TwitterConfig | None = None):
        self.cfg = config or TwitterConfig()
        self.save_dir = RAW_DIR / "twitter"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_query(self) -> str:
        """Build X API search query from oil keywords."""
        terms = " OR ".join(self.cfg.oil_keywords)
        return f"({terms}) lang:en -is:retweet"

    def fetch_tweets(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Fetch tweets from X API v2.

        In production, uses academic research endpoint for historical data
        or recent search for real-time. Returns raw tweet data with
        author follower counts.

        For development: use generate_synthetic_tweets().
        """
        raise NotImplementedError(
            "Live X API fetch requires bearer token. "
            "Use generate_synthetic_tweets() for development."
        )

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the three Twitter feature types from raw tweet data.

        Input DataFrame needs: timestamp_utc, text, sentiment_score,
                               follower_count

        Output: daily DataFrame with volume spike, sentiment velocity,
                and elite/retail split features.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        df["is_elite"] = df["follower_count"] >= self.cfg.elite_follower_threshold

        # ── Feature 1: Volume spike ratio ────────────────────────────────
        daily_volume = df.groupby("date").size().rename("tweet_count")
        daily_volume = daily_volume.to_frame()
        rolling_avg = daily_volume["tweet_count"].rolling(
            window=self.cfg.volume_rolling_window_days,
            min_periods=1,
        ).mean()
        daily_volume["tw_volume_spike_ratio"] = (
            daily_volume["tweet_count"] / rolling_avg.replace(0, 1)
        )

        # ── Feature 2: Sentiment velocity ────────────────────────────────
        daily_sentiment = df.groupby("date")["sentiment_score"].mean()
        daily_sentiment = daily_sentiment.to_frame("tw_sentiment_level")

        # Velocity = rate of change (first difference, not the level)
        daily_sentiment["tw_sentiment_velocity"] = (
            daily_sentiment["tw_sentiment_level"].diff()
        )
        # Acceleration = second derivative
        daily_sentiment["tw_sentiment_accel"] = (
            daily_sentiment["tw_sentiment_velocity"].diff()
        )

        # ── Feature 3: Elite vs retail split ─────────────────────────────
        elite = df[df["is_elite"]].groupby("date").agg(
            tw_elite_count=("text", "count"),
            tw_elite_sentiment=("sentiment_score", "mean"),
        )
        retail = df[~df["is_elite"]].groupby("date").agg(
            tw_retail_count=("text", "count"),
            tw_retail_sentiment=("sentiment_score", "mean"),
        )

        # Elite-retail divergence: when elites go negative but retail
        # stays positive (or vice versa), something is about to break
        combined = pd.concat(
            [daily_volume, daily_sentiment, elite, retail],
            axis=1,
        )
        combined.fillna(0, inplace=True)

        combined["tw_elite_retail_divergence"] = (
            combined.get("tw_elite_sentiment", 0)
            - combined.get("tw_retail_sentiment", 0)
        )

        # Elite-weighted composite: elite sentiment gets 3x weight
        w = self.cfg.elite_weight_multiplier
        e_count = combined.get("tw_elite_count", 0)
        r_count = combined.get("tw_retail_count", 0)
        e_sent = combined.get("tw_elite_sentiment", 0)
        r_sent = combined.get("tw_retail_sentiment", 0)

        total_weight = (e_count * w + r_count).replace(0, 1)
        combined["tw_weighted_sentiment"] = (
            (e_sent * e_count * w + r_sent * r_count) / total_weight
        )

        combined.index.name = "date"
        return combined

    def compute_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score sentiment on raw tweets using VADER + transformer ensemble.

        If 'sentiment_score' column already exists (pre-scored data),
        skips re-computation. Otherwise:
        - Tries transformer NLP engine (VADER 0.3 + sentence-transformers 0.7)
        - Falls back to VADER-only if transformers unavailable
        - Falls back to zeros if neither available

        This runs BEFORE compute_features() — it produces the
        sentiment_score column that compute_features() consumes.
        """
        if "sentiment_score" in df.columns:
            logger.info("sentiment_score already present — skipping scoring")
            return df

        out = df.copy()
        texts = out["text"].tolist()

        # Try transformer ensemble first
        try:
            from crudenerve.utils.nlp_engine import NLPEngine
            engine = NLPEngine()
            logger.info(f"Scoring {len(texts)} tweets with transformer NLP engine...")
            results = engine.analyze_batch(texts, show_progress=False)
            out["sentiment_score"] = [r.sentiment for r in results]
            out["sentiment_vader"] = [r.sentiment_vader for r in results]
            out["sentiment_transformer"] = [r.sentiment_transformer for r in results]
            logger.info("Twitter sentiment scored with transformer ensemble")
            return out
        except Exception as e:
            logger.info(f"Transformer unavailable ({e}), trying VADER-only")

        # Fallback: VADER only
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
            out["sentiment_score"] = [
                vader.polarity_scores(t)["compound"] for t in texts
            ]
            out["sentiment_vader"] = out["sentiment_score"]
            out["sentiment_transformer"] = None
            logger.info("Twitter sentiment scored with VADER-only fallback")
            return out
        except ImportError:
            logger.warning("No sentiment model available — using zeros")

        out["sentiment_score"] = 0.0
        return out

    def fetch_and_process(
        self,
        start: str,
        end: str,
        save: bool = True,
    ) -> pd.DataFrame:
        """Fetch tweets + score sentiment + compute features."""
        try:
            raw = self.fetch_tweets(start, end)
        except NotImplementedError:
            logger.info("Using synthetic Twitter data for development")
            raw = generate_synthetic_tweets(n=5000, start=start, end=end)

        # Score sentiment BEFORE computing features
        raw = self.compute_sentiment(raw)

        features = self.compute_features(raw)

        if save and not features.empty:
            out_path = self.save_dir / "twitter_features.parquet"
            features.to_parquet(out_path)
            logger.info(f"Saved Twitter features → {out_path}")

        return features

    def load_cached(self) -> pd.DataFrame | None:
        path = self.save_dir / "twitter_features.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


# ─── Synthetic data generator ────────────────────────────────────────────────

def generate_synthetic_tweets(
    n: int = 5000,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
) -> pd.DataFrame:
    """Generate synthetic oil-related tweet data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, end, freq="D")

    records = []
    for _ in range(n):
        day = rng.choice(dates)
        hour = rng.integers(0, 24)

        # Sentiment: mostly neutral-ish, with spikes during "events"
        # Simulate event days (every ~30 days) where sentiment goes negative
        day_of_range = (day - dates[0]).days
        is_event_day = (day_of_range % 30) < 3
        base_sentiment = -0.5 if is_event_day else 0.1
        sentiment = np.clip(rng.normal(base_sentiment, 0.3), -1, 1)

        # Follower count: power law (most accounts small, few are elite)
        followers = int(10 ** rng.uniform(2, 7))

        # Volume spikes on event days
        records.append({
            "tweet_id": f"tw_{rng.integers(1_000_000_000):010d}",
            "text": "synthetic oil tweet",
            "timestamp_utc": day + pd.Timedelta(
                hours=int(hour), minutes=int(rng.integers(0, 60))
            ),
            "sentiment_score": round(float(sentiment), 3),
            "follower_count": followers,
        })

    # Add extra tweets on "event days" to simulate volume spikes
    event_days = dates[::30][:6]
    for day in event_days:
        spike_n = rng.integers(50, 200)
        for _ in range(spike_n):
            records.append({
                "tweet_id": f"tw_{rng.integers(1_000_000_000):010d}",
                "text": "synthetic spike tweet",
                "timestamp_utc": day + pd.Timedelta(
                    hours=int(rng.integers(0, 24)),
                    minutes=int(rng.integers(0, 60)),
                ),
                "sentiment_score": round(float(rng.normal(-0.6, 0.25)), 3),
                "follower_count": int(10 ** rng.uniform(2, 7)),
            })

    df = pd.DataFrame(records)
    df.sort_values("timestamp_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing D3: Twitter/X sentiment pipeline\n")

    ingestor = TwitterIngestor()
    raw = generate_synthetic_tweets(5000)
    print(f"Synthetic tweets: {len(raw)}")

    features = ingestor.compute_features(raw)
    print(f"Daily feature rows: {len(features)}")
    print(f"\nColumns: {list(features.columns)}")
    print(f"\nVolume spike ratio stats:")
    print(features["tw_volume_spike_ratio"].describe())
    print(f"\nSentiment velocity stats:")
    print(features["tw_sentiment_velocity"].describe())
    print(f"\nElite-retail divergence stats:")
    print(features["tw_elite_retail_divergence"].describe())
    print(f"\nSample (days with highest volume spike):")
    print(features.nlargest(5, "tw_volume_spike_ratio"))
