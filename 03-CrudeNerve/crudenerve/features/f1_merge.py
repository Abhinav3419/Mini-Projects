"""
F1: Time-indexed merge engine.

Aligns all 5 data streams (D1-D5) into a unified daily DataFrame.
Handles timezone normalization, gap filling, and column namespacing.

This is the single source of truth for all downstream feature engineering
and model training. Every feature module (F2-F6) reads from this DataFrame.

Usage:
    from crudenerve.features.f1_merge import TimeIndexedMerge
    merger = TimeIndexedMerge()
    unified = merger.merge_all(
        gdelt_daily=..., truth_social_daily=...,
        twitter_daily=..., supply_daily=..., price_vix=...,
    )
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crudenerve.config.settings import (
    BACKFILL_START,
    BACKFILL_END,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


class TimeIndexedMerge:
    """
    Merges all CrudeNerve data streams into a single time-indexed DataFrame.

    Responsibilities:
        - Normalize all indices to daily datetime (UTC midnight)
        - Forward-fill weekly/monthly data (with limits)
        - Namespace columns to avoid collisions
        - Track data availability per stream per day
        - Save the unified DataFrame for downstream consumption
    """

    def __init__(self):
        self.save_dir = PROCESSED_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_index(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Ensure DatetimeIndex at daily frequency, UTC, named 'date'."""
        if df.empty:
            logger.warning(f"Empty DataFrame for {name} — skipping")
            return df

        df = df.copy()

        # Handle various index types
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Strip timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Normalize to midnight
        df.index = df.index.normalize()
        df.index.name = "date"

        # Deduplicate (take last value for each date)
        if df.index.duplicated().any():
            logger.info(f"{name}: {df.index.duplicated().sum()} duplicate dates — keeping last")
            df = df[~df.index.duplicated(keep="last")]

        df.sort_index(inplace=True)
        return df

    def merge_all(
        self,
        gdelt_daily: pd.DataFrame | None = None,
        truth_social_daily: pd.DataFrame | None = None,
        twitter_daily: pd.DataFrame | None = None,
        supply_daily: pd.DataFrame | None = None,
        price_vix: pd.DataFrame | None = None,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Merge all streams into a unified daily DataFrame.

        Each stream is optional — missing streams get NaN columns
        that downstream modules can handle. This lets you build
        incrementally (start with just D5 + D1, add others later).
        """
        # Create the master date index
        full_index = pd.date_range(start, end, freq="D", name="date")
        unified = pd.DataFrame(index=full_index)

        # Track which streams are available each day
        stream_presence = pd.DataFrame(index=full_index)

        # ── Stream 1: GDELT (D1) ────────────────────────────────────────
        if gdelt_daily is not None and not gdelt_daily.empty:
            gdelt = self._normalize_index(gdelt_daily, "GDELT")
            unified = unified.join(gdelt, how="left")
            stream_presence["has_gdelt"] = unified["gdelt_event_count"].notna().astype(int) \
                if "gdelt_event_count" in unified.columns else 0
            logger.info(f"Merged GDELT: {gdelt.shape[1]} columns, "
                       f"{stream_presence.get('has_gdelt', pd.Series()).sum()} days with data")
        else:
            stream_presence["has_gdelt"] = 0

        # ── Stream 2: Truth Social (D2) ──────────────────────────────────
        if truth_social_daily is not None and not truth_social_daily.empty:
            ts = self._normalize_index(truth_social_daily, "TruthSocial")
            unified = unified.join(ts, how="left")
            stream_presence["has_truth_social"] = unified["ts_post_count"].notna().astype(int) \
                if "ts_post_count" in unified.columns else 0
            logger.info(f"Merged Truth Social: {ts.shape[1]} columns")
        else:
            stream_presence["has_truth_social"] = 0

        # ── Stream 3: Twitter/X (D3) ────────────────────────────────────
        if twitter_daily is not None and not twitter_daily.empty:
            tw = self._normalize_index(twitter_daily, "Twitter")
            unified = unified.join(tw, how="left")
            stream_presence["has_twitter"] = unified["tw_volume_spike_ratio"].notna().astype(int) \
                if "tw_volume_spike_ratio" in unified.columns else 0
            logger.info(f"Merged Twitter: {tw.shape[1]} columns")
        else:
            stream_presence["has_twitter"] = 0

        # ── Stream 4: Physical supply (D4) ───────────────────────────────
        if supply_daily is not None and not supply_daily.empty:
            sup = self._normalize_index(supply_daily, "Supply")
            unified = unified.join(sup, how="left")
            stream_presence["has_supply"] = unified["eia_inventory_kbbl"].notna().astype(int) \
                if "eia_inventory_kbbl" in unified.columns else 0
            logger.info(f"Merged Supply: {sup.shape[1]} columns")
        else:
            stream_presence["has_supply"] = 0

        # ── Stream 5: Price + VIX (D5) ───────────────────────────────────
        if price_vix is not None and not price_vix.empty:
            pv = self._normalize_index(price_vix, "PriceVIX")
            unified = unified.join(pv, how="left")
            stream_presence["has_price_vix"] = unified["^VIX_close"].notna().astype(int) \
                if "^VIX_close" in unified.columns else 0
            logger.info(f"Merged Price/VIX: {pv.shape[1]} columns")
        else:
            stream_presence["has_price_vix"] = 0

        # ── Data quality metrics ─────────────────────────────────────────
        stream_presence["stream_count"] = stream_presence[
            [c for c in stream_presence.columns if c.startswith("has_")]
        ].sum(axis=1)

        unified = unified.join(stream_presence)

        # Forward-fill with limits (don't fill through genuine gaps)
        # Price data: 5 days (weekends + holidays)
        price_cols = [c for c in unified.columns if c.startswith(("BZ=F", "CL=F", "^VIX"))]
        if price_cols:
            unified[price_cols] = unified[price_cols].ffill(limit=5)

        # Weekly data (EIA, SPR): 10 days
        weekly_cols = [c for c in unified.columns if c.startswith(("eia_", "spr_"))]
        if weekly_cols:
            unified[weekly_cols] = unified[weekly_cols].ffill(limit=10)

        # Monthly data (OPEC): 35 days
        monthly_cols = [c for c in unified.columns if c.startswith("opec_")]
        if monthly_cols:
            unified[monthly_cols] = unified[monthly_cols].ffill(limit=35)

        # Social media + GDELT: no fill — missing means no signal that day
        # (which is itself informative)

        # ── Summary ──────────────────────────────────────────────────────
        logger.info(
            f"Unified DataFrame: {unified.shape[0]} days × {unified.shape[1]} columns"
        )
        logger.info(
            f"Date range: {unified.index.min()} → {unified.index.max()}"
        )
        logger.info(
            f"Stream coverage:\n"
            f"  GDELT:        {stream_presence['has_gdelt'].sum()} days\n"
            f"  Truth Social: {stream_presence['has_truth_social'].sum()} days\n"
            f"  Twitter:      {stream_presence['has_twitter'].sum()} days\n"
            f"  Supply:       {stream_presence['has_supply'].sum()} days\n"
            f"  Price/VIX:    {stream_presence['has_price_vix'].sum()} days"
        )

        if save:
            out_path = self.save_dir / "unified_daily.parquet"
            unified.to_parquet(out_path)
            logger.info(f"Saved unified DataFrame → {out_path}")

        return unified

    def load_cached(self) -> pd.DataFrame | None:
        path = self.save_dir / "unified_daily.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Testing F1: Time-indexed merge with synthetic data from all streams")
    print("=" * 70)

    # Import all ingestors
    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
    from crudenerve.data_ingest.d2_truth_social import (
        TruthSocialPipeline, generate_synthetic_posts,
    )
    from crudenerve.data_ingest.d3_twitter import TwitterIngestor, generate_synthetic_tweets
    from crudenerve.data_ingest.d4_supply import SupplyIngestor
    from crudenerve.data_ingest.d5_price_vix import PriceVIXIngestor

    START = "2024-01-01"
    END = "2024-06-30"

    # D1: GDELT
    gdelt_ingestor = GDELTIngestor()
    gdelt_raw = _generate_synthetic_gdelt(500)
    gdelt_raw = gdelt_ingestor._tag_entities(gdelt_raw)
    gdelt_daily = gdelt_ingestor.aggregate_daily(gdelt_raw)
    print(f"\nD1 GDELT daily: {gdelt_daily.shape}")

    # D2: Truth Social
    ts_pipeline = TruthSocialPipeline()
    ts_posts = generate_synthetic_posts(200)
    ts_enriched = ts_pipeline.run_full_pipeline(ts_posts, save=False)
    ts_daily = ts_pipeline.aggregate_daily(ts_enriched)
    print(f"D2 Truth Social daily: {ts_daily.shape}")

    # D3: Twitter
    tw_ingestor = TwitterIngestor()
    tw_raw = generate_synthetic_tweets(5000, START, END)
    tw_daily = tw_ingestor.compute_features(tw_raw)
    print(f"D3 Twitter daily: {tw_daily.shape}")

    # D4: Supply
    supply_ingestor = SupplyIngestor()
    supply_daily = supply_ingestor.fetch_all(START, END, save=False)
    print(f"D4 Supply daily: {supply_daily.shape}")

    # D5: Price + VIX (synthetic since no API key)
    # Generate synthetic price data
    dates = pd.date_range(START, END, freq="B")
    rng = np.random.default_rng(99)
    price_vix = pd.DataFrame({
        "BZ=F_close": 75 + np.cumsum(rng.normal(0, 1, len(dates))),
        "CL=F_close": 72 + np.cumsum(rng.normal(0, 1, len(dates))),
        "^VIX_close": 18 + np.cumsum(rng.normal(0, 0.5, len(dates))),
        "BZ=F_volume": rng.integers(100000, 500000, len(dates)),
    }, index=dates)
    price_vix.index.name = "date"
    print(f"D5 Price/VIX: {price_vix.shape}")

    # F1: Merge everything
    print("\n" + "=" * 70)
    merger = TimeIndexedMerge()
    unified = merger.merge_all(
        gdelt_daily=gdelt_daily,
        truth_social_daily=ts_daily,
        twitter_daily=tw_daily,
        supply_daily=supply_daily,
        price_vix=price_vix,
        start=START,
        end=END,
        save=False,
    )

    print(f"\n--- Unified DataFrame ---")
    print(f"Shape: {unified.shape}")
    print(f"Columns ({len(unified.columns)}):")
    for col in sorted(unified.columns):
        non_null = unified[col].notna().sum()
        print(f"  {col}: {non_null}/{len(unified)} non-null")

    print(f"\nSample (5 rows):\n{unified.iloc[30:35][['gdelt_event_count', 'ts_post_count', 'tw_volume_spike_ratio', 'hormuz_throughput_kbbl', '^VIX_close', 'stream_count']]}")
