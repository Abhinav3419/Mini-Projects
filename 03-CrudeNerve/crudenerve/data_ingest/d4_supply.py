"""
D4: Physical supply indicators.

Four sub-signals:
  1. EIA crude oil inventory (weekly) — surprise vs consensus
  2. US Strategic Petroleum Reserve releases (weekly)
  3. Strait of Hormuz tanker throughput (AIS proxy)
  4. OPEC production quota vs actual gap

Usage:
    from crudenerve.data_ingest.d4_supply import SupplyIngestor
    ingestor = SupplyIngestor()
    df = ingestor.fetch_all(start="2024-01-01", end="2024-06-30")
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None

from crudenerve.config.settings import (
    SupplyConfig,
    RAW_DIR,
    BACKFILL_START,
    BACKFILL_END,
    FRED_API_KEY,
)

logger = logging.getLogger(__name__)


class SupplyIngestor:
    """Fetches and combines physical oil supply indicators."""

    def __init__(self, config: SupplyConfig | None = None):
        self.cfg = config or SupplyConfig()
        self.save_dir = RAW_DIR / "supply"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── EIA crude inventory ──────────────────────────────────────────────

    def fetch_eia_inventory(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
    ) -> pd.DataFrame:
        """
        Fetch US crude oil inventory from FRED (EIA weekly data).
        Series: PET.WCESTUS1.W (thousand barrels)
        """
        if Fred is None or not FRED_API_KEY:
            logger.warning("FRED unavailable — using synthetic EIA data")
            return self._synthetic_eia(start, end)

        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(self.cfg.eia_series_id, start, end)
        df = series.to_frame(name="eia_inventory_kbbl")
        df.index.name = "date"
        df.index = pd.to_datetime(df.index)

        # Week-over-week change (the "build" or "draw")
        df["eia_inventory_change"] = df["eia_inventory_kbbl"].diff()
        # Inventory surprise = change relative to rolling average
        rolling_mean = df["eia_inventory_change"].rolling(8, min_periods=1).mean()
        df["eia_inventory_surprise"] = df["eia_inventory_change"] - rolling_mean

        return df

    # ── SPR releases ─────────────────────────────────────────────────────

    def fetch_spr(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
    ) -> pd.DataFrame:
        """
        Fetch US Strategic Petroleum Reserve levels from FRED.
        Series: PET.WCSSTUS1.W (thousand barrels)
        """
        if Fred is None or not FRED_API_KEY:
            logger.warning("FRED unavailable — using synthetic SPR data")
            return self._synthetic_spr(start, end)

        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(self.cfg.spr_series_id, start, end)
        df = series.to_frame(name="spr_level_kbbl")
        df.index.name = "date"
        df.index = pd.to_datetime(df.index)

        # SPR change (negative = release into market)
        df["spr_change"] = df["spr_level_kbbl"].diff()
        df["spr_is_release"] = (df["spr_change"] < -500).astype(int)

        return df

    # ── Hormuz throughput (AIS proxy) ────────────────────────────────────

    def fetch_hormuz_throughput(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
    ) -> pd.DataFrame:
        """
        Proxy for Strait of Hormuz tanker throughput.

        In production, this would pull from AIS data providers
        (MarineTraffic, Spire, VesselFinder). For now, generates
        synthetic data with realistic patterns.
        """
        return self._synthetic_hormuz(start, end)

    # ── OPEC compliance gap ──────────────────────────────────────────────

    def fetch_opec_compliance(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
    ) -> pd.DataFrame:
        """
        OPEC production quota vs actual production gap.

        Monthly data. Positive gap = producing above quota (bearish).
        Negative gap = producing below quota (bullish).
        In production, scraped from OPEC monthly reports.
        """
        return self._synthetic_opec(start, end)

    # ── combined fetch ───────────────────────────────────────────────────

    def fetch_all(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
        save: bool = True,
    ) -> pd.DataFrame:
        """Fetch all supply indicators and merge on daily index."""
        eia = self.fetch_eia_inventory(start, end)
        spr = self.fetch_spr(start, end)
        hormuz = self.fetch_hormuz_throughput(start, end)
        opec = self.fetch_opec_compliance(start, end)

        # Create daily index and merge everything
        full_index = pd.date_range(start, end, freq="D", name="date")
        merged = pd.DataFrame(index=full_index)

        for df in [eia, spr, hormuz, opec]:
            if not df.empty:
                merged = merged.join(df, how="left")

        # Forward-fill weekly/monthly data to daily
        merged.ffill(inplace=True)
        merged.bfill(limit=7, inplace=True)  # backfill start of series

        if save:
            out_path = self.save_dir / "supply_indicators.parquet"
            merged.to_parquet(out_path)
            logger.info(f"Saved {len(merged)} supply rows → {out_path}")

        return merged

    # ── synthetic data generators ────────────────────────────────────────

    def _synthetic_eia(self, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range(start, end, freq="W-WED", name="date")
        inventory = 430_000 + np.cumsum(rng.normal(0, 2000, len(dates)))
        df = pd.DataFrame({"eia_inventory_kbbl": inventory}, index=dates)
        df["eia_inventory_change"] = df["eia_inventory_kbbl"].diff()
        rolling = df["eia_inventory_change"].rolling(8, min_periods=1).mean()
        df["eia_inventory_surprise"] = df["eia_inventory_change"] - rolling
        return df

    def _synthetic_spr(self, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(43)
        dates = pd.date_range(start, end, freq="W-FRI", name="date")
        # SPR trending down over time (drawdown periods)
        level = 600_000 - np.arange(len(dates)) * 200 + rng.normal(0, 500, len(dates))
        level = np.maximum(level, 350_000)
        df = pd.DataFrame({"spr_level_kbbl": level}, index=dates)
        df["spr_change"] = df["spr_level_kbbl"].diff()
        df["spr_is_release"] = (df["spr_change"] < -500).astype(int)
        return df

    def _synthetic_hormuz(self, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(44)
        dates = pd.date_range(start, end, freq="D", name="date")
        # ~18-20 million bbl/day through Hormuz normally
        base = 19_000 + rng.normal(0, 300, len(dates))
        # Simulate disruption events (~every 60 days, lasting 3-5 days)
        for i in range(0, len(dates), 60):
            if rng.random() > 0.5:
                duration = rng.integers(3, 6)
                drop = rng.integers(2000, 5000)
                end_idx = min(i + duration, len(dates))
                base[i:end_idx] -= drop

        df = pd.DataFrame({"hormuz_throughput_kbbl": base}, index=dates)
        rolling = df["hormuz_throughput_kbbl"].rolling(30, min_periods=1)
        mean = rolling.mean()
        std = rolling.std().fillna(1)
        df["hormuz_zscore"] = (df["hormuz_throughput_kbbl"] - mean) / std
        df["hormuz_is_anomaly"] = (
            df["hormuz_zscore"].abs() > self.cfg.hormuz_anomaly_std_threshold
        ).astype(int)
        return df

    def _synthetic_opec(self, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(45)
        dates = pd.date_range(start, end, freq="MS", name="date")
        # Gap in thousand bbl/day: positive = overproduction
        gap = rng.normal(200, 400, len(dates))
        df = pd.DataFrame({
            "opec_quota_gap_kbbl": gap,
            "opec_compliance_pct": np.clip(100 - gap / 50, 80, 110),
        }, index=dates)
        return df

    def load_cached(self) -> pd.DataFrame | None:
        path = self.save_dir / "supply_indicators.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing D4: Physical supply indicators\n")

    ingestor = SupplyIngestor()
    df = ingestor.fetch_all(start="2024-01-01", end="2024-06-30", save=False)

    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nHormuz anomaly days: {df.get('hormuz_is_anomaly', pd.Series()).sum()}")
    print(f"SPR release days: {df.get('spr_is_release', pd.Series()).sum()}")
    print(f"\nSample:\n{df.tail(10)}")
