"""
D5: Price + VIX history ingestion.

Pulls daily OHLCV for Brent crude, WTI crude, and VIX from Yahoo Finance.
Falls back to FRED for VIX if Yahoo fails.
Computes forward returns at 24h/48h/72h horizons for label generation.

Usage:
    from crudenerve.data_ingest.d5_price_vix import PriceVIXIngestor
    ingestor = PriceVIXIngestor()
    df = ingestor.fetch_all()
    df = ingestor.add_forward_returns(df)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from fredapi import Fred
except ImportError:
    Fred = None

from crudenerve.config.settings import (
    PriceConfig,
    RAW_DIR,
    BACKFILL_START,
    BACKFILL_END,
    PREDICTION_HORIZONS,
    VIX_DIRECTION_THRESHOLD,
)

logger = logging.getLogger(__name__)


class PriceVIXIngestor:
    """Fetches and stores Brent crude, WTI, and VIX daily price data."""

    def __init__(self, config: PriceConfig | None = None):
        self.cfg = config or PriceConfig()
        self.save_dir = RAW_DIR / "price_vix"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── core fetch ───────────────────────────────────────────────────────

    def fetch_yahoo(
        self,
        ticker: str,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
    ) -> pd.DataFrame:
        """Pull daily OHLCV from Yahoo Finance for one ticker."""
        if yf is None:
            raise ImportError("yfinance not installed: pip install yfinance")

        logger.info(f"Fetching {ticker} from Yahoo Finance [{start} → {end}]")
        raw = yf.download(
            ticker, start=start, end=end,
            interval=self.cfg.granularity, progress=False,
        )

        if raw.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # yfinance sometimes returns MultiIndex columns — flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = [f"{ticker}_{c.lower()}" for c in df.columns]
        df.index.name = "date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    def fetch_vix_fred(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
    ) -> pd.DataFrame:
        """Fallback: pull VIX close from FRED (series VIXCLS)."""
        if Fred is None:
            raise ImportError("fredapi not installed: pip install fredapi")

        from crudenerve.config.settings import FRED_API_KEY
        if not FRED_API_KEY:
            raise ValueError("FRED_API_KEY env var not set")

        logger.info("Fetching VIX from FRED (VIXCLS)")
        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series("VIXCLS", start, end)
        df = series.to_frame(name="^VIX_close")
        df.index.name = "date"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # ── combined fetch ───────────────────────────────────────────────────

    def fetch_all(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all tickers and merge into a single DataFrame on date index.

        Returns a DataFrame with columns like:
            BZ=F_open, BZ=F_high, ..., CL=F_close, ^VIX_close, etc.
        """
        frames = []
        for ticker in self.cfg.tickers:
            try:
                df = self.fetch_yahoo(ticker, start, end)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.error(f"Yahoo fetch failed for {ticker}: {e}")
                # VIX fallback
                if ticker == "^VIX":
                    try:
                        frames.append(self.fetch_vix_fred(start, end))
                    except Exception as e2:
                        logger.error(f"FRED fallback also failed: {e2}")

        if not frames:
            logger.error("No price data fetched at all")
            return pd.DataFrame()

        merged = pd.concat(frames, axis=1, join="outer")
        merged.sort_index(inplace=True)

        # Forward-fill weekends/holidays (max 5 days to avoid filling
        # through actual data gaps)
        merged.ffill(limit=5, inplace=True)

        if save:
            out_path = self.save_dir / "price_vix_daily.parquet"
            merged.to_parquet(out_path)
            logger.info(f"Saved {len(merged)} rows → {out_path}")

        return merged

    # ── label generation ─────────────────────────────────────────────────

    def add_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward returns and binary direction labels for VIX.

        For each horizon in PREDICTION_HORIZONS (24h, 48h, 72h on daily data
        that means 1, 2, 3 trading days):
            - vix_return_{h}h:    pct change in VIX close
            - vix_direction_{h}h: 1 if return > threshold, 0 otherwise
            - brent_return_{h}h:  pct change in Brent close
        """
        vix_col = "^VIX_close"
        brent_col = "BZ=F_close"

        if vix_col not in df.columns:
            logger.warning("VIX close column not found — skipping labels")
            return df

        out = df.copy()
        for hours in PREDICTION_HORIZONS:
            days = max(1, hours // 24)  # 24h=1d, 48h=2d, 72h=3d

            # VIX forward return
            fwd = out[vix_col].shift(-days)
            ret = (fwd - out[vix_col]) / out[vix_col]
            out[f"vix_return_{hours}h"] = ret

            # Binary direction: 1 = up by more than threshold
            out[f"vix_direction_{hours}h"] = (
                ret.abs() > VIX_DIRECTION_THRESHOLD
            ).astype(int)

            # sign-aware: +1 = up beyond threshold, -1 = down, 0 = flat
            out[f"vix_signed_{hours}h"] = np.where(
                ret > VIX_DIRECTION_THRESHOLD, 1,
                np.where(ret < -VIX_DIRECTION_THRESHOLD, -1, 0)
            )

            # Brent forward return
            if brent_col in df.columns:
                brent_fwd = out[brent_col].shift(-days)
                out[f"brent_return_{hours}h"] = (
                    (brent_fwd - out[brent_col]) / out[brent_col]
                )

        return out

    # ── convenience ──────────────────────────────────────────────────────

    def load_cached(self) -> pd.DataFrame | None:
        """Load from parquet if available."""
        path = self.save_dir / "price_vix_daily.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = PriceVIXIngestor()
    df = ingestor.fetch_all(start="2024-01-01", end="2024-06-30")
    if not df.empty:
        df = ingestor.add_forward_returns(df)
        print(f"\nShape: {df.shape}")
        print(f"Date range: {df.index.min()} → {df.index.max()}")
        print(f"\nColumns:\n{list(df.columns)}")
        print(f"\nSample (last 5 rows):\n{df.tail()}")
        print(f"\nVIX direction 24h distribution:")
        if "vix_direction_24h" in df.columns:
            print(df["vix_direction_24h"].value_counts())
