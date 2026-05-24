"""
D1: GDELT event ingestion pipeline.

Queries the GDELT 2.0 API for events involving oil-relevant geopolitical actors.
Filters by entity names and CAMEO event codes.
Returns daily event records with Goldstein scale, tone, and actor metadata.

Usage:
    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor
    ingestor = GDELTIngestor()
    df = ingestor.fetch_range("2024-01-01", "2024-06-30")
"""

import json
import logging
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from crudenerve.config.settings import (
    GDELTConfig,
    RAW_DIR,
    BACKFILL_START,
    BACKFILL_END,
)

logger = logging.getLogger(__name__)

# GDELT 2.0 DOC API endpoint
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
# GDELT Events export (for CAMEO-coded structured events)
GDELT_EVENTS_URL = "http://data.gdeltproject.org/events"


class GDELTIngestor:
    """Fetches geopolitical events from GDELT filtered for CrudeNerve entities."""

    def __init__(self, config: GDELTConfig | None = None):
        self.cfg = config or GDELTConfig()
        self.save_dir = RAW_DIR / "gdelt"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    # ── GDELT DOC API query builder ──────────────────────────────────────

    def _build_query(self, entity_group: list[str]) -> str:
        """Build GDELT query string from entity list."""
        # OR together all entities — GDELT uses space-separated OR
        terms = " OR ".join(f'"{e}"' for e in entity_group)
        return terms

    def _fetch_doc_api(
        self,
        query: str,
        start_date: str,
        end_date: str,
        max_records: int = 250,
    ) -> pd.DataFrame:
        """
        Query GDELT DOC 2.0 API.

        Returns articles/events mentioning the query terms within the date range.
        The API returns JSON with tone, themes, locations, and source metadata.
        """
        params = {
            "query": query,
            "mode": self.cfg.query_mode,
            "maxrecords": str(max_records),
            "format": "json",
            "startdatetime": start_date.replace("-", "") + "000000",
            "enddatetime": end_date.replace("-", "") + "235959",
            "sort": "DateDesc",
        }

        try:
            resp = self.session.get(GDELT_DOC_API, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"GDELT API request failed: {e}")
            return pd.DataFrame()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.error("GDELT returned non-JSON response")
            return pd.DataFrame()

        articles = data.get("articles", [])
        if not articles:
            logger.info(f"No articles found for query window {start_date}-{end_date}")
            return pd.DataFrame()

        records = []
        for art in articles:
            records.append({
                "datetime": art.get("seendate", ""),
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("domain", ""),
                "language": art.get("language", ""),
                "tone": art.get("tone", 0.0),
                "socialimage": art.get("socialimage", ""),
            })

        df = pd.DataFrame(records)
        if not df.empty and "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["date"] = df["datetime"].dt.date
        return df

    # ── batch fetch over date range ──────────────────────────────────────

    def fetch_range(
        self,
        start: str = BACKFILL_START,
        end: str = BACKFILL_END,
        chunk_days: int = 7,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch GDELT events in weekly chunks to avoid API limits.

        Queries each entity group separately and deduplicates.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        query = self._build_query(self.cfg.filter_entities)
        all_frames = []
        current = start_dt

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            s = current.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"Fetching GDELT chunk: {s} → {e}")
            chunk_df = self._fetch_doc_api(
                query, s, e,
                max_records=self.cfg.max_records_per_query,
            )

            if not chunk_df.empty:
                all_frames.append(chunk_df)

            current = chunk_end + timedelta(days=1)
            time.sleep(1)  # rate limit courtesy

        if not all_frames:
            logger.warning("No GDELT data fetched across entire range")
            return pd.DataFrame()

        merged = pd.concat(all_frames, ignore_index=True)
        merged.drop_duplicates(subset=["url"], inplace=True)
        merged.sort_values("datetime", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        # Tag which entities each article mentions
        merged = self._tag_entities(merged)

        if save:
            out_path = self.save_dir / "gdelt_events.parquet"
            merged.to_parquet(out_path)
            logger.info(f"Saved {len(merged)} GDELT events → {out_path}")

        return merged

    def _tag_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add boolean columns for each filter entity found in the title."""
        title_lower = df["title"].str.lower().fillna("")
        for entity in self.cfg.filter_entities:
            col_name = f"entity_{entity.lower().replace(' ', '_')}"
            df[col_name] = title_lower.str.contains(
                entity.lower(), regex=False
            )
        return df

    # ── daily aggregation (for F2 tension index) ─────────────────────────

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw GDELT events to daily summary.

        Output columns:
            - gdelt_event_count: number of relevant articles that day
            - gdelt_mean_tone: average tone (negative = hostile)
            - gdelt_tone_std: tone variance (high = conflicting signals)
            - gdelt_entity_*_count: per-entity mention counts
        """
        if df.empty or "date" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date").agg(
            gdelt_event_count=("title", "count"),
            gdelt_mean_tone=("tone", "mean"),
            gdelt_tone_std=("tone", "std"),
        )

        # Per-entity daily counts
        entity_cols = [c for c in df.columns if c.startswith("entity_")]
        for col in entity_cols:
            count_name = col.replace("entity_", "gdelt_") + "_count"
            daily[count_name] = df.groupby("date")[col].sum()

        daily["gdelt_tone_std"] = daily["gdelt_tone_std"].fillna(0)
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"

        return daily

    # ── convenience ──────────────────────────────────────────────────────

    def load_cached(self) -> pd.DataFrame | None:
        path = self.save_dir / "gdelt_events.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


# ─── Quick test with synthetic data ──────────────────────────────────────────

def _generate_synthetic_gdelt(n: int = 500) -> pd.DataFrame:
    """Generate synthetic GDELT-like data for testing without API access."""
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")

    entities = [
        "Iran", "Israel", "OPEC", "Hormuz", "sanctions",
        "Houthis", "crude oil",
    ]
    templates = [
        "{} announces new sanctions against Iran nuclear program",
        "Tensions rise in Strait of Hormuz after {} incident",
        "OPEC considers production cut amid {} pressure",
        "Israel responds to {} with diplomatic measures",
        "{} crude oil prices surge on supply concerns",
        "Houthis target {} shipping in Red Sea corridor",
        "Iran enrichment levels reach {} percent threshold",
    ]

    records = []
    for _ in range(n):
        day = rng.choice(dates)
        entity = rng.choice(entities)
        template = rng.choice(templates)
        # Tone: negative = hostile, range roughly -10 to +10
        tone = rng.normal(-2, 4)
        records.append({
            "datetime": day + pd.Timedelta(hours=rng.integers(0, 24)),
            "title": template.format(entity),
            "url": f"https://example.com/{rng.integers(100000)}",
            "source": rng.choice(["reuters.com", "bbc.co.uk", "aljazeera.com"]),
            "language": "English",
            "tone": round(tone, 2),
            "date": pd.Timestamp(day).date(),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing D1 with synthetic GDELT data...\n")

    ingestor = GDELTIngestor()
    synthetic = _generate_synthetic_gdelt(500)
    synthetic = ingestor._tag_entities(synthetic)
    daily = ingestor.aggregate_daily(synthetic)

    print(f"Raw events: {len(synthetic)}")
    print(f"Daily rows: {len(daily)}")
    print(f"Date range: {daily.index.min()} → {daily.index.max()}")
    print(f"\nColumns: {list(daily.columns)}")
    print(f"\nSample:\n{daily.head(10)}")
    print(f"\nDaily event count stats:\n{daily['gdelt_event_count'].describe()}")
