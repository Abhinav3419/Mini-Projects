"""
F6: Supply disruption features.

Transforms raw D4 physical supply indicators into model-ready features:
  - Inventory surprise magnitude and direction
  - SPR release momentum (cumulative releases over 30d)
  - Hormuz disruption score (anomaly severity × duration)
  - OPEC discipline index (compliance trend)
  - Supply stress composite score

Usage:
    from crudenerve.features.f6_supply_features import SupplyFeatureBuilder
    builder = SupplyFeatureBuilder()
    df = builder.compute(unified_df)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SupplyFeatureBuilder:
    """
    Converts raw supply indicators into features for the ML pipeline.

    Raw D4 data is in physical units (thousand barrels, z-scores).
    This module normalizes and derives actionable signals.
    """

    def __init__(
        self,
        inventory_surprise_threshold: float = 2.0,
        spr_release_lookback_days: int = 30,
        hormuz_disruption_min_days: int = 2,
    ):
        self.inv_surprise_thresh = inventory_surprise_threshold
        self.spr_lookback = spr_release_lookback_days
        self.hormuz_min_days = hormuz_disruption_min_days

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute supply disruption features.

        Adds columns:
            - f6_inventory_surprise_signal: standardized surprise direction
            - f6_spr_release_momentum:      30d cumulative release pressure
            - f6_hormuz_disruption_score:    severity × duration composite
            - f6_opec_discipline:            compliance trend direction
            - f6_supply_stress:              composite supply stress score
        """
        out = df.copy()

        # ── Inventory surprise signal ────────────────────────────────────
        surprise = out.get(
            "eia_inventory_surprise", pd.Series(0, index=out.index)
        ).fillna(0)
        # Standardize by rolling std
        rolling_std = surprise.rolling(26, min_periods=4).std().replace(0, 1)
        out["f6_inventory_surprise_signal"] = surprise / rolling_std
        # Big surprise flag (>2σ above or below)
        out["f6_inventory_big_surprise"] = (
            out["f6_inventory_surprise_signal"].abs() > self.inv_surprise_thresh
        ).astype(int)

        # ── SPR release momentum ─────────────────────────────────────────
        spr_change = out.get("spr_change", pd.Series(0, index=out.index)).fillna(0)
        # Cumulative release pressure over lookback window
        # Only count releases (negative changes)
        releases = spr_change.clip(upper=0)  # keep only negatives
        out["f6_spr_release_momentum"] = (
            releases.rolling(self.spr_lookback, min_periods=1).sum()
        )
        # Normalize by typical SPR level
        spr_level = out.get("spr_level_kbbl", pd.Series(500_000, index=out.index)).fillna(500_000)
        out["f6_spr_release_pct"] = (
            out["f6_spr_release_momentum"] / spr_level.replace(0, 500_000) * 100
        )

        # ── Hormuz disruption score ──────────────────────────────────────
        hormuz_anomaly = out.get(
            "hormuz_is_anomaly", pd.Series(0, index=out.index)
        ).fillna(0)
        hormuz_zscore = out.get(
            "hormuz_zscore", pd.Series(0, index=out.index)
        ).fillna(0)

        # Duration: rolling count of anomaly days
        anomaly_duration = hormuz_anomaly.rolling(7, min_periods=1).sum()
        # Severity × duration composite
        out["f6_hormuz_disruption_score"] = (
            hormuz_zscore.abs() * anomaly_duration
        ).clip(0, 50)
        # Binary disruption flag (anomaly persisting ≥ min_days)
        out["f6_hormuz_disruption_flag"] = (
            anomaly_duration >= self.hormuz_min_days
        ).astype(int)

        # ── OPEC discipline index ────────────────────────────────────────
        opec_gap = out.get(
            "opec_quota_gap_kbbl", pd.Series(0, index=out.index)
        ).fillna(0)
        # Positive gap = overproduction (bearish for prices)
        # Trend: is OPEC becoming more or less disciplined?
        out["f6_opec_discipline"] = -opec_gap.rolling(30, min_periods=1).mean()
        # Direction of discipline change
        out["f6_opec_discipline_trend"] = out["f6_opec_discipline"].diff(7)

        # ── Composite supply stress score ────────────────────────────────
        # Higher = more supply disruption risk (bullish for prices)
        stress_components = pd.DataFrame(index=out.index)

        # Inventory draw (negative surprise = bullish)
        stress_components["inv"] = (
            -out["f6_inventory_surprise_signal"].clip(-3, 3) / 3.0
        ).clip(0, 1)

        # SPR releases (more releases = perceived shortage)
        stress_components["spr"] = (
            (-out["f6_spr_release_momentum"]).clip(0, 10000) / 10000
        )

        # Hormuz disruption
        stress_components["hormuz"] = (
            out["f6_hormuz_disruption_score"].clip(0, 20) / 20
        )

        # OPEC underproduction (negative gap = bullish)
        stress_components["opec"] = (
            out["f6_opec_discipline"].clip(-1000, 1000) / 1000
        ).clip(0, 1)

        out["f6_supply_stress"] = (
            stress_components["inv"] * 0.25
            + stress_components["spr"] * 0.20
            + stress_components["hormuz"] * 0.35
            + stress_components["opec"] * 0.20
        )

        f6_cols = [c for c in out.columns if c.startswith("f6_")]
        logger.info(f"Supply features computed: {len(f6_cols)} columns")
        return out


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing F6: Supply Disruption Features\n")

    from crudenerve.data_ingest.d4_supply import SupplyIngestor

    ingestor = SupplyIngestor()
    supply = ingestor.fetch_all("2024-01-01", "2024-06-30", save=False)

    builder = SupplyFeatureBuilder()
    result = builder.compute(supply)

    f6_cols = [c for c in result.columns if c.startswith("f6_")]
    print(f"F6 columns ({len(f6_cols)}):")
    for col in f6_cols:
        vals = result[col]
        if vals.dtype in ["float64", "int64"]:
            print(f"  {col:40s} mean={vals.mean():.3f}  max={vals.max():.3f}")

    print(f"\nHormuz disruption days: {result['f6_hormuz_disruption_flag'].sum()}")
    print(f"Big inventory surprise days: {result['f6_inventory_big_surprise'].sum()}")
    print(f"\nSupply stress stats:")
    print(result["f6_supply_stress"].describe())
