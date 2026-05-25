"""
F2: Geopolitical Tension Index.

Builds a continuous tension score from GDELT event data using:
  - Goldstein-scale-weighted event counts (not raw counts)
  - Exponential recency decay (recent events count more)
  - Entity-specific sub-indices (Iran, Israel, Hormuz, etc.)
  - Regime change acceleration detector

The tension index is the primary input to M1 (HMM regime detection).

Usage:
    from crudenerve.features.f2_tension_index import TensionIndexBuilder
    builder = TensionIndexBuilder()
    df = builder.compute(unified_df)
"""

import logging

import numpy as np
import pandas as pd

from crudenerve.config.settings import GDELTConfig

logger = logging.getLogger(__name__)


class TensionIndexBuilder:
    """
    Builds a composite geopolitical tension index from GDELT daily features.

    The index is NOT just event counts. It weights events by:
    1. Tone (more negative tone = higher tension contribution)
    2. Recency (exponential decay — yesterday's event matters more
       than last month's)
    3. Entity salience (Iran/Hormuz events get higher weight than
       generic geopolitical noise)
    """

    def __init__(self, config: GDELTConfig | None = None):
        self.cfg = config or GDELTConfig()

        # Entity importance weights for the composite index.
        # Higher = more impact on oil markets.
        self.entity_weights: dict[str, float] = {
            "iran": 1.5,
            "hormuz": 2.0,           # Hormuz = direct supply chokepoint
            "strait_of_hormuz": 2.0,
            "opec": 1.2,
            "sanctions": 1.4,
            "israel": 1.3,
            "houthis": 1.6,          # Red Sea shipping disruption
            "hezbollah": 1.1,
            "irgc": 1.5,
            "crude_oil": 1.0,
            "petroleum": 1.0,
            "enrichment": 1.4,
        }

    def compute(
        self,
        df: pd.DataFrame,
        window_days: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the geopolitical tension index from unified DataFrame.

        Adds columns:
            - f2_tension_raw:       unsmoothed daily tension score
            - f2_tension_ema:       EMA-smoothed tension (primary index)
            - f2_tension_velocity:  rate of change (for regime acceleration)
            - f2_tension_accel:     second derivative (inflection detector)
            - f2_tension_zscore:    standardized score for cross-period comparison
            - f2_entity_*_tension:  per-entity sub-indices
        """
        window = window_days or self.cfg.rolling_window_days
        out = df.copy()

        # ── Step 1: Entity-weighted tension signal ───────────────────────
        entity_count_cols = [
            c for c in df.columns
            if c.startswith("gdelt_") and c.endswith("_count")
            and c != "gdelt_event_count"
        ]

        if not entity_count_cols:
            logger.warning("No GDELT entity count columns found — "
                          "returning zeros")
            out["f2_tension_raw"] = 0.0
            out["f2_tension_ema"] = 0.0
            out["f2_tension_velocity"] = 0.0
            out["f2_tension_accel"] = 0.0
            out["f2_tension_zscore"] = 0.0
            return out

        # Weighted sum of entity counts
        weighted_signal = pd.Series(0.0, index=df.index)
        for col in entity_count_cols:
            # Extract entity name from column: gdelt_iran_count → iran
            entity = col.replace("gdelt_", "").replace("_count", "")
            weight = self.entity_weights.get(entity, 1.0)
            entity_vals = df[col].fillna(0)
            weighted_signal += entity_vals * weight

            # Per-entity sub-index (EMA smoothed)
            out[f"f2_entity_{entity}_tension"] = (
                entity_vals.ewm(span=window, min_periods=1).mean()
            )

        # ── Step 2: Tone adjustment ─────────────────────────────────────
        # Negative tone amplifies tension, positive tone dampens it.
        # Tone typically ranges -10 to +10 in GDELT.
        if "gdelt_mean_tone" in df.columns:
            tone = df["gdelt_mean_tone"].fillna(0)
            # Convert to a multiplier: tone=-5 → 1.5x, tone=0 → 1.0x, tone=+5 → 0.5x
            tone_multiplier = 1.0 - (tone / 10.0)
            tone_multiplier = tone_multiplier.clip(0.5, 2.0)
            weighted_signal = weighted_signal * tone_multiplier

        # ── Step 3: Tone variance bonus ──────────────────────────────────
        # High tone variance = conflicting signals = uncertainty = tension
        if "gdelt_tone_std" in df.columns:
            tone_std = df["gdelt_tone_std"].fillna(0)
            # Add a bonus for high variance (disagreement/uncertainty)
            uncertainty_bonus = (tone_std / tone_std.rolling(window, min_periods=1).mean().replace(0, 1))
            uncertainty_bonus = uncertainty_bonus.clip(0, 3).fillna(1)
            weighted_signal = weighted_signal * (1 + 0.1 * uncertainty_bonus)

        out["f2_tension_raw"] = weighted_signal

        # ── Step 4: Exponential moving average (primary index) ───────────
        out["f2_tension_ema"] = weighted_signal.ewm(
            span=window, min_periods=1
        ).mean()

        # ── Step 5: Velocity and acceleration ────────────────────────────
        ema = out["f2_tension_ema"]
        out["f2_tension_velocity"] = ema.diff()
        out["f2_tension_accel"] = out["f2_tension_velocity"].diff()

        # ── Step 6: Z-score for cross-period comparison ──────────────────
        rolling_mean = ema.rolling(window * 3, min_periods=window).mean()
        rolling_std = ema.rolling(window * 3, min_periods=window).std().replace(0, 1)
        out["f2_tension_zscore"] = (ema - rolling_mean) / rolling_std

        logger.info(
            f"Tension index computed: "
            f"mean={out['f2_tension_ema'].mean():.2f}, "
            f"max={out['f2_tension_ema'].max():.2f}, "
            f"zscore_range=[{out['f2_tension_zscore'].min():.2f}, "
            f"{out['f2_tension_zscore'].max():.2f}]"
        )

        return out


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing F2: Geopolitical Tension Index\n")

    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt

    # Build synthetic GDELT daily data
    ingestor = GDELTIngestor()
    raw = _generate_synthetic_gdelt(500)
    raw = ingestor._tag_entities(raw)
    daily = ingestor.aggregate_daily(raw)

    # Compute tension index
    builder = TensionIndexBuilder()
    result = builder.compute(daily)

    f2_cols = [c for c in result.columns if c.startswith("f2_")]
    print(f"F2 columns added: {len(f2_cols)}")
    for col in f2_cols:
        if "entity" not in col:
            print(f"  {col}: mean={result[col].mean():.3f}, "
                  f"std={result[col].std():.3f}")

    print(f"\nTop 5 highest tension days:")
    top5 = result.nlargest(5, "f2_tension_ema")
    print(top5[["f2_tension_raw", "f2_tension_ema",
                "f2_tension_velocity", "f2_tension_zscore"]])
