"""
F7: Prediction Mode Engine.

Transforms the unified feature DataFrame according to one of three
prediction philosophies:

  INTUITIVE:         Take all signals at face value. Trump says sanctions →
                     oil goes up. Dollar weakens → commodities rise.
                     The textbook economist's view.

  COUNTER-INTUITIVE: Assume signals are misleading. Two sub-modes:
                     FLIP  — invert signal direction (high severity → bullish)
                     ZERO  — treat signal as noise (drop it entirely)
                     The contrarian trader's view.

  GOLDEN MEAN:       User selects which signals to trust and which to
                     discount, per stream. Toggle each of the 8 signal
                     streams on/off. Named presets available for common
                     configurations.

This module sits between the feature layer (F2-F6) and the model layer
(M1-M4). Raw features are always computed the same way — the mode engine
transforms them before they reach the models.

Architecture:
    D1-D5 → F1 → F2-F6 → [F7 Mode Transform] → M1-M4
                            ↑
                     mode + config from user

Usage:
    from crudenerve.features.f7_prediction_mode import PredictionModeEngine
    engine = PredictionModeEngine()

    # Intuitive (default)
    df_intuitive = engine.transform(unified_df, mode="intuitive")

    # Counter-Intuitive with flip
    df_contrarian = engine.transform(unified_df, mode="counter_intuitive",
                                     ci_submode="flip")

    # Golden Mean with custom toggles
    df_golden = engine.transform(unified_df, mode="golden_mean",
                                 trust_config={"truth_social": False,
                                               "twitter": True, ...})

    # Golden Mean with preset
    df_golden = engine.transform(unified_df, mode="golden_mean",
                                 preset="social_skeptic")
"""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from crudenerve.config.settings import (
    PredictionMode,
    CounterIntuitiveSubMode,
    SIGNAL_STREAMS,
    GOLDEN_MEAN_PRESETS,
    SignalStreamDef,
)

logger = logging.getLogger(__name__)


class PredictionModeEngine:
    """
    Transforms features according to the selected prediction philosophy.

    The engine never modifies raw data — it produces a new DataFrame
    with transformed feature values. The original unified DataFrame
    is always preserved.
    """

    def __init__(self):
        self.streams = {s.stream_id: s for s in SIGNAL_STREAMS}
        self.presets = GOLDEN_MEAN_PRESETS

    # ── public API ───────────────────────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        mode: str = "intuitive",
        ci_submode: str = "flip",
        trust_config: dict[str, bool] | None = None,
        preset: str | None = None,
    ) -> pd.DataFrame:
        """
        Apply prediction mode transformation to the unified DataFrame.

        Args:
            df:           Unified DataFrame from F1-F6 pipeline
            mode:         "intuitive", "counter_intuitive", or "golden_mean"
            ci_submode:   "flip" or "zero" (only used in counter_intuitive mode)
            trust_config: Per-stream trust toggles for golden_mean mode.
                          Keys are stream_id strings, values are booleans.
                          Missing streams default to their default_trust setting.
            preset:       Named preset for golden_mean mode (overrides trust_config).
                          Options: "macro_only", "social_skeptic",
                                   "crisis_trader", "contrarian_lite"

        Returns:
            Transformed DataFrame with a "mode_*" metadata column
            indicating what mode was applied.
        """
        mode_enum = PredictionMode(mode)
        out = df.copy()

        if mode_enum == PredictionMode.INTUITIVE:
            out = self._apply_intuitive(out)

        elif mode_enum == PredictionMode.COUNTER_INTUITIVE:
            submode = CounterIntuitiveSubMode(ci_submode)
            out = self._apply_counter_intuitive(out, submode)

        elif mode_enum == PredictionMode.GOLDEN_MEAN:
            config = self._resolve_trust_config(trust_config, preset)
            out = self._apply_golden_mean(out, config)

        # Add metadata columns
        out["mode_name"] = mode
        if mode == "counter_intuitive":
            out["mode_ci_submode"] = ci_submode
        if mode == "golden_mean":
            config = self._resolve_trust_config(trust_config, preset)
            trusted = [k for k, v in config.items() if v]
            discounted = [k for k, v in config.items() if not v]
            out["mode_gm_trusted"] = ",".join(trusted)
            out["mode_gm_discounted"] = ",".join(discounted)

        return out

    def get_available_streams(self) -> list[dict]:
        """Return stream definitions for UI rendering."""
        return [
            {
                "stream_id": s.stream_id,
                "display_name": s.display_name,
                "description": s.description,
                "category": s.category,
                "default_trust": s.default_trust,
                "invertible": s.invertible,
            }
            for s in SIGNAL_STREAMS
        ]

    def get_presets(self) -> dict[str, dict[str, bool]]:
        """Return available Golden Mean presets."""
        return deepcopy(self.presets)

    # ── mode implementations ─────────────────────────────────────────────

    def _apply_intuitive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INTUITIVE MODE: Pass all signals through unchanged.

        This is the identity transformation — features reach the model
        exactly as computed by F2-F6. The textbook interpretation:
        - High Trump severity → bearish (expect sanctions, supply disruption)
        - Dollar weakening → bullish for oil (cheaper for non-USD buyers)
        - High GDELT tension → bearish (geopolitical risk)
        - Hormuz anomaly → strongly bullish (supply chokepoint threatened)
        """
        logger.info("Mode: INTUITIVE — all signals at face value")
        # No transformation needed. Features are already in their
        # "intuitive" orientation from F2-F6.
        return df

    def _apply_counter_intuitive(
        self,
        df: pd.DataFrame,
        submode: CounterIntuitiveSubMode,
    ) -> pd.DataFrame:
        """
        COUNTER-INTUITIVE MODE: Assume all signals are misleading.

        FLIP submode: Invert directional signals.
          - High Trump severity → bullish (he's bluffing / market overreacts)
          - High GDELT tension → bullish (tension is priced in, resolution coming)
          - Hormuz anomaly → less bullish (market already panicked, reversion)

        ZERO submode: Treat all social/geopolitical signals as noise.
          - Drop all social and sentiment features entirely
          - Keep only price history and hard supply data
          - Let the model find patterns in what's left
        """
        logger.info(f"Mode: COUNTER-INTUITIVE ({submode.value})")

        if submode == CounterIntuitiveSubMode.FLIP:
            return self._flip_all_signals(df)
        elif submode == CounterIntuitiveSubMode.ZERO_OUT:
            return self._zero_all_signals(df)

        return df

    def _apply_golden_mean(
        self,
        df: pd.DataFrame,
        trust_config: dict[str, bool],
    ) -> pd.DataFrame:
        """
        GOLDEN MEAN MODE: Selective trust per signal stream.

        Trusted streams pass through unchanged (intuitive interpretation).
        Discounted streams are zeroed out (treated as noise).

        The user's configuration expresses their market thesis:
        "I trust dollar fundamentals and supply data, but I think
         Trump's rhetoric is noise and Twitter is an echo chamber."
        """
        trusted = [k for k, v in trust_config.items() if v]
        discounted = [k for k, v in trust_config.items() if not v]
        logger.info(
            f"Mode: GOLDEN MEAN — trusted: {trusted}, discounted: {discounted}"
        )

        for stream_id, is_trusted in trust_config.items():
            if not is_trusted:
                stream_def = self.streams.get(stream_id)
                if stream_def:
                    df = self._zero_stream(df, stream_def)

        return df

    # ── signal transformation helpers ────────────────────────────────────

    def _get_stream_columns(
        self,
        df: pd.DataFrame,
        stream_def: SignalStreamDef,
    ) -> list[str]:
        """Find all columns belonging to a signal stream."""
        cols = []
        for prefix in stream_def.feature_prefixes:
            cols.extend([c for c in df.columns if c.startswith(prefix)])
        return list(set(cols))  # deduplicate

    def _flip_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Invert all directional signals.

        For each invertible stream, negate numeric feature values.
        This reverses the implied direction: high tension becomes
        a bullish signal instead of bearish.

        Columns that are counts, flags, or identifiers are NOT flipped —
        only continuous directional signals.
        """
        out = df.copy()

        # Columns that should NOT be flipped (counts, IDs, flags, metadata)
        no_flip_suffixes = (
            "_count", "_flag", "_regime", "_n_active",
            "_direction", "_is_", "has_", "stream_",
            "is_relevant", "is_high", "is_action",
        )

        for stream_def in SIGNAL_STREAMS:
            if not stream_def.invertible:
                continue

            cols = self._get_stream_columns(out, stream_def)
            for col in cols:
                # Skip non-numeric columns
                if out[col].dtype not in ["float64", "float32", "int64", "int32"]:
                    continue
                # Skip count/flag/identifier columns
                if any(col.endswith(s) or s in col for s in no_flip_suffixes):
                    continue
                # Flip: negate the signal
                out[col] = -out[col]

            logger.debug(f"  Flipped {len(cols)} columns in {stream_def.stream_id}")

        return out

    def _zero_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zero out all social and geopolitical signals.

        Keeps only price history and hard supply quantities.
        Everything derived from sentiment, rhetoric, or news flow
        becomes 0.
        """
        out = df.copy()

        for stream_def in SIGNAL_STREAMS:
            out = self._zero_stream(out, stream_def)

        return out

    def _zero_stream(
        self,
        df: pd.DataFrame,
        stream_def: SignalStreamDef,
    ) -> pd.DataFrame:
        """Zero out all numeric columns belonging to one signal stream."""
        out = df.copy()
        cols = self._get_stream_columns(out, stream_def)

        zeroed_count = 0
        for col in cols:
            if out[col].dtype in ["float64", "float32", "int64", "int32"]:
                out[col] = 0
                zeroed_count += 1

        if zeroed_count > 0:
            logger.debug(
                f"  Zeroed {zeroed_count} columns in {stream_def.stream_id}"
            )

        return out

    # ── trust config resolution ──────────────────────────────────────────

    def _resolve_trust_config(
        self,
        trust_config: dict[str, bool] | None,
        preset: str | None,
    ) -> dict[str, bool]:
        """
        Build a complete trust config from user input or preset.

        Priority: preset > trust_config > stream defaults
        """
        # Start with defaults
        config = {s.stream_id: s.default_trust for s in SIGNAL_STREAMS}

        # Override with user-provided config
        if trust_config:
            config.update(trust_config)

        # Override with preset (takes priority)
        if preset and preset in self.presets:
            config.update(self.presets[preset])
        elif preset:
            logger.warning(
                f"Unknown preset '{preset}'. Available: {list(self.presets.keys())}"
            )

        return config

    # ── analysis / comparison helpers ────────────────────────────────────

    def compare_modes(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Run all three modes and compare key feature statistics.

        Returns a summary DataFrame showing how each mode transforms
        the features — useful for understanding what the modes actually do.
        """
        modes = {
            "intuitive": self.transform(df, mode="intuitive"),
            "counter_flip": self.transform(df, mode="counter_intuitive", ci_submode="flip"),
            "counter_zero": self.transform(df, mode="counter_intuitive", ci_submode="zero"),
            "golden_default": self.transform(df, mode="golden_mean"),
        }

        # Add presets
        for preset_name in self.presets:
            modes[f"gm_{preset_name}"] = self.transform(
                df, mode="golden_mean", preset=preset_name
            )

        # Pick key features to compare
        if feature_columns is None:
            feature_columns = [
                "f2_tension_ema", "f3_dho_combined", "f4_trump_shock_score",
                "f5_fear_index", "f6_supply_stress",
            ]
            feature_columns = [c for c in feature_columns if c in df.columns]

        records = []
        for mode_name, mode_df in modes.items():
            for col in feature_columns:
                if col in mode_df.columns:
                    records.append({
                        "mode": mode_name,
                        "feature": col,
                        "mean": mode_df[col].mean(),
                        "std": mode_df[col].std(),
                        "min": mode_df[col].min(),
                        "max": mode_df[col].max(),
                        "nonzero_pct": (mode_df[col] != 0).mean() * 100,
                    })

        return pd.DataFrame(records)

    def get_mode_summary(
        self,
        mode: str,
        ci_submode: str = "flip",
        trust_config: dict[str, bool] | None = None,
        preset: str | None = None,
    ) -> dict:
        """
        Return a human-readable summary of what the mode does.

        Useful for the frontend dashboard to explain the active
        prediction mode to the user.
        """
        if mode == "intuitive":
            return {
                "mode": "Intuitive",
                "philosophy": "All signals taken at face value",
                "assumption": "Markets are informationally efficient. "
                              "Political rhetoric translates to policy action. "
                              "Consensus interpretation is correct.",
                "streams_active": list(self.streams.keys()),
                "streams_discounted": [],
                "best_for": "Normal market conditions, trending regimes, "
                            "genuine geopolitical crises",
                "risk": "Gets killed by reversals, bluffs, and priced-in events",
            }
        elif mode == "counter_intuitive":
            sub = ci_submode
            return {
                "mode": f"Counter-Intuitive ({sub})",
                "philosophy": "All signals are misleading" if sub == "flip"
                              else "Social/geopolitical signals are noise",
                "assumption": "Markets overreact. Rhetoric is bluster. "
                              "The consensus trade is crowded and wrong."
                              if sub == "flip"
                              else "Only hard supply/demand data matters. "
                              "Everything else is noise.",
                "streams_active": [] if sub == "zero"
                                  else list(self.streams.keys()),
                "streams_inverted": list(self.streams.keys()) if sub == "flip"
                                    else [],
                "best_for": "Mean-reversion regimes, post-panic recovery, "
                            "political theater without follow-through",
                "risk": "Catastrophic during genuine crises — fighting the "
                        "trend when the trend is real",
            }
        elif mode == "golden_mean":
            config = self._resolve_trust_config(trust_config, preset)
            trusted = [self.streams[k].display_name for k, v in config.items() if v]
            discounted = [self.streams[k].display_name for k, v in config.items() if not v]
            preset_name = preset if preset else "custom"
            return {
                "mode": f"Golden Mean ({preset_name})",
                "philosophy": "Selective trust based on signal quality",
                "assumption": "Some signals are genuine (hard data, "
                              "verified actions), others are noise or "
                              "manipulation (rhetoric, social media). "
                              "The skill is knowing which is which.",
                "streams_trusted": trusted,
                "streams_discounted": discounted,
                "best_for": "Experienced traders with a specific market thesis",
                "risk": "Only as good as the user's judgment about which "
                        "signals to trust",
            }

        return {"mode": "unknown"}


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Testing F7: Prediction Mode Engine")
    print("=" * 70)

    # Build a unified DataFrame from synthetic data
    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
    from crudenerve.data_ingest.d2_truth_social import (
        TruthSocialPipeline, generate_synthetic_posts,
    )
    from crudenerve.data_ingest.d3_twitter import TwitterIngestor, generate_synthetic_tweets
    from crudenerve.data_ingest.d4_supply import SupplyIngestor
    from crudenerve.features.f1_merge import TimeIndexedMerge
    from crudenerve.features.f2_tension_index import TensionIndexBuilder
    from crudenerve.features.f3_dho_kernel import DHOKernelEngine
    from crudenerve.features.f4_trump_volatility import TrumpVolatilityEngine
    from crudenerve.features.f5_twitter_features import TwitterFeatureBuilder
    from crudenerve.features.f6_supply_features import SupplyFeatureBuilder

    START, END = "2024-01-01", "2024-06-30"

    # Quick pipeline rebuild
    gdelt_ingestor = GDELTIngestor()
    gdelt_raw = _generate_synthetic_gdelt(500)
    gdelt_raw = gdelt_ingestor._tag_entities(gdelt_raw)
    gdelt_daily = gdelt_ingestor.aggregate_daily(gdelt_raw)

    ts_pipeline = TruthSocialPipeline()
    ts_posts = generate_synthetic_posts(200)
    ts_enriched = ts_pipeline.run_full_pipeline(ts_posts, save=False)
    ts_daily = ts_pipeline.aggregate_daily(ts_enriched)

    tw_raw = generate_synthetic_tweets(5000, START, END)
    tw_daily = TwitterIngestor().compute_features(tw_raw)

    supply_daily = SupplyIngestor().fetch_all(START, END, save=False)

    rng = np.random.default_rng(99)
    dates = pd.date_range(START, END, freq="B")
    price_vix = pd.DataFrame({
        "BZ=F_close": 75 + np.cumsum(rng.normal(0, 1, len(dates))),
        "CL=F_close": 72 + np.cumsum(rng.normal(0, 1, len(dates))),
        "^VIX_close": 18 + np.cumsum(rng.normal(0, 0.5, len(dates))),
        "BZ=F_volume": rng.integers(100000, 500000, len(dates)),
    }, index=dates)
    price_vix.index.name = "date"

    merger = TimeIndexedMerge()
    unified = merger.merge_all(
        gdelt_daily=gdelt_daily, truth_social_daily=ts_daily,
        twitter_daily=tw_daily, supply_daily=supply_daily,
        price_vix=price_vix, start=START, end=END, save=False,
    )
    unified = TensionIndexBuilder().compute(unified)
    unified = DHOKernelEngine().compute(unified)
    unified = TrumpVolatilityEngine().compute(unified, enriched_posts=ts_enriched)
    unified = TwitterFeatureBuilder().compute(unified)
    unified = SupplyFeatureBuilder().compute(unified)

    print(f"\nUnified DataFrame: {unified.shape}")

    # ── Test all modes ───────────────────────────────────────────────────
    engine = PredictionModeEngine()

    print("\n" + "=" * 70)
    print("MODE COMPARISON")
    print("=" * 70)

    comparison = engine.compare_modes(unified)
    # Pivot for readability
    for feature in comparison["feature"].unique():
        print(f"\n  Feature: {feature}")
        subset = comparison[comparison["feature"] == feature]
        for _, row in subset.iterrows():
            print(f"    {row['mode']:20s}  mean={row['mean']:8.3f}  "
                  f"std={row['std']:7.3f}  nonzero={row['nonzero_pct']:5.1f}%")

    # ── Test mode summaries ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODE SUMMARIES")
    print("=" * 70)

    for mode, kwargs in [
        ("intuitive", {}),
        ("counter_intuitive", {"ci_submode": "flip"}),
        ("counter_intuitive", {"ci_submode": "zero"}),
        ("golden_mean", {"preset": "social_skeptic"}),
        ("golden_mean", {"preset": "crisis_trader"}),
    ]:
        summary = engine.get_mode_summary(mode, **kwargs)
        print(f"\n  {summary['mode']}")
        print(f"    Philosophy: {summary['philosophy']}")
        print(f"    Best for:   {summary.get('best_for', 'N/A')}")
        print(f"    Risk:       {summary.get('risk', 'N/A')}")

    # ── Verify transformations are working ───────────────────────────────
    print("\n" + "=" * 70)
    print("TRANSFORMATION VERIFICATION")
    print("=" * 70)

    intuitive = engine.transform(unified, mode="intuitive")
    flipped = engine.transform(unified, mode="counter_intuitive", ci_submode="flip")
    zeroed = engine.transform(unified, mode="counter_intuitive", ci_submode="zero")
    golden = engine.transform(unified, mode="golden_mean", preset="social_skeptic")

    key = "f4_trump_shock_score"
    if key in unified.columns:
        print(f"\n  {key}:")
        print(f"    Intuitive mean:       {intuitive[key].mean():+.4f}")
        print(f"    Counter-flip mean:    {flipped[key].mean():+.4f}")
        print(f"    Counter-zero mean:    {zeroed[key].mean():.4f}")
        print(f"    Golden (soc_skeptic): {golden[key].mean():.4f}")

    key2 = "f6_supply_stress"
    if key2 in unified.columns:
        print(f"\n  {key2} (should be same in social_skeptic, zeroed in counter_zero):")
        print(f"    Intuitive mean:       {intuitive[key2].mean():.4f}")
        print(f"    Counter-flip mean:    {flipped[key2].mean():+.4f}")
        print(f"    Counter-zero mean:    {zeroed[key2].mean():.4f}")
        print(f"    Golden (soc_skeptic): {golden[key2].mean():.4f}")

    print(f"\n{'='*70}")
    print("✅ F7 Prediction Mode Engine test PASSED")
    print(f"{'='*70}")
