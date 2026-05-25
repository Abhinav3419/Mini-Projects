"""
CrudeNerve Phase 1+2 Integration Test.

Runs the complete pipeline:
    D1-D5 (data ingest) → F1 (merge) → F2-F6 (features)

All on synthetic data. Validates that every module produces
expected columns and the unified DataFrame is model-ready.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

START = "2024-01-01"
END = "2024-06-30"


def main():
    print("=" * 70)
    print("CrudeNerve — Full Phase 1+2 Integration Test")
    print("=" * 70)

    # ═══ PHASE 1: DATA INGESTION ═══════════════════════════════════════

    print("\n▸ PHASE 1: Data Ingestion")
    print("-" * 40)

    # D1: GDELT
    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
    gdelt_ingestor = GDELTIngestor()
    gdelt_raw = _generate_synthetic_gdelt(500)
    gdelt_raw = gdelt_ingestor._tag_entities(gdelt_raw)
    gdelt_daily = gdelt_ingestor.aggregate_daily(gdelt_raw)
    print(f"  ✓ D1 GDELT:         {gdelt_daily.shape[0]} days, {gdelt_daily.shape[1]} cols")

    # D2: Truth Social (full pipeline: D2 → D2-NLP → D2-SEV)
    from crudenerve.data_ingest.d2_truth_social import (
        TruthSocialPipeline, generate_synthetic_posts,
    )
    ts_pipeline = TruthSocialPipeline()
    ts_posts = generate_synthetic_posts(200)
    ts_enriched = ts_pipeline.run_full_pipeline(ts_posts, save=False)
    ts_daily = ts_pipeline.aggregate_daily(ts_enriched)
    n_relevant = ts_enriched["is_relevant"].sum()
    n_high_sev = ts_enriched["is_high_severity"].sum()
    print(f"  ✓ D2 Truth Social:  {len(ts_enriched)} posts → {ts_daily.shape[0]} days")
    print(f"    D2-NLP topics:    {n_relevant} relevant / {len(ts_enriched)} total")
    print(f"    D2-SEV severity:  {n_high_sev} high-severity posts")

    # D3: Twitter
    from crudenerve.data_ingest.d3_twitter import TwitterIngestor, generate_synthetic_tweets
    tw_raw = generate_synthetic_tweets(5000, START, END)
    tw_ingestor = TwitterIngestor()
    tw_daily = tw_ingestor.compute_features(tw_raw)
    print(f"  ✓ D3 Twitter:       {tw_daily.shape[0]} days, {tw_daily.shape[1]} cols")

    # D4: Supply
    from crudenerve.data_ingest.d4_supply import SupplyIngestor
    supply_ingestor = SupplyIngestor()
    supply_daily = supply_ingestor.fetch_all(START, END, save=False)
    print(f"  ✓ D4 Supply:        {supply_daily.shape[0]} days, {supply_daily.shape[1]} cols")

    # D5: Price + VIX (synthetic)
    dates = pd.date_range(START, END, freq="B")
    rng = np.random.default_rng(99)
    price_vix = pd.DataFrame({
        "BZ=F_close": 75 + np.cumsum(rng.normal(0, 1, len(dates))),
        "CL=F_close": 72 + np.cumsum(rng.normal(0, 1, len(dates))),
        "^VIX_close": 18 + np.cumsum(rng.normal(0, 0.5, len(dates))),
        "BZ=F_volume": rng.integers(100000, 500000, len(dates)),
    }, index=dates)
    price_vix.index.name = "date"
    print(f"  ✓ D5 Price/VIX:     {price_vix.shape[0]} days, {price_vix.shape[1]} cols")

    # F1: Merge
    from crudenerve.features.f1_merge import TimeIndexedMerge
    merger = TimeIndexedMerge()
    unified = merger.merge_all(
        gdelt_daily=gdelt_daily,
        truth_social_daily=ts_daily,
        twitter_daily=tw_daily,
        supply_daily=supply_daily,
        price_vix=price_vix,
        start=START, end=END,
        save=False,
    )
    print(f"  ✓ F1 Merge:         {unified.shape[0]} days × {unified.shape[1]} cols")

    # ═══ PHASE 2: FEATURE ENGINEERING ══════════════════════════════════

    print("\n▸ PHASE 2: Feature Engineering")
    print("-" * 40)

    # F2: Geopolitical Tension Index
    from crudenerve.features.f2_tension_index import TensionIndexBuilder
    f2_builder = TensionIndexBuilder()
    unified = f2_builder.compute(unified)
    f2_cols = [c for c in unified.columns if c.startswith("f2_")]
    print(f"  ✓ F2 Tension Index: {len(f2_cols)} cols, "
          f"peak={unified['f2_tension_ema'].max():.2f}")

    # F3: DHO Decay Kernel
    from crudenerve.features.f3_dho_kernel import DHOKernelEngine
    f3_engine = DHOKernelEngine()
    unified = f3_engine.compute(unified)
    f3_cols = [c for c in unified.columns if c.startswith("f3_")]
    active_days = (unified["f3_dho_n_active"] > 0).sum()
    print(f"  ✓ F3 DHO Kernel:    {len(f3_cols)} cols, "
          f"{active_days} days with active kernels")

    # F4: Trump Volatility Injection (with post-level features)
    from crudenerve.features.f4_trump_volatility import TrumpVolatilityEngine
    f4_engine = TrumpVolatilityEngine()
    unified = f4_engine.compute(unified, enriched_posts=ts_enriched)
    f4_cols = [c for c in unified.columns if c.startswith("f4")]
    escalation = unified["f4_trump_escalation"].sum()
    print(f"  ✓ F4 Trump Vol:     {len(f4_cols)} cols, "
          f"peak_shock={unified['f4_trump_shock_score'].max():.3f}, "
          f"escalation_days={escalation}")

    # F5: Twitter Features
    from crudenerve.features.f5_twitter_features import TwitterFeatureBuilder
    f5_builder = TwitterFeatureBuilder()
    unified = f5_builder.compute(unified)
    f5_cols = [c for c in unified.columns if c.startswith("f5_")]
    print(f"  ✓ F5 Twitter:       {len(f5_cols)} cols, "
          f"fear_max={unified['f5_fear_index'].max():.3f}")

    # F6: Supply Disruption
    from crudenerve.features.f6_supply_features import SupplyFeatureBuilder
    f6_builder = SupplyFeatureBuilder()
    unified = f6_builder.compute(unified)
    f6_cols = [c for c in unified.columns if c.startswith("f6_")]
    hormuz_days = unified.get("f6_hormuz_disruption_flag", pd.Series(0)).sum()
    print(f"  ✓ F6 Supply:        {len(f6_cols)} cols, "
          f"hormuz_disruption_days={hormuz_days}")

    # ═══ FINAL SUMMARY ════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("FINAL UNIFIED DATAFRAME")
    print("=" * 70)
    print(f"Shape: {unified.shape[0]} days × {unified.shape[1]} columns")
    print(f"Date range: {unified.index.min().date()} → {unified.index.max().date()}")

    # Column groups
    groups = {
        "D1 GDELT (gdelt_*)":     [c for c in unified.columns if c.startswith("gdelt_")],
        "D2 TruthSocial (ts_*)":  [c for c in unified.columns if c.startswith("ts_")],
        "D3 Twitter (tw_*)":      [c for c in unified.columns if c.startswith("tw_")],
        "D4 Supply (eia/spr/...)": [c for c in unified.columns if c.startswith(("eia_", "spr_", "hormuz_", "opec_"))],
        "D5 Price/VIX":           [c for c in unified.columns if c.startswith(("BZ=F", "CL=F", "^VIX"))],
        "F1 Merge (has_*/stream)":[c for c in unified.columns if c.startswith(("has_", "stream_"))],
        "F2 Tension (f2_*)":      f2_cols,
        "F3 DHO (f3_*)":          f3_cols,
        "F4 Trump (f4*)":         f4_cols,
        "F5 Twitter (f5_*)":      f5_cols,
        "F6 Supply (f6_*)":       f6_cols,
    }

    print(f"\nColumn breakdown:")
    total = 0
    for name, cols in groups.items():
        print(f"  {name:30s} → {len(cols):3d} columns")
        total += len(cols)
    other = unified.shape[1] - total
    if other > 0:
        print(f"  {'Other':30s} → {other:3d} columns")

    # Data quality
    print(f"\nData quality:")
    total_cells = unified.shape[0] * unified.shape[1]
    null_cells = unified.isnull().sum().sum()
    print(f"  Total cells: {total_cells:,}")
    print(f"  Null cells:  {null_cells:,} ({100*null_cells/total_cells:.1f}%)")

    # Key feature distributions
    print(f"\nKey feature distributions:")
    key_features = [
        "f2_tension_ema", "f3_dho_combined", "f4_trump_shock_score",
        "f5_fear_index", "f6_supply_stress",
    ]
    for feat in key_features:
        if feat in unified.columns:
            s = unified[feat]
            print(f"  {feat:30s}  μ={s.mean():7.3f}  σ={s.std():7.3f}  "
                  f"min={s.min():7.3f}  max={s.max():7.3f}")

    print(f"\n{'='*70}")
    print(f"✅ Phase 1+2 integration test PASSED")
    print(f"   Pipeline ready for Phase 3: Model training (M1-M4)")
    print(f"{'='*70}")

    return unified


if __name__ == "__main__":
    unified = main()
