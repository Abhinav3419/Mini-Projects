"""
O1: CrudeNerve FastAPI prediction endpoint.

Orchestrates the full pipeline:
    D1-D5 (ingest) → F1 (merge) → F2-F6 (features)
    → F8 (war barometer) → F7 (prediction mode) → model inference

Endpoints:
    POST /predict          — Run full prediction with mode + war level
    GET  /war-assessment   — Dynamic war tension assessment (no prediction)
    GET  /modes            — List available prediction modes + descriptions
    GET  /streams          — List toggleable signal streams for Golden Mean
    GET  /war-levels       — List all 10 escalation levels
    GET  /presets          — List Golden Mean presets
    GET  /health           — Pipeline health check
    GET  /pipeline-status  — Current data freshness + pipeline state

Usage:
    uvicorn crudenerve.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crudenerve.config.settings import (
    PREDICTION_HORIZONS,
    QUANTILE_LEVELS,
    VIX_DIRECTION_THRESHOLD,
    GOLDEN_MEAN_PRESETS,
    SIGNAL_STREAMS,
)
from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
from crudenerve.data_ingest.d2_truth_social import (
    TruthSocialPipeline,
    generate_synthetic_posts,
)
from crudenerve.data_ingest.d3_twitter import (
    TwitterIngestor,
    generate_synthetic_tweets,
)
from crudenerve.data_ingest.d4_supply import SupplyIngestor
from crudenerve.data_ingest.d5_price_vix import PriceVIXIngestor
from crudenerve.features.f1_merge import TimeIndexedMerge
from crudenerve.features.f2_tension_index import TensionIndexBuilder
from crudenerve.features.f3_dho_kernel import DHOKernelEngine
from crudenerve.features.f4_trump_volatility import TrumpVolatilityEngine
from crudenerve.features.f5_twitter_features import TwitterFeatureBuilder
from crudenerve.features.f6_supply_features import SupplyFeatureBuilder
from crudenerve.features.f7_prediction_mode import PredictionModeEngine
from crudenerve.features.f8_war_barometer import WarTensionBarometer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class PredictionModeEnum(str, Enum):
    intuitive = "intuitive"
    counter_intuitive = "counter_intuitive"
    golden_mean = "golden_mean"


class CISubModeEnum(str, Enum):
    flip = "flip"
    zero = "zero"


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    mode: PredictionModeEnum = Field(
        default=PredictionModeEnum.intuitive,
        description="Prediction philosophy: intuitive (face value), "
                    "counter_intuitive (contrarian), golden_mean (selective trust)",
    )
    ci_submode: CISubModeEnum = Field(
        default=CISubModeEnum.flip,
        description="Counter-Intuitive sub-mode: flip (invert signals) "
                    "or zero (drop signals as noise). Only used when mode=counter_intuitive",
    )
    war_level: int | None = Field(
        default=None,
        ge=1, le=10,
        description="USA-Iran-Israel war tension level (1-10). "
                    "If null, uses the dynamically recommended level.",
    )
    trust_config: dict[str, bool] | None = Field(
        default=None,
        description="Per-stream trust toggles for Golden Mean mode. "
                    "Keys: truth_social, twitter, gdelt_tension, dho_decay, "
                    "eia_inventory, spr_releases, hormuz_shipping, opec_compliance",
    )
    preset: str | None = Field(
        default=None,
        description="Golden Mean preset name: macro_only, social_skeptic, "
                    "crisis_trader, contrarian_lite. Overrides trust_config.",
    )
    horizons: list[int] = Field(
        default=PREDICTION_HORIZONS,
        description="Prediction horizons in hours (default: [24, 48, 72])",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "mode": "intuitive",
                "war_level": 5,
            },
            {
                "mode": "counter_intuitive",
                "ci_submode": "flip",
                "war_level": 6,
            },
            {
                "mode": "golden_mean",
                "preset": "social_skeptic",
                "war_level": None,
            },
            {
                "mode": "golden_mean",
                "trust_config": {
                    "truth_social": False,
                    "twitter": True,
                    "gdelt_tension": True,
                    "dho_decay": True,
                    "eia_inventory": True,
                    "spr_releases": True,
                    "hormuz_shipping": True,
                    "opec_compliance": True,
                },
                "war_level": 4,
            },
        ]
    }}


class HorizonPrediction(BaseModel):
    """Prediction result for a single time horizon."""
    horizon_hours: int
    vix_direction_prob: float = Field(
        description="Probability of VIX moving >2% in either direction",
    )
    vix_up_prob: float = Field(
        description="Probability of VIX rising >2%",
    )
    vix_down_prob: float = Field(
        description="Probability of VIX falling >2%",
    )
    brent_return_quantiles: dict[str, float] = Field(
        description="Expected Brent crude return distribution "
                    "(keys: q10, q25, q50, q75, q90)",
    )


class WarAssessmentResponse(BaseModel):
    """War tension barometer assessment."""
    recommended_level: int
    level_name: str
    confidence: float
    sub_scores: dict[str, float]
    evidence: dict
    brent_impact_range: str
    vix_impact_range: str
    data_freshness: dict


class TopFeatureDriver(BaseModel):
    """A single SHAP-ranked feature contributing to the prediction."""
    feature: str
    shap_value: float
    raw_value: float
    direction: str = Field(description="'bullish' or 'bearish'")


class ModeInfo(BaseModel):
    """Description of the applied prediction mode."""
    mode: str
    philosophy: str
    streams_active: list[str] | None = None
    streams_discounted: list[str] | None = None
    streams_inverted: list[str] | None = None
    best_for: str | None = None
    risk: str | None = None


class PredictResponse(BaseModel):
    """Full prediction response."""
    timestamp: str
    pipeline_version: str = "v3.0"

    # Mode + regime
    mode_info: ModeInfo
    war_assessment: WarAssessmentResponse

    # Predictions per horizon
    predictions: list[HorizonPrediction]

    # Top feature drivers (SHAP)
    top_drivers: list[TopFeatureDriver]

    # Pipeline metadata
    data_date_range: dict[str, str]
    feature_count: int
    processing_time_ms: float


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class CrudeNervePipeline:
    """
    Orchestrates the full CrudeNerve pipeline.

    Manages state: caches the unified DataFrame, rebuilds only when
    new data arrives, applies F8 + F7 transformations per-request.
    """

    def __init__(self):
        self.merger = TimeIndexedMerge()
        self.f2 = TensionIndexBuilder()
        self.f3 = DHOKernelEngine()
        self.f4 = TrumpVolatilityEngine()
        self.f5 = TwitterFeatureBuilder()
        self.f6 = SupplyFeatureBuilder()
        self.f7 = PredictionModeEngine()
        self.f8 = WarTensionBarometer()

        # Cached state
        self._unified_df: pd.DataFrame | None = None
        self._enriched_posts: pd.DataFrame | None = None
        self._last_rebuild: datetime | None = None

    def rebuild_data(self, start: str = "2024-01-01", end: str | None = None):
        """
        Rebuild the full data pipeline from ingestion through features.

        In production, this runs on a schedule (e.g., every hour).
        For dev, uses synthetic data.
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Rebuilding pipeline: {start} → {end}")
        t0 = time.time()

        # ── D1: GDELT ────────────────────────────────────────────────────
        gdelt_ingestor = GDELTIngestor()
        try:
            gdelt_raw = gdelt_ingestor.fetch_range(start, end, save=False)
        except Exception:
            logger.warning("GDELT API unavailable — using synthetic data")
            gdelt_raw = _generate_synthetic_gdelt(500)
        gdelt_raw = gdelt_ingestor._tag_entities(gdelt_raw)
        gdelt_daily = gdelt_ingestor.aggregate_daily(gdelt_raw)

        # ── D2: Truth Social ─────────────────────────────────────────────
        ts_pipeline = TruthSocialPipeline()
        try:
            ts_posts = ts_pipeline.scraper.load_cached()
            if ts_posts is None:
                raise FileNotFoundError
        except Exception:
            ts_posts = generate_synthetic_posts(200)
        self._enriched_posts = ts_pipeline.run_full_pipeline(ts_posts, save=False)
        ts_daily = ts_pipeline.aggregate_daily(self._enriched_posts)

        # ── D3: Twitter ──────────────────────────────────────────────────
        tw_ingestor = TwitterIngestor()
        try:
            tw_daily = tw_ingestor.load_cached()
            if tw_daily is None:
                raise FileNotFoundError
        except Exception:
            tw_raw = generate_synthetic_tweets(5000, start, end)
            tw_daily = tw_ingestor.compute_features(tw_raw)

        # ── D4: Supply ───────────────────────────────────────────────────
        supply_daily = SupplyIngestor().fetch_all(start, end, save=False)

        # ── D5: Price + VIX ──────────────────────────────────────────────
        price_ingestor = PriceVIXIngestor()
        try:
            price_vix = price_ingestor.fetch_all(start, end, save=False)
        except Exception:
            logger.warning("Yahoo Finance unavailable — using synthetic prices")
            rng = np.random.default_rng(99)
            dates = pd.date_range(start, end, freq="B")
            price_vix = pd.DataFrame({
                "BZ=F_close": 75 + np.cumsum(rng.normal(0, 1, len(dates))),
                "CL=F_close": 72 + np.cumsum(rng.normal(0, 1, len(dates))),
                "^VIX_close": 18 + np.cumsum(rng.normal(0, 0.5, len(dates))),
                "BZ=F_volume": rng.integers(100000, 500000, len(dates)),
            }, index=dates)
            price_vix.index.name = "date"

        # ── F1: Merge ────────────────────────────────────────────────────
        unified = self.merger.merge_all(
            gdelt_daily=gdelt_daily,
            truth_social_daily=ts_daily,
            twitter_daily=tw_daily,
            supply_daily=supply_daily,
            price_vix=price_vix,
            start=start, end=end,
            save=False,
        )

        # ── F2-F6: Feature engineering ───────────────────────────────────
        unified = self.f2.compute(unified)
        unified = self.f3.compute(unified)
        unified = self.f4.compute(unified, enriched_posts=self._enriched_posts)
        unified = self.f5.compute(unified)
        unified = self.f6.compute(unified)

        self._unified_df = unified
        self._last_rebuild = datetime.now()

        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"Pipeline rebuilt in {elapsed:.0f}ms: "
            f"{unified.shape[0]} days × {unified.shape[1]} columns"
        )

    def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Run prediction with the specified mode and war level.

        Pipeline order: unified_df → F8 (war level) → F7 (mode) → model
        """
        t0 = time.time()

        if self._unified_df is None:
            self.rebuild_data()

        df = self._unified_df.copy()

        # ── F8: War tension barometer ────────────────────────────────────
        war_assessment = self.f8.assess(df)

        # Use user-specified level or the dynamic recommendation
        war_level = request.war_level or war_assessment.recommended_level
        df = self.f8.apply_level(df, level=war_level)

        war_details = self.f8.get_level_details(war_level)

        # ── F7: Prediction mode transform ────────────────────────────────
        df = self.f7.transform(
            df,
            mode=request.mode.value,
            ci_submode=request.ci_submode.value,
            trust_config=request.trust_config,
            preset=request.preset,
        )

        mode_summary = self.f7.get_mode_summary(
            mode=request.mode.value,
            ci_submode=request.ci_submode.value,
            trust_config=request.trust_config,
            preset=request.preset,
        )

        # ── Model inference ──────────────────────────────────────────────
        # M1-M4 are not yet built. For now, generate intelligent stub
        # predictions derived from the actual feature values so the API
        # contract is fully exercised and the frontend can be built.
        predictions = self._stub_predictions(df, request.horizons)
        top_drivers = self._stub_shap_drivers(df)

        elapsed_ms = (time.time() - t0) * 1000

        # ── Build response ───────────────────────────────────────────────
        return PredictResponse(
            timestamp=datetime.now().isoformat(),
            mode_info=ModeInfo(
                mode=mode_summary.get("mode", request.mode.value),
                philosophy=mode_summary.get("philosophy", ""),
                streams_active=mode_summary.get("streams_active"),
                streams_discounted=mode_summary.get("streams_discounted",
                                                     mode_summary.get("streams_discounted")),
                streams_inverted=mode_summary.get("streams_inverted"),
                best_for=mode_summary.get("best_for"),
                risk=mode_summary.get("risk"),
            ),
            war_assessment=WarAssessmentResponse(
                recommended_level=war_assessment.recommended_level,
                level_name=war_details.name,
                confidence=war_assessment.confidence,
                sub_scores=war_assessment.sub_scores,
                evidence=self._sanitize_evidence(war_assessment.evidence),
                brent_impact_range=war_details.brent_impact_range,
                vix_impact_range=war_details.vix_impact_range,
                data_freshness=war_assessment.data_freshness,
            ),
            predictions=predictions,
            top_drivers=top_drivers,
            data_date_range={
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            },
            feature_count=df.shape[1],
            processing_time_ms=round(elapsed_ms, 1),
        )

    # ── Stub model predictions (replace with M1-M4 when built) ───────────

    def _stub_predictions(
        self,
        df: pd.DataFrame,
        horizons: list[int],
    ) -> list[HorizonPrediction]:
        """
        Generate feature-derived stub predictions.

        These are NOT random — they're computed from the actual feature
        state so the API response is internally consistent. When M1-M4
        are built, this method gets replaced with real model inference.
        """
        predictions = []
        latest = df.iloc[-1] if len(df) > 0 else pd.Series(dtype=float)

        # Base probability from feature signals
        tension = latest.get("f2_tension_ema", 0)
        shock = latest.get("f4_trump_shock_score", 0)
        fear = latest.get("f5_fear_index", 0)
        supply_stress = latest.get("f6_supply_stress", 0)
        war_level = latest.get("f8_war_level", 4)

        # Composite signal: higher = more likely VIX spike
        composite = (
            min(tension / 20, 1.0) * 0.25
            + min(shock, 1.0) * 0.25
            + min(fear, 1.0) * 0.20
            + min(supply_stress, 1.0) * 0.15
            + min(war_level / 10, 1.0) * 0.15
        )

        for horizon in horizons:
            # Longer horizons = higher probability of a move
            horizon_factor = 1.0 + (horizon - 24) / 120
            base_prob = min(0.95, composite * horizon_factor)

            # Directional split: tension + shock → more likely up than down
            up_bias = 0.5 + composite * 0.3
            vix_up = base_prob * up_bias
            vix_down = base_prob * (1 - up_bias)

            # Brent return quantiles — wider spread at higher tension
            spread = 0.02 + composite * 0.08
            center = composite * 0.03 * horizon_factor

            predictions.append(HorizonPrediction(
                horizon_hours=horizon,
                vix_direction_prob=round(base_prob, 4),
                vix_up_prob=round(vix_up, 4),
                vix_down_prob=round(vix_down, 4),
                brent_return_quantiles={
                    "q10": round(center - 2 * spread, 4),
                    "q25": round(center - spread, 4),
                    "q50": round(center, 4),
                    "q75": round(center + spread, 4),
                    "q90": round(center + 2 * spread, 4),
                },
            ))

        return predictions

    def _stub_shap_drivers(
        self,
        df: pd.DataFrame,
        n_top: int = 8,
    ) -> list[TopFeatureDriver]:
        """
        Generate stub SHAP feature importance from actual feature values.

        Uses feature magnitudes as proxy for importance until M4 SHAP
        is built. The ranking is directionally correct — features with
        extreme values will rank higher.
        """
        if df.empty:
            return []

        latest = df.iloc[-1]
        key_features = [
            ("f2_tension_ema", "Geopolitical tension index"),
            ("f3_dho_combined", "Event decay signal"),
            ("f4_trump_shock_score", "Trump policy-shock"),
            ("f5_fear_index", "Twitter fear index"),
            ("f6_supply_stress", "Supply disruption stress"),
            ("f6_hormuz_disruption_score", "Hormuz disruption"),
            ("f8_war_level", "War tension level"),
            ("tw_volume_spike_ratio", "Twitter volume spike"),
            ("tw_elite_retail_divergence", "Elite-retail divergence"),
            ("ts_max_severity", "Trump max severity"),
            ("hormuz_zscore", "Hormuz anomaly z-score"),
            ("f2_tension_velocity", "Tension acceleration"),
        ]

        drivers = []
        for col, name in key_features:
            val = latest.get(col, 0)
            if pd.isna(val):
                val = 0
            val = float(val)
            # Proxy SHAP: absolute value normalized, sign determines direction
            shap_proxy = val / max(abs(val) + 1, 1)
            drivers.append(TopFeatureDriver(
                feature=name,
                shap_value=round(shap_proxy, 4),
                raw_value=round(val, 4),
                direction="bearish" if val > 0 else "bullish",
            ))

        # Sort by absolute SHAP value
        drivers.sort(key=lambda d: abs(d.shap_value), reverse=True)
        return drivers[:n_top]

    def _sanitize_evidence(self, evidence: dict) -> dict:
        """Convert numpy types to Python natives for JSON serialization."""
        clean = {}
        for key, val in evidence.items():
            if isinstance(val, dict):
                clean[key] = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in val.items()
                }
            elif hasattr(val, 'item'):
                clean[key] = float(val)
            else:
                clean[key] = val
        return clean


# ═══════════════════════════════════════════════════════════════════════════
# APP FACTORY
# ═══════════════════════════════════════════════════════════════════════════

pipeline = CrudeNervePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline on startup."""
    logger.info("CrudeNerve starting — building initial pipeline...")
    pipeline.rebuild_data()
    logger.info("Pipeline ready.")
    yield
    logger.info("CrudeNerve shutting down.")


app = FastAPI(
    title="CrudeNerve",
    description=(
        "Geopolitical signal processing engine for crude oil volatility prediction. "
        "Predicts VIX direction and Brent crude return distributions using "
        "5 data streams, physics-informed DHO decay kernels, a 10-level "
        "USA-Iran-Israel war tension barometer, and 3 prediction modes "
        "(Intuitive / Counter-Intuitive / Golden Mean)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run the full CrudeNerve prediction pipeline.

    The pipeline executes in order:
    1. Load cached unified DataFrame (D1-D5 → F1 → F2-F6)
    2. F8: Apply war tension level (dynamic or user-specified)
    3. F7: Apply prediction mode transformation
    4. Model inference (M1-M4 — stub until models are trained)
    5. Return predictions + SHAP drivers + war assessment

    Modes:
    - **intuitive**: All signals at face value. The textbook view.
    - **counter_intuitive**: Signals are misleading. Sub-modes: flip or zero.
    - **golden_mean**: User picks which of 8 signal streams to trust.

    War level (1-10) scales ALL features before mode transform.
    If not specified, the barometer recommends one dynamically.
    """
    try:
        return pipeline.predict(request)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/war-assessment", response_model=WarAssessmentResponse)
async def war_assessment():
    """
    Get the current dynamic war tension assessment without running predictions.

    Analyzes the last 14 days of data across all streams and recommends
    a severity level (1-10) with confidence score and evidence breakdown.
    """
    if pipeline._unified_df is None:
        pipeline.rebuild_data()

    assessment = pipeline.f8.assess(pipeline._unified_df)
    details = pipeline.f8.get_level_details(assessment.recommended_level)

    return WarAssessmentResponse(
        recommended_level=assessment.recommended_level,
        level_name=details.name,
        confidence=assessment.confidence,
        sub_scores=assessment.sub_scores,
        evidence=pipeline._sanitize_evidence(assessment.evidence),
        brent_impact_range=details.brent_impact_range,
        vix_impact_range=details.vix_impact_range,
        data_freshness=assessment.data_freshness,
    )


@app.get("/war-levels")
async def war_levels():
    """List all 10 escalation levels with descriptions and market impact."""
    return pipeline.f8.get_all_levels_summary()


@app.get("/modes")
async def modes():
    """List available prediction modes with descriptions."""
    return {
        "modes": [
            pipeline.f7.get_mode_summary("intuitive"),
            pipeline.f7.get_mode_summary("counter_intuitive", ci_submode="flip"),
            pipeline.f7.get_mode_summary("counter_intuitive", ci_submode="zero"),
            pipeline.f7.get_mode_summary("golden_mean"),
        ]
    }


@app.get("/streams")
async def streams():
    """List toggleable signal streams for Golden Mean mode."""
    return {"streams": pipeline.f7.get_available_streams()}


@app.get("/presets")
async def presets():
    """List available Golden Mean presets."""
    return {
        "presets": {
            name: {
                "trust_config": config,
                "description": {
                    "macro_only": "Trust fundamentals, ignore all social/political noise",
                    "social_skeptic": "Trust everything except Trump rhetoric",
                    "crisis_trader": "Only trust real-time crisis signals",
                    "contrarian_lite": "Trust only hard supply data",
                }.get(name, ""),
            }
            for name, config in GOLDEN_MEAN_PRESETS.items()
        }
    }


@app.get("/health")
async def health():
    """Pipeline health check."""
    has_data = pipeline._unified_df is not None
    return {
        "status": "healthy" if has_data else "no_data",
        "pipeline_version": "v3.0",
        "data_loaded": has_data,
        "last_rebuild": pipeline._last_rebuild.isoformat()
        if pipeline._last_rebuild else None,
        "data_shape": {
            "days": pipeline._unified_df.shape[0],
            "features": pipeline._unified_df.shape[1],
        } if has_data else None,
    }


@app.get("/pipeline-status")
async def pipeline_status():
    """Detailed pipeline status with data freshness per stream."""
    if pipeline._unified_df is None:
        return {"status": "not_initialized", "message": "Call /predict first"}

    df = pipeline._unified_df
    assessment = pipeline.f8.assess(df)

    return {
        "status": "ready",
        "last_rebuild": pipeline._last_rebuild.isoformat()
        if pipeline._last_rebuild else None,
        "data_range": {
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "days": len(df),
        },
        "feature_count": df.shape[1],
        "stream_coverage": {
            "gdelt_days": int(df.get("has_gdelt", pd.Series(0)).sum()),
            "truth_social_days": int(df.get("has_truth_social", pd.Series(0)).sum()),
            "twitter_days": int(df.get("has_twitter", pd.Series(0)).sum()),
            "supply_days": int(df.get("has_supply", pd.Series(0)).sum()),
            "price_vix_days": int(df.get("has_price_vix", pd.Series(0)).sum()),
        },
        "data_freshness": assessment.data_freshness,
        "current_war_assessment": {
            "level": assessment.recommended_level,
            "name": assessment.level_details.name,
            "confidence": assessment.confidence,
        },
    }


@app.post("/rebuild")
async def rebuild():
    """Force a pipeline rebuild from fresh data."""
    pipeline.rebuild_data()
    return {
        "status": "rebuilt",
        "timestamp": datetime.now().isoformat(),
        "shape": {
            "days": pipeline._unified_df.shape[0],
            "features": pipeline._unified_df.shape[1],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
