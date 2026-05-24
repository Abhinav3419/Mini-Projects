"""
CrudeNerve configuration — single source of truth for all parameters.

Edit values here instead of hunting through module code.
API keys should come from environment variables in production.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "saved_models"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── API Keys (from env vars — never hardcode) ──────────────────────────────

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2"
X_API_BEARER_TOKEN = os.getenv("X_API_BEARER_TOKEN", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


# ─── Time boundaries ────────────────────────────────────────────────────────

BACKFILL_START = "2015-01-01"
BACKFILL_END = datetime.now().strftime("%Y-%m-%d")
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"


# ─── D1: GDELT parameters ───────────────────────────────────────────────────

@dataclass
class GDELTConfig:
    """Editable GDELT ingestion parameters."""
    filter_entities: list[str] = field(default_factory=lambda: [
        "Iran", "Hormuz", "OPEC", "sanctions", "Israel",
        "Houthis", "Hezbollah", "crude oil", "petroleum",
        "Strait of Hormuz", "IRGC", "enrichment",
    ])
    # CAMEO event codes for military actions (190-195) and sanctions
    military_event_codes: list[str] = field(default_factory=lambda: [
        "190", "191", "192", "193", "194", "195",  # military force
        "163", "164",                                # sanctions
    ])
    rolling_window_days: int = 30
    recency_decay: str = "exponential"  # or "linear"
    goldstein_weight: float = 1.0
    max_records_per_query: int = 250
    query_mode: str = "ArtList"  # GDELT API mode


# ─── D2: Truth Social parameters ────────────────────────────────────────────

@dataclass
class TruthSocialConfig:
    """Editable Truth Social scraper + NLP parameters."""
    base_url: str = "https://truthsocial.com/@realDonaldTrump"
    poll_interval_waking_min: int = 15   # 6am-2am ET
    poll_interval_overnight_min: int = 60
    waking_hours_et: tuple[int, int] = (6, 26)  # 6am to 2am next day
    engagement_snapshot_hours: list[int] = field(
        default_factory=lambda: [0, 1, 6, 24]
    )
    backfill_start: str = "2022-02-01"

    # Layer 2B: Topic taxonomy
    topic_categories: dict = field(default_factory=lambda: {
        "ENERGY": [
            "oil_price_direct", "sanctions_iran",
            "sanctions_russia", "energy_policy",
        ],
        "MILITARY": [
            "iran_military", "israel_conflict", "general_threat",
        ],
        "TRADE": [
            "tariff_china", "tariff_general", "trade_deal",
        ],
        "DOMESTIC": [
            "fed_monetary", "fiscal_policy", "election_rhetoric",
        ],
    })
    topic_relevance_threshold: float = 0.3

    # Layer 2C: Severity axis weights (must sum to 1.0)
    severity_weight_urgency: float = 0.40
    severity_weight_extremity: float = 0.35
    severity_weight_targeting: float = 0.25

    # Layer 2E: Temporal pattern thresholds
    burst_count_threshold: int = 3
    burst_window_hours: int = 6
    friday_cutoff_hour_et: int = 16  # 4pm ET
    pre_announcement_window_hours: int = 48
    silence_threshold_hours: int = 12
    silence_min_severity: float = 4.0


# ─── D3: Twitter/X parameters ───────────────────────────────────────────────

@dataclass
class TwitterConfig:
    """Editable Twitter/X sentiment parameters."""
    oil_keywords: list[str] = field(default_factory=lambda: [
        "crude", "oil", "brent", "wti", "opec", "iran",
        "hormuz", "sanctions", "tanker", "petroleum",
        "barrel", "drilling", "refinery",
    ])
    volume_rolling_window_days: int = 7
    elite_follower_threshold: int = 100_000
    elite_weight_multiplier: float = 3.0
    max_results_per_query: int = 100
    sentiment_velocity_window_hours: int = 6


# ─── D4: Physical supply parameters ─────────────────────────────────────────

@dataclass
class SupplyConfig:
    """Editable physical supply indicator parameters."""
    eia_series_id: str = "PET.WCESTUS1.W"  # US crude inventory weekly
    spr_series_id: str = "PET.WCSSTUS1.W"  # SPR weekly
    hormuz_anomaly_std_threshold: float = 2.0
    opec_source: str = "manual"  # or "api" when available


# ─── D5: Price + VIX parameters ─────────────────────────────────────────────

@dataclass
class PriceConfig:
    """Editable price/VIX ingestion parameters."""
    tickers: list[str] = field(default_factory=lambda: [
        "BZ=F",   # Brent crude futures
        "CL=F",   # WTI crude futures
        "^VIX",   # VIX
    ])
    granularity: str = "1d"
    vix_threshold_pct: float = 2.0  # binary direction threshold
    horizons_hours: list[int] = field(
        default_factory=lambda: [24, 48, 72]
    )


# ─── DHO Kernel parameters ──────────────────────────────────────────────────

@dataclass
class DHOParams:
    """Single row in the DHO event-type table."""
    event_type: str
    zeta: float         # damping ratio
    omega_n: float      # natural frequency (rad/day)
    typical_decay: str  # human-readable


DHO_EVENT_TABLE: list[DHOParams] = [
    DHOParams("sanctions_announced", 0.15, 0.3, "weeks"),
    DHOParams("military_strike",     0.6,  1.2, "2-4 days"),
    DHOParams("opec_quota_change",   0.3,  0.5, "1-2 weeks"),
    DHOParams("hormuz_disruption",   0.2,  0.9, "1-2 weeks"),
]

# Severity-graded Trump DHO (Layer 2F)
TRUMP_DHO_TABLE: list[dict] = [
    {"severity_min": 1.0, "severity_max": 2.0, "zeta": 0.7,  "omega_n": 0.5,  "decay": "1-2 days"},
    {"severity_min": 2.0, "severity_max": 3.0, "zeta": 0.5,  "omega_n": 0.7,  "decay": "2-3 days"},
    {"severity_min": 3.0, "severity_max": 4.0, "zeta": 0.35, "omega_n": 0.9,  "decay": "3-5 days"},
    {"severity_min": 4.0, "severity_max": 4.5, "zeta": 0.2,  "omega_n": 1.1,  "decay": "5-7 days"},
    {"severity_min": 4.5, "severity_max": 5.0, "zeta": 0.1,  "omega_n": 1.4,  "decay": "7-14 days"},
]


# ─── Prediction targets ─────────────────────────────────────────────────────

PREDICTION_HORIZONS = [24, 48, 72]  # hours
VIX_DIRECTION_THRESHOLD = 0.02      # 2% move
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


# ─── F7: Prediction Mode configuration ──────────────────────────────────────

from enum import Enum


class PredictionMode(Enum):
    """Three prediction philosophies."""
    INTUITIVE = "intuitive"
    COUNTER_INTUITIVE = "counter_intuitive"
    GOLDEN_MEAN = "golden_mean"


class CounterIntuitiveSubMode(Enum):
    """How to apply contrarian logic to a signal."""
    FLIP = "flip"       # invert the signal direction
    ZERO_OUT = "zero"   # treat signal as pure noise, drop it


@dataclass
class SignalStreamDef:
    """
    Defines one toggleable signal stream for mode control.

    Each stream maps to specific feature columns in the unified DataFrame.
    The mode engine uses this mapping to know what to transform.
    """
    stream_id: str
    display_name: str
    description: str
    feature_prefixes: list[str]   # column prefixes to match
    category: str                 # "social", "macro", "supply", "geopolitical"
    default_trust: bool = True    # default state in Golden Mean mode
    invertible: bool = True       # can this stream be meaningfully flipped?


# The 8 toggleable signal streams
SIGNAL_STREAMS: list[SignalStreamDef] = [
    SignalStreamDef(
        stream_id="truth_social",
        display_name="Truth Social / Trump Rhetoric",
        description="Trump posts: topic classification, severity scoring, "
                    "engagement velocity, temporal patterns, policy-shock score",
        feature_prefixes=["ts_", "f4_", "f4eng_", "f4temp_"],
        category="social",
        default_trust=False,   # Golden Mean default: discount political rhetoric
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="twitter",
        display_name="Twitter/X Sentiment",
        description="Twitter volume spikes, sentiment velocity, "
                    "elite-retail divergence, fear index",
        feature_prefixes=["tw_", "f5_"],
        category="social",
        default_trust=True,
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="gdelt_tension",
        display_name="GDELT Geopolitical Tension",
        description="Entity-weighted tension index, velocity, z-score "
                    "from global news event flow",
        feature_prefixes=["gdelt_", "f2_"],
        category="geopolitical",
        default_trust=True,
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="dho_decay",
        display_name="DHO Event Decay Kernels",
        description="Physics-modeled event persistence — how long "
                    "geopolitical shocks affect markets",
        feature_prefixes=["f3_"],
        category="geopolitical",
        default_trust=True,
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="eia_inventory",
        display_name="EIA Crude Inventory",
        description="Weekly US crude oil inventory changes and surprises",
        feature_prefixes=["eia_", "f6_inventory"],
        category="supply",
        default_trust=True,
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="spr_releases",
        display_name="Strategic Petroleum Reserve",
        description="SPR level changes, release momentum",
        feature_prefixes=["spr_", "f6_spr"],
        category="supply",
        default_trust=True,
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="hormuz_shipping",
        display_name="Strait of Hormuz Throughput",
        description="Tanker traffic anomalies and disruption scores",
        feature_prefixes=["hormuz_", "f6_hormuz"],
        category="supply",
        default_trust=True,
        invertible=True,
    ),
    SignalStreamDef(
        stream_id="opec_compliance",
        display_name="OPEC Production Compliance",
        description="Quota gap and discipline trend",
        feature_prefixes=["opec_", "f6_opec"],
        category="supply",
        default_trust=True,
        invertible=True,
    ),
]

# Golden Mean presets — named configurations for quick selection
GOLDEN_MEAN_PRESETS: dict[str, dict[str, bool]] = {
    "macro_only": {
        # Trust fundamentals, ignore all social/political noise
        "truth_social": False,
        "twitter": False,
        "gdelt_tension": True,
        "dho_decay": True,
        "eia_inventory": True,
        "spr_releases": True,
        "hormuz_shipping": True,
        "opec_compliance": True,
    },
    "social_skeptic": {
        # Trust everything except Trump rhetoric
        "truth_social": False,
        "twitter": True,
        "gdelt_tension": True,
        "dho_decay": True,
        "eia_inventory": True,
        "spr_releases": True,
        "hormuz_shipping": True,
        "opec_compliance": True,
    },
    "crisis_trader": {
        # Only trust real-time crisis signals, ignore slow-moving data
        "truth_social": True,
        "twitter": True,
        "gdelt_tension": True,
        "dho_decay": True,
        "eia_inventory": False,
        "spr_releases": False,
        "hormuz_shipping": True,
        "opec_compliance": False,
    },
    "contrarian_lite": {
        # Trust supply data, discount everything social and geopolitical
        "truth_social": False,
        "twitter": False,
        "gdelt_tension": False,
        "dho_decay": False,
        "eia_inventory": True,
        "spr_releases": True,
        "hormuz_shipping": True,
        "opec_compliance": True,
    },
}

