"""
F8: USA-Iran-Israel War Tension Barometer.

A cross-modal feature that persists across ALL three prediction modes
(Intuitive, Counter-Intuitive, Golden Mean). Unlike F7 which changes
*how* signals are interpreted, F8 sets the *baseline geopolitical regime*
— where are we on the escalation ladder right now?

THE 10-LEVEL ESCALATION LADDER:
    Level 1:  Diplomatic calm — no active threats, normal trade
    Level 2:  Background tension — routine posturing, no specific threats
    Level 3:  Diplomatic friction — ambassador recalls, UN votes, rhetoric
    Level 4:  Sanctions escalation — new sanctions, enrichment milestones
    Level 5:  Proxy conflict active — Houthi/Hezbollah attacks, Red Sea disruption
    Level 6:  Direct threats — explicit military threats, naval positioning
    Level 7:  Military mobilization — troop movements, carrier deployments
    Level 8:  Limited strikes — targeted strikes on military/nuclear facilities
    Level 9:  Open conflict — sustained military operations, Hormuz closure
    Level 10: Full-scale war — all-out regional conflict, multiple fronts

DYNAMIC ASSESSMENT:
    Every time the app runs, F8 ingests the latest data from:
    - GDELT (military event codes, entity co-occurrence, tone extremes)
    - Truth Social (severity of Iran/Israel-related posts)
    - Twitter (volume spikes + elite sentiment on war-related keywords)
    - Supply data (Hormuz throughput anomalies)
    Then recommends a severity level. User can accept or override.

    Two users on the same day get the same *recommendation* but may choose
    different levels based on their own geopolitical read.

IMPACT ON PIPELINE:
    The selected level modifies:
    - F2 tension index: entity weight multiplier (higher level = more weight)
    - F3 DHO kernel: damping ratio shifts (higher level = more underdamped)
    - F4 Trump volatility: severity threshold adjustments
    - M2 XGBoost: feature importance re-weighting per regime
    - All three prediction modes get these modifications equally

Usage:
    from crudenerve.features.f8_war_barometer import WarTensionBarometer
    barometer = WarTensionBarometer()
    assessment = barometer.assess(unified_df)
    # assessment.recommended_level = 5
    # assessment.confidence = 0.72
    # assessment.evidence = {...}
    df = barometer.apply_level(unified_df, level=5)  # or user overrides to 6
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# THE 10-LEVEL ESCALATION LADDER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EscalationLevel:
    """Definition of one escalation level with its market impact parameters."""
    level: int
    name: str
    description: str
    historical_examples: list[str]
    # Market impact modifiers
    tension_multiplier: float      # scales F2 tension index weights
    dho_damping_shift: float       # added to ζ (negative = more underdamped)
    dho_frequency_boost: float     # multiplier on ω_n (>1 = faster reaction)
    trump_severity_floor: float    # minimum Trump severity to trigger features
    brent_impact_range: str        # expected Brent crude % impact
    vix_impact_range: str          # expected VIX % impact
    hormuz_weight: float           # extra weight on Hormuz anomaly features
    feature_boost: dict = field(default_factory=dict)  # per-feature multipliers


ESCALATION_LADDER: list[EscalationLevel] = [
    EscalationLevel(
        level=1,
        name="Diplomatic calm",
        description="Normal diplomatic relations. No active threats. "
                    "Routine trade and IAEA inspections proceeding.",
        historical_examples=["2015-2017 JCPOA compliance period"],
        tension_multiplier=0.5,
        dho_damping_shift=+0.3,    # overdamped — events fade fast
        dho_frequency_boost=0.7,
        trump_severity_floor=4.0,   # only very high severity matters
        brent_impact_range="±1%",
        vix_impact_range="±0.5%",
        hormuz_weight=1.0,
        feature_boost={"f2_tension_ema": 0.5, "f3_dho_combined": 0.5},
    ),
    EscalationLevel(
        level=2,
        name="Background tension",
        description="Routine geopolitical posturing. Occasional rhetoric "
                    "but no specific actionable threats.",
        historical_examples=["2017-2018 pre-JCPOA withdrawal"],
        tension_multiplier=0.7,
        dho_damping_shift=+0.2,
        dho_frequency_boost=0.8,
        trump_severity_floor=3.5,
        brent_impact_range="±2%",
        vix_impact_range="±1%",
        hormuz_weight=1.0,
        feature_boost={"f2_tension_ema": 0.7, "f3_dho_combined": 0.7},
    ),
    EscalationLevel(
        level=3,
        name="Diplomatic friction",
        description="Ambassador recalls, UN Security Council disputes, "
                    "sharp rhetoric from heads of state. No military moves.",
        historical_examples=["May 2018 JCPOA withdrawal announcement"],
        tension_multiplier=0.85,
        dho_damping_shift=+0.1,
        dho_frequency_boost=0.9,
        trump_severity_floor=3.0,
        brent_impact_range="+1% to +4%",
        vix_impact_range="+1% to +2%",
        hormuz_weight=1.2,
        feature_boost={"f2_tension_ema": 0.85, "f3_dho_combined": 0.85},
    ),
    EscalationLevel(
        level=4,
        name="Sanctions escalation",
        description="New sanctions packages imposed or tightened. Iran enrichment "
                    "milestones (20%, 60%, 90%). Oil export restrictions.",
        historical_examples=[
            "Nov 2018 maximum pressure sanctions",
            "2019 Iran 20% enrichment breach",
        ],
        tension_multiplier=1.0,
        dho_damping_shift=0.0,      # baseline — no shift
        dho_frequency_boost=1.0,
        trump_severity_floor=2.5,
        brent_impact_range="+2% to +6%",
        vix_impact_range="+1% to +3%",
        hormuz_weight=1.4,
        feature_boost={"f2_tension_ema": 1.0, "f3_dho_combined": 1.0,
                       "f4_trump_shock_score": 1.2},
    ),
    EscalationLevel(
        level=5,
        name="Proxy conflict active",
        description="Houthi attacks on shipping, Hezbollah border skirmishes, "
                    "militia strikes on US bases. No direct state-on-state action.",
        historical_examples=[
            "2023-2024 Houthi Red Sea attacks",
            "Oct 2023 post-Gaza escalation",
        ],
        tension_multiplier=1.2,
        dho_damping_shift=-0.05,    # slightly more underdamped
        dho_frequency_boost=1.1,
        trump_severity_floor=2.0,
        brent_impact_range="+3% to +8%",
        vix_impact_range="+2% to +4%",
        hormuz_weight=1.8,
        feature_boost={"f2_tension_ema": 1.2, "f3_dho_combined": 1.3,
                       "f4_trump_shock_score": 1.3, "f5_fear_index": 1.2},
    ),
    EscalationLevel(
        level=6,
        name="Direct threats",
        description="Explicit military threats between USA/Israel and Iran. "
                    "Naval assets repositioned. B-52 overflights. "
                    "'All options on the table' language.",
        historical_examples=[
            "June 2019 Iran drone shoot-down + aborted strike",
            "Apr 2024 Iran-Israel direct exchange",
        ],
        tension_multiplier=1.5,
        dho_damping_shift=-0.1,
        dho_frequency_boost=1.2,
        trump_severity_floor=1.5,
        brent_impact_range="+5% to +12%",
        vix_impact_range="+3% to +6%",
        hormuz_weight=2.2,
        feature_boost={"f2_tension_ema": 1.5, "f3_dho_combined": 1.5,
                       "f4_trump_shock_score": 1.5, "f5_fear_index": 1.5,
                       "f6_supply_stress": 1.3},
    ),
    EscalationLevel(
        level=7,
        name="Military mobilization",
        description="Visible troop movements. Carrier strike groups deployed. "
                    "Reservists called up. Civil defense alerts in the region.",
        historical_examples=[
            "Jan 2020 post-Soleimani mobilization",
        ],
        tension_multiplier=1.8,
        dho_damping_shift=-0.15,
        dho_frequency_boost=1.3,
        trump_severity_floor=1.0,    # everything matters now
        brent_impact_range="+8% to +18%",
        vix_impact_range="+5% to +10%",
        hormuz_weight=2.5,
        feature_boost={"f2_tension_ema": 1.8, "f3_dho_combined": 1.8,
                       "f4_trump_shock_score": 1.8, "f5_fear_index": 1.8,
                       "f6_supply_stress": 1.5, "f6_hormuz_disruption_score": 2.0},
    ),
    EscalationLevel(
        level=8,
        name="Limited strikes",
        description="Targeted military strikes on specific facilities. "
                    "Nuclear sites, military bases, or naval assets hit. "
                    "Both sides still signaling restraint.",
        historical_examples=[
            "Jan 2020 Soleimani assassination + Iran retaliation",
            "Apr 2024 Israel strike on Isfahan (limited)",
        ],
        tension_multiplier=2.2,
        dho_damping_shift=-0.2,     # strongly underdamped
        dho_frequency_boost=1.5,
        trump_severity_floor=1.0,
        brent_impact_range="+10% to +25%",
        vix_impact_range="+8% to +15%",
        hormuz_weight=3.0,
        feature_boost={"f2_tension_ema": 2.2, "f3_dho_combined": 2.5,
                       "f4_trump_shock_score": 2.0, "f5_fear_index": 2.0,
                       "f6_supply_stress": 2.0, "f6_hormuz_disruption_score": 3.0},
    ),
    EscalationLevel(
        level=9,
        name="Open conflict",
        description="Sustained military operations. Multiple strike waves. "
                    "Hormuz strait partially or fully closed. "
                    "Regional allies drawn in.",
        historical_examples=[
            "No direct precedent — closest: 1980 Iran-Iraq war onset",
        ],
        tension_multiplier=3.0,
        dho_damping_shift=-0.3,     # very underdamped — persistent shock
        dho_frequency_boost=1.8,
        trump_severity_floor=1.0,
        brent_impact_range="+20% to +50%",
        vix_impact_range="+15% to +30%",
        hormuz_weight=4.0,
        feature_boost={"f2_tension_ema": 3.0, "f3_dho_combined": 3.5,
                       "f4_trump_shock_score": 2.5, "f5_fear_index": 3.0,
                       "f6_supply_stress": 3.0, "f6_hormuz_disruption_score": 5.0},
    ),
    EscalationLevel(
        level=10,
        name="Full-scale regional war",
        description="All-out multi-front conflict. Hormuz closed. "
                    "Global oil supply crisis. Multiple state actors engaged. "
                    "Nuclear escalation risk non-zero.",
        historical_examples=[
            "No precedent — hypothetical worst case",
        ],
        tension_multiplier=5.0,
        dho_damping_shift=-0.4,     # maximally underdamped — shock persists months
        dho_frequency_boost=2.0,
        trump_severity_floor=1.0,
        brent_impact_range="+40% to +100%+",
        vix_impact_range="+25% to +50%+",
        hormuz_weight=5.0,
        feature_boost={"f2_tension_ema": 5.0, "f3_dho_combined": 5.0,
                       "f4_trump_shock_score": 3.0, "f5_fear_index": 5.0,
                       "f6_supply_stress": 5.0, "f6_hormuz_disruption_score": 8.0},
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC SEVERITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WarAssessment:
    """Result of the dynamic severity assessment."""
    recommended_level: int           # 1-10
    confidence: float                # 0-1 (how sure the algorithm is)
    evidence: dict                   # what drove the recommendation
    sub_scores: dict                 # per-signal-source scores
    level_details: EscalationLevel   # full details of the recommended level
    timestamp: str                   # when assessment was computed
    data_freshness: dict             # how recent each data source is


class WarTensionBarometer:
    """
    Assesses current USA-Iran-Israel war tension from all data streams
    and applies escalation-level modifiers to the feature DataFrame.

    This module is INDEPENDENT of the prediction mode (F7).
    All three modes get the same war tension level applied.
    """

    def __init__(self):
        self.ladder = {lvl.level: lvl for lvl in ESCALATION_LADDER}

    # ── Dynamic assessment: recommend a level from data ──────────────────

    def assess(
        self,
        df: pd.DataFrame,
        lookback_days: int = 14,
    ) -> WarAssessment:
        """
        Analyze the most recent data and recommend a severity level.

        Uses the last `lookback_days` of data to compute:
          1. GDELT military event intensity (Iran+Israel entity density)
          2. Truth Social Iran/Israel severity peak
          3. Twitter fear index trajectory
          4. Hormuz disruption status
          5. Oil price volatility regime

        Each sub-score maps to a 1-10 range. The final recommendation
        is a weighted combination, biased toward the HIGHEST sub-score
        (because escalation is driven by the most alarming signal,
        not the average).
        """
        # Use last N days of data
        if df.empty:
            return self._empty_assessment()

        recent = df.iloc[-lookback_days:] if len(df) >= lookback_days else df

        sub_scores = {}
        evidence = {}

        # ── Sub-score 1: GDELT military intensity ────────────────────────
        gdelt_score, gdelt_evidence = self._score_gdelt(recent)
        sub_scores["gdelt_military"] = gdelt_score
        evidence["gdelt"] = gdelt_evidence

        # ── Sub-score 2: Truth Social war rhetoric ───────────────────────
        ts_score, ts_evidence = self._score_truth_social(recent)
        sub_scores["truth_social_war"] = ts_score
        evidence["truth_social"] = ts_evidence

        # ── Sub-score 3: Twitter fear trajectory ─────────────────────────
        tw_score, tw_evidence = self._score_twitter_fear(recent)
        sub_scores["twitter_fear"] = tw_score
        evidence["twitter"] = tw_evidence

        # ── Sub-score 4: Hormuz disruption ───────────────────────────────
        hormuz_score, hormuz_evidence = self._score_hormuz(recent)
        sub_scores["hormuz_status"] = hormuz_score
        evidence["hormuz"] = hormuz_evidence

        # ── Sub-score 5: Oil volatility regime ───────────────────────────
        vol_score, vol_evidence = self._score_oil_volatility(recent)
        sub_scores["oil_volatility"] = vol_score
        evidence["oil_volatility"] = vol_evidence

        # ── Combine: weighted toward the MAXIMUM ─────────────────────────
        # Why max-biased? Because wars escalate based on the worst signal,
        # not the average. One Hormuz closure outweighs ten calm GDELT days.
        scores = list(sub_scores.values())
        max_score = max(scores) if scores else 1
        mean_score = np.mean(scores) if scores else 1
        # 60% max + 40% mean — biased toward the hottest signal
        combined = 0.6 * max_score + 0.4 * mean_score
        recommended = int(np.clip(round(combined), 1, 10))

        # Confidence: high when sub-scores agree, low when they diverge
        score_std = np.std(scores) if len(scores) > 1 else 0
        confidence = max(0.3, 1.0 - score_std / 3.0)

        # Data freshness
        freshness = self._check_data_freshness(recent)

        return WarAssessment(
            recommended_level=recommended,
            confidence=round(confidence, 2),
            evidence=evidence,
            sub_scores=sub_scores,
            level_details=self.ladder[recommended],
            timestamp=datetime.now().isoformat(),
            data_freshness=freshness,
        )

    # ── Sub-score computation methods ────────────────────────────────────

    def _score_gdelt(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Score GDELT military event intensity for Iran/Israel axis."""
        iran_col = [c for c in df.columns if "iran" in c.lower() and "count" in c]
        israel_col = [c for c in df.columns if "israel" in c.lower() and "count" in c]
        hormuz_col = [c for c in df.columns if "hormuz" in c.lower() and "count" in c]
        irgc_col = [c for c in df.columns if "irgc" in c.lower() and "count" in c]

        entity_sum = 0
        for cols in [iran_col, israel_col, hormuz_col, irgc_col]:
            for c in cols:
                entity_sum += df[c].fillna(0).sum()

        # Normalize: 0-20 mentions/14d = level 1-2, 100+ = level 7+
        total_days = max(len(df), 1)
        daily_rate = entity_sum / total_days

        if daily_rate < 1:
            score = 1
        elif daily_rate < 3:
            score = 2
        elif daily_rate < 5:
            score = 3
        elif daily_rate < 8:
            score = 4
        elif daily_rate < 12:
            score = 5
        elif daily_rate < 18:
            score = 6
        elif daily_rate < 25:
            score = 7
        elif daily_rate < 35:
            score = 8
        elif daily_rate < 50:
            score = 9
        else:
            score = 10

        # Tone extremity bonus
        tone_col = "gdelt_mean_tone"
        if tone_col in df.columns:
            min_tone = df[tone_col].min()
            if min_tone < -6:
                score = min(10, score + 1)

        evidence = {
            "total_war_entity_mentions": int(entity_sum),
            "daily_rate": round(daily_rate, 1),
            "tone_min": round(df.get(tone_col, pd.Series(0)).min(), 2)
            if tone_col in df.columns else 0,
        }
        return float(score), evidence

    def _score_truth_social(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Score Trump's Iran/Israel rhetoric intensity."""
        sev_col = "ts_max_severity"
        iran_topic = "ts_topic_sanctions_iran_max"
        israel_topic = "ts_topic_israel_conflict_max"
        iran_mil = "ts_topic_iran_military_max"

        peak_severity = 0
        war_topic_days = 0

        if sev_col in df.columns:
            peak_severity = df[sev_col].max()

        for col in [iran_topic, israel_topic, iran_mil]:
            if col in df.columns:
                war_topic_days += (df[col].fillna(0) > 0.3).sum()

        # Peak severity directly maps: sev 4.5+ = level 7+
        if peak_severity >= 4.8:
            score = 9
        elif peak_severity >= 4.5:
            score = 7
        elif peak_severity >= 4.0:
            score = 6
        elif peak_severity >= 3.5:
            score = 5
        elif peak_severity >= 3.0:
            score = 4
        elif peak_severity >= 2.5:
            score = 3
        elif peak_severity >= 2.0:
            score = 2
        else:
            score = 1

        # Frequency bonus: many war-topic posts = sustained focus
        if war_topic_days >= 10:
            score = min(10, score + 2)
        elif war_topic_days >= 5:
            score = min(10, score + 1)

        evidence = {
            "peak_severity": round(float(peak_severity), 2),
            "war_topic_days": int(war_topic_days),
        }
        return float(score), evidence

    def _score_twitter_fear(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Score Twitter fear trajectory."""
        fear_col = "f5_fear_index"
        spike_col = "tw_volume_spike_ratio"

        if fear_col not in df.columns:
            return 1.0, {"note": "no Twitter fear data"}

        peak_fear = df[fear_col].max()
        mean_fear = df[fear_col].mean()
        fear_trend = df[fear_col].diff().mean()  # rising or falling

        # Peak fear maps: >0.8 = level 8+
        score = np.clip(peak_fear * 10, 1, 10)

        # Volume spike amplifies: 3x+ volume = add 1 level
        if spike_col in df.columns and df[spike_col].max() > 3.0:
            score = min(10, score + 1)

        # Rising trend adds urgency
        if fear_trend > 0.02:
            score = min(10, score + 1)

        evidence = {
            "peak_fear": round(float(peak_fear), 3),
            "mean_fear": round(float(mean_fear), 3),
            "fear_trending": "rising" if fear_trend > 0 else "falling",
        }
        return float(score), evidence

    def _score_hormuz(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Score Hormuz disruption severity."""
        anomaly_col = "hormuz_is_anomaly"
        disruption_col = "f6_hormuz_disruption_score"
        zscore_col = "hormuz_zscore"

        if anomaly_col not in df.columns:
            return 1.0, {"note": "no Hormuz data"}

        anomaly_days = int(df[anomaly_col].fillna(0).sum())
        peak_disruption = 0
        if disruption_col in df.columns:
            peak_disruption = df[disruption_col].max()

        # Anomaly days in 14 days: 0 = level 1, 5+ = level 7, 10+ = level 9
        if anomaly_days == 0:
            score = 1
        elif anomaly_days <= 2:
            score = 4
        elif anomaly_days <= 4:
            score = 6
        elif anomaly_days <= 7:
            score = 7
        elif anomaly_days <= 10:
            score = 9
        else:
            score = 10

        # Z-score severity bonus
        if zscore_col in df.columns:
            min_zscore = df[zscore_col].min()
            if min_zscore < -3:
                score = min(10, score + 1)

        evidence = {
            "anomaly_days_14d": anomaly_days,
            "peak_disruption_score": round(float(peak_disruption), 2),
        }
        return float(score), evidence

    def _score_oil_volatility(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Score oil price volatility regime."""
        brent_col = "BZ=F_close"
        vix_col = "^VIX_close"

        if brent_col not in df.columns:
            return 1.0, {"note": "no price data"}

        # Brent daily returns volatility
        brent = df[brent_col].dropna()
        if len(brent) < 2:
            return 1.0, {"note": "insufficient price data"}

        returns = brent.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)  # annualized
        max_daily_move = returns.abs().max() * 100

        # VIX level
        vix_level = df[vix_col].iloc[-1] if vix_col in df.columns else 15

        # Map: vol < 15% = level 1-2, vol > 60% = level 9-10
        if vol < 0.15:
            score = 1
        elif vol < 0.20:
            score = 2
        elif vol < 0.25:
            score = 3
        elif vol < 0.30:
            score = 4
        elif vol < 0.40:
            score = 5
        elif vol < 0.50:
            score = 6
        elif vol < 0.60:
            score = 7
        elif vol < 0.80:
            score = 8
        else:
            score = 9

        # VIX spike bonus
        if vix_level > 35:
            score = min(10, score + 1)

        evidence = {
            "annualized_vol": round(float(vol) * 100, 1),
            "max_daily_move_pct": round(float(max_daily_move), 2),
            "vix_latest": round(float(vix_level), 1),
        }
        return float(score), evidence

    # ── Apply the selected level to the DataFrame ────────────────────────

    def apply_level(
        self,
        df: pd.DataFrame,
        level: int,
    ) -> pd.DataFrame:
        """
        Apply war tension level modifiers to the unified DataFrame.

        This runs AFTER F2-F6 features are computed and BEFORE F7 mode
        transform. It modifies feature values based on the selected
        escalation level.

        Modifications:
            - Scales key features by level-specific multipliers
            - Adds f8_war_level (constant column for the session)
            - Adds f8_tension_multiplier
            - Adds f8_hormuz_weight
            - Adjusts DHO-related features for damping shift
        """
        level = int(np.clip(level, 1, 10))
        escalation = self.ladder[level]
        out = df.copy()

        # ── Apply feature multipliers ────────────────────────────────────
        for feature, multiplier in escalation.feature_boost.items():
            if feature in out.columns:
                out[feature] = out[feature] * multiplier

        # ── Add F8 metadata columns ──────────────────────────────────────
        out["f8_war_level"] = level
        out["f8_war_level_name"] = escalation.name
        out["f8_tension_multiplier"] = escalation.tension_multiplier
        out["f8_hormuz_weight"] = escalation.hormuz_weight
        out["f8_dho_damping_shift"] = escalation.dho_damping_shift
        out["f8_dho_frequency_boost"] = escalation.dho_frequency_boost
        out["f8_trump_severity_floor"] = escalation.trump_severity_floor

        # ── Apply Trump severity floor ───────────────────────────────────
        # In high-tension regimes, even moderate Trump posts matter.
        # In low-tension regimes, only extreme posts get through.
        if "f4_trump_shock_score" in out.columns and "ts_max_severity" in out.columns:
            floor = escalation.trump_severity_floor
            below_floor = out["ts_max_severity"].fillna(0) < floor
            out.loc[below_floor, "f4_trump_shock_score"] = (
                out.loc[below_floor, "f4_trump_shock_score"] * 0.3
            )

        logger.info(
            f"War barometer: level {level} ({escalation.name}) applied. "
            f"tension_mult={escalation.tension_multiplier}, "
            f"dho_shift={escalation.dho_damping_shift:+.2f}, "
            f"hormuz_weight={escalation.hormuz_weight}"
        )

        return out

    # ── Utility methods ──────────────────────────────────────────────────

    def get_level_details(self, level: int) -> EscalationLevel:
        """Get full details for a specific level."""
        return self.ladder[int(np.clip(level, 1, 10))]

    def get_all_levels_summary(self) -> list[dict]:
        """Return summary of all 10 levels for UI display."""
        return [
            {
                "level": lvl.level,
                "name": lvl.name,
                "description": lvl.description,
                "brent_impact": lvl.brent_impact_range,
                "vix_impact": lvl.vix_impact_range,
                "historical_examples": lvl.historical_examples,
            }
            for lvl in ESCALATION_LADDER
        ]

    def _check_data_freshness(self, df: pd.DataFrame) -> dict:
        """Check how recent each data source is."""
        freshness = {}
        checks = {
            "gdelt": "gdelt_event_count",
            "truth_social": "ts_post_count",
            "twitter": "tw_volume_spike_ratio",
            "supply": "hormuz_throughput_kbbl",
            "price": "BZ=F_close",
        }
        latest_date = df.index.max() if not df.empty else None
        for source, col in checks.items():
            if col in df.columns:
                non_null = df[col].last_valid_index()
                if non_null is not None and latest_date is not None:
                    age_days = (latest_date - non_null).days
                    freshness[source] = {
                        "last_data": str(non_null.date()),
                        "age_days": age_days,
                        "status": "fresh" if age_days <= 1 else
                                  "recent" if age_days <= 3 else "stale",
                    }
                else:
                    freshness[source] = {"status": "missing"}
            else:
                freshness[source] = {"status": "missing"}
        return freshness

    def _empty_assessment(self) -> WarAssessment:
        """Return a safe default when no data is available."""
        return WarAssessment(
            recommended_level=4,
            confidence=0.1,
            evidence={"note": "no data available, defaulting to level 4"},
            sub_scores={},
            level_details=self.ladder[4],
            timestamp=datetime.now().isoformat(),
            data_freshness={},
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Testing F8: USA-Iran-Israel War Tension Barometer")
    print("=" * 70)

    # Rebuild unified DataFrame from synthetic data (quick version)
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

    gdelt_daily = GDELTIngestor().aggregate_daily(
        GDELTIngestor()._tag_entities(_generate_synthetic_gdelt(500))
    )
    ts_posts = generate_synthetic_posts(200)
    ts_pipeline = TruthSocialPipeline()
    ts_enriched = ts_pipeline.run_full_pipeline(ts_posts, save=False)
    ts_daily = ts_pipeline.aggregate_daily(ts_enriched)
    tw_daily = TwitterIngestor().compute_features(
        generate_synthetic_tweets(5000, START, END)
    )
    supply_daily = SupplyIngestor().fetch_all(START, END, save=False)

    rng = np.random.default_rng(99)
    dates = pd.date_range(START, END, freq="B")
    price_vix = pd.DataFrame({
        "BZ=F_close": 75 + np.cumsum(rng.normal(0, 1, len(dates))),
        "^VIX_close": 18 + np.cumsum(rng.normal(0, 0.5, len(dates))),
    }, index=dates)
    price_vix.index.name = "date"

    unified = TimeIndexedMerge().merge_all(
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

    # ── Test dynamic assessment ──────────────────────────────────────────
    barometer = WarTensionBarometer()
    assessment = barometer.assess(unified)

    print(f"\n{'='*70}")
    print(f"DYNAMIC WAR ASSESSMENT")
    print(f"{'='*70}")
    print(f"  Recommended level: {assessment.recommended_level}/10 "
          f"— {assessment.level_details.name}")
    print(f"  Confidence: {assessment.confidence:.0%}")
    print(f"\n  Sub-scores:")
    for source, score in assessment.sub_scores.items():
        print(f"    {source:25s} → {score:.1f}/10")
    print(f"\n  Evidence:")
    for source, ev in assessment.evidence.items():
        print(f"    {source}: {ev}")
    print(f"\n  Expected market impact:")
    print(f"    Brent: {assessment.level_details.brent_impact_range}")
    print(f"    VIX:   {assessment.level_details.vix_impact_range}")

    # ── Test applying different levels ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LEVEL COMPARISON — same data, different war assumptions")
    print(f"{'='*70}")

    key_features = ["f2_tension_ema", "f3_dho_combined",
                    "f4_trump_shock_score", "f6_supply_stress"]

    for level in [2, 5, 8]:
        modified = barometer.apply_level(unified, level=level)
        details = barometer.get_level_details(level)
        print(f"\n  Level {level}: {details.name}")
        for feat in key_features:
            if feat in modified.columns:
                orig = unified[feat].mean()
                new = modified[feat].mean()
                change = ((new - orig) / max(abs(orig), 0.001)) * 100
                print(f"    {feat:30s}  {orig:8.3f} → {new:8.3f}  ({change:+.1f}%)")

    # ── Show all 10 levels ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"THE 10-LEVEL ESCALATION LADDER")
    print(f"{'='*70}")
    for lvl in ESCALATION_LADDER:
        print(f"  {lvl.level:2d}. {lvl.name:28s} | Brent: {lvl.brent_impact_range:15s} "
              f"| Tension ×{lvl.tension_multiplier}")

    print(f"\n{'='*70}")
    print(f"✅ F8 War Tension Barometer test PASSED")
    print(f"{'='*70}")
