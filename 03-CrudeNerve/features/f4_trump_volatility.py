"""
F4: Trump Volatility Injection.

Combines all Truth Social layers into a composite feature set:
  F4-ENG:  Engagement velocity (re-Truth speed, media lag, cross-platform echo)
  F4-TEMP: Temporal patterns (burst detection, Friday-4pm, pre-announcement
           ramp, silence-after-storm)
  F4:      Composite Trump policy-shock score

This module reads from enriched Truth Social data (D2 → D2-NLP → D2-SEV)
and produces daily features for the unified DataFrame.

Usage:
    from crudenerve.features.f4_trump_volatility import TrumpVolatilityEngine
    engine = TrumpVolatilityEngine()
    df = engine.compute(unified_df, enriched_posts_df)
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from crudenerve.config.settings import TruthSocialConfig, TRUMP_DHO_TABLE

logger = logging.getLogger(__name__)


class EngagementVelocityComputer:
    """
    F4-ENG: Engagement velocity features.

    Tracks how fast and how far a Trump post propagates:
    1. Re-Truth velocity (first hour — early virality signal)
    2. Reply sentiment ratio (base enthusiasm vs mainstream pushback)
    3. Media amplification lag (time to Reuters/AP pickup)
    4. Cross-platform echo (Twitter spike within 2h of post)
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()

    def compute_post_level(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engagement velocity features at the post level.

        Input: enriched posts DataFrame from D2 pipeline with columns:
            retruth_count_0h, retruth_count_1h (if available),
            reply_count_0h, reply_sentiment (if available)

        Output: same DataFrame with added f4eng_* columns.
        """
        out = posts_df.copy()

        # Re-Truth velocity (1-hour growth rate)
        if "retruth_count_0h" in out.columns:
            rt_0h = out["retruth_count_0h"].fillna(0).clip(lower=1)
            # If we have 1h snapshot, compute growth rate
            if "retruth_count_1h" in out.columns:
                rt_1h = out["retruth_count_1h"].fillna(0)
                out["f4eng_retruth_velocity"] = (rt_1h - rt_0h) / rt_0h
            else:
                # Proxy: use 0h count normalized by median as virality signal
                median_rt = rt_0h.median()
                out["f4eng_retruth_velocity"] = (rt_0h / max(median_rt, 1)) - 1

        # Reply sentiment ratio (if available; synthetic placeholder otherwise)
        if "reply_sentiment" in out.columns:
            out["f4eng_reply_sentiment_ratio"] = out["reply_sentiment"]
        else:
            # Proxy: high-severity posts tend to get more polarized replies
            if "sev_severity_combined" in out.columns:
                sev = out["sev_severity_combined"].fillna(1)
                # Higher severity → more negative reply sentiment
                out["f4eng_reply_sentiment_ratio"] = 1 - (sev / 5.0)
            else:
                out["f4eng_reply_sentiment_ratio"] = 0.5

        # Engagement intensity score (composite)
        rt_vel = out.get("f4eng_retruth_velocity", pd.Series(0, index=out.index))
        reply_ratio = out.get("f4eng_reply_sentiment_ratio", pd.Series(0.5, index=out.index))
        out["f4eng_intensity"] = (
            rt_vel.clip(-2, 10).fillna(0) * 0.6
            + (1 - reply_ratio.fillna(0.5)) * 0.4  # negative replies = more intensity
        )

        return out

    def aggregate_daily(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate post-level engagement to daily features."""
        if posts_df.empty:
            return pd.DataFrame()

        df = posts_df.copy()
        df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.normalize()

        eng_cols = [c for c in df.columns if c.startswith("f4eng_")]
        if not eng_cols:
            return pd.DataFrame()

        agg_dict = {}
        for col in eng_cols:
            agg_dict[f"{col}_max"] = (col, "max")
            agg_dict[f"{col}_mean"] = (col, "mean")

        daily = df.groupby("date").agg(**agg_dict)
        daily.index.name = "date"
        return daily


class TemporalPatternDetector:
    """
    F4-TEMP: Temporal pattern features.

    Detects behavioral signatures in Trump's posting patterns
    that predict major policy announcements:

    1. Burst detection: >3 posts on same sub-topic within 6h
       (repeating himself = committed, not venting)
    2. Friday-4pm flag: policy posts after market close on Friday
       (thin markets, weekend to amplify)
    3. Pre-announcement ramp: increasing severity over 48h on same topic
       (telegraphing a major move)
    4. Silence-after-storm: >12h gap after severity ≥4 post
       (action happening behind the scenes)
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()

    def detect_patterns(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect temporal patterns at the post level.

        Adds boolean/float pattern columns to each post.
        """
        out = posts_df.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"])
        out.sort_values("timestamp_utc", inplace=True)

        # ── Pattern 1: Burst detection ───────────────────────────────────
        out["f4temp_is_burst"] = False
        if "topic_dominant" in out.columns:
            window_hours = self.cfg.burst_window_hours
            threshold = self.cfg.burst_count_threshold

            for i, row in out.iterrows():
                topic = row.get("topic_dominant", "none")
                if topic == "none" or topic == "election_rhetoric":
                    continue
                ts = row["timestamp_utc"]
                window_start = ts - timedelta(hours=window_hours)

                same_topic_mask = (
                    (out["timestamp_utc"] >= window_start)
                    & (out["timestamp_utc"] <= ts)
                    & (out["topic_dominant"] == topic)
                )
                if same_topic_mask.sum() >= threshold:
                    out.loc[i, "f4temp_is_burst"] = True

        # ── Pattern 2: Friday-4pm ────────────────────────────────────────
        # Posts after 4pm ET on Friday (markets thin, impact amplified)
        utc_offset = 5  # ET = UTC-5 (approx; ignoring DST for now)
        et_hour = out["timestamp_utc"].dt.hour - utc_offset
        et_weekday = out["timestamp_utc"].dt.weekday  # 4 = Friday

        out["f4temp_friday_4pm"] = (
            (et_weekday == 4)
            & (et_hour >= self.cfg.friday_cutoff_hour_et)
            & out.get("is_relevant", pd.Series(True, index=out.index))
        )

        # ── Pattern 3: Pre-announcement ramp ─────────────────────────────
        # Increasing severity on same topic over 48h window
        out["f4temp_pre_announcement"] = False
        if "sev_severity_combined" in out.columns and "topic_dominant" in out.columns:
            window_h = self.cfg.pre_announcement_window_hours

            for i, row in out.iterrows():
                topic = row.get("topic_dominant", "none")
                if topic == "none":
                    continue

                ts = row["timestamp_utc"]
                sev = row.get("sev_severity_combined", 0)
                window_start = ts - timedelta(hours=window_h)

                prior_mask = (
                    (out["timestamp_utc"] >= window_start)
                    & (out["timestamp_utc"] < ts)
                    & (out["topic_dominant"] == topic)
                )
                prior_posts = out.loc[prior_mask]

                if len(prior_posts) >= 2:
                    prior_sevs = prior_posts["sev_severity_combined"].values
                    # Check if severity is monotonically increasing
                    if all(prior_sevs[j] <= prior_sevs[j + 1]
                           for j in range(len(prior_sevs) - 1)):
                        if sev > prior_sevs[-1]:
                            out.loc[i, "f4temp_pre_announcement"] = True

        # ── Pattern 4: Silence-after-storm ───────────────────────────────
        out["f4temp_silence_after_storm"] = False
        if "sev_severity_combined" in out.columns:
            silence_h = self.cfg.silence_threshold_hours
            min_sev = self.cfg.silence_min_severity

            for i in range(len(out) - 1):
                row = out.iloc[i]
                sev = row.get("sev_severity_combined", 0)
                if sev >= min_sev:
                    ts = row["timestamp_utc"]
                    next_ts = out.iloc[i + 1]["timestamp_utc"]
                    gap_hours = (next_ts - ts).total_seconds() / 3600
                    if gap_hours > silence_h:
                        out.iloc[i, out.columns.get_loc("f4temp_silence_after_storm")] = True

        return out

    def aggregate_daily(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate temporal pattern detections to daily features."""
        if posts_df.empty:
            return pd.DataFrame()

        df = posts_df.copy()
        df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.normalize()

        pattern_cols = [c for c in df.columns if c.startswith("f4temp_")]
        if not pattern_cols:
            return pd.DataFrame()

        daily = df.groupby("date")[pattern_cols].agg("sum")
        daily.columns = [f"{c}_count" for c in daily.columns]
        daily.index.name = "date"

        # Binary flags: any pattern detected that day
        for col in pattern_cols:
            daily[f"{col}_flag"] = (daily[f"{col}_count"] > 0).astype(int)

        return daily


class TrumpVolatilityEngine:
    """
    F4: Master Trump volatility injection engine.

    Orchestrates F4-ENG and F4-TEMP, then computes the composite
    Trump policy-shock score used by M2 and M3.
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()
        self.engagement = EngagementVelocityComputer(self.cfg)
        self.temporal = TemporalPatternDetector(self.cfg)

    def compute(
        self,
        unified_df: pd.DataFrame,
        enriched_posts: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute all Trump volatility features.

        If enriched_posts is provided, computes engagement velocity
        and temporal patterns from post-level data. Otherwise uses
        only the daily-aggregated Truth Social features already in
        the unified DataFrame.

        Adds columns:
            - f4_trump_shock_score:   composite daily policy-shock score
            - f4_trump_shock_72h:     72h trailing shock accumulation
            - f4_trump_escalation:    binary flag for escalation state
            - f4eng_* :               engagement velocity features
            - f4temp_* :              temporal pattern features
        """
        out = unified_df.copy()

        # ── Process post-level features if available ─────────────────────
        if enriched_posts is not None and not enriched_posts.empty:
            # F4-ENG: engagement velocity
            posts_with_eng = self.engagement.compute_post_level(enriched_posts)
            eng_daily = self.engagement.aggregate_daily(posts_with_eng)

            # F4-TEMP: temporal patterns
            posts_with_patterns = self.temporal.detect_patterns(posts_with_eng)
            temp_daily = self.temporal.aggregate_daily(posts_with_patterns)

            # Merge into unified
            if not eng_daily.empty:
                out = out.join(eng_daily, how="left")
            if not temp_daily.empty:
                out = out.join(temp_daily, how="left")

            logger.info(
                f"Post-level features: {eng_daily.shape[1]} engagement + "
                f"{temp_daily.shape[1]} temporal columns"
            )

        # ── Composite Trump policy-shock score ───────────────────────────
        # Combines severity, engagement, and temporal signals into
        # a single daily score.

        shock_components = pd.DataFrame(index=out.index)

        # Component 1: Max severity that day (0-5 scale, weight 0.4)
        severity = out.get("ts_max_severity", pd.Series(0, index=out.index)).fillna(0)
        shock_components["sev"] = severity / 5.0  # normalize to 0-1

        # Component 2: Relevant post count (log-scaled, weight 0.15)
        relevant = out.get("ts_relevant_count", pd.Series(0, index=out.index)).fillna(0)
        shock_components["volume"] = np.log1p(relevant) / np.log1p(10)  # normalize

        # Component 3: Engagement intensity (weight 0.15)
        eng_intensity = out.get(
            "f4eng_intensity_max",
            pd.Series(0, index=out.index),
        ).fillna(0)
        shock_components["engagement"] = eng_intensity.clip(0, 3) / 3.0

        # Component 4: Temporal pattern strength (weight 0.15)
        pattern_flags = [c for c in out.columns if c.startswith("f4temp_") and c.endswith("_flag")]
        if pattern_flags:
            shock_components["patterns"] = out[pattern_flags].sum(axis=1) / max(len(pattern_flags), 1)
        else:
            shock_components["patterns"] = 0.0

        # Component 5: Actionable post flag (weight 0.15)
        actionable = out.get("ts_actionable_count", pd.Series(0, index=out.index)).fillna(0)
        shock_components["actionable"] = (actionable > 0).astype(float)

        # Weighted composite
        weights = {
            "sev": 0.40,
            "volume": 0.15,
            "engagement": 0.15,
            "patterns": 0.15,
            "actionable": 0.15,
        }
        out["f4_trump_shock_score"] = sum(
            shock_components[k] * w for k, w in weights.items()
        )

        # 72h trailing accumulation (captures multi-day escalation)
        out["f4_trump_shock_72h"] = (
            out["f4_trump_shock_score"].rolling(3, min_periods=1).sum()
        )

        # Escalation flag: shock score above 0.5 for 2+ consecutive days
        above_threshold = (out["f4_trump_shock_score"] > 0.5).astype(int)
        consecutive = above_threshold.rolling(2, min_periods=1).min()
        out["f4_trump_escalation"] = consecutive.astype(int)

        # Log summary
        peak_shock = out["f4_trump_shock_score"].max()
        escalation_days = out["f4_trump_escalation"].sum()
        logger.info(
            f"Trump volatility: peak_shock={peak_shock:.3f}, "
            f"escalation_days={escalation_days}"
        )

        return out


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Testing F4: Trump Volatility Injection Engine")
    print("=" * 70)

    from crudenerve.data_ingest.d2_truth_social import (
        TruthSocialPipeline, generate_synthetic_posts,
    )

    # Generate and enrich posts
    posts = generate_synthetic_posts(200)
    pipeline = TruthSocialPipeline()
    enriched = pipeline.run_full_pipeline(posts, save=False)
    ts_daily = pipeline.aggregate_daily(enriched)
    print(f"\nEnriched posts: {len(enriched)}")
    print(f"Daily Truth Social rows: {len(ts_daily)}")

    # Build a minimal unified DataFrame
    dates = pd.date_range("2024-01-01", "2024-06-30", freq="D", name="date")
    unified = pd.DataFrame(index=dates).join(ts_daily, how="left")

    # Run F4
    engine = TrumpVolatilityEngine()
    result = engine.compute(unified, enriched_posts=enriched)

    f4_cols = [c for c in result.columns if c.startswith("f4")]
    print(f"\n--- F4 Features ({len(f4_cols)} columns) ---")
    for col in sorted(f4_cols):
        vals = result[col]
        if vals.dtype in ["float64", "int64", "float32", "int32"]:
            print(f"  {col:40s} mean={vals.mean():.3f}  max={vals.max():.3f}")

    print(f"\n--- Pattern detections ---")
    burst_cols = [c for c in f4_cols if "burst" in c]
    friday_cols = [c for c in f4_cols if "friday" in c]
    preann_cols = [c for c in f4_cols if "pre_announcement" in c]
    silence_cols = [c for c in f4_cols if "silence" in c]

    print(f"  Burst days: {sum(result.get(c, pd.Series(0)).sum() for c in burst_cols if 'flag' in c):.0f}")
    print(f"  Friday-4pm days: {sum(result.get(c, pd.Series(0)).sum() for c in friday_cols if 'flag' in c):.0f}")
    print(f"  Pre-announcement ramps: {sum(result.get(c, pd.Series(0)).sum() for c in preann_cols if 'flag' in c):.0f}")
    print(f"  Silence-after-storm: {sum(result.get(c, pd.Series(0)).sum() for c in silence_cols if 'flag' in c):.0f}")

    print(f"\n--- Trump shock score ---")
    print(result["f4_trump_shock_score"].describe())
    print(f"\nEscalation days: {result['f4_trump_escalation'].sum()}")
    print(f"\nTop 5 shock days:")
    top5 = result.nlargest(5, "f4_trump_shock_score")
    print(top5[["ts_max_severity", "ts_relevant_count",
                "f4_trump_shock_score", "f4_trump_shock_72h",
                "f4_trump_escalation"]])
