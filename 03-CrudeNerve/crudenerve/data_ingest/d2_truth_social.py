"""
D2: Truth Social scraper + NLP pipeline.

Three layers in one module:
  D2     — Raw post ingestion (Selenium scraper or API proxy)
  D2-NLP — Multi-label topic classification (12 sub-topics)
  D2-SEV — Three-axis severity scoring (urgency × extremity × targeting)

Usage:
    from crudenerve.data_ingest.d2_truth_social import TruthSocialPipeline
    pipeline = TruthSocialPipeline()
    df = pipeline.run_full_pipeline(posts_df)  # classify + score
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from crudenerve.config.settings import TruthSocialConfig, RAW_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER D2: Raw scraper / ingestion
# ═══════════════════════════════════════════════════════════════════════════

class TruthSocialScraper:
    """
    Scrapes Trump's Truth Social posts via Selenium.

    In production, this uses headless Chrome + Selenium to poll
    truthsocial.com/@realDonaldTrump at configured intervals.
    For development/testing, loads from cached JSON or synthetic data.
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()
        self.save_dir = RAW_DIR / "truth_social"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def scrape_latest(self) -> pd.DataFrame:
        """
        Scrape most recent posts from Truth Social.

        Returns DataFrame with columns:
            post_id, text, timestamp_utc, media_flag,
            retruth_count_0h, reply_count_0h
        """
        # Selenium implementation goes here in production.
        # For now, raise NotImplementedError with clear instructions.
        raise NotImplementedError(
            "Live scraping requires Selenium + Chrome driver setup. "
            "Use load_cached() or generate_synthetic() for development."
        )

    def load_cached(self) -> pd.DataFrame | None:
        path = self.save_dir / "truth_social_posts.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# LAYER D2-NLP: Multi-label topic classification
# ═══════════════════════════════════════════════════════════════════════════

# Keyword dictionaries for rule-based MVP classifier.
# In production, replace with fine-tuned DistilBERT.

TOPIC_KEYWORDS: dict[str, list[str]] = {
    # ENERGY parent
    "oil_price_direct": [
        "drill", "drilling", "oil", "gas", "gasoline", "energy",
        "opec", "barrel", "petroleum", "pipeline", "lng", "fuel",
        "price", "prices",
    ],
    "sanctions_iran": [
        "iran", "iranian", "sanctions", "nuclear", "enrichment",
        "tehran", "khamenei", "irgc", "raisi", "pezeshkian",
    ],
    "sanctions_russia": [
        "russia", "russian", "putin", "gazprom", "nord stream",
        "pipeline", "sanctions",
    ],
    "energy_policy": [
        "epa", "regulation", "permit", "permits", "keystone",
        "clean energy", "coal", "renewable", "climate",
    ],
    # MILITARY parent
    "iran_military": [
        "iran", "strike", "military", "navy", "hormuz", "irgc",
        "missile", "drone", "attack", "retaliate",
    ],
    "israel_conflict": [
        "israel", "israeli", "gaza", "hamas", "hezbollah",
        "netanyahu", "idf", "lebanon", "west bank",
    ],
    "general_threat": [
        "military", "troops", "defense", "army", "war",
        "nato", "deploy", "combat", "threat",
    ],
    # TRADE parent
    "tariff_china": [
        "china", "chinese", "tariff", "tariffs", "trade war",
        "beijing", "xi", "jinping", "import", "duty",
    ],
    "tariff_general": [
        "tariff", "tariffs", "import", "duty", "trade",
        "eu", "europe", "canada", "mexico",
    ],
    "trade_deal": [
        "deal", "agreement", "negotiate", "negotiation",
        "bilateral", "trade deal", "partnership",
    ],
    # DOMESTIC parent
    "fed_monetary": [
        "fed", "federal reserve", "interest rate", "powell",
        "inflation", "monetary", "basis points",
    ],
    "fiscal_policy": [
        "tax", "taxes", "spending", "budget", "debt ceiling",
        "deficit", "fiscal",
    ],
    "election_rhetoric": [
        "vote", "election", "ballot", "campaign", "democrat",
        "republican", "biden", "maga", "rally",
    ],
}

# Parent category mapping
TOPIC_PARENTS: dict[str, str] = {
    "oil_price_direct": "ENERGY",
    "sanctions_iran": "ENERGY",
    "sanctions_russia": "ENERGY",
    "energy_policy": "ENERGY",
    "iran_military": "MILITARY",
    "israel_conflict": "MILITARY",
    "general_threat": "MILITARY",
    "tariff_china": "TRADE",
    "tariff_general": "TRADE",
    "trade_deal": "TRADE",
    "fed_monetary": "DOMESTIC",
    "fiscal_policy": "DOMESTIC",
    "election_rhetoric": "DOMESTIC",
}


class TopicClassifier:
    """
    Layer D2-NLP: Multi-label topic classification for Trump posts.

    MVP: keyword-based scoring with TF-IDF-like normalization.
    Production: fine-tuned DistilBERT multi-label classifier.

    Each post gets a probability vector across all 12 sub-topics,
    NOT a single label. A post about "Iran sanctions and oil" scores
    on both sanctions_iran AND oil_price_direct.
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()
        self.keywords = TOPIC_KEYWORDS
        self.parents = TOPIC_PARENTS

    def classify(self, text: str) -> dict[str, float]:
        """
        Score a single post against all 12 sub-topics.

        Returns dict like:
            {"sanctions_iran": 0.72, "oil_price_direct": 0.45, ...}

        Scores are normalized to [0, 1] range within each post.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        scores: dict[str, float] = {}
        for topic, kw_list in self.keywords.items():
            # Count keyword hits, weighting multi-word phrases higher
            hits = 0
            for kw in kw_list:
                if " " in kw:
                    # Multi-word phrase — exact substring match
                    if kw in text_lower:
                        hits += 2  # phrases are stronger signals
                else:
                    if kw in words:
                        hits += 1
            # Normalize by keyword list length to avoid bias toward
            # topics with more keywords
            scores[topic] = hits / max(len(kw_list), 1)

        # Rescale so max score per post = 1.0
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: round(v / max_score, 3) for k, v in scores.items()}
        else:
            scores = {k: 0.0 for k in scores}

        return scores

    def classify_batch(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Add topic probability columns to a DataFrame of posts.

        Adds columns: topic_sanctions_iran, topic_oil_price_direct, etc.
        Also adds: topic_parent (highest-scoring parent category),
                   topic_max_score (confidence of top topic),
                   is_relevant (any non-domestic topic > threshold).
        """
        topic_scores = df[text_col].apply(self.classify)
        topic_df = pd.DataFrame(topic_scores.tolist(), index=df.index)
        topic_df.columns = [f"topic_{c}" for c in topic_df.columns]

        # Find dominant sub-topic and parent
        raw_scores = pd.DataFrame(topic_scores.tolist(), index=df.index)
        topic_df["topic_dominant"] = raw_scores.idxmax(axis=1)
        topic_df["topic_max_score"] = raw_scores.max(axis=1)
        topic_df["topic_parent"] = topic_df["topic_dominant"].map(self.parents)

        # Relevance flag: any energy/military/trade topic above threshold
        relevant_topics = [
            c for c in raw_scores.columns
            if self.parents.get(c) != "DOMESTIC"
        ]
        topic_df["is_relevant"] = (
            raw_scores[relevant_topics].max(axis=1)
            > self.cfg.topic_relevance_threshold
        )

        return pd.concat([df, topic_df], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER D2-SEV: Three-axis severity scoring
# ═══════════════════════════════════════════════════════════════════════════

# Keyword dictionaries for each severity axis at each level.

URGENCY_PATTERNS: dict[int, list[str]] = {
    1: ["should", "could", "might", "would", "may", "will consider",
        "looking at", "thinking about"],
    2: ["soon", "very soon", "in the coming", "shortly",
        "in the near future", "not long"],
    3: ["tomorrow", "monday", "tuesday", "wednesday", "thursday",
        "friday", "this week", "next week", "by end of"],
    4: ["right now", "as we speak", "effective immediately",
        "starting today", "at this moment", "currently"],
    5: ["i have just", "we have", "it is done", "just signed",
        "just announced", "completed", "executed", "finished"],
}

EXTREMITY_PATTERNS: dict[int, list[str]] = {
    1: ["review", "look at", "consider", "study", "examine",
        "evaluate", "assess", "monitor"],
    2: ["increase", "strengthen", "expand", "improve", "enhance",
        "boost", "raise"],
    3: ["maximum", "all", "total", "complete", "full", "massive",
        "enormous", "huge", "major"],
    4: ["destroy", "obliterate", "end", "crush", "eliminate",
        "devastate", "annihilate", "wipe out"],
    5: ["like never before", "biggest ever", "the likes of which",
        "never seen", "unprecedented", "most powerful in history",
        "strongest ever"],
}

TARGETING_PATTERNS: dict[int, list[str]] = {
    1: ["bad actors", "our enemies", "those who", "anyone who",
        "certain countries", "some people"],
    2: ["middle east", "asia", "europe", "region", "area",
        "gulf", "pacific"],
    3: ["iran", "china", "russia", "north korea", "venezuela",
        "turkey", "iraq", "syria"],
    4: ["irgc", "petrochina", "huawei", "hamas", "hezbollah",
        "rosneft", "opec", "houthis"],
    5: ["khamenei", "xi", "putin", "kim", "maduro", "erdogan",
        "netanyahu", "specific named official"],
}


class SeverityScorer:
    """
    Layer D2-SEV: Three-axis severity analysis.

    Each post gets three independent scores:
        urgency:   1-5 (is this talk or action?)
        extremity: 1-5 (mild or total?)
        targeting:  1-5 (vague or naming names?)

    Combined severity = urgency×0.4 + extremity×0.35 + targeting×0.25
    Output is continuous 1.0 to 5.0.
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()

    def _score_axis(
        self,
        text: str,
        patterns: dict[int, list[str]],
    ) -> int:
        """Score text against one severity axis. Returns highest matching level."""
        text_lower = text.lower()
        best = 1  # default baseline
        for level in sorted(patterns.keys(), reverse=True):
            for phrase in patterns[level]:
                if phrase in text_lower:
                    return level  # highest matching level wins
        return best

    def score(self, text: str) -> dict[str, float]:
        """
        Score a single post on all three axes.

        Returns:
            {
                "urgency": 3,
                "extremity": 4,
                "targeting": 5,
                "severity_combined": 3.85,
            }
        """
        urgency = self._score_axis(text, URGENCY_PATTERNS)
        extremity = self._score_axis(text, EXTREMITY_PATTERNS)
        targeting = self._score_axis(text, TARGETING_PATTERNS)

        combined = (
            urgency * self.cfg.severity_weight_urgency
            + extremity * self.cfg.severity_weight_extremity
            + targeting * self.cfg.severity_weight_targeting
        )

        return {
            "urgency": urgency,
            "extremity": extremity,
            "targeting": targeting,
            "severity_combined": round(combined, 2),
        }

    def score_batch(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Add severity columns to a DataFrame of posts."""
        severity_scores = df[text_col].apply(self.score)
        severity_df = pd.DataFrame(severity_scores.tolist(), index=df.index)
        severity_df.columns = [f"sev_{c}" for c in severity_df.columns]
        return pd.concat([df, severity_df], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TruthSocialPipeline:
    """
    Runs D2 → D2-NLP → D2-SEV in sequence.

    Uses the production NLP engine (sentence-transformers + VADER ensemble)
    when available, falls back to keyword classifiers if transformers
    can't be loaded.

    Input:  DataFrame with at least 'text' and 'timestamp_utc' columns
    Output: Same DataFrame enriched with topic probabilities + severity scores
    """

    def __init__(self, config: TruthSocialConfig | None = None):
        self.cfg = config or TruthSocialConfig()
        self.scraper = TruthSocialScraper(self.cfg)
        self.save_dir = RAW_DIR / "truth_social"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Try to load production NLP engine; fall back to keyword classifiers
        self._nlp_engine = None
        self._use_transformers = False
        try:
            from crudenerve.utils.nlp_engine import NLPEngine
            self._nlp_engine = NLPEngine(
                relevance_threshold=self.cfg.topic_relevance_threshold
            )
            self._use_transformers = True
            logger.info("TruthSocialPipeline: using transformer NLP engine")
        except Exception as e:
            logger.warning(f"Transformer NLP unavailable ({e}), using keyword fallback")
            self.classifier = TopicClassifier(self.cfg)
            self.scorer = SeverityScorer(self.cfg)

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Run topic classification + severity scoring on posts DataFrame.

        If transformer NLP engine is available:
            - Single embedding pass per post (efficient)
            - Semantic topic classification via cosine similarity
            - Negation-aware severity scoring
            - VADER + transformer sentiment ensemble
            - Entity extraction with negation context

        Fallback:
            - Keyword-based topic classification
            - Pattern-based severity scoring
            - No sentiment, no entity extraction, no negation handling
        """
        logger.info(f"Running Truth Social NLP pipeline on {len(df)} posts "
                    f"(engine={'transformer' if self._use_transformers else 'keyword'})")

        if self._use_transformers and self._nlp_engine is not None:
            df = self._run_transformer_pipeline(df)
        else:
            df = self._run_keyword_pipeline(df)

        # Derived convenience columns (same for both paths)
        df["is_high_severity"] = df["sev_severity_combined"] >= 4.0
        df["is_actionable"] = df["is_relevant"] & df["is_high_severity"]

        n_relevant = df["is_relevant"].sum()
        n_high = df["is_high_severity"].sum()
        n_actionable = df["is_actionable"].sum()
        logger.info(
            f"Pipeline complete: {n_relevant} relevant, {n_high} high-severity, "
            f"{n_actionable} actionable out of {len(df)} posts"
        )

        if save:
            out_path = self.save_dir / "truth_social_enriched.parquet"
            # Drop embedding column before saving (not serializable to parquet)
            save_df = df.drop(columns=["embedding"], errors="ignore")
            save_df.to_parquet(out_path)
            logger.info(f"Saved enriched posts → {out_path}")

        return df

    def _run_transformer_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full transformer-based NLP analysis."""
        texts = df["text"].tolist()
        results = self._nlp_engine.analyze_batch(texts, show_progress=False)

        # Convert NLPResults to DataFrame columns
        nlp_rows = [self._nlp_engine.result_to_series(r) for r in results]
        nlp_df = pd.DataFrame(nlp_rows, index=df.index)

        # Merge NLP columns into the posts DataFrame
        out = pd.concat([df, nlp_df], axis=1)

        # Store embeddings for potential downstream use (F4-ENG cross-platform echo)
        out["embedding"] = [r.embedding for r in results]

        return out

    def _run_keyword_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: keyword-based topic + severity classification."""
        df = self.classifier.classify_batch(df)
        df = self.scorer.score_batch(df)
        return df

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate enriched posts to daily features for F4 consumption.

        Output columns:
            - ts_post_count: total posts that day
            - ts_relevant_count: market-relevant posts
            - ts_max_severity: highest severity score that day
            - ts_mean_severity: average severity of relevant posts
            - ts_dominant_topic: most frequent relevant topic
            - ts_actionable_count: high-severity + relevant posts
            - ts_topic_*_max: max score per topic that day
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.date
        df["date"] = pd.to_datetime(df["date"])

        # Basic daily aggregation
        daily = df.groupby("date").agg(
            ts_post_count=("text", "count"),
            ts_relevant_count=("is_relevant", "sum"),
            ts_max_severity=("sev_severity_combined", "max"),
            ts_actionable_count=("is_actionable", "sum"),
        )

        # Mean severity of relevant posts only
        relevant = df[df["is_relevant"]].groupby("date")
        if len(relevant) > 0:
            daily["ts_mean_severity_relevant"] = relevant[
                "sev_severity_combined"
            ].mean()
        else:
            daily["ts_mean_severity_relevant"] = 0.0

        # Dominant topic per day
        if "topic_dominant" in df.columns:
            relevant_posts = df[df["is_relevant"]]
            if not relevant_posts.empty:
                dominant = relevant_posts.groupby("date")["topic_dominant"].agg(
                    lambda x: x.mode().iloc[0] if len(x) > 0 else "none"
                )
                daily["ts_dominant_topic"] = dominant

        # Per-topic max scores
        topic_cols = [c for c in df.columns if c.startswith("topic_") and c not in [
            "topic_dominant", "topic_max_score", "topic_parent",
        ]]
        for col in topic_cols:
            daily[f"ts_{col}_max"] = df.groupby("date")[col].max()

        # Severity axes daily max
        for axis in ["urgency", "extremity", "targeting"]:
            col = f"sev_{axis}"
            if col in df.columns:
                daily[f"ts_{axis}_max"] = df.groupby("date")[col].max()

        # Fill NaN: numeric columns with 0, string columns with "none"
        numeric_cols = daily.select_dtypes(include="number").columns
        daily[numeric_cols] = daily[numeric_cols].fillna(0)
        string_cols = daily.select_dtypes(include=["object", "string"]).columns
        if len(string_cols) > 0:
            daily[string_cols] = daily[string_cols].fillna("none")
        daily.index.name = "date"
        return daily


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR (for testing without scraper)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_posts(n: int = 200) -> pd.DataFrame:
    """
    Generate realistic synthetic Trump Truth Social posts for testing.

    Mixes high-severity geopolitical posts with low-severity noise
    to simulate the real distribution (most posts are noise).
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")

    # Template pools by severity band
    high_severity = [
        "I have just imposed the STRONGEST sanctions in history on Iran. "
        "Their nuclear program will be CRUSHED. Effective immediately!",

        "Just spoke with Netanyahu. The United States stands with Israel "
        "against Iran and Hezbollah. We will DESTROY any threat to our "
        "allies. Right now!",

        "OPEC thinks they can control oil prices? WRONG. I am ordering "
        "maximum drilling on all federal lands. Drill Baby Drill! Starting today!",

        "Iran's Khamenei made a very bad mistake. The United States military "
        "is the most powerful in history. He will find out very soon what "
        "happens when you threaten us.",

        "Just signed the biggest tariff increase on China EVER. 60% on all "
        "Chinese goods. The likes of which the world has never seen! "
        "Effective immediately!",
    ]

    medium_severity = [
        "We are looking very strongly at Iran and their nuclear program. "
        "Major sanctions coming soon. Very soon!",

        "Oil prices are too high. OPEC must increase production or there "
        "will be consequences. I will expand drilling enormously.",

        "China is ripping us off on trade. We will strengthen our tariff "
        "position. Major announcement this week!",

        "The situation in the Middle East is very serious. Our military "
        "is monitoring everything. Iran should be very careful.",

        "Energy independence is our goal. We will review all EPA "
        "regulations that slow down drilling. Big changes coming!",
    ]

    low_severity = [
        "Great rally last night in Ohio! Biggest crowd ever. MAGA!",
        "The Fake News media is at it again. So dishonest!",
        "Our economy is the strongest in the world. Thank you!",
        "Happy Birthday to our amazing First Lady!",
        "Just had a wonderful meeting with supporters. America First!",
        "The Democrats have no idea what they are doing. SAD!",
        "Congratulations to the Kansas City Chiefs!",
        "Looking forward to a great week ahead for our country.",
    ]

    records = []
    for _ in range(n):
        day = rng.choice(dates)
        hour = rng.integers(6, 24)  # post during waking hours

        # Distribution: ~15% high, ~25% medium, ~60% low (noise)
        r = rng.random()
        if r < 0.15:
            text = rng.choice(high_severity)
        elif r < 0.40:
            text = rng.choice(medium_severity)
        else:
            text = rng.choice(low_severity)

        records.append({
            "post_id": f"ts_{rng.integers(1_000_000):07d}",
            "text": text,
            "timestamp_utc": day + pd.Timedelta(hours=int(hour),
                                                 minutes=int(rng.integers(0, 60))),
            "media_flag": bool(rng.random() > 0.7),
            "retruth_count_0h": int(rng.integers(100, 50000)),
            "reply_count_0h": int(rng.integers(50, 20000)),
        })

    df = pd.DataFrame(records)
    df.sort_values("timestamp_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Testing D2 + D2-NLP + D2-SEV pipeline with synthetic data")
    print("=" * 70)

    # Generate synthetic posts
    posts = generate_synthetic_posts(200)
    print(f"\nGenerated {len(posts)} synthetic posts")
    print(f"Date range: {posts['timestamp_utc'].min()} → {posts['timestamp_utc'].max()}")

    # Run full pipeline
    pipeline = TruthSocialPipeline()
    enriched = pipeline.run_full_pipeline(posts, save=False)

    # Show results
    print(f"\n--- Topic Classification ---")
    print(f"Relevant posts: {enriched['is_relevant'].sum()}/{len(enriched)}")
    if "topic_parent" in enriched.columns:
        print(f"Parent category distribution:")
        print(enriched[enriched["is_relevant"]]["topic_parent"].value_counts())

    print(f"\n--- Severity Scoring ---")
    print(f"High severity (>=4.0): {enriched['is_high_severity'].sum()}")
    print(f"Actionable (relevant + high sev): {enriched['is_actionable'].sum()}")
    print(f"\nSeverity stats:")
    print(enriched["sev_severity_combined"].describe())

    print(f"\n--- Sample high-severity post ---")
    high = enriched[enriched["is_high_severity"]].iloc[0] if enriched["is_high_severity"].any() else None
    if high is not None:
        print(f"Text: {high['text'][:100]}...")
        print(f"Topic: {high.get('topic_dominant', 'N/A')} "
              f"({high.get('topic_max_score', 0):.2f})")
        print(f"Urgency: {high['sev_urgency']}, "
              f"Extremity: {high['sev_extremity']}, "
              f"Targeting: {high['sev_targeting']}")
        print(f"Combined: {high['sev_severity_combined']}")

    # Daily aggregation
    daily = pipeline.aggregate_daily(enriched)
    print(f"\n--- Daily Aggregation ---")
    print(f"Daily rows: {len(daily)}")
    print(f"\nSample:\n{daily.head()}")
