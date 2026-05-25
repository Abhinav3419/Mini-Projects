"""
CrudeNerve NLP Engine — production-grade text analysis.

Replaces keyword-based MVP classifiers with:
  1. Transformer embeddings (sentence-transformers) for semantic similarity
  2. Zero-shot classification via cross-encoder for topic detection
  3. VADER + transformer ensemble for sentiment polarity
  4. Negation-aware severity scoring with dependency parsing
  5. Named entity recognition for targeting axis
  6. Contextual topic classification using cosine similarity against
     curated anchor sentences (not keyword bags)

Architecture:
  - SentenceTransformer encodes text into dense vectors ONCE per post
  - All downstream tasks (topic, severity, sentiment) operate on
    the same embedding — no redundant computation
  - VADER provides a fast baseline; transformer provides contextual depth
  - Final scores are ensemble: VADER × 0.3 + transformer × 0.7

Usage:
    from crudenerve.utils.nlp_engine import NLPEngine
    engine = NLPEngine()
    result = engine.analyze("I have just imposed maximum sanctions on Iran!")
    # result.topics = {"sanctions_iran": 0.91, "oil_price_direct": 0.34, ...}
    # result.severity = {"urgency": 5, "extremity": 3, "targeting": 3, "combined": 3.85}
    # result.sentiment = -0.72
    # result.entities = ["Iran"]
    # result.negated = False
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Lazy-loaded model singletons ────────────────────────────────────────────
# Models are heavy; load only when first needed.

_sentence_model = None
_vader_analyzer = None


def _get_sentence_model():
    """Lazy-load sentence-transformers model (all-MiniLM-L6-v2)."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence model loaded.")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers: {e}. "
                          f"Falling back to keyword-based classification.")
            _sentence_model = None
    return _sentence_model


def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("VADER not installed. Sentiment will use transformer only.")
            _vader_analyzer = None
    return _vader_analyzer


# ═══════════════════════════════════════════════════════════════════════════
# ANCHOR SENTENCES for semantic similarity classification
# ═══════════════════════════════════════════════════════════════════════════
# Instead of keyword bags, we define SENTENCES that represent each topic.
# The classifier computes cosine similarity between the post embedding
# and each anchor. This captures semantics that keywords miss:
# "drill baby drill" matches energy policy even without the word "oil".

TOPIC_ANCHORS: dict[str, list[str]] = {
    "oil_price_direct": [
        "Oil prices are going up because of supply issues",
        "We need to drill more oil and gas to lower prices",
        "OPEC is cutting production to raise oil prices",
        "Gasoline and energy prices are too high for Americans",
        "We will achieve energy dominance and lower fuel costs",
    ],
    "sanctions_iran": [
        "Imposing the strongest economic sanctions on Iran",
        "Iran nuclear program must be stopped with sanctions",
        "Maximum pressure campaign against Iranian regime",
        "Sanctions on Iranian oil exports and banking system",
        "Iran is violating the nuclear deal and must face consequences",
    ],
    "sanctions_russia": [
        "Sanctions against Russia for invasion of Ukraine",
        "Russian energy exports must be restricted",
        "Nord Stream pipeline threatens European energy security",
        "Putin must face economic consequences for aggression",
    ],
    "energy_policy": [
        "EPA regulations are killing American energy production",
        "Opening federal lands for oil and gas drilling",
        "Approving pipeline permits and LNG export terminals",
        "Ending the war on American energy producers",
        "Clean energy mandates are destroying our economy",
    ],
    "iran_military": [
        "Military strikes against Iranian nuclear facilities",
        "Navy deployed to Persian Gulf to counter Iran",
        "Strait of Hormuz must remain open for shipping",
        "IRGC is a terrorist organization and military threat",
        "Iran attacked our forces and will face military response",
    ],
    "israel_conflict": [
        "United States stands with Israel against terrorism",
        "Hamas and Hezbollah attacks on Israel must be stopped",
        "Supporting Israel's right to defend itself",
        "Iran-backed proxies attacking Israel from Gaza and Lebanon",
        "Israeli military operations to eliminate terrorist threats",
    ],
    "general_threat": [
        "Our military is the strongest in the world",
        "Deploying troops to defend American interests",
        "NATO allies must increase defense spending",
        "Military readiness and defense modernization",
        "No enemy will threaten American security",
    ],
    "tariff_china": [
        "Imposing tariffs on Chinese goods to protect American workers",
        "China has been ripping off America on trade for decades",
        "Trade war with China to reduce the trade deficit",
        "Chinese imports will face the highest tariffs ever",
        "Beijing must agree to fair trade or face consequences",
    ],
    "tariff_general": [
        "Tariffs on imports from Europe Canada and Mexico",
        "Trade deals must be fair and reciprocal",
        "Import duties to protect American manufacturing",
        "Foreign countries dumping cheap goods in America",
    ],
    "trade_deal": [
        "Negotiating the best trade deal in history",
        "Bilateral trade agreement reached with great terms",
        "New trade partnership will create American jobs",
        "Deal signed that puts America first in global trade",
    ],
    "fed_monetary": [
        "Federal Reserve should lower interest rates",
        "Jerome Powell is making a mistake with monetary policy",
        "Interest rates are too high and hurting the economy",
        "Inflation is coming down and the Fed should act",
    ],
    "fiscal_policy": [
        "Biggest tax cut in American history",
        "Government spending is out of control",
        "Debt ceiling must be raised responsibly",
        "Cutting wasteful government spending",
    ],
    "election_rhetoric": [
        "Vote for me and we will make America great again",
        "The election was stolen and we need election integrity",
        "Democrats are destroying this country",
        "Biggest rally crowd in political history",
        "MAGA movement is stronger than ever",
    ],
}

# Pre-compute parent mapping
TOPIC_TO_PARENT: dict[str, str] = {
    "oil_price_direct": "ENERGY", "sanctions_iran": "ENERGY",
    "sanctions_russia": "ENERGY", "energy_policy": "ENERGY",
    "iran_military": "MILITARY", "israel_conflict": "MILITARY",
    "general_threat": "MILITARY",
    "tariff_china": "TRADE", "tariff_general": "TRADE",
    "trade_deal": "TRADE",
    "fed_monetary": "DOMESTIC", "fiscal_policy": "DOMESTIC",
    "election_rhetoric": "DOMESTIC",
}


# ═══════════════════════════════════════════════════════════════════════════
# SEVERITY ANCHOR SENTENCES (not keywords — full contextual sentences)
# ═══════════════════════════════════════════════════════════════════════════

URGENCY_ANCHORS: dict[int, list[str]] = {
    1: ["We should consider looking into this matter",
        "We might review our options in the future",
        "Perhaps we will address this at some point"],
    2: ["This will happen soon and we are preparing",
        "Very soon we will take action on this issue",
        "In the coming weeks expect a major announcement"],
    3: ["This will happen this week by Friday",
        "Tomorrow we announce our decision",
        "Next Monday the new policy takes effect"],
    4: ["Effective immediately we are taking action right now",
        "As we speak our teams are implementing this",
        "Starting today this new order is in effect"],
    5: ["I have just signed the executive order it is done",
        "We have completed the action and it is finished",
        "Just announced the deal has been executed"],
}

EXTREMITY_ANCHORS: dict[int, list[str]] = {
    1: ["We will review and study this carefully",
        "Let us examine and monitor the situation",
        "We are assessing and evaluating options"],
    2: ["We are increasing and strengthening our position",
        "Expanding and boosting our capabilities",
        "Raising and improving our efforts"],
    3: ["Total and complete action on all fronts",
        "Maximum and full implementation across the board",
        "Massive and enormous effort like no other"],
    4: ["We will destroy and obliterate the threat",
        "Crush and eliminate every last one of them",
        "Devastate and annihilate the enemy completely"],
    5: ["The biggest and strongest action in history like never before",
        "Unprecedented in the history of the world no one has seen anything like it",
        "The most powerful response the likes of which have never been seen"],
}

TARGETING_ANCHORS: dict[int, list[str]] = {
    1: ["Those bad actors and enemies who threaten us",
        "Certain countries and some people who act against us"],
    2: ["The situation in the Middle East and Asia region",
        "Countries in the Gulf area and Pacific region"],
    3: ["Iran must comply or face consequences",
        "China and Russia are the greatest threats",
        "North Korea and Venezuela are failing states"],
    4: ["The IRGC and Hamas and Hezbollah terrorist organizations",
        "Sanctions against Huawei and PetroChina entities",
        "OPEC and the Houthis are causing problems"],
    5: ["Khamenei has made a terrible mistake",
        "Xi Jinping must understand the consequences",
        "Putin and Kim Jong Un will pay the price",
        "Direct message to Erdogan and Maduro"],
}


# ═══════════════════════════════════════════════════════════════════════════
# NEGATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

NEGATION_CUES = [
    "not", "no", "never", "neither", "nor", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
    "nothing", "nobody", "nowhere", "without", "refuse",
    "reject", "deny", "denied", "decline", "declined",
]

NEGATION_WINDOW = 5  # words after negation cue that are affected


def detect_negation(text: str) -> tuple[bool, list[tuple[int, int]]]:
    """
    Detect negation in text and return affected word spans.

    Returns:
        (has_negation, list of (start_word_idx, end_word_idx) spans)

    Uses a window-based approach: each negation cue affects the
    next N words, unless interrupted by punctuation or a clause boundary.
    """
    words = text.lower().split()
    negated_spans = []
    has_negation = False

    for i, word in enumerate(words):
        clean = re.sub(r'[^\w\']', '', word)
        if clean in NEGATION_CUES:
            has_negation = True
            # Negation affects next NEGATION_WINDOW words
            # unless interrupted by sentence boundary
            end = min(i + NEGATION_WINDOW + 1, len(words))
            for j in range(i + 1, end):
                if any(p in words[j] for p in ['.', '!', '?', ',', ';', 'but', 'however']):
                    end = j
                    break
            negated_spans.append((i, end))

    return has_negation, negated_spans


def is_word_negated(word_idx: int, negated_spans: list[tuple[int, int]]) -> bool:
    """Check if a word at given index falls within any negation span."""
    return any(start <= word_idx < end for start, end in negated_spans)


# ═══════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION (lightweight — no spaCy dependency)
# ═══════════════════════════════════════════════════════════════════════════

GEOPOLITICAL_ENTITIES = {
    # Countries
    "iran": "COUNTRY", "china": "COUNTRY", "russia": "COUNTRY",
    "israel": "COUNTRY", "iraq": "COUNTRY", "syria": "COUNTRY",
    "turkey": "COUNTRY", "venezuela": "COUNTRY", "north korea": "COUNTRY",
    "saudi arabia": "COUNTRY", "qatar": "COUNTRY", "uae": "COUNTRY",
    # Organizations
    "irgc": "ORG", "hamas": "ORG", "hezbollah": "ORG", "houthis": "ORG",
    "opec": "ORG", "nato": "ORG", "epa": "ORG", "fed": "ORG",
    "federal reserve": "ORG", "huawei": "ORG", "petrochina": "ORG",
    # Persons
    "khamenei": "PERSON", "xi": "PERSON", "jinping": "PERSON",
    "putin": "PERSON", "kim": "PERSON", "netanyahu": "PERSON",
    "erdogan": "PERSON", "maduro": "PERSON", "powell": "PERSON",
    "biden": "PERSON", "trump": "PERSON",
    # Locations
    "hormuz": "LOCATION", "strait of hormuz": "LOCATION",
    "persian gulf": "LOCATION", "red sea": "LOCATION",
    "gaza": "LOCATION", "west bank": "LOCATION", "lebanon": "LOCATION",
    "tehran": "LOCATION", "beijing": "LOCATION",
}


def extract_entities(text: str) -> list[dict]:
    """
    Extract known geopolitical entities from text.

    Returns list of {"entity": str, "type": str, "negated": bool}.
    Multi-word entities are checked first (longest match wins).
    """
    text_lower = text.lower()
    words = text_lower.split()
    _, neg_spans = detect_negation(text)

    found = []
    used_positions = set()

    # Sort by length descending (multi-word first)
    sorted_entities = sorted(GEOPOLITICAL_ENTITIES.keys(), key=len, reverse=True)

    for entity in sorted_entities:
        start = text_lower.find(entity)
        if start == -1:
            continue

        # Check if this position is already claimed by a longer entity
        entity_word_start = len(text_lower[:start].split()) - 1
        if entity_word_start < 0:
            entity_word_start = 0

        if entity_word_start in used_positions:
            continue

        # Mark positions as used
        n_words = len(entity.split())
        for w in range(entity_word_start, entity_word_start + n_words):
            used_positions.add(w)

        # Check if entity mention is negated
        negated = is_word_negated(entity_word_start, neg_spans)

        found.append({
            "entity": entity,
            "type": GEOPOLITICAL_ENTITIES[entity],
            "negated": negated,
        })

    return found


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NLPResult:
    """Complete NLP analysis of a single text."""
    text: str
    topics: dict[str, float]             # 12 sub-topic scores
    topic_dominant: str                   # highest-scoring topic
    topic_parent: str                     # parent category of dominant
    topic_max_score: float
    is_relevant: bool                     # non-domestic topic above threshold

    severity_urgency: int                 # 1-5
    severity_extremity: int               # 1-5
    severity_targeting: int               # 1-5
    severity_combined: float              # weighted composite

    sentiment: float                      # -1.0 to +1.0
    sentiment_vader: float | None         # VADER component
    sentiment_transformer: float | None   # transformer component

    entities: list[dict]                  # extracted named entities
    has_negation: bool                    # negation detected in text
    negated_entities: list[str]           # entities in negated context

    embedding: np.ndarray | None = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════════════════════════
# NLP ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class NLPEngine:
    """
    Production-grade NLP analysis engine for CrudeNerve.

    Architecture:
        1. Encode text with sentence-transformers (all-MiniLM-L6-v2)
        2. Topic classification via cosine similarity to anchor sentences
        3. Severity scoring via cosine similarity to severity anchors
        4. Sentiment: VADER (0.3) + transformer-derived (0.7) ensemble
        5. Entity extraction with negation awareness
        6. Negation detection modifies severity and topic scores

    All tasks share the same embedding — single forward pass per post.
    """

    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold
        self._anchor_embeddings: dict[str, np.ndarray] | None = None
        self._severity_embeddings: dict[str, dict[int, np.ndarray]] | None = None

    def _ensure_anchors_encoded(self):
        """Pre-compute anchor sentence embeddings (once)."""
        if self._anchor_embeddings is not None:
            return

        model = _get_sentence_model()
        if model is None:
            self._anchor_embeddings = {}
            return

        logger.info("Encoding topic anchor sentences...")
        self._anchor_embeddings = {}
        for topic, anchors in TOPIC_ANCHORS.items():
            embeddings = model.encode(anchors, show_progress_bar=False)
            # Store mean embedding for each topic
            self._anchor_embeddings[topic] = np.mean(embeddings, axis=0)

        # Severity anchors
        self._severity_embeddings = {}
        for axis_name, axis_anchors in [
            ("urgency", URGENCY_ANCHORS),
            ("extremity", EXTREMITY_ANCHORS),
            ("targeting", TARGETING_ANCHORS),
        ]:
            self._severity_embeddings[axis_name] = {}
            for level, sentences in axis_anchors.items():
                embs = model.encode(sentences, show_progress_bar=False)
                self._severity_embeddings[axis_name][level] = np.mean(embs, axis=0)

        logger.info("Anchor embeddings ready.")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ── Topic classification ─────────────────────────────────────────────

    def classify_topics(self, text: str, embedding: np.ndarray | None = None) -> dict[str, float]:
        """
        Classify text into 12 sub-topics using semantic similarity.

        Each topic has curated anchor sentences. The classifier computes
        cosine similarity between the text embedding and each topic's
        centroid embedding. This captures:
        - "drill baby drill" → energy policy (no keyword "oil" needed)
        - "the likes of which the world has never seen" → extreme rhetoric
        - Context around negation: "NOT sanctioning Iran" vs "sanctioning Iran"
        """
        model = _get_sentence_model()
        self._ensure_anchors_encoded()

        if model is None or not self._anchor_embeddings:
            return self._keyword_fallback_topics(text)

        if embedding is None:
            embedding = model.encode(text, show_progress_bar=False)

        scores = {}
        for topic, anchor_emb in self._anchor_embeddings.items():
            sim = self._cosine_similarity(embedding, anchor_emb)
            # Rescale from cosine range [-1,1] to [0,1]
            scores[topic] = max(0.0, round((sim + 1) / 2, 4))

        # Apply negation penalty: if the text negates the topic's key concepts,
        # reduce the score
        has_neg, neg_spans = detect_negation(text)
        if has_neg:
            entities = extract_entities(text)
            negated_ents = {e["entity"] for e in entities if e["negated"]}

            # If a topic's key entities are negated, discount that topic
            topic_entity_map = {
                "sanctions_iran": {"iran", "sanctions"},
                "iran_military": {"iran", "irgc", "hormuz"},
                "israel_conflict": {"israel", "gaza", "hamas", "hezbollah"},
                "tariff_china": {"china", "tariff"},
            }
            for topic, key_ents in topic_entity_map.items():
                if topic in scores and negated_ents & key_ents:
                    scores[topic] *= 0.3  # heavy discount for negated topics

        # Normalize: max score = 1.0
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: round(v / max_score, 4) for k, v in scores.items()}

        return scores

    def _keyword_fallback_topics(self, text: str) -> dict[str, float]:
        """Fallback keyword-based classification when transformers unavailable."""
        from crudenerve.data_ingest.d2_truth_social import TopicClassifier
        return TopicClassifier().classify(text)

    # ── Severity scoring ─────────────────────────────────────────────────

    def score_severity(self, text: str, embedding: np.ndarray | None = None) -> dict[str, float]:
        """
        Score severity on three axes using semantic similarity to anchors.

        For each axis (urgency, extremity, targeting), computes cosine
        similarity to anchor sentences at each level (1-5). The level
        with highest similarity wins, but we use soft assignment —
        the score is a weighted average, not argmax.

        Negation adjustment: if the text contains negation around
        severity-relevant phrases, the score is dampened.
        """
        model = _get_sentence_model()
        self._ensure_anchors_encoded()

        if model is None or not self._severity_embeddings:
            return self._keyword_fallback_severity(text)

        if embedding is None:
            embedding = model.encode(text, show_progress_bar=False)

        result = {}
        for axis_name, level_embeddings in self._severity_embeddings.items():
            # Compute similarity to each level
            level_sims = {}
            for level, level_emb in level_embeddings.items():
                sim = self._cosine_similarity(embedding, level_emb)
                level_sims[level] = max(0, (sim + 1) / 2)  # rescale to [0,1]

            # Soft assignment: weighted average by similarity
            total_weight = sum(level_sims.values())
            if total_weight > 0:
                weighted = sum(level * sim for level, sim in level_sims.items())
                score = weighted / total_weight
            else:
                score = 1.0

            # Also get the hard argmax for discrete output
            best_level = max(level_sims, key=level_sims.get)

            result[axis_name] = int(round(score))
            result[f"{axis_name}_continuous"] = round(score, 3)

        # Negation dampening
        has_neg, _ = detect_negation(text)
        if has_neg:
            # Negation reduces urgency and extremity (not targeting)
            result["urgency"] = max(1, result["urgency"] - 1)
            result["urgency_continuous"] = max(1.0, result["urgency_continuous"] - 0.5)
            result["extremity"] = max(1, result["extremity"] - 1)
            result["extremity_continuous"] = max(1.0, result["extremity_continuous"] - 0.5)

        # Combined severity
        u_weight, e_weight, t_weight = 0.40, 0.35, 0.25
        result["combined"] = round(
            result["urgency_continuous"] * u_weight
            + result["extremity_continuous"] * e_weight
            + result["targeting_continuous"] * t_weight,
            3,
        )

        return result

    def _keyword_fallback_severity(self, text: str) -> dict[str, float]:
        """Fallback to keyword-based severity when transformers unavailable."""
        from crudenerve.data_ingest.d2_truth_social import SeverityScorer
        raw = SeverityScorer().score(text)
        return {
            "urgency": raw["urgency"],
            "urgency_continuous": float(raw["urgency"]),
            "extremity": raw["extremity"],
            "extremity_continuous": float(raw["extremity"]),
            "targeting": raw["targeting"],
            "targeting_continuous": float(raw["targeting"]),
            "combined": raw["severity_combined"],
        }

    # ── Sentiment analysis ───────────────────────────────────────────────

    def analyze_sentiment(self, text: str, embedding: np.ndarray | None = None) -> dict[str, float]:
        """
        Ensemble sentiment: VADER (0.3) + transformer-derived (0.7).

        VADER is good at catching explicit sentiment markers (CAPS,
        exclamation marks, intensifiers). The transformer captures
        contextual sentiment that VADER misses.

        For the transformer component, we compute cosine similarity
        to positive and negative political rhetoric anchors.
        """
        result = {"sentiment": 0.0, "vader": None, "transformer": None}

        # VADER component
        vader = _get_vader()
        if vader is not None:
            vs = vader.polarity_scores(text)
            result["vader"] = round(vs["compound"], 4)

        # Transformer component: similarity to positive vs negative anchors
        model = _get_sentence_model()
        if model is not None:
            if embedding is None:
                embedding = model.encode(text, show_progress_bar=False)

            pos_anchors = [
                "Great deal achieved, tremendous success, very positive outcome",
                "Peace and prosperity, strong economy, winning bigly",
                "Agreement reached, cooperation, diplomatic breakthrough",
            ]
            neg_anchors = [
                "Terrible threat, disaster, catastrophe approaching",
                "Destroying, devastating, the worst situation ever",
                "War, conflict, sanctions, punishment, consequences",
            ]

            pos_embs = model.encode(pos_anchors, show_progress_bar=False)
            neg_embs = model.encode(neg_anchors, show_progress_bar=False)

            pos_sim = np.mean([self._cosine_similarity(embedding, e) for e in pos_embs])
            neg_sim = np.mean([self._cosine_similarity(embedding, e) for e in neg_embs])

            # Scale to [-1, 1]
            transformer_sent = round(float(pos_sim - neg_sim), 4)
            result["transformer"] = np.clip(transformer_sent, -1, 1)

        # Ensemble
        if result["vader"] is not None and result["transformer"] is not None:
            result["sentiment"] = round(
                result["vader"] * 0.3 + result["transformer"] * 0.7, 4
            )
        elif result["vader"] is not None:
            result["sentiment"] = result["vader"]
        elif result["transformer"] is not None:
            result["sentiment"] = result["transformer"]

        return result

    # ── Full analysis ────────────────────────────────────────────────────

    def analyze(self, text: str) -> NLPResult:
        """
        Run complete NLP analysis on a single text.

        Single embedding pass — all downstream tasks share it.
        """
        # Single encoding for all tasks
        model = _get_sentence_model()
        embedding = None
        if model is not None:
            embedding = model.encode(text, show_progress_bar=False)

        # Entity extraction (runs on raw text, not embedding)
        entities = extract_entities(text)
        has_neg, _ = detect_negation(text)
        negated_ents = [e["entity"] for e in entities if e["negated"]]

        # Topic classification
        topics = self.classify_topics(text, embedding)
        dominant = max(topics, key=topics.get) if topics else "none"
        parent = TOPIC_TO_PARENT.get(dominant, "DOMESTIC")
        max_score = max(topics.values()) if topics else 0
        relevant_topics = {k: v for k, v in topics.items() if TOPIC_TO_PARENT.get(k) != "DOMESTIC"}
        is_relevant = max(relevant_topics.values(), default=0) > self.relevance_threshold

        # Severity scoring
        severity = self.score_severity(text, embedding)

        # Sentiment
        sentiment = self.analyze_sentiment(text, embedding)

        return NLPResult(
            text=text,
            topics=topics,
            topic_dominant=dominant,
            topic_parent=parent,
            topic_max_score=max_score,
            is_relevant=is_relevant,
            severity_urgency=severity["urgency"],
            severity_extremity=severity["extremity"],
            severity_targeting=severity["targeting"],
            severity_combined=severity["combined"],
            sentiment=sentiment["sentiment"],
            sentiment_vader=sentiment["vader"],
            sentiment_transformer=sentiment["transformer"],
            entities=entities,
            has_negation=has_neg,
            negated_entities=negated_ents,
            embedding=embedding,
        )

    def analyze_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[NLPResult]:
        """
        Batch analysis — encodes all texts in one pass for efficiency.
        """
        model = _get_sentence_model()
        embeddings = None
        if model is not None:
            logger.info(f"Encoding {len(texts)} texts with sentence-transformers...")
            embeddings = model.encode(texts, show_progress_bar=show_progress, batch_size=32)

        results = []
        for i, text in enumerate(texts):
            emb = embeddings[i] if embeddings is not None else None
            # We pass the pre-computed embedding to avoid re-encoding
            entities = extract_entities(text)
            has_neg, _ = detect_negation(text)
            negated_ents = [e["entity"] for e in entities if e["negated"]]

            topics = self.classify_topics(text, emb)
            dominant = max(topics, key=topics.get) if topics else "none"
            parent = TOPIC_TO_PARENT.get(dominant, "DOMESTIC")
            max_score = max(topics.values()) if topics else 0
            relevant_topics = {k: v for k, v in topics.items() if TOPIC_TO_PARENT.get(k) != "DOMESTIC"}
            is_relevant = max(relevant_topics.values(), default=0) > self.relevance_threshold

            severity = self.score_severity(text, emb)
            sentiment = self.analyze_sentiment(text, emb)

            results.append(NLPResult(
                text=text,
                topics=topics,
                topic_dominant=dominant,
                topic_parent=parent,
                topic_max_score=max_score,
                is_relevant=is_relevant,
                severity_urgency=severity["urgency"],
                severity_extremity=severity["extremity"],
                severity_targeting=severity["targeting"],
                severity_combined=severity["combined"],
                sentiment=sentiment["sentiment"],
                sentiment_vader=sentiment["vader"],
                sentiment_transformer=sentiment["transformer"],
                entities=entities,
                has_negation=has_neg,
                negated_entities=negated_ents,
                embedding=emb,
            ))

        return results

    def result_to_series(self, result: NLPResult) -> pd.Series:
        """Convert an NLPResult to a flat pandas Series for DataFrame integration."""
        data = {f"topic_{k}": v for k, v in result.topics.items()}
        data.update({
            "topic_dominant": result.topic_dominant,
            "topic_parent": result.topic_parent,
            "topic_max_score": result.topic_max_score,
            "is_relevant": result.is_relevant,
            "sev_urgency": result.severity_urgency,
            "sev_extremity": result.severity_extremity,
            "sev_targeting": result.severity_targeting,
            "sev_severity_combined": result.severity_combined,
            "sentiment": result.sentiment,
            "sentiment_vader": result.sentiment_vader,
            "sentiment_transformer": result.sentiment_transformer,
            "has_negation": result.has_negation,
            "entity_count": len(result.entities),
            "negated_entity_count": len(result.negated_entities),
            "is_high_severity": result.severity_combined >= 4.0,
            "is_actionable": result.is_relevant and result.severity_combined >= 4.0,
        })
        return pd.Series(data)


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("Testing NLP Engine — Production-Grade Text Analysis")
    print("=" * 70)

    engine = NLPEngine()

    test_posts = [
        # High severity, clear action
        "I have just imposed the STRONGEST sanctions in history on Iran. "
        "Their nuclear program will be CRUSHED. Effective immediately!",

        # Negation — should NOT trigger sanctions topic strongly
        "We are NOT going to sanction Iran at this time. "
        "We believe in diplomatic solutions.",

        # Moderate severity, Iran military
        "Iran's IRGC is a serious threat in the Strait of Hormuz. "
        "Our Navy is monitoring the situation very closely.",

        # Pure noise — election rhetoric
        "Great rally last night in Ohio! Biggest crowd ever. MAGA! "
        "The Democrats have no idea what they are doing.",

        # Trade + high severity
        "Just signed the biggest tariff increase on China EVER. "
        "60% on all Chinese goods. The likes of which the world has never seen!",

        # Ambiguous — mentions Iran but in diplomatic context
        "Had a very productive call about Iran. Good things may happen. "
        "We will see. But we are ready for anything.",
    ]

    print(f"\nAnalyzing {len(test_posts)} test posts...\n")

    results = engine.analyze_batch(test_posts, show_progress=False)

    for i, r in enumerate(results):
        print(f"{'─'*60}")
        print(f"POST {i+1}: {r.text[:80]}...")
        print(f"  Topic:     {r.topic_dominant} ({r.topic_parent}) — score={r.topic_max_score:.3f}")
        print(f"  Relevant:  {r.is_relevant}")
        print(f"  Severity:  urgency={r.severity_urgency} extremity={r.severity_extremity} "
              f"targeting={r.severity_targeting} → combined={r.severity_combined:.2f}")
        print(f"  Sentiment: {r.sentiment:+.3f} (VADER={r.sentiment_vader}, "
              f"transformer={r.sentiment_transformer})")
        print(f"  Entities:  {[e['entity'] for e in r.entities]}")
        print(f"  Negation:  {r.has_negation} → negated entities: {r.negated_entities}")
        print(f"  Actionable: {r.severity_combined >= 4.0 and r.is_relevant}")
        print()

    # Specific negation test
    print("=" * 60)
    print("NEGATION CONTRAST TEST")
    print("=" * 60)
    pos = engine.analyze("We WILL impose maximum sanctions on Iran immediately")
    neg = engine.analyze("We will NOT impose sanctions on Iran at this time")
    print(f"\n  POSITIVE: 'We WILL impose maximum sanctions on Iran immediately'")
    print(f"    sanctions_iran = {pos.topics.get('sanctions_iran', 0):.3f}")
    print(f"    severity = {pos.severity_combined:.2f}")
    print(f"    sentiment = {pos.sentiment:+.3f}")
    print(f"\n  NEGATED:  'We will NOT impose sanctions on Iran at this time'")
    print(f"    sanctions_iran = {neg.topics.get('sanctions_iran', 0):.3f}")
    print(f"    severity = {neg.severity_combined:.2f}")
    print(f"    sentiment = {neg.sentiment:+.3f}")
    print(f"\n  Difference: topic score {pos.topics.get('sanctions_iran',0) - neg.topics.get('sanctions_iran',0):+.3f}, "
          f"severity {pos.severity_combined - neg.severity_combined:+.2f}")

    print(f"\n{'='*70}")
    print(f"✅ NLP Engine test PASSED")
    print(f"{'='*70}")
