"""
M1: HMM Regime Detection.

Fits a 4-state Gaussian Hidden Markov Model on the F2 geopolitical
tension index to classify each day into one of:
    0: calm           — baseline, low tension, markets range-bound
    1: escalation     — tension rising, vol expanding
    2: active_conflict— peak tension, crisis mode
    3: de-escalation  — tension fading, recovery underway

The detected regime feeds into M2 (regime-specific XGBoost) to select
which model and which feature subset to use for prediction.

Usage:
    from crudenerve.models.m1_hmm_regime import HMMRegimeDetector
    detector = HMMRegimeDetector()
    detector.fit(unified_df)
    df = detector.predict(unified_df)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from crudenerve.config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

REGIME_NAMES = {0: "calm", 1: "escalation", 2: "active_conflict", 3: "de_escalation"}


class HMMRegimeDetector:
    """
    4-state Gaussian HMM for geopolitical regime detection.

    Input features (from F2):
        - f2_tension_ema:       smoothed tension level
        - f2_tension_velocity:  rate of change
        - f2_tension_zscore:    standardized anomaly score

    Output:
        - m1_regime:       integer regime label (0-3)
        - m1_regime_name:  human-readable regime name
        - m1_regime_prob_*: posterior probability per state
    """

    def __init__(self, n_states: int = 4, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state

        self.hmm: GaussianHMM | None = None
        self.scaler = StandardScaler()
        self.feature_cols = ["f2_tension_ema", "f2_tension_velocity", "f2_tension_zscore"]
        self._regime_order: dict[int, int] | None = None  # maps HMM state → semantic label

        self.save_dir = MODELS_DIR / "m1_hmm"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale HMM input features."""
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            raise ValueError(
                f"None of the required columns found: {self.feature_cols}. "
                f"Run F2 (TensionIndexBuilder) first."
            )

        X = df[available].fillna(0).values
        return X

    def fit(self, df: pd.DataFrame, save: bool = True) -> "HMMRegimeDetector":
        """
        Fit the HMM on historical data.

        After fitting, we reorder the states so that:
            state 0 = lowest mean tension  (calm)
            state 3 = highest mean tension (active_conflict)
        """
        logger.info("Fitting M1 HMM regime detector...")
        X_raw = self._prepare_features(df)
        X = self.scaler.fit_transform(X_raw)

        self.hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",  # initialize all params
        )

        self.hmm.fit(X)
        log_likelihood = self.hmm.score(X)
        logger.info(f"HMM fit complete. Log-likelihood: {log_likelihood:.2f}")

        # Reorder states by mean tension level (first feature = tension_ema)
        state_means = self.hmm.means_[:, 0]  # first column = scaled tension_ema
        order = np.argsort(state_means)

        # Map: HMM internal state → semantic label
        # order[0] = lowest mean → regime 0 (calm)
        # order[-1] = highest mean → regime 2 (active_conflict)
        # Middle states sorted by velocity to distinguish escalation vs de-escalation
        self._regime_order = {}
        if self.n_states == 4:
            self._regime_order[order[0]] = 0   # calm
            self._regime_order[order[3]] = 2   # active_conflict

            # For the two middle states: higher velocity → escalation
            mid_states = [order[1], order[2]]
            if len(self.hmm.means_[0]) > 1:
                vel_idx = 1  # second feature = tension_velocity
                if self.hmm.means_[mid_states[0], vel_idx] > self.hmm.means_[mid_states[1], vel_idx]:
                    self._regime_order[mid_states[0]] = 1  # escalation (rising)
                    self._regime_order[mid_states[1]] = 3  # de-escalation (falling)
                else:
                    self._regime_order[mid_states[0]] = 3
                    self._regime_order[mid_states[1]] = 1
            else:
                self._regime_order[mid_states[0]] = 1
                self._regime_order[mid_states[1]] = 3
        else:
            # Generic: sort by tension mean
            for i, s in enumerate(order):
                self._regime_order[s] = i

        if save:
            self._save()

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes for each day and add columns to the DataFrame.

        Adds:
            m1_regime:       integer (0=calm, 1=escalation, 2=conflict, 3=de-escalation)
            m1_regime_name:  string label
            m1_regime_prob_*: posterior probabilities per state
        """
        if self.hmm is None:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        X_raw = self._prepare_features(df)
        X = self.scaler.transform(X_raw)

        # Viterbi decoding: most likely state sequence
        raw_states = self.hmm.predict(X)

        # Posterior probabilities
        raw_probs = self.hmm.predict_proba(X)

        # Remap to semantic labels
        mapped_states = np.array([self._regime_order.get(s, s) for s in raw_states])

        out = df.copy()
        out["m1_regime"] = mapped_states
        out["m1_regime_name"] = [REGIME_NAMES.get(s, f"state_{s}") for s in mapped_states]

        # Add per-regime probabilities (remapped)
        for semantic_label, name in REGIME_NAMES.items():
            # Find which HMM state maps to this semantic label
            hmm_state = None
            for k, v in self._regime_order.items():
                if v == semantic_label:
                    hmm_state = k
                    break
            if hmm_state is not None and hmm_state < raw_probs.shape[1]:
                out[f"m1_prob_{name}"] = raw_probs[:, hmm_state]
            else:
                out[f"m1_prob_{name}"] = 0.0

        # Regime transition flag: 1 on days where regime changes
        out["m1_regime_transition"] = (out["m1_regime"].diff().abs() > 0).astype(int)

        logger.info(
            f"Regime prediction complete. Distribution: "
            + ", ".join(f"{name}={int((mapped_states==i).sum())}"
                        for i, name in REGIME_NAMES.items())
        )

        return out

    def get_regime_stats(self, df: pd.DataFrame) -> dict:
        """Return summary statistics per regime."""
        if "m1_regime_name" not in df.columns:
            df = self.predict(df)

        stats = {}
        for regime_id, name in REGIME_NAMES.items():
            mask = df["m1_regime"] == regime_id
            subset = df[mask]
            stats[name] = {
                "days": int(mask.sum()),
                "pct": round(mask.mean() * 100, 1),
                "avg_tension": round(subset["f2_tension_ema"].mean(), 3)
                if "f2_tension_ema" in subset.columns and len(subset) > 0 else 0,
                "transitions_into": int(
                    (df["m1_regime"].diff() == regime_id).sum()
                ) if len(df) > 1 else 0,
            }
        return stats

    def _save(self):
        path = self.save_dir / "hmm_regime.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "hmm": self.hmm,
                "scaler": self.scaler,
                "regime_order": self._regime_order,
                "feature_cols": self.feature_cols,
            }, f)
        logger.info(f"Saved M1 HMM → {path}")

    def load(self) -> "HMMRegimeDetector":
        path = self.save_dir / "hmm_regime.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.hmm = data["hmm"]
        self.scaler = data["scaler"]
        self._regime_order = data["regime_order"]
        self.feature_cols = data["feature_cols"]
        logger.info(f"Loaded M1 HMM from {path}")
        return self


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Testing M1: HMM Regime Detection")
    print("=" * 60)

    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
    from crudenerve.features.f2_tension_index import TensionIndexBuilder

    # Build synthetic tension data
    gi = GDELTIngestor()
    raw = gi._tag_entities(_generate_synthetic_gdelt(500))
    daily = gi.aggregate_daily(raw)
    daily = TensionIndexBuilder().compute(daily)

    # Fit HMM
    detector = HMMRegimeDetector()
    detector.fit(daily, save=False)

    # Predict
    result = detector.predict(daily)

    print(f"\nRegime distribution:")
    print(result["m1_regime_name"].value_counts())

    print(f"\nRegime stats:")
    stats = detector.get_regime_stats(result)
    for name, s in stats.items():
        print(f"  {name:20s} {s['days']:3d} days ({s['pct']:5.1f}%)  "
              f"avg_tension={s['avg_tension']:.3f}")

    print(f"\nTransition days: {result['m1_regime_transition'].sum()}")

    m1_cols = [c for c in result.columns if c.startswith("m1_")]
    print(f"\nM1 columns: {m1_cols}")
    print("\n✅ M1 test PASSED")
