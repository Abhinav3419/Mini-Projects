"""
F3: Damped Harmonic Oscillator (DHO) decay kernel.

Physics-informed feature: models how geopolitical events decay in market
impact over time using DHO impulse response from vibration theory.

    response(t) = A × exp(-ζ×ω_n×t) × cos(ω_d×t + φ)

where:
    A     = amplitude (event severity)
    ζ     = damping ratio (how fast the effect fades)
    ω_n   = natural frequency (market reaction speed)
    ω_d   = damped frequency = ω_n × sqrt(1 - ζ²)
    φ     = phase offset

Key insight from Applied Mechanics (Abhinav's M.Tech background):
    ζ < 1  → underdamped (oscillates — event persists, market keeps reacting)
    ζ = 1  → critically damped (fastest decay without oscillation)
    ζ > 1  → overdamped (slow monotonic decay — event fades without drama)

Trump posts get severity-graded parameters:
    Low severity  → overdamped (noise, fades fast)
    High severity → underdamped (persists 7-14 days, follow-ups re-excite)

Usage:
    from crudenerve.features.f3_dho_kernel import DHOKernelEngine
    engine = DHOKernelEngine()
    df = engine.compute(unified_df)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crudenerve.config.settings import DHO_EVENT_TABLE, TRUMP_DHO_TABLE

logger = logging.getLogger(__name__)


@dataclass
class DHOResponse:
    """Single DHO impulse response result."""
    event_type: str
    zeta: float
    omega_n: float
    amplitude: float
    kernel: np.ndarray  # decay values over time


class DHOKernelEngine:
    """
    Applies DHO impulse response kernels to geopolitical events.

    For each event detected in the unified DataFrame, convolves
    a decay kernel forward in time to model the persisting market
    impact. The kernel shape (oscillatory vs monotonic) depends
    on event type and severity.
    """

    def __init__(self, max_decay_days: int = 30):
        self.max_decay_days = max_decay_days
        self.event_params = {
            p.event_type: (p.zeta, p.omega_n)
            for p in DHO_EVENT_TABLE
        }
        self.trump_params = TRUMP_DHO_TABLE

    # ── Core physics: DHO impulse response ───────────────────────────────

    def _impulse_response(
        self,
        zeta: float,
        omega_n: float,
        amplitude: float = 1.0,
        n_days: int | None = None,
        phi: float = 0.0,
    ) -> np.ndarray:
        """
        Compute the DHO impulse response kernel.

        Args:
            zeta:      damping ratio (0 < ζ < 1 = underdamped, ζ ≥ 1 = overdamped)
            omega_n:   natural frequency (rad/day)
            amplitude: event severity scaling factor
            n_days:    kernel length (defaults to max_decay_days)
            phi:       phase offset (radians)

        Returns:
            1D array of decay values, one per day from event day forward.
        """
        n = n_days or self.max_decay_days
        t = np.arange(n, dtype=float)

        if zeta >= 1.0:
            # Overdamped: monotonic exponential decay (no oscillation)
            # response = A × exp(-ζ×ω_n×t) — simplified
            kernel = amplitude * np.exp(-zeta * omega_n * t)
        else:
            # Underdamped: oscillatory decay
            omega_d = omega_n * np.sqrt(1 - zeta**2)
            kernel = amplitude * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)

        # Normalize so kernel[0] = amplitude (event day impact)
        if kernel[0] != 0:
            kernel = kernel * (amplitude / abs(kernel[0]))

        return kernel

    def _get_trump_dho_params(self, severity: float) -> tuple[float, float]:
        """Look up severity-graded DHO parameters for Trump posts."""
        for row in self.trump_params:
            if row["severity_min"] <= severity < row["severity_max"]:
                return row["zeta"], row["omega_n"]
        # Fallback: highest severity band
        last = self.trump_params[-1]
        return last["zeta"], last["omega_n"]

    # ── Event detection from unified DataFrame ───────────────────────────

    def _detect_gdelt_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect significant geopolitical events from GDELT features.

        An "event" is a day where entity mention count exceeds
        2σ above its rolling mean (i.e., an unusual spike).
        """
        events = []
        entity_cols = [c for c in df.columns
                       if c.startswith("gdelt_") and c.endswith("_count")
                       and c != "gdelt_event_count"]

        for col in entity_cols:
            entity = col.replace("gdelt_", "").replace("_count", "")
            vals = df[col].fillna(0)
            rolling_mean = vals.rolling(30, min_periods=7).mean()
            rolling_std = vals.rolling(30, min_periods=7).std().fillna(1)
            threshold = rolling_mean + 2 * rolling_std

            spike_days = df.index[vals > threshold]
            for day in spike_days:
                # Map entity to event type for DHO params
                event_type = self._entity_to_event_type(entity)
                amplitude = float(vals.loc[day])
                events.append({
                    "date": day,
                    "source": "gdelt",
                    "entity": entity,
                    "event_type": event_type,
                    "amplitude": amplitude,
                })

        return pd.DataFrame(events) if events else pd.DataFrame(
            columns=["date", "source", "entity", "event_type", "amplitude"]
        )

    def _detect_trump_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect significant Trump Truth Social events.

        Any day with max severity >= 3.0 triggers a DHO kernel.
        Severity determines the kernel shape (overdamped vs underdamped).
        """
        events = []
        if "ts_max_severity" not in df.columns:
            return pd.DataFrame(
                columns=["date", "source", "severity", "event_type", "amplitude"]
            )

        severity = df["ts_max_severity"].fillna(0)
        event_days = df.index[severity >= 3.0]

        for day in event_days:
            sev = float(severity.loc[day])
            events.append({
                "date": day,
                "source": "truth_social",
                "severity": sev,
                "event_type": "trump_post",
                "amplitude": sev,  # severity IS the amplitude
            })

        return pd.DataFrame(events) if events else pd.DataFrame(
            columns=["date", "source", "severity", "event_type", "amplitude"]
        )

    def _entity_to_event_type(self, entity: str) -> str:
        """Map GDELT entity names to DHO event types."""
        mapping = {
            "iran": "sanctions_announced",
            "irgc": "military_strike",
            "hormuz": "hormuz_disruption",
            "strait_of_hormuz": "hormuz_disruption",
            "opec": "opec_quota_change",
            "sanctions": "sanctions_announced",
            "israel": "military_strike",
            "houthis": "hormuz_disruption",
            "hezbollah": "military_strike",
        }
        return mapping.get(entity, "sanctions_announced")

    # ── Main compute: convolve kernels forward ───────────────────────────

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect events and apply DHO decay kernels to the unified DataFrame.

        Adds columns:
            - f3_dho_gdelt:       sum of all active GDELT decay signals
            - f3_dho_trump:       sum of all active Trump decay signals
            - f3_dho_combined:    total decay signal (GDELT + Trump)
            - f3_dho_max_active:  highest single active kernel value
            - f3_dho_n_active:    number of currently-active decay kernels
        """
        out = df.copy()
        n = len(df)

        # Initialize accumulator arrays
        gdelt_signal = np.zeros(n)
        trump_signal = np.zeros(n)
        active_count = np.zeros(n)

        # ── Process GDELT events ─────────────────────────────────────────
        gdelt_events = self._detect_gdelt_events(df)
        if not gdelt_events.empty:
            for _, event in gdelt_events.iterrows():
                event_type = event["event_type"]
                zeta, omega_n = self.event_params.get(
                    event_type, (0.5, 0.7)
                )
                amplitude = event["amplitude"]
                kernel = self._impulse_response(zeta, omega_n, amplitude)

                # Find position of event day in index
                event_date = event["date"]
                if event_date in df.index:
                    start_idx = df.index.get_loc(event_date)
                    end_idx = min(start_idx + len(kernel), n)
                    k_len = end_idx - start_idx
                    gdelt_signal[start_idx:end_idx] += kernel[:k_len]
                    active_count[start_idx:end_idx] += (kernel[:k_len] > 0.01).astype(float)

            logger.info(f"Applied {len(gdelt_events)} GDELT event kernels")

        # ── Process Trump events (severity-graded DHO) ───────────────────
        trump_events = self._detect_trump_events(df)
        if not trump_events.empty:
            for _, event in trump_events.iterrows():
                severity = event.get("severity", 3.0)
                zeta, omega_n = self._get_trump_dho_params(severity)
                amplitude = severity  # severity directly scales amplitude
                kernel = self._impulse_response(zeta, omega_n, amplitude)

                event_date = event["date"]
                if event_date in df.index:
                    start_idx = df.index.get_loc(event_date)
                    end_idx = min(start_idx + len(kernel), n)
                    k_len = end_idx - start_idx
                    trump_signal[start_idx:end_idx] += kernel[:k_len]
                    active_count[start_idx:end_idx] += (kernel[:k_len] > 0.01).astype(float)

            logger.info(f"Applied {len(trump_events)} Trump event kernels "
                       f"(severity-graded)")

        # ── Assign features ──────────────────────────────────────────────
        out["f3_dho_gdelt"] = gdelt_signal
        out["f3_dho_trump"] = trump_signal
        out["f3_dho_combined"] = gdelt_signal + trump_signal
        out["f3_dho_n_active"] = active_count

        # Max active kernel value on each day
        combined = gdelt_signal + trump_signal
        # Rolling max over a short window to capture peak persistence
        out["f3_dho_max_active"] = (
            pd.Series(combined, index=df.index)
            .rolling(3, min_periods=1).max()
        )

        logger.info(
            f"DHO features computed: "
            f"max_combined={out['f3_dho_combined'].max():.2f}, "
            f"days_with_active_kernels="
            f"{(out['f3_dho_n_active'] > 0).sum()}"
        )

        return out


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing F3: DHO Decay Kernel\n")

    # Show kernel shapes for different event types
    engine = DHOKernelEngine()

    print("--- Kernel shapes (first 10 days) ---")
    for etype, (z, w) in engine.event_params.items():
        k = engine._impulse_response(z, w, amplitude=1.0, n_days=15)
        vals = ", ".join(f"{v:.3f}" for v in k[:10])
        regime = "underdamped" if z < 1 else "overdamped"
        print(f"  {etype:25s} ζ={z:.2f} ω={w:.1f} ({regime}): [{vals}]")

    print("\n--- Trump severity-graded kernels ---")
    for sev in [1.5, 2.5, 3.5, 4.2, 4.8]:
        z, w = engine._get_trump_dho_params(sev)
        k = engine._impulse_response(z, w, amplitude=sev, n_days=15)
        vals = ", ".join(f"{v:.3f}" for v in k[:10])
        regime = "underdamped" if z < 1 else "overdamped"
        print(f"  severity={sev:.1f} ζ={z:.2f} ω={w:.1f} ({regime}): [{vals}]")

    # Integration test with synthetic data
    print("\n--- Integration with unified DataFrame ---")
    from crudenerve.data_ingest.d1_gdelt import GDELTIngestor, _generate_synthetic_gdelt
    from crudenerve.data_ingest.d2_truth_social import (
        TruthSocialPipeline, generate_synthetic_posts,
    )

    gdelt_ingestor = GDELTIngestor()
    raw = _generate_synthetic_gdelt(500)
    raw = gdelt_ingestor._tag_entities(raw)
    gdelt_daily = gdelt_ingestor.aggregate_daily(raw)

    ts_pipeline = TruthSocialPipeline()
    ts_posts = generate_synthetic_posts(200)
    ts_enriched = ts_pipeline.run_full_pipeline(ts_posts, save=False)
    ts_daily = ts_pipeline.aggregate_daily(ts_enriched)

    # Simple merge for test
    combined = gdelt_daily.join(ts_daily, how="outer")
    result = engine.compute(combined)

    f3_cols = [c for c in result.columns if c.startswith("f3_")]
    print(f"\nF3 columns: {f3_cols}")
    for col in f3_cols:
        print(f"  {col}: mean={result[col].mean():.3f}, max={result[col].max():.3f}")

    print(f"\nDays with active decay kernels: "
          f"{(result['f3_dho_n_active'] > 0).sum()}/{len(result)}")
