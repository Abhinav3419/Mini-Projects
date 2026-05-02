"""
manning_model.py — Module 2: Manning's Equation Flood Arrival Time Prediction
Physics-based flood propagation model for the Yamuna river through Delhi.

Given upstream rainfall intensity and current water level, predicts:
1. Discharge at each monitoring station
2. When floodwater reaches downstream zones
3. Whether danger level will be breached and by how much

Uses Manning's equation: Q = (1/n) × A × R^(2/3) × S^(1/2)
and kinematic wave approximation for downstream propagation.

Author: Abhinav Pandey
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from config import YAMUNA_STATIONS, MANNING_PARAMS, FLOOD_ZONES


@dataclass
class StationPrediction:
    """Flood prediction for a single Yamuna monitoring station."""
    name: str
    lat: float
    lon: float
    km_marker: float
    danger_level_m: float
    # Predictions
    predicted_depth_m: float
    predicted_discharge_m3s: float
    predicted_velocity_ms: float
    arrival_time_hours: float       # Hours from now until flood peak arrives
    arrival_datetime: str           # Human-readable arrival time
    above_danger: bool              # True if predicted depth > danger level
    excess_depth_m: float           # How much above danger (0 if below)
    risk_level: str                 # LOW / MEDIUM / HIGH / CRITICAL
    # Context
    flow_area_m2: float
    hydraulic_radius_m: float
    wetted_perimeter_m: float


@dataclass
class FloodPrediction:
    """Complete flood prediction for all Yamuna stations."""
    timestamp: str
    upstream_rainfall_mm: float
    base_water_level_m: float
    stations: List[StationPrediction]
    peak_discharge_m3s: float
    peak_station: str
    highest_risk: str
    summary: str
    warning_message: str


class ManningFloodModel:
    """
    Physics-based flood prediction using Manning's equation
    and kinematic wave approximation.

    Manning's equation for open-channel flow:
        Q = (1/n) × A × R^(2/3) × S^(1/2)

    Where:
        Q = discharge (m³/s)
        n = Manning's roughness coefficient
        A = cross-sectional flow area (m²)
        R = hydraulic radius = A / P (m)
        P = wetted perimeter (m)
        S = channel slope (m/m)

    The Yamuna through Delhi is modeled as a trapezoidal channel
    with known dimensions. Flood propagation uses the kinematic
    wave celerity: c = (5/3) × V, where V = Q/A.
    """

    def __init__(self):
        self.n = MANNING_PARAMS["n"]
        self.slope = MANNING_PARAMS["channel_slope"]
        self.base_width = MANNING_PARAMS["channel_width_m"]
        self.base_depth = MANNING_PARAMS["avg_depth_m"]
        self.floodplain_width = MANNING_PARAMS["floodplain_width_m"]
        self.danger_discharge = MANNING_PARAMS["danger_discharge_m3s"]

        # Trapezoidal channel side slope (1V:2H typical for natural rivers)
        self.side_slope = 2.0

    def predict(self, upstream_rainfall_mm: float,
                base_water_level_m: float = None,
                forecast_hours: int = 48) -> FloodPrediction:
        """
        Predict flood propagation through Delhi's Yamuna stretch.

        Parameters:
            upstream_rainfall_mm: Cumulative rainfall (mm) in upstream catchment
            base_water_level_m: Current water level at Palla (m above datum).
                                If None, estimated from normal flow.
            forecast_hours: Prediction window in hours
        """
        if base_water_level_m is None:
            base_water_level_m = 202.0  # Normal Yamuna level at Palla

        # ============================================
        # STEP 1: Convert rainfall to runoff discharge
        # ============================================
        # Rational method: Q_peak = C × I × A_catchment
        # C = runoff coefficient (0.3-0.5 for Delhi's Yamuna catchment)
        # I = rainfall intensity (mm/hr, assume 6-hour event)
        # A = catchment area upstream of Delhi (~25,000 km²)

        catchment_area_km2 = 25000
        runoff_coefficient = 0.08  # Calibrated: large catchment with reservoirs upstream (Hathnikund, Tajewala)
        rainfall_intensity_mmhr = upstream_rainfall_mm / 12.0  # 12-hour event (typical monsoon burst)
        
        # Convert to m³/s: Q = C × I(m/s) × A(m²)
        rainfall_discharge_m3s = (
            runoff_coefficient *
            (rainfall_intensity_mmhr / (3600 * 1000)) *  # mm/hr → m/s
            (catchment_area_km2 * 1e6)  # km² → m²
        )

        # Base flow (normal Yamuna discharge through Delhi)
        base_discharge = self._manning_discharge(self.base_depth)

        # Total discharge at upstream boundary (Palla)
        total_discharge_palla = base_discharge + rainfall_discharge_m3s

        # ============================================
        # STEP 2: Compute hydraulics at each station
        # ============================================
        stations_list = list(YAMUNA_STATIONS.items())
        predictions = []
        
        for i, (name, station) in enumerate(stations_list):
            # Distance from upstream boundary
            distance_km = station["km_marker"]
            distance_m = distance_km * 1000

            # Discharge attenuation (flood wave attenuates downstream)
            # Simple exponential decay: Q(x) = Q₀ × e^(-α×x)
            # α is attenuation coefficient (~0.00002 per meter for Yamuna)
            attenuation = np.exp(-0.00002 * distance_m)
            station_discharge = total_discharge_palla * attenuation

            # Add local tributary inflow (Najafgarh drain adds ~500 m³/s during floods)
            if name == "Nizamuddin_Bridge":
                station_discharge += 300 * (upstream_rainfall_mm / 100)  # Scale with rainfall
            if name == "Okhla_Barrage":
                station_discharge += 200 * (upstream_rainfall_mm / 100)

            # ============================================
            # STEP 3: Inverse Manning's — find depth from discharge
            # ============================================
            predicted_depth = self._depth_from_discharge(station_discharge)

            # Compute flow area and velocity
            flow_area = self._flow_area(predicted_depth)
            wetted_perimeter = self._wetted_perimeter(predicted_depth)
            hydraulic_radius = flow_area / wetted_perimeter if wetted_perimeter > 0 else 0
            velocity = station_discharge / flow_area if flow_area > 0 else 0

            # ============================================
            # STEP 4: Flood wave travel time (kinematic wave)
            # ============================================
            # Kinematic wave celerity: c = (5/3) × V
            wave_celerity = (5.0 / 3.0) * velocity if velocity > 0 else 1.0
            travel_time_seconds = distance_m / wave_celerity if wave_celerity > 0 else 0
            travel_time_hours = travel_time_seconds / 3600.0

            # Convert to arrival datetime
            arrival_dt = datetime.now() + timedelta(hours=travel_time_hours)
            arrival_str = arrival_dt.strftime("%Y-%m-%d %H:%M IST")

            # ============================================
            # STEP 5: Danger level assessment
            # ============================================
            # Convert predicted depth to absolute level
            # Datum offset: Palla datum is ~198.5m, each station has its own
            datum_offsets = {
                "Palla": 198.50,
                "Old_Railway_Bridge": 199.00,
                "ITO_Barrage": 199.50,
                "Nizamuddin_Bridge": 198.80,
                "Okhla_Barrage": 197.50,
            }
            datum = datum_offsets.get(name, 198.5)
            absolute_level = datum + predicted_depth

            above_danger = absolute_level > station["danger_level_m"]
            excess = max(0, absolute_level - station["danger_level_m"])

            # Risk classification
            if excess > 2.0:
                risk = "CRITICAL"
            elif excess > 0.5:
                risk = "HIGH"
            elif excess > 0:
                risk = "MEDIUM"
            elif predicted_depth > self.base_depth * 1.5:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            predictions.append(StationPrediction(
                name=name.replace("_", " "),
                lat=station["lat"],
                lon=station["lon"],
                km_marker=station["km_marker"],
                danger_level_m=station["danger_level_m"],
                predicted_depth_m=round(predicted_depth, 2),
                predicted_discharge_m3s=round(station_discharge, 1),
                predicted_velocity_ms=round(velocity, 2),
                arrival_time_hours=round(travel_time_hours, 1),
                arrival_datetime=arrival_str,
                above_danger=above_danger,
                excess_depth_m=round(excess, 2),
                risk_level=risk,
                flow_area_m2=round(flow_area, 1),
                hydraulic_radius_m=round(hydraulic_radius, 2),
                wetted_perimeter_m=round(wetted_perimeter, 1)
            ))

        # ============================================
        # STEP 6: Generate summary and warning
        # ============================================
        peak_station = max(predictions, key=lambda p: p.predicted_discharge_m3s)
        highest_risk = max(predictions, key=lambda p: ["LOW","MEDIUM","HIGH","CRITICAL"].index(p.risk_level))

        summary = self._generate_summary(predictions, upstream_rainfall_mm)
        warning = self._generate_warning(predictions, upstream_rainfall_mm)

        return FloodPrediction(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            upstream_rainfall_mm=upstream_rainfall_mm,
            base_water_level_m=base_water_level_m,
            stations=predictions,
            peak_discharge_m3s=round(peak_station.predicted_discharge_m3s, 1),
            peak_station=peak_station.name,
            highest_risk=highest_risk.risk_level,
            summary=summary,
            warning_message=warning
        )

    def _flow_area(self, depth: float) -> float:
        """Trapezoidal channel cross-sectional area."""
        # A = (b + z×d) × d, where b=base width, z=side slope, d=depth
        if depth <= self.base_depth:
            return (self.base_width + self.side_slope * depth) * depth
        else:
            # Main channel + floodplain
            main = (self.base_width + self.side_slope * self.base_depth) * self.base_depth
            flood_depth = depth - self.base_depth
            flood = self.floodplain_width * flood_depth
            return main + flood

    def _wetted_perimeter(self, depth: float) -> float:
        """Trapezoidal channel wetted perimeter."""
        if depth <= self.base_depth:
            # P = b + 2d × sqrt(1 + z²)
            return self.base_width + 2 * depth * np.sqrt(1 + self.side_slope**2)
        else:
            main_p = self.base_width + 2 * self.base_depth * np.sqrt(1 + self.side_slope**2)
            flood_depth = depth - self.base_depth
            flood_p = 2 * flood_depth + self.floodplain_width
            return main_p + flood_p

    def _manning_discharge(self, depth: float) -> float:
        """Calculate discharge using Manning's equation for given depth."""
        A = self._flow_area(depth)
        P = self._wetted_perimeter(depth)
        if P <= 0:
            return 0
        R = A / P
        Q = (1.0 / self.n) * A * R**(2.0/3.0) * self.slope**0.5
        return Q

    def _depth_from_discharge(self, target_Q: float,
                               tol: float = 0.01,
                               max_iter: int = 100) -> float:
        """
        Inverse Manning's: find depth that produces target discharge.
        Uses bisection method (robust, guaranteed convergence).
        """
        d_low, d_high = 0.1, 15.0  # Search range: 0.1m to 15m

        for _ in range(max_iter):
            d_mid = (d_low + d_high) / 2.0
            Q_mid = self._manning_discharge(d_mid)

            if abs(Q_mid - target_Q) < tol:
                return d_mid
            elif Q_mid < target_Q:
                d_low = d_mid
            else:
                d_high = d_mid

        return (d_low + d_high) / 2.0

    def _generate_summary(self, predictions: List[StationPrediction],
                          rainfall_mm: float) -> str:
        """Generate technical summary."""
        danger_stations = [p for p in predictions if p.above_danger]
        if danger_stations:
            names = ", ".join(p.name for p in danger_stations)
            return (
                f"With {rainfall_mm}mm upstream rainfall, {len(danger_stations)} of "
                f"{len(predictions)} Yamuna stations are predicted to breach danger level: "
                f"{names}. Peak discharge: {max(p.predicted_discharge_m3s for p in predictions):.0f} m\u00B3/s. "
                f"Earliest breach at {danger_stations[0].name} in {danger_stations[0].arrival_time_hours:.1f} hours."
            )
        else:
            return (
                f"With {rainfall_mm}mm upstream rainfall, no Yamuna stations are predicted to breach "
                f"danger level. Peak discharge: {max(p.predicted_discharge_m3s for p in predictions):.0f} m\u00B3/s. "
                f"Continue monitoring."
            )

    def _generate_warning(self, predictions: List[StationPrediction],
                          rainfall_mm: float) -> str:
        """Generate farmer/citizen-facing warning message."""
        danger_stations = [p for p in predictions if p.above_danger]

        if not danger_stations:
            if any(p.risk_level == "MEDIUM" for p in predictions):
                return (
                    f"\u26A0\uFE0F YELLOW ALERT: Yamuna water levels rising due to {rainfall_mm}mm upstream rainfall. "
                    f"No danger level breach expected, but low-lying areas near the riverbank should stay alert. "
                    f"Next update in 3 hours."
                )
            return (
                f"\u2705 ALL CLEAR: {rainfall_mm}mm upstream rainfall detected. Yamuna water levels within normal range. "
                f"No flood risk for Delhi at this time."
            )

        # Build station-specific warnings
        warnings = []
        for p in sorted(danger_stations, key=lambda x: x.arrival_time_hours):
            warnings.append(
                f"{p.name}: water level predicted {p.excess_depth_m:.1f}m ABOVE danger mark, "
                f"arriving in {p.arrival_time_hours:.1f} hours ({p.arrival_datetime})"
            )

        highest = max(danger_stations, key=lambda p: p.excess_depth_m)
        alert_level = "\U0001F534 RED ALERT" if highest.excess_depth_m > 1.5 else "\U0001F7E0 ORANGE ALERT"

        return (
            f"{alert_level}: YAMUNA FLOOD WARNING FOR DELHI\n"
            f"Upstream rainfall: {rainfall_mm}mm | "
            f"Peak discharge: {max(p.predicted_discharge_m3s for p in predictions):.0f} m\u00B3/s\n\n"
            + "\n".join(warnings) +
            f"\n\nIMMEDIATE ACTION: Evacuate low-lying areas near {', '.join(p.name for p in danger_stations[:2])}. "
            f"Move to higher ground. Avoid crossing Yamuna bridges."
        )

    def scenario_analysis(self, rainfall_values: List[float] = None) -> Dict:
        """
        Run multiple rainfall scenarios to show flood risk curve.
        Useful for dashboard: "What happens if we get 50mm? 100mm? 150mm?"
        """
        if rainfall_values is None:
            rainfall_values = [10, 25, 50, 75, 100, 125, 150, 200]

        results = []
        for rain in rainfall_values:
            pred = self.predict(rain)
            danger_count = sum(1 for s in pred.stations if s.above_danger)
            max_excess = max((s.excess_depth_m for s in pred.stations), default=0)
            results.append({
                "rainfall_mm": rain,
                "peak_discharge_m3s": pred.peak_discharge_m3s,
                "stations_above_danger": danger_count,
                "max_excess_depth_m": round(max_excess, 2),
                "highest_risk": pred.highest_risk,
                "earliest_arrival_hrs": round(min(s.arrival_time_hours for s in pred.stations if s.above_danger), 1) if danger_count > 0 else None
            })

        return {
            "scenarios": results,
            "danger_threshold_mm": next(
                (r["rainfall_mm"] for r in results if r["stations_above_danger"] > 0),
                None
            )
        }


# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    model = ManningFloodModel()

    # Test 1: Moderate rainfall (50mm)
    print("=" * 60)
    print("SCENARIO 1: 50mm upstream rainfall")
    print("=" * 60)
    pred = model.predict(upstream_rainfall_mm=50)
    print(f"\n{pred.warning_message}\n")
    print(f"{'Station':<25} {'Depth(m)':>8} {'Q(m3/s)':>10} {'V(m/s)':>8} {'Arrival':>8} {'Risk':>10}")
    print("-" * 75)
    for s in pred.stations:
        danger_mark = " ***" if s.above_danger else ""
        print(f"{s.name:<25} {s.predicted_depth_m:>8.2f} {s.predicted_discharge_m3s:>10.1f} "
              f"{s.predicted_velocity_ms:>8.2f} {s.arrival_time_hours:>7.1f}h {s.risk_level:>10}{danger_mark}")

    # Test 2: Heavy rainfall (120mm — like 2023 event)
    print("\n" + "=" * 60)
    print("SCENARIO 2: 120mm upstream rainfall (2023-like event)")
    print("=" * 60)
    pred2 = model.predict(upstream_rainfall_mm=120)
    print(f"\n{pred2.warning_message}\n")
    for s in pred2.stations:
        danger_mark = " ***" if s.above_danger else ""
        print(f"{s.name:<25} depth={s.predicted_depth_m:.2f}m, "
              f"Q={s.predicted_discharge_m3s:.0f} m\u00B3/s, "
              f"arrives in {s.arrival_time_hours:.1f}h [{s.risk_level}]{danger_mark}")

    # Test 3: Scenario analysis
    print("\n" + "=" * 60)
    print("SCENARIO ANALYSIS: Rainfall vs Flood Risk")
    print("=" * 60)
    analysis = model.scenario_analysis()
    print(f"\n{'Rain(mm)':>10} {'Peak Q':>10} {'Danger':>8} {'Excess(m)':>10} {'Risk':>10}")
    print("-" * 55)
    for r in analysis["scenarios"]:
        print(f"{r['rainfall_mm']:>10} {r['peak_discharge_m3s']:>10.0f} "
              f"{r['stations_above_danger']:>8} {r['max_excess_depth_m']:>10.2f} {r['highest_risk']:>10}")

    threshold = analysis["danger_threshold_mm"]
    if threshold:
        print(f"\n\u26A0\uFE0F Danger threshold: {threshold}mm upstream rainfall")
    else:
        print(f"\n\u2705 No danger breach in tested range (up to 200mm)")

    print("\n\u2713 Module 2 (Manning's Equation Flood Prediction) complete.")
