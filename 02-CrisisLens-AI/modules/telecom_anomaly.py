"""
telecom_anomaly.py — Module 4B: Telecom Anomaly Detection
Simulated cell tower network for New Delhi with realistic diurnal
traffic patterns, crisis-induced spike injection, and Z-score
anomaly detection for early crisis warning.

NOTE: Real telecom data is restricted under TRAI regulations.
This module uses a physics-informed simulation based on published
cell tower capacity parameters and known diurnal traffic curves.
The anomaly detection algorithm is real and transferable to any
time-series spike detection problem.

Author: Abhinav Pandey
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from config import CITY_CENTER, CITY_BOUNDS


@dataclass
class CellTower:
    """A single cell tower with traffic data."""
    id: str
    lat: float
    lon: float
    zone: str                    # Delhi zone name
    capacity: int                # Max simultaneous calls
    current_load: float          # Current load (0-1 fraction)
    baseline_load: float         # Expected load at this hour
    z_score: float               # How many std devs above baseline
    is_anomaly: bool             # True if z_score > threshold
    load_history_24h: List[float] = field(default_factory=list)


@dataclass
class TelecomAnomaly:
    """Detected anomaly cluster."""
    cluster_id: str
    center_lat: float
    center_lon: float
    radius_km: float
    tower_count: int             # Towers in anomaly cluster
    avg_z_score: float
    max_z_score: float
    confidence: float            # 0-1
    severity: str                # LOW / MEDIUM / HIGH / CRITICAL
    timestamp: str
    message: str


@dataclass
class TelecomSnapshot:
    """Full network snapshot at a point in time."""
    timestamp: str
    total_towers: int
    anomaly_towers: int
    anomaly_clusters: List[TelecomAnomaly]
    towers: List[CellTower]
    network_avg_load: float
    network_max_load: float
    crisis_detected: bool


class TelecomNetwork:
    """
    Simulated cell tower network for New Delhi.

    Physics-informed model:
    - Tower density: higher in central/commercial areas, lower in outer zones
    - Each tower: ~200 simultaneous call capacity
    - Diurnal pattern: load(t) = A × sin(2π×t/24 + φ) + B + noise
      peaks at 10AM and 7PM, troughs at 3AM
    - Crisis injection: nearby towers spike to 3-5x baseline

    Anomaly detection:
    - Rolling Z-score: z = (current - baseline_mean) / baseline_std
    - Threshold: z > 3.0 = anomaly (99.7% confidence)
    - Spatial clustering: 3+ anomalous towers within 2km = crisis alert
    """

    # Delhi zones with tower density (towers per km²)
    ZONES = {
        "Connaught Place / Central": {
            "lat": 28.632, "lon": 77.220,
            "radius_km": 2.5, "density": 25
        },
        "Chandni Chowk / Old Delhi": {
            "lat": 28.656, "lon": 77.230,
            "radius_km": 2.0, "density": 20
        },
        "South Delhi (Saket/GK)": {
            "lat": 28.530, "lon": 77.220,
            "radius_km": 3.0, "density": 18
        },
        "Dwarka / West Delhi": {
            "lat": 28.592, "lon": 77.047,
            "radius_km": 4.0, "density": 10
        },
        "Rohini / North Delhi": {
            "lat": 28.732, "lon": 77.118,
            "radius_km": 3.5, "density": 12
        },
        "Noida Border / East Delhi": {
            "lat": 28.630, "lon": 77.310,
            "radius_km": 3.0, "density": 14
        },
        "Mayur Vihar / Yamuna East": {
            "lat": 28.609, "lon": 77.298,
            "radius_km": 2.0, "density": 15
        },
        "IGI Airport / Aerocity": {
            "lat": 28.556, "lon": 77.100,
            "radius_km": 2.0, "density": 8
        },
    }

    # Anomaly detection threshold
    Z_THRESHOLD = 3.0
    CLUSTER_RADIUS_KM = 2.0
    MIN_CLUSTER_SIZE = 3

    def __init__(self, n_towers: int = 500, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.towers = self._generate_tower_grid(n_towers)

    def _generate_tower_grid(self, n_towers: int) -> List[CellTower]:
        """Generate realistic tower placement across Delhi zones."""
        towers = []
        tower_id = 0

        # Distribute towers proportional to zone density
        total_density = sum(z["density"] for z in self.ZONES.values())

        for zone_name, zone in self.ZONES.items():
            zone_towers = max(5, int(n_towers * zone["density"] / total_density))

            for _ in range(zone_towers):
                # Random position within zone radius
                angle = self.rng.uniform(0, 2 * np.pi)
                r = zone["radius_km"] * np.sqrt(self.rng.uniform(0, 1))
                lat_offset = r * np.cos(angle) / 111.32
                lon_offset = r * np.sin(angle) / (111.32 * np.cos(np.radians(zone["lat"])))

                towers.append(CellTower(
                    id=f"TWR_{tower_id:04d}",
                    lat=round(zone["lat"] + lat_offset, 5),
                    lon=round(zone["lon"] + lon_offset, 5),
                    zone=zone_name,
                    capacity=self.rng.choice([150, 200, 250, 300]),
                    current_load=0.0,
                    baseline_load=0.0,
                    z_score=0.0,
                    is_anomaly=False,
                    load_history_24h=[]
                ))
                tower_id += 1

                if tower_id >= n_towers:
                    return towers

        return towers

    def _diurnal_load(self, hour: float, zone: str) -> float:
        """
        Compute expected network load for a given hour.

        Model: load(t) = A₁×sin(2π(t-10)/24) + A₂×sin(2π(t-19)/12) + B + noise

        Two peaks: morning business (10AM) and evening social (7PM).
        Trough at 3AM. Commercial zones peak earlier; residential peak later.
        """
        # Zone-specific phase shift
        phase_shift = {
            "Connaught Place / Central": -0.5,    # Peaks earlier (business)
            "Chandni Chowk / Old Delhi": 0.0,
            "South Delhi (Saket/GK)": 0.5,        # Peaks slightly later
            "Dwarka / West Delhi": 1.0,            # Residential, peaks evening
            "Rohini / North Delhi": 1.0,
            "Noida Border / East Delhi": 0.5,
            "Mayur Vihar / Yamuna East": 0.5,
            "IGI Airport / Aerocity": -1.0,        # Early morning travel peak
        }
        phase = phase_shift.get(zone, 0)

        # Double-peak diurnal model
        morning_peak = 0.20 * np.sin(2 * np.pi * (hour - 10 + phase) / 24)
        evening_peak = 0.15 * np.sin(2 * np.pi * (hour - 19 + phase) / 12)

        # Base load: 0.45 (never drops below 30% even at 3AM)
        base = 0.45

        load = base + morning_peak + evening_peak
        return np.clip(load, 0.15, 0.85)

    def simulate_snapshot(self, crisis_location: Tuple[float, float] = None,
                          crisis_intensity: float = 0.0,
                          current_hour: float = None) -> TelecomSnapshot:
        """
        Generate a network snapshot with optional crisis injection.

        Parameters:
            crisis_location: (lat, lon) of crisis event. If None, normal traffic.
            crisis_intensity: 0-1, how severe the crisis is (1.0 = max spike)
            current_hour: Hour of day (0-24). If None, use current time.
        """
        if current_hour is None:
            now = datetime.now()
            current_hour = now.hour + now.minute / 60.0

        anomaly_towers = 0

        for tower in self.towers:
            # 1. Compute baseline load for this hour and zone
            baseline = self._diurnal_load(current_hour, tower.zone)
            noise = self.rng.normal(0, 0.05)  # ±5% normal variation
            normal_load = np.clip(baseline + noise, 0.1, 0.90)

            tower.baseline_load = round(baseline, 3)

            # 2. Crisis spike injection
            crisis_boost = 0.0
            if crisis_location and crisis_intensity > 0:
                dist_km = self._haversine_km(
                    (tower.lat, tower.lon), crisis_location
                )
                # Inverse-square falloff: towers closer to crisis spike more
                if dist_km < 5.0:
                    proximity = max(0, 1.0 - (dist_km / 5.0))
                    crisis_boost = crisis_intensity * proximity**2 * self.rng.uniform(0.6, 1.0)

            tower.current_load = round(min(1.0, normal_load + crisis_boost), 3)

            # 3. Generate 24h history for Z-score calculation
            history = []
            for h in range(24):
                hist_load = self._diurnal_load(h, tower.zone)
                hist_noise = self.rng.normal(0, 0.05)
                history.append(np.clip(hist_load + hist_noise, 0.1, 0.90))
            tower.load_history_24h = history

            # 4. Z-score anomaly detection
            hist_mean = np.mean(history)
            hist_std = max(0.01, np.std(history))
            tower.z_score = round((tower.current_load - hist_mean) / hist_std, 2)
            tower.is_anomaly = tower.z_score > self.Z_THRESHOLD

            if tower.is_anomaly:
                anomaly_towers += 1

        # 5. Spatial clustering of anomalies
        clusters = self._cluster_anomalies()

        # 6. Build snapshot
        loads = [t.current_load for t in self.towers]

        return TelecomSnapshot(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            total_towers=len(self.towers),
            anomaly_towers=anomaly_towers,
            anomaly_clusters=clusters,
            towers=self.towers,
            network_avg_load=round(np.mean(loads), 3),
            network_max_load=round(np.max(loads), 3),
            crisis_detected=len(clusters) > 0
        )

    def _cluster_anomalies(self) -> List[TelecomAnomaly]:
        """Cluster nearby anomalous towers into crisis alerts."""
        anomalous = [t for t in self.towers if t.is_anomaly]

        if len(anomalous) < self.MIN_CLUSTER_SIZE:
            return []

        # Simple DBSCAN-like clustering
        visited = set()
        clusters = []
        cluster_id = 0

        for tower in anomalous:
            if tower.id in visited:
                continue

            # Find all anomalous towers within cluster radius
            nearby = []
            for other in anomalous:
                if other.id not in visited:
                    dist = self._haversine_km(
                        (tower.lat, tower.lon),
                        (other.lat, other.lon)
                    )
                    if dist <= self.CLUSTER_RADIUS_KM:
                        nearby.append(other)
                        visited.add(other.id)

            if len(nearby) >= self.MIN_CLUSTER_SIZE:
                # Compute cluster center and stats
                lats = [t.lat for t in nearby]
                lons = [t.lon for t in nearby]
                z_scores = [t.z_score for t in nearby]

                center_lat = np.mean(lats)
                center_lon = np.mean(lons)
                avg_z = np.mean(z_scores)
                max_z = np.max(z_scores)

                # Confidence based on cluster size and Z-score
                confidence = min(1.0, (len(nearby) / 10.0) * (avg_z / 5.0))

                if avg_z > 5.0:
                    severity = "CRITICAL"
                elif avg_z > 4.0:
                    severity = "HIGH"
                elif avg_z > 3.5:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"

                # Compute cluster radius
                max_dist = max(
                    self._haversine_km((center_lat, center_lon), (t.lat, t.lon))
                    for t in nearby
                )

                # Identify likely zone
                zone_counts = {}
                for t in nearby:
                    zone_counts[t.zone] = zone_counts.get(t.zone, 0) + 1
                likely_zone = max(zone_counts, key=zone_counts.get)

                clusters.append(TelecomAnomaly(
                    cluster_id=f"CLU_{cluster_id:03d}",
                    center_lat=round(center_lat, 5),
                    center_lon=round(center_lon, 5),
                    radius_km=round(max_dist, 2),
                    tower_count=len(nearby),
                    avg_z_score=round(avg_z, 2),
                    max_z_score=round(max_z, 2),
                    confidence=round(confidence, 3),
                    severity=severity,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
                    message=(
                        f"Unusual call activity detected near {likely_zone}. "
                        f"{len(nearby)} towers showing {avg_z:.1f}\u03C3 above baseline. "
                        f"Possible emergency event — recommend field verification."
                    )
                ))
                cluster_id += 1

        return clusters

    def _haversine_km(self, p1: Tuple[float, float],
                      p2: Tuple[float, float]) -> float:
        """Distance in km."""
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))


# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("CrisisLens AI — Telecom Anomaly Detection (New Delhi)")
    print("=" * 60)

    network = TelecomNetwork(n_towers=500, seed=42)
    print(f"Network: {len(network.towers)} towers across {len(network.ZONES)} zones\n")

    # Test 1: Normal traffic (no crisis)
    print("SCENARIO 1: Normal traffic (no crisis)")
    snap1 = network.simulate_snapshot()
    print(f"  Avg load: {snap1.network_avg_load:.3f}")
    print(f"  Max load: {snap1.network_max_load:.3f}")
    print(f"  Anomaly towers: {snap1.anomaly_towers}")
    print(f"  Crisis clusters: {len(snap1.anomaly_clusters)}")
    print(f"  Crisis detected: {snap1.crisis_detected}")

    # Test 2: Crisis at ITO (flood)
    print("\nSCENARIO 2: Crisis at ITO Barrage (flood event, intensity=0.8)")
    snap2 = network.simulate_snapshot(
        crisis_location=(28.629, 77.249),
        crisis_intensity=0.8
    )
    print(f"  Avg load: {snap2.network_avg_load:.3f}")
    print(f"  Max load: {snap2.network_max_load:.3f}")
    print(f"  Anomaly towers: {snap2.anomaly_towers}")
    print(f"  Crisis clusters: {len(snap2.anomaly_clusters)}")
    print(f"  Crisis detected: {snap2.crisis_detected}")

    for c in snap2.anomaly_clusters:
        print(f"\n  CLUSTER {c.cluster_id} [{c.severity}]:")
        print(f"    Location: ({c.center_lat:.4f}, {c.center_lon:.4f})")
        print(f"    Towers: {c.tower_count}, Avg Z: {c.avg_z_score}, Max Z: {c.max_z_score}")
        print(f"    Confidence: {c.confidence:.3f}")
        print(f"    {c.message}")

    # Test 3: Crisis at Bawana (fire at night)
    print("\nSCENARIO 3: Crisis at Bawana industrial zone (fire, intensity=0.9, 2AM)")
    snap3 = network.simulate_snapshot(
        crisis_location=(28.728, 77.133),
        crisis_intensity=0.9,
        current_hour=2.0  # 2AM — low baseline makes spikes more detectable
    )
    print(f"  Anomaly towers: {snap3.anomaly_towers}")
    print(f"  Crisis clusters: {len(snap3.anomaly_clusters)}")
    for c in snap3.anomaly_clusters:
        print(f"  [{c.severity}] {c.message}")

    print(f"\n\u2713 Module 4B (Telecom Anomaly Detection) complete.")
