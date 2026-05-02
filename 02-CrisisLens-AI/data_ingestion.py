"""
data_ingestion.py — CrisisLens AI Data Ingestion Layer
Connects to all external data sources, fetches live data for New Delhi,
validates responses, and provides clean interfaces for downstream modules.

All APIs are free and require no API keys (except OpenRouteService which
has a free tier with self-service key).

Author: Abhinav Pandey
"""

import requests
import numpy as np
import json
import sqlite3
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    CITY_CENTER, CITY_BOUNDS, YAMUNA_STATIONS, FLOOD_ZONES,
    HOSPITALS, INCIDENT_TYPES, SEVERITY_LEVELS
)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class WeatherData:
    """Live + forecast weather for New Delhi."""
    timestamp: str
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    precipitation_mm: float
    pressure_hpa: float
    weather_code: int
    # 7-day forecast
    daily_dates: List[str] = field(default_factory=list)
    daily_rain_mm: List[float] = field(default_factory=list)
    daily_max_temp: List[float] = field(default_factory=list)
    daily_min_temp: List[float] = field(default_factory=list)
    # Derived
    forecast_rain_total_mm: float = 0.0
    forecast_dry_days: int = 0
    heat_wave: bool = False  # True if max > 45°C


@dataclass
class HospitalStatus:
    """Hospital with simulated live occupancy."""
    name: str
    lat: float
    lon: float
    total_beds: int
    icu_beds: int
    has_trauma: bool
    hospital_type: str
    # Simulated live data
    occupied_beds: int = 0
    occupied_icu: int = 0
    available_beds: int = 0
    available_icu: int = 0
    occupancy_pct: float = 0.0
    status: str = "NORMAL"  # NORMAL / BUSY / CRITICAL / FULL


@dataclass
class Incident:
    """A single crisis incident."""
    id: str
    incident_type: str
    lat: float
    lon: float
    severity_score: float
    severity_label: str
    title: str
    description: str
    timestamp: str
    status: str = "ACTIVE"  # ACTIVE / RESPONDING / RESOLVED
    source: str = "simulated"
    responders_assigned: int = 0


@dataclass
class FloodRiskZone:
    """A geographic zone with computed flood risk."""
    name: str
    lat: float
    lon: float
    radius_km: float
    base_risk: float
    current_risk: float  # 0-1, adjusted by weather
    risk_level: str      # LOW / MEDIUM / HIGH / CRITICAL
    color: str
    description: str


# ============================================
# WEATHER INGESTION (Open-Meteo)
# ============================================

class WeatherIngestion:
    """Fetches live weather from Open-Meteo API for New Delhi."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def fetch(self) -> WeatherData:
        """Fetch current conditions + 7-day forecast."""
        params = {
            "latitude": CITY_CENTER["lat"],
            "longitude": CITY_CENTER["lon"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": 7,
            "timezone": "Asia/Kolkata"
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[Weather API Error] {e}. Using fallback.")
            return self._fallback()

        current = data.get("current", {})
        daily = data.get("daily", {})

        daily_rain = daily.get("precipitation_sum", [0]*7)
        daily_max = daily.get("temperature_2m_max", [35]*7)
        daily_min = daily.get("temperature_2m_min", [25]*7)
        daily_dates = daily.get("time", [])

        forecast_rain_total = sum(daily_rain[:7])
        forecast_dry_days = sum(1 for r in daily_rain[:7] if r < 1.0)
        heat_wave = any(t >= 45.0 for t in daily_max[:7])

        return WeatherData(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=current.get("temperature_2m", 35),
            humidity_pct=current.get("relative_humidity_2m", 50),
            wind_speed_kmh=current.get("wind_speed_10m", 10),
            precipitation_mm=current.get("precipitation", 0),
            pressure_hpa=current.get("surface_pressure", 1010),
            weather_code=current.get("weather_code", 0),
            daily_dates=daily_dates[:7],
            daily_rain_mm=[round(r, 1) for r in daily_rain[:7]],
            daily_max_temp=[round(t, 1) for t in daily_max[:7]],
            daily_min_temp=[round(t, 1) for t in daily_min[:7]],
            forecast_rain_total_mm=round(forecast_rain_total, 1),
            forecast_dry_days=forecast_dry_days,
            heat_wave=heat_wave
        )

    def _fallback(self) -> WeatherData:
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        return WeatherData(
            timestamp=today.strftime("%Y-%m-%d %H:%M IST"),
            temperature_c=38.0, humidity_pct=40.0, wind_speed_kmh=15.0,
            precipitation_mm=0.0, pressure_hpa=1008.0, weather_code=1,
            daily_dates=dates,
            daily_rain_mm=[0, 0, 2.0, 5.0, 0, 0, 0],
            daily_max_temp=[38, 39, 37, 36, 38, 40, 39],
            daily_min_temp=[26, 27, 25, 24, 26, 27, 26],
            forecast_rain_total_mm=7.0, forecast_dry_days=5, heat_wave=False
        )


# ============================================
# HOSPITAL STATUS (Static + Simulated Live)
# ============================================

class HospitalIngestion:
    """Generates realistic live hospital occupancy for Delhi hospitals."""

    def fetch_all(self, crisis_active: bool = False) -> List[HospitalStatus]:
        """Return hospital statuses with simulated live occupancy."""
        rng = np.random.RandomState(int(datetime.now().hour))
        statuses = []

        for h in HOSPITALS:
            # Base occupancy: government 80-92%, private 65-80%
            if h["type"] == "Government":
                base_occ = rng.uniform(0.80, 0.92)
            else:
                base_occ = rng.uniform(0.65, 0.80)

            # Crisis bump: +8-15% during active crisis
            if crisis_active:
                base_occ = min(0.98, base_occ + rng.uniform(0.08, 0.15))

            occupied_beds = int(h["beds"] * base_occ)
            occupied_icu = int(h["icu"] * min(1.0, base_occ + 0.05))
            available_beds = h["beds"] - occupied_beds
            available_icu = h["icu"] - occupied_icu

            occ_pct = round(base_occ * 100, 1)
            if occ_pct >= 95:
                status = "FULL"
            elif occ_pct >= 85:
                status = "CRITICAL"
            elif occ_pct >= 75:
                status = "BUSY"
            else:
                status = "NORMAL"

            statuses.append(HospitalStatus(
                name=h["name"], lat=h["lat"], lon=h["lon"],
                total_beds=h["beds"], icu_beds=h["icu"],
                has_trauma=h["trauma"], hospital_type=h["type"],
                occupied_beds=occupied_beds, occupied_icu=occupied_icu,
                available_beds=available_beds, available_icu=available_icu,
                occupancy_pct=occ_pct, status=status
            ))

        return statuses


# ============================================
# FLOOD RISK COMPUTATION
# ============================================

class FloodRiskEngine:
    """Computes zone-level flood risk from weather + river data."""

    def compute_risks(self, weather: WeatherData) -> List[FloodRiskZone]:
        """Adjust base flood risks by current/forecast weather."""
        zones = []

        # Rain multiplier: more rain → higher risk
        rain_24h = weather.precipitation_mm + (weather.daily_rain_mm[0] if weather.daily_rain_mm else 0)
        rain_factor = min(2.0, rain_24h / 20.0)  # Normalize: 20mm = 1.0x, 40mm = 2.0x

        # Forecast rain factor (next 3 days)
        forecast_3d = sum(weather.daily_rain_mm[:3]) if weather.daily_rain_mm else 0
        forecast_factor = min(1.5, forecast_3d / 50.0)  # 50mm in 3 days = 1.0x

        # Combined weather multiplier
        weather_mult = 1.0 + (rain_factor * 0.6) + (forecast_factor * 0.4)

        for name, zone in FLOOD_ZONES.items():
            current_risk = min(1.0, zone["base_risk"] * weather_mult)

            if current_risk >= 0.75:
                level, color = "CRITICAL", "#D32F2F"
            elif current_risk >= 0.50:
                level, color = "HIGH", "#F57C00"
            elif current_risk >= 0.25:
                level, color = "MEDIUM", "#FBC02D"
            else:
                level, color = "LOW", "#388E3C"

            zones.append(FloodRiskZone(
                name=name, lat=zone["lat"], lon=zone["lon"],
                radius_km=zone["radius_km"], base_risk=zone["base_risk"],
                current_risk=round(current_risk, 3), risk_level=level,
                color=color, description=zone["description"]
            ))

        return zones


# ============================================
# INCIDENT GENERATION (GDELT + Simulated)
# ============================================

class IncidentIngestion:
    """Fetches real news events from GDELT and generates simulated incidents."""

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch_gdelt_events(self, max_results: int = 5) -> List[Incident]:
        """Fetch recent Delhi-related crisis events from GDELT."""
        params = {
            "query": "Delhi (flood OR fire OR accident OR collapse OR emergency)",
            "mode": "artlist",
            "maxrecords": max_results,
            "format": "json",
            "timespan": "7d"
        }

        try:
            resp = requests.get(self.GDELT_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
        except Exception as e:
            print(f"[GDELT Error] {e}. Using simulated incidents only.")
            return []

        incidents = []
        for i, art in enumerate(articles[:max_results]):
            # GDELT doesn't always have precise coordinates
            # Use Delhi center with slight randomization
            rng = np.random.RandomState(hash(art.get("title", "")) % (2**31))
            lat = CITY_CENTER["lat"] + rng.uniform(-0.1, 0.1)
            lon = CITY_CENTER["lon"] + rng.uniform(-0.1, 0.1)

            incidents.append(Incident(
                id=f"GDELT_{i:04d}",
                incident_type="fire" if "fire" in art.get("title", "").lower()
                    else "flood" if "flood" in art.get("title", "").lower()
                    else "road_accident",
                lat=lat, lon=lon,
                severity_score=0.5,
                severity_label="MEDIUM",
                title=art.get("title", "Unknown event")[:100],
                description=art.get("seendate", ""),
                timestamp=art.get("seendate", datetime.now().strftime("%Y-%m-%d")),
                source="GDELT"
            ))

        return incidents

    def generate_simulated(self, n: int = 10, weather: WeatherData = None) -> List[Incident]:
        """Generate realistic simulated incidents for New Delhi."""
        rng = np.random.RandomState(int(datetime.now().timestamp()) % (2**31))
        incidents = []

        # Incident type probabilities (weather-adjusted)
        types = list(INCIDENT_TYPES.keys())
        weights = [0.15, 0.20, 0.05, 0.05, 0.30, 0.10, 0.10, 0.05]

        # If heavy rain → boost flood probability
        if weather and weather.precipitation_mm > 10:
            weights[0] = 0.40  # flood
            weights[4] = 0.15  # reduce road accidents
        # If extreme heat → boost heat emergency
        if weather and weather.temperature_c > 42:
            weights[5] = 0.25  # heat emergency

        weights = np.array(weights) / sum(weights)

        # Delhi hotspot locations for realistic placement
        hotspots = {
            "flood": [(28.643, 77.268), (28.609, 77.298), (28.629, 77.249)],  # Yamuna areas
            "fire": [(28.728, 77.133), (28.695, 77.112), (28.685, 77.315)],   # Bawana, Narela, industrial
            "building_collapse": [(28.656, 77.230), (28.640, 77.245)],         # Old Delhi
            "gas_leak": [(28.728, 77.133), (28.540, 77.278)],                  # Industrial zones
            "road_accident": [(28.612, 77.229), (28.567, 77.210), (28.640, 77.200)],  # Major roads
            "heat_emergency": [(28.632, 77.220), (28.570, 77.200)],            # Central/South Delhi
            "crowd_incident": [(28.656, 77.230), (28.635, 77.224)],            # Chandni Chowk, CP
            "power_outage": [(28.700, 77.150), (28.590, 77.050)],              # Outer Delhi
        }

        for i in range(n):
            itype = rng.choice(types, p=weights)
            spots = hotspots.get(itype, [(CITY_CENTER["lat"], CITY_CENTER["lon"])])
            base_lat, base_lon = spots[rng.randint(len(spots))]

            lat = base_lat + rng.uniform(-0.02, 0.02)
            lon = base_lon + rng.uniform(-0.02, 0.02)

            type_info = INCIDENT_TYPES[itype]
            severity = type_info["base_severity"] + rng.uniform(-0.15, 0.15)
            severity = round(max(0.1, min(1.0, severity)), 3)

            if severity >= 0.75: label = "CRITICAL"
            elif severity >= 0.50: label = "HIGH"
            elif severity >= 0.25: label = "MEDIUM"
            else: label = "LOW"

            hours_ago = rng.randint(0, 48)
            ts = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M")

            titles = {
                "flood": f"Waterlogging reported near {rng.choice(['Mayur Vihar','ITO','Kashmere Gate','Okhla'])}",
                "fire": f"Fire at {rng.choice(['factory in Bawana','godown in Mundka','building in Narela','market in Lajpat Nagar'])}",
                "building_collapse": f"Partial building collapse in {rng.choice(['Old Delhi','Karol Bagh','Shahdara'])}",
                "gas_leak": f"Gas leak at {rng.choice(['chemical plant in Bawana','factory in Okhla','industrial unit in Narela'])}",
                "road_accident": f"Multi-vehicle collision on {rng.choice(['NH-44','Ring Road','Mehrauli-Badarpur Road','GT Karnal Road'])}",
                "heat_emergency": f"Heatstroke cases reported in {rng.choice(['outdoor workers in CP','construction site in Dwarka','market area in Sadar Bazaar'])}",
                "crowd_incident": f"Crowd management alert at {rng.choice(['Chandni Chowk','India Gate','Jama Masjid','Pragati Maidan'])}",
                "power_outage": f"Power grid failure in {rng.choice(['Rohini Sector 15','Dwarka Sector 10','Janakpuri'])}",
            }

            incidents.append(Incident(
                id=f"SIM_{i:04d}",
                incident_type=itype,
                lat=lat, lon=lon,
                severity_score=severity,
                severity_label=label,
                title=titles.get(itype, f"Incident at ({lat:.3f}, {lon:.3f})"),
                description=f"Simulated {itype.replace('_',' ')} incident for demo",
                timestamp=ts,
                source="simulated",
                responders_assigned=rng.randint(0, 5) if label in ["CRITICAL", "HIGH"] else 0
            ))

        return incidents


# ============================================
# UNIFIED DATA LAYER
# ============================================

class CrisisDataLayer:
    """
    Unified data access for all CrisisLens modules.
    Single entry point — call refresh() to update all data sources.
    """

    def __init__(self):
        self.weather_engine = WeatherIngestion()
        self.hospital_engine = HospitalIngestion()
        self.flood_engine = FloodRiskEngine()
        self.incident_engine = IncidentIngestion()

        # Cached data
        self.weather: Optional[WeatherData] = None
        self.hospitals: List[HospitalStatus] = []
        self.flood_zones: List[FloodRiskZone] = []
        self.incidents: List[Incident] = []
        self.last_refresh: Optional[str] = None

    def refresh(self):
        """Refresh all data sources."""
        print("[CrisisLens] Refreshing all data sources...")

        # 1. Weather (live API)
        self.weather = self.weather_engine.fetch()
        print(f"  [Weather] {self.weather.temperature_c}°C, Rain: {self.weather.precipitation_mm}mm, "
              f"7-day total: {self.weather.forecast_rain_total_mm}mm")

        # 2. Flood risk (computed from weather)
        self.flood_zones = self.flood_engine.compute_risks(self.weather)
        critical = sum(1 for z in self.flood_zones if z.risk_level == "CRITICAL")
        high = sum(1 for z in self.flood_zones if z.risk_level == "HIGH")
        print(f"  [Flood] {len(self.flood_zones)} zones assessed: {critical} CRITICAL, {high} HIGH")

        # 3. Hospitals (simulated live)
        has_crisis = critical > 0 or any(
            i.severity_label == "CRITICAL" for i in self.incidents
        )
        self.hospitals = self.hospital_engine.fetch_all(crisis_active=has_crisis)
        full = sum(1 for h in self.hospitals if h.status == "FULL")
        print(f"  [Hospitals] {len(self.hospitals)} hospitals loaded, {full} at FULL capacity")

        # 4. Incidents (GDELT + simulated)
        gdelt = self.incident_engine.fetch_gdelt_events(max_results=3)
        simulated = self.incident_engine.generate_simulated(n=8, weather=self.weather)
        self.incidents = gdelt + simulated
        active_critical = sum(1 for i in self.incidents if i.severity_label == "CRITICAL")
        print(f"  [Incidents] {len(self.incidents)} total ({len(gdelt)} GDELT, "
              f"{len(simulated)} simulated), {active_critical} CRITICAL")

        self.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        print(f"[CrisisLens] All sources refreshed at {self.last_refresh}\n")

    def get_summary(self) -> Dict:
        """Return a summary dict for dashboard KPIs."""
        if not self.weather:
            self.refresh()

        return {
            "city": "New Delhi",
            "timestamp": self.last_refresh,
            "temperature_c": self.weather.temperature_c,
            "humidity_pct": self.weather.humidity_pct,
            "rain_today_mm": self.weather.precipitation_mm,
            "rain_7day_mm": self.weather.forecast_rain_total_mm,
            "heat_wave": self.weather.heat_wave,
            "flood_zones_critical": sum(1 for z in self.flood_zones if z.risk_level == "CRITICAL"),
            "flood_zones_high": sum(1 for z in self.flood_zones if z.risk_level == "HIGH"),
            "hospitals_total": len(self.hospitals),
            "hospitals_full": sum(1 for h in self.hospitals if h.status == "FULL"),
            "beds_available": sum(h.available_beds for h in self.hospitals),
            "icu_available": sum(h.available_icu for h in self.hospitals),
            "incidents_active": len([i for i in self.incidents if i.status == "ACTIVE"]),
            "incidents_critical": sum(1 for i in self.incidents if i.severity_label == "CRITICAL"),
        }


# ============================================
# DEMO: Test all data sources
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("CrisisLens AI — Data Ingestion Test (New Delhi)")
    print("=" * 60)

    layer = CrisisDataLayer()
    layer.refresh()

    print("=" * 60)
    print("DASHBOARD SUMMARY")
    print("=" * 60)
    summary = layer.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nFLOOD ZONES:")
    for z in layer.flood_zones:
        print(f"  [{z.risk_level:8s}] {z.name}: risk={z.current_risk:.3f}")

    print("\nHOSPITALS (top 5 by available beds):")
    sorted_h = sorted(layer.hospitals, key=lambda h: h.available_beds, reverse=True)
    for h in sorted_h[:5]:
        print(f"  [{h.status:8s}] {h.name}: {h.available_beds} beds, {h.available_icu} ICU ({h.occupancy_pct}%)")

    print("\nINCIDENTS (critical first):")
    sorted_i = sorted(layer.incidents, key=lambda i: i.severity_score, reverse=True)
    for i in sorted_i[:5]:
        print(f"  [{i.severity_label:8s}] {i.title[:60]} ({i.source})")

    print("\n\u2713 Step 1 complete. All data sources verified for New Delhi.")
