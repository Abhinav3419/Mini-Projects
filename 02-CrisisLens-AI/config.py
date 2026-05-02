"""
config.py — CrisisLens AI Configuration
All constants, zone definitions, and API endpoints for New Delhi.

Author: Abhinav Pandey
"""

# ============================================
# NEW DELHI GEOGRAPHY
# ============================================
CITY = "New Delhi"
CITY_CENTER = {"lat": 28.6139, "lon": 77.2090}
CITY_BOUNDS = {
    "north": 28.88,
    "south": 28.40,
    "east": 77.35,
    "west": 76.84
}

# Yamuna River monitoring points (upstream → downstream)
YAMUNA_STATIONS = {
    "Palla": {"lat": 28.808, "lon": 77.132, "km_marker": 0, "danger_level_m": 204.83},
    "Old_Railway_Bridge": {"lat": 28.672, "lon": 77.243, "km_marker": 18, "danger_level_m": 205.33},
    "ITO_Barrage": {"lat": 28.629, "lon": 77.249, "km_marker": 24, "danger_level_m": 205.65},
    "Nizamuddin_Bridge": {"lat": 28.589, "lon": 77.248, "km_marker": 30, "danger_level_m": 204.50},
    "Okhla_Barrage": {"lat": 28.541, "lon": 77.278, "km_marker": 40, "danger_level_m": 203.00},
}

# Delhi flood-prone zones with risk weights
FLOOD_ZONES = {
    "Yamuna Floodplain (East)": {
        "lat": 28.643, "lon": 77.268, "radius_km": 3.0,
        "base_risk": 0.8, "description": "Low-lying areas along Yamuna east bank"
    },
    "Mayur Vihar": {
        "lat": 28.609, "lon": 77.298, "radius_km": 2.0,
        "base_risk": 0.6, "description": "Flood-prone residential area near Yamuna"
    },
    "ITO / Pragati Maidan": {
        "lat": 28.629, "lon": 77.249, "radius_km": 1.5,
        "base_risk": 0.7, "description": "Historic flooding zone near ITO barrage"
    },
    "Kashmere Gate / Civil Lines": {
        "lat": 28.668, "lon": 77.228, "radius_km": 2.0,
        "base_risk": 0.5, "description": "North Delhi near Old Railway Bridge"
    },
    "South Delhi (Sarita Vihar / Okhla)": {
        "lat": 28.541, "lon": 77.278, "radius_km": 2.5,
        "base_risk": 0.5, "description": "Near Okhla barrage, industrial zone"
    },
    "Central Delhi (Connaught Place)": {
        "lat": 28.632, "lon": 77.220, "radius_km": 2.0,
        "base_risk": 0.15, "description": "Elevated terrain, drainage issues only"
    },
    "West Delhi (Dwarka)": {
        "lat": 28.592, "lon": 77.047, "radius_km": 3.0,
        "base_risk": 0.1, "description": "Far from Yamuna, low flood risk"
    },
    "North Delhi (Rohini / Pitampura)": {
        "lat": 28.732, "lon": 77.118, "radius_km": 3.0,
        "base_risk": 0.2, "description": "Moderate risk from Najafgarh drain overflow"
    },
}

# ============================================
# MANNING'S EQUATION PARAMETERS (YAMUNA)
# ============================================
MANNING_PARAMS = {
    "n": 0.035,                    # Manning's roughness coefficient (natural river with vegetation)
    "channel_slope": 0.0002,       # Yamuna slope through Delhi (m/m) — very gentle
    "channel_width_m": 350,        # Average Yamuna width in Delhi
    "avg_depth_m": 3.5,            # Average depth at normal flow
    "floodplain_width_m": 1200,    # Width when flooded
    "danger_discharge_m3s": 8500,  # Discharge at danger level (approx)
}

# ============================================
# API ENDPOINTS (All free, no keys required)
# ============================================
APIS = {
    "weather": {
        "name": "Open-Meteo",
        "base_url": "https://api.open-meteo.com/v1/forecast",
        "cost": "Free",
        "key_required": False,
    },
    "routing": {
        "name": "OpenRouteService",
        "base_url": "https://api.openrouteservice.org/v2/directions",
        "cost": "Free (2000 req/day)",
        "key_required": True,
        "note": "Free API key from openrouteservice.org"
    },
    "geocoding": {
        "name": "Nominatim (OpenStreetMap)",
        "base_url": "https://nominatim.openstreetmap.org/search",
        "cost": "Free",
        "key_required": False,
    },
    "news_events": {
        "name": "GDELT",
        "base_url": "https://api.gdeltproject.org/api/v2/doc/doc",
        "cost": "Free",
        "key_required": False,
    },
}

# ============================================
# HOSPITAL DATA (Top 20 Delhi hospitals)
# ============================================
HOSPITALS = [
    {"name": "AIIMS Delhi", "lat": 28.5672, "lon": 77.2100, "beds": 2478, "icu": 250, "trauma": True, "type": "Government"},
    {"name": "Safdarjung Hospital", "lat": 28.5686, "lon": 77.2064, "beds": 1600, "icu": 120, "trauma": True, "type": "Government"},
    {"name": "Ram Manohar Lohia Hospital", "lat": 28.6270, "lon": 77.2000, "beds": 1350, "icu": 100, "trauma": True, "type": "Government"},
    {"name": "GTB Hospital", "lat": 28.6860, "lon": 77.3120, "beds": 1500, "icu": 110, "trauma": True, "type": "Government"},
    {"name": "Lok Nayak Hospital", "lat": 28.6380, "lon": 77.2400, "beds": 2000, "icu": 150, "trauma": True, "type": "Government"},
    {"name": "Apollo Hospital (Sarita Vihar)", "lat": 28.5310, "lon": 77.2840, "beds": 700, "icu": 120, "trauma": True, "type": "Private"},
    {"name": "Max Hospital (Saket)", "lat": 28.5270, "lon": 77.2120, "beds": 500, "icu": 80, "trauma": True, "type": "Private"},
    {"name": "Fortis Escorts (Okhla)", "lat": 28.5560, "lon": 77.2620, "beds": 310, "icu": 60, "trauma": True, "type": "Private"},
    {"name": "Sir Ganga Ram Hospital", "lat": 28.6420, "lon": 77.1890, "beds": 675, "icu": 90, "trauma": True, "type": "Private"},
    {"name": "BLK-Max Hospital", "lat": 28.6530, "lon": 77.1830, "beds": 700, "icu": 85, "trauma": True, "type": "Private"},
    {"name": "Maulana Azad Medical College", "lat": 28.6350, "lon": 77.2370, "beds": 1800, "icu": 130, "trauma": True, "type": "Government"},
    {"name": "Hindu Rao Hospital", "lat": 28.6800, "lon": 77.2050, "beds": 900, "icu": 50, "trauma": False, "type": "Government"},
    {"name": "Deen Dayal Upadhyay Hospital", "lat": 28.5920, "lon": 77.1640, "beds": 800, "icu": 60, "trauma": True, "type": "Government"},
    {"name": "Medanta (Gurugram)", "lat": 28.4400, "lon": 77.0420, "beds": 1250, "icu": 200, "trauma": True, "type": "Private"},
    {"name": "Max Hospital (Patparganj)", "lat": 28.6340, "lon": 77.3100, "beds": 300, "icu": 50, "trauma": False, "type": "Private"},
    {"name": "Rajiv Gandhi Super Specialty", "lat": 28.6980, "lon": 77.2970, "beds": 650, "icu": 70, "trauma": True, "type": "Government"},
    {"name": "Lady Hardinge Medical College", "lat": 28.6330, "lon": 77.2080, "beds": 800, "icu": 60, "trauma": False, "type": "Government"},
    {"name": "Fortis Hospital (Shalimar Bagh)", "lat": 28.7170, "lon": 77.1580, "beds": 262, "icu": 45, "trauma": False, "type": "Private"},
    {"name": "Primus Super Specialty", "lat": 28.5700, "lon": 77.1740, "beds": 150, "icu": 30, "trauma": False, "type": "Private"},
    {"name": "Batra Hospital (Tughlakabad)", "lat": 28.5270, "lon": 77.2500, "beds": 400, "icu": 50, "trauma": True, "type": "Private"},
]

# ============================================
# INCIDENT TYPES & SEVERITY WEIGHTS
# ============================================
INCIDENT_TYPES = {
    "flood": {"icon": "water", "color": "blue", "base_severity": 0.7},
    "fire": {"icon": "fire", "color": "red", "base_severity": 0.8},
    "building_collapse": {"icon": "home", "color": "darkred", "base_severity": 0.9},
    "gas_leak": {"icon": "cloud", "color": "purple", "base_severity": 0.85},
    "road_accident": {"icon": "car", "color": "orange", "base_severity": 0.5},
    "heat_emergency": {"icon": "sun", "color": "red", "base_severity": 0.6},
    "crowd_incident": {"icon": "users", "color": "cadetblue", "base_severity": 0.65},
    "power_outage": {"icon": "bolt", "color": "gray", "base_severity": 0.3},
}

# Severity thresholds
SEVERITY_LEVELS = {
    "CRITICAL": {"min_score": 0.75, "color": "#D32F2F", "response_mins": 10},
    "HIGH":     {"min_score": 0.50, "color": "#F57C00", "response_mins": 20},
    "MEDIUM":   {"min_score": 0.25, "color": "#FBC02D", "response_mins": 45},
    "LOW":      {"min_score": 0.00, "color": "#388E3C", "response_mins": 90},
}

# Critical infrastructure proximity radius (meters)
CRITICAL_INFRA_RADIUS_M = 500  # If incident within 500m of hospital/school → bump severity
