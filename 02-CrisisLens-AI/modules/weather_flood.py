"""
weather_flood.py — Module 1: Live Weather & Flood Risk Dashboard
Generates interactive Folium map with weather overlay and
color-coded flood risk zones for New Delhi.

Author: Abhinav Pandey
"""

import folium
import numpy as np
from folium.plugins import HeatMap
from typing import List, Dict, Optional
from datetime import datetime

from config import CITY_CENTER, YAMUNA_STATIONS, FLOOD_ZONES
from data_ingestion import WeatherData, FloodRiskZone


class WeatherFloodMap:
    """
    Generates the Weather + Flood Risk interactive map.
    Shows: current conditions, 7-day forecast bar,
    flood risk zones (color-coded circles), Yamuna monitoring stations.
    """

    WEATHER_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
    }

    def generate(self, weather: WeatherData,
                 flood_zones: List[FloodRiskZone]) -> folium.Map:
        """Generate the complete weather + flood risk map."""

        # Base map centered on New Delhi
        m = folium.Map(
            location=[CITY_CENTER["lat"], CITY_CENTER["lon"]],
            zoom_start=11,
            tiles="CartoDB positron"
        )

        # Add satellite tile option
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite View", overlay=False
        ).add_to(m)

        folium.TileLayer("OpenStreetMap", name="Street Map", overlay=False).add_to(m)

        # --- FLOOD RISK ZONES ---
        flood_layer = folium.FeatureGroup(name="Flood Risk Zones")

        for zone in flood_zones:
            popup_html = self._zone_popup(zone, weather)

            folium.Circle(
                location=[zone.lat, zone.lon],
                radius=zone.radius_km * 1000,
                color=zone.color,
                weight=2,
                fill=True,
                fill_color=zone.color,
                fill_opacity=0.25,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{zone.name}: {zone.risk_level} ({zone.current_risk:.2f})"
            ).add_to(flood_layer)

        flood_layer.add_to(m)

        # --- YAMUNA MONITORING STATIONS ---
        yamuna_layer = folium.FeatureGroup(name="Yamuna Stations")

        for name, station in YAMUNA_STATIONS.items():
            folium.Marker(
                location=[station["lat"], station["lon"]],
                popup=(
                    f"<b>{name.replace('_', ' ')}</b><br>"
                    f"Danger Level: {station['danger_level_m']}m<br>"
                    f"Distance from Palla: {station['km_marker']}km"
                ),
                tooltip=f"Yamuna: {name.replace('_', ' ')}",
                icon=folium.Icon(color="blue", icon="tint", prefix="fa")
            ).add_to(yamuna_layer)

        # Draw Yamuna river line (approximate)
        yamuna_coords = [[s["lat"], s["lon"]] for s in YAMUNA_STATIONS.values()]
        folium.PolyLine(
            locations=yamuna_coords,
            color="#1565C0",
            weight=4,
            opacity=0.7,
            tooltip="Yamuna River"
        ).add_to(yamuna_layer)

        yamuna_layer.add_to(m)

        # --- WEATHER INFO BOX ---
        self._add_weather_box(m, weather)

        # --- LEGEND ---
        self._add_legend(m, flood_zones)

        # Layer control
        folium.LayerControl().add_to(m)

        return m

    def _zone_popup(self, zone: FloodRiskZone, weather: WeatherData) -> str:
        """Build HTML popup for a flood zone."""
        risk_bar_width = int(zone.current_risk * 100)
        return f"""
        <div style="font-family:Arial;font-size:12px;width:250px;">
            <h4 style="margin:0 0 6px 0;color:#333;">{zone.name}</h4>
            <p style="margin:2px 0;color:#666;">{zone.description}</p>
            <div style="background:#eee;border-radius:4px;height:16px;margin:6px 0;">
                <div style="background:{zone.color};width:{risk_bar_width}%;height:100%;
                            border-radius:4px;text-align:center;color:white;font-size:10px;
                            line-height:16px;font-weight:bold;">
                    {zone.current_risk:.0%}
                </div>
            </div>
            <table style="width:100%;font-size:11px;">
                <tr><td><b>Risk Level:</b></td>
                    <td style="color:{zone.color};font-weight:bold;">{zone.risk_level}</td></tr>
                <tr><td><b>Base Risk:</b></td><td>{zone.base_risk:.2f}</td></tr>
                <tr><td><b>Current Risk:</b></td><td>{zone.current_risk:.3f}</td></tr>
                <tr><td><b>Rain Today:</b></td><td>{weather.precipitation_mm}mm</td></tr>
                <tr><td><b>7-Day Forecast:</b></td><td>{weather.forecast_rain_total_mm}mm</td></tr>
            </table>
        </div>
        """

    def _add_weather_box(self, m: folium.Map, weather: WeatherData):
        """Add a floating weather info box to the map."""
        condition = self.WEATHER_CODES.get(weather.weather_code, "Unknown")
        heat_badge = '<span style="background:#D32F2F;color:white;padding:2px 6px;border-radius:3px;font-size:10px;">HEAT WAVE</span>' if weather.heat_wave else ''

        html = f"""
        <div style="
            position:fixed; top:10px; right:10px; z-index:1000;
            background:white; padding:14px 18px; border-radius:10px;
            box-shadow:0 2px 10px rgba(0,0,0,0.2); font-family:Arial;
            font-size:12px; line-height:1.7; min-width:220px;
        ">
            <b style="font-size:14px;">New Delhi Weather</b> {heat_badge}<br>
            <span style="color:#888;">{weather.timestamp}</span><br>
            <hr style="margin:4px 0;border-color:#eee;">
            <b>{weather.temperature_c}\u00B0C</b> | {condition}<br>
            Humidity: {weather.humidity_pct}% | Wind: {weather.wind_speed_kmh} km/h<br>
            Rain now: {weather.precipitation_mm}mm<br>
            <hr style="margin:4px 0;border-color:#eee;">
            <b>7-Day Forecast:</b><br>
            Total rain: {weather.forecast_rain_total_mm}mm |
            Dry days: {weather.forecast_dry_days}/7<br>
            Max: {max(weather.daily_max_temp) if weather.daily_max_temp else 'N/A'}\u00B0C |
            Min: {min(weather.daily_min_temp) if weather.daily_min_temp else 'N/A'}\u00B0C
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))

    def _add_legend(self, m: folium.Map, zones: List[FloodRiskZone]):
        """Add flood risk legend."""
        counts = {}
        for z in zones:
            counts[z.risk_level] = counts.get(z.risk_level, 0) + 1

        html = """
        <div style="
            position:fixed; bottom:30px; left:30px; z-index:1000;
            background:white; padding:12px 16px; border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial;
            font-size:12px; line-height:1.8;
        ">
            <b style="font-size:13px;">Flood Risk Legend</b><br>
            <i style="background:#D32F2F;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;CRITICAL — {crit}<br>
            <i style="background:#F57C00;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;HIGH — {high}<br>
            <i style="background:#FBC02D;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;MEDIUM — {med}<br>
            <i style="background:#388E3C;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;LOW — {low}
        </div>
        """.format(
            crit=counts.get("CRITICAL", 0),
            high=counts.get("HIGH", 0),
            med=counts.get("MEDIUM", 0),
            low=counts.get("LOW", 0)
        )
        m.get_root().html.add_child(folium.Element(html))


# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    from data_ingestion import WeatherIngestion, FloodRiskEngine

    print("Generating Weather + Flood Risk map for New Delhi...")
    wx = WeatherIngestion().fetch()
    zones = FloodRiskEngine().compute_risks(wx)

    gen = WeatherFloodMap()
    m = gen.generate(wx, zones)
    m.save("./data/weather_flood_map.html")
    print(f"Map saved: ./data/weather_flood_map.html")
    print("\u2713 Module 1 (Weather + Flood Risk) complete.")
