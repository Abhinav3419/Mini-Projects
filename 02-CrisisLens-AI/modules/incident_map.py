"""
incident_map.py — Module 4: Incident Map with Severity Classification
Displays live map of all active incidents in New Delhi — fires, floods,
accidents, collapses — as geotagged markers color-coded by severity.

Includes rule-based severity classifier that boosts severity when
incidents are near critical infrastructure (hospitals, schools, power).

Author: Abhinav Pandey
"""

import folium
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

from config import (
    CITY_CENTER, INCIDENT_TYPES, SEVERITY_LEVELS,
    CRITICAL_INFRA_RADIUS_M, HOSPITALS
)
from data_ingestion import Incident, HospitalStatus


class SeverityClassifier:
    """
    Rule-based incident severity classifier.

    Base severity comes from incident type (fire > road accident).
    Modifiers:
    - Proximity to hospital/school: +0.15
    - During extreme weather: +0.10
    - Night time (10PM-6AM): +0.05 (slower response)
    - Multiple incidents in same zone: +0.10
    """

    def classify(self, incident: Incident,
                 hospitals: List[HospitalStatus] = None,
                 all_incidents: List[Incident] = None,
                 weather_severe: bool = False) -> Incident:
        """Reclassify incident severity with context modifiers."""

        base = incident.severity_score
        modifiers = []

        # 1. Proximity to critical infrastructure
        if hospitals:
            for h in hospitals:
                dist = self._haversine(
                    (incident.lat, incident.lon),
                    (h.lat, h.lon)
                )
                if dist <= CRITICAL_INFRA_RADIUS_M:
                    base += 0.15
                    modifiers.append(f"Near {h.name} ({dist:.0f}m)")
                    break  # Only count nearest

        # 2. Severe weather amplification
        if weather_severe:
            base += 0.10
            modifiers.append("Severe weather active")

        # 3. Night time penalty
        hour = datetime.now().hour
        if hour >= 22 or hour <= 6:
            base += 0.05
            modifiers.append("Night hours (delayed response)")

        # 4. Cluster detection — multiple incidents nearby
        if all_incidents:
            nearby = sum(
                1 for other in all_incidents
                if other.id != incident.id and
                self._haversine(
                    (incident.lat, incident.lon),
                    (other.lat, other.lon)
                ) < 2000  # 2km radius
            )
            if nearby >= 2:
                base += 0.10
                modifiers.append(f"{nearby} other incidents within 2km")

        # Clamp and reclassify
        base = round(min(1.0, max(0.0, base)), 3)

        if base >= 0.75:
            label = "CRITICAL"
        elif base >= 0.50:
            label = "HIGH"
        elif base >= 0.25:
            label = "MEDIUM"
        else:
            label = "LOW"

        incident.severity_score = base
        incident.severity_label = label

        return incident

    def _haversine(self, p1: Tuple[float, float],
                   p2: Tuple[float, float]) -> float:
        """Distance in meters between two lat/lon points."""
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371000 * 2 * np.arcsin(np.sqrt(a))


class IncidentMap:
    """
    Generates interactive Folium map with all active incidents
    color-coded by severity, with popups showing details.
    """

    SEVERITY_COLORS = {
        "CRITICAL": "#D32F2F",
        "HIGH": "#F57C00",
        "MEDIUM": "#FBC02D",
        "LOW": "#388E3C",
    }

    SEVERITY_ICONS = {
        "CRITICAL": "exclamation-triangle",
        "HIGH": "exclamation-circle",
        "MEDIUM": "info-circle",
        "LOW": "check-circle",
    }

    def generate(self, incidents: List[Incident],
                 hospitals: List[HospitalStatus] = None) -> folium.Map:
        """Generate incident map for New Delhi."""

        m = folium.Map(
            location=[CITY_CENTER["lat"], CITY_CENTER["lon"]],
            zoom_start=11,
            tiles="CartoDB dark_matter"
        )

        folium.TileLayer("CartoDB positron", name="Light Map", overlay=False).add_to(m)
        folium.TileLayer("OpenStreetMap", name="Street Map", overlay=False).add_to(m)

        # --- INCIDENT MARKERS ---
        # Group by severity for layer control
        layers = {
            "CRITICAL": folium.FeatureGroup(name="CRITICAL Incidents"),
            "HIGH": folium.FeatureGroup(name="HIGH Incidents"),
            "MEDIUM": folium.FeatureGroup(name="MEDIUM Incidents"),
            "LOW": folium.FeatureGroup(name="LOW Incidents"),
        }

        for inc in incidents:
            color = self.SEVERITY_COLORS.get(inc.severity_label, "#666")
            icon_name = INCIDENT_TYPES.get(inc.incident_type, {}).get("icon", "info-circle")

            popup_html = self._incident_popup(inc)

            # Pulsing circle for CRITICAL incidents
            if inc.severity_label == "CRITICAL":
                folium.CircleMarker(
                    location=[inc.lat, inc.lon],
                    radius=18,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.15,
                ).add_to(layers[inc.severity_label])

            folium.Marker(
                location=[inc.lat, inc.lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"[{inc.severity_label}] {inc.title[:50]}",
                icon=folium.Icon(
                    color="red" if inc.severity_label == "CRITICAL"
                    else "orange" if inc.severity_label == "HIGH"
                    else "beige" if inc.severity_label == "MEDIUM"
                    else "green",
                    icon=icon_name,
                    prefix="fa"
                )
            ).add_to(layers[inc.severity_label])

        for layer in layers.values():
            layer.add_to(m)

        # --- HOSPITAL MARKERS (faded, for context) ---
        if hospitals:
            hosp_layer = folium.FeatureGroup(name="Hospitals", show=False)
            for h in hospitals:
                folium.CircleMarker(
                    location=[h.lat, h.lon],
                    radius=5,
                    color="#1565C0",
                    fill=True,
                    fill_color="#1565C0",
                    fill_opacity=0.5,
                    tooltip=f"{h.name} ({h.available_beds} beds free)"
                ).add_to(hosp_layer)
            hosp_layer.add_to(m)

        # --- SUMMARY BOX ---
        self._add_summary_box(m, incidents)

        # --- LEGEND ---
        self._add_legend(m, incidents)

        folium.LayerControl(collapsed=False).add_to(m)

        return m

    def _incident_popup(self, inc: Incident) -> str:
        """Build HTML popup for incident marker."""
        color = self.SEVERITY_COLORS.get(inc.severity_label, "#666")
        type_display = inc.incident_type.replace("_", " ").title()
        responder_text = f"{inc.responders_assigned} units assigned" if inc.responders_assigned > 0 else "No responders yet"

        return f"""
        <div style="font-family:Arial;font-size:12px;width:270px;">
            <div style="background:{color};color:white;padding:6px 10px;
                        border-radius:6px 6px 0 0;font-weight:bold;font-size:13px;">
                {inc.severity_label} — {type_display}
            </div>
            <div style="padding:8px 10px;background:#fafafa;border-radius:0 0 6px 6px;">
                <b>{inc.title}</b><br>
                <span style="color:#888;font-size:11px;">{inc.timestamp} | {inc.source}</span>
                <hr style="margin:6px 0;border-color:#eee;">
                <table style="width:100%;font-size:11px;">
                    <tr><td><b>Severity Score:</b></td><td>{inc.severity_score:.3f}</td></tr>
                    <tr><td><b>Status:</b></td><td>{inc.status}</td></tr>
                    <tr><td><b>Responders:</b></td><td>{responder_text}</td></tr>
                    <tr><td><b>Location:</b></td><td>{inc.lat:.4f}, {inc.lon:.4f}</td></tr>
                </table>
            </div>
        </div>
        """

    def _add_summary_box(self, m: folium.Map, incidents: List[Incident]):
        """Floating incident summary."""
        counts = {}
        for i in incidents:
            counts[i.severity_label] = counts.get(i.severity_label, 0) + 1

        type_counts = {}
        for i in incidents:
            t = i.incident_type.replace("_", " ").title()
            type_counts[t] = type_counts.get(t, 0) + 1
        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        type_str = " | ".join(f"{t}: {c}" for t, c in top_types)

        html = f"""
        <div style="
            position:fixed; top:10px; right:10px; z-index:1000;
            background:rgba(30,30,30,0.92); padding:14px 18px; border-radius:10px;
            box-shadow:0 2px 10px rgba(0,0,0,0.5); font-family:Arial;
            font-size:12px; line-height:1.7; min-width:230px; color:white;
        ">
            <b style="font-size:14px;">Incident Command</b><br>
            <span style="color:#aaa;">New Delhi | {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            <hr style="margin:4px 0;border-color:#444;">
            <span style="color:#D32F2F;font-weight:bold;">{counts.get('CRITICAL', 0)} CRITICAL</span> |
            <span style="color:#F57C00;">{counts.get('HIGH', 0)} HIGH</span> |
            <span style="color:#FBC02D;">{counts.get('MEDIUM', 0)} MEDIUM</span> |
            <span style="color:#388E3C;">{counts.get('LOW', 0)} LOW</span><br>
            <hr style="margin:4px 0;border-color:#444;">
            <span style="color:#aaa;font-size:11px;">{type_str}</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))

    def _add_legend(self, m: folium.Map, incidents: List[Incident]):
        """Incident type legend."""
        html = """
        <div style="
            position:fixed; bottom:30px; left:30px; z-index:1000;
            background:rgba(30,30,30,0.92); padding:12px 16px; border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.5); font-family:Arial;
            font-size:12px; line-height:1.8; color:white;
        ">
            <b style="font-size:13px;">Severity Legend</b><br>
            <i style="background:#D32F2F;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;CRITICAL (score \u2265 0.75)<br>
            <i style="background:#F57C00;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;HIGH (score \u2265 0.50)<br>
            <i style="background:#FBC02D;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;MEDIUM (score \u2265 0.25)<br>
            <i style="background:#388E3C;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;LOW (score < 0.25)
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))


# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from data_ingestion import IncidentIngestion, HospitalIngestion, WeatherIngestion

    print("Generating Incident Map for New Delhi...")

    wx = WeatherIngestion().fetch()
    hospitals = HospitalIngestion().fetch_all()
    incidents = IncidentIngestion().generate_simulated(n=12, weather=wx)

    # Reclassify with context
    classifier = SeverityClassifier()
    weather_severe = wx.precipitation_mm > 20 or wx.temperature_c > 43
    for inc in incidents:
        classifier.classify(inc, hospitals=hospitals,
                          all_incidents=incidents,
                          weather_severe=weather_severe)

    gen = IncidentMap()
    m = gen.generate(incidents, hospitals=hospitals)
    m.save("./data/incident_map.html")

    counts = {}
    for i in incidents:
        counts[i.severity_label] = counts.get(i.severity_label, 0) + 1
    print(f"  {len(incidents)} incidents: {counts}")
    print(f"  Map saved: ./data/incident_map.html")
    print("\u2713 Module 4 (Incident Map) complete.")
