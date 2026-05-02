"""
hospital_panel.py — Module 3: Hospital Capacity & Resource Panel
Generates interactive map showing Delhi hospitals with real-time
(simulated) bed occupancy, ICU availability, and trauma center status.

Author: Abhinav Pandey
"""

import folium
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import CITY_CENTER
from data_ingestion import HospitalStatus


class HospitalCapacityMap:
    """
    Generates the Hospital Capacity interactive map.
    Each hospital is a marker color-coded by occupancy status.
    Click for bed/ICU counts, occupancy %, and trauma availability.
    """

    STATUS_COLORS = {
        "NORMAL": "green",
        "BUSY": "orange",
        "CRITICAL": "red",
        "FULL": "darkred",
    }

    STATUS_FILL = {
        "NORMAL": "#388E3C",
        "BUSY": "#F57C00",
        "CRITICAL": "#D32F2F",
        "FULL": "#B71C1C",
    }

    def generate(self, hospitals: List[HospitalStatus],
                 incident_location: Tuple[float, float] = None) -> folium.Map:
        """Generate hospital capacity map."""

        m = folium.Map(
            location=[CITY_CENTER["lat"], CITY_CENTER["lon"]],
            zoom_start=11,
            tiles="CartoDB positron"
        )

        folium.TileLayer("OpenStreetMap", name="Street Map", overlay=False).add_to(m)

        # --- HOSPITAL MARKERS ---
        gov_layer = folium.FeatureGroup(name="Government Hospitals")
        pvt_layer = folium.FeatureGroup(name="Private Hospitals")

        for h in hospitals:
            popup_html = self._hospital_popup(h)
            marker_color = self.STATUS_COLORS.get(h.status, "gray")

            icon = folium.Icon(
                color=marker_color,
                icon="plus-square" if h.has_trauma else "hospital-o",
                prefix="fa"
            )

            marker = folium.Marker(
                location=[h.lat, h.lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{h.name} | {h.status} | {h.available_beds} beds free",
                icon=icon
            )

            if h.hospital_type == "Government":
                marker.add_to(gov_layer)
            else:
                marker.add_to(pvt_layer)

        gov_layer.add_to(m)
        pvt_layer.add_to(m)

        # --- INCIDENT MARKER (if provided) ---
        if incident_location:
            folium.Marker(
                location=incident_location,
                popup="<b>Incident Location</b>",
                tooltip="Incident",
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")
            ).add_to(m)

            # Draw lines to 3 nearest hospitals with available beds
            nearest = self._find_nearest(hospitals, incident_location, n=3)
            for rank, h in enumerate(nearest, 1):
                folium.PolyLine(
                    locations=[incident_location, [h.lat, h.lon]],
                    color=self.STATUS_FILL.get(h.status, "#666"),
                    weight=3,
                    opacity=0.6,
                    dash_array="8",
                    tooltip=f"#{rank}: {h.name} ({h.available_beds} beds, {self._distance(incident_location, (h.lat, h.lon)):.1f}km)"
                ).add_to(m)

        # --- SUMMARY BOX ---
        self._add_summary_box(m, hospitals)

        # --- LEGEND ---
        self._add_legend(m, hospitals)

        folium.LayerControl().add_to(m)

        return m

    def _hospital_popup(self, h: HospitalStatus) -> str:
        """Build HTML popup for a hospital marker."""
        fill_color = self.STATUS_FILL.get(h.status, "#666")
        occ_width = int(h.occupancy_pct)
        trauma_badge = '<span style="background:#1565C0;color:white;padding:2px 6px;border-radius:3px;font-size:10px;">TRAUMA CENTER</span>' if h.has_trauma else ''

        return f"""
        <div style="font-family:Arial;font-size:12px;width:270px;">
            <h4 style="margin:0 0 4px 0;color:#333;">{h.name}</h4>
            <span style="background:{fill_color};color:white;padding:2px 8px;
                         border-radius:3px;font-size:11px;font-weight:bold;">
                {h.status}
            </span> {trauma_badge}
            <span style="color:#888;font-size:11px;margin-left:6px;">{h.hospital_type}</span>

            <div style="background:#eee;border-radius:4px;height:14px;margin:8px 0 4px 0;">
                <div style="background:{fill_color};width:{min(occ_width, 100)}%;height:100%;
                            border-radius:4px;text-align:center;color:white;font-size:9px;
                            line-height:14px;font-weight:bold;">
                    {h.occupancy_pct}%
                </div>
            </div>

            <table style="width:100%;font-size:11px;margin-top:6px;">
                <tr style="background:#f5f5f5;">
                    <td></td><td><b>Total</b></td><td><b>Used</b></td>
                    <td><b>Free</b></td>
                </tr>
                <tr>
                    <td><b>Beds</b></td>
                    <td>{h.total_beds}</td>
                    <td>{h.occupied_beds}</td>
                    <td style="color:{fill_color};font-weight:bold;">{h.available_beds}</td>
                </tr>
                <tr style="background:#f5f5f5;">
                    <td><b>ICU</b></td>
                    <td>{h.icu_beds}</td>
                    <td>{h.occupied_icu}</td>
                    <td style="color:{fill_color};font-weight:bold;">{h.available_icu}</td>
                </tr>
            </table>
        </div>
        """

    def _find_nearest(self, hospitals: List[HospitalStatus],
                      location: Tuple[float, float],
                      n: int = 3) -> List[HospitalStatus]:
        """Find n nearest hospitals with available beds."""
        available = [h for h in hospitals if h.available_beds > 0]
        available.sort(key=lambda h: self._distance(location, (h.lat, h.lon)))
        return available[:n]

    def _distance(self, p1: Tuple[float, float],
                  p2: Tuple[float, float]) -> float:
        """Haversine distance in km."""
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    def _add_summary_box(self, m: folium.Map, hospitals: List[HospitalStatus]):
        """Add floating hospital summary box."""
        total_beds = sum(h.total_beds for h in hospitals)
        free_beds = sum(h.available_beds for h in hospitals)
        total_icu = sum(h.icu_beds for h in hospitals)
        free_icu = sum(h.available_icu for h in hospitals)
        full_count = sum(1 for h in hospitals if h.status == "FULL")
        critical_count = sum(1 for h in hospitals if h.status == "CRITICAL")

        bed_color = "#D32F2F" if free_beds < 500 else "#F57C00" if free_beds < 1000 else "#388E3C"
        icu_color = "#D32F2F" if free_icu < 50 else "#F57C00" if free_icu < 100 else "#388E3C"

        html = f"""
        <div style="
            position:fixed; top:10px; right:10px; z-index:1000;
            background:white; padding:14px 18px; border-radius:10px;
            box-shadow:0 2px 10px rgba(0,0,0,0.2); font-family:Arial;
            font-size:12px; line-height:1.7; min-width:200px;
        ">
            <b style="font-size:14px;">Hospital Capacity</b><br>
            <span style="color:#888;">{len(hospitals)} hospitals | New Delhi NCR</span>
            <hr style="margin:4px 0;border-color:#eee;">
            Beds: <b style="color:{bed_color};">{free_beds}</b> / {total_beds} available<br>
            ICU: <b style="color:{icu_color};">{free_icu}</b> / {total_icu} available<br>
            <hr style="margin:4px 0;border-color:#eee;">
            <span style="color:#B71C1C;"><b>{full_count}</b> FULL</span> |
            <span style="color:#D32F2F;"><b>{critical_count}</b> CRITICAL</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))

    def _add_legend(self, m: folium.Map, hospitals: List[HospitalStatus]):
        """Add hospital status legend."""
        counts = {}
        for h in hospitals:
            counts[h.status] = counts.get(h.status, 0) + 1

        html = f"""
        <div style="
            position:fixed; bottom:30px; left:30px; z-index:1000;
            background:white; padding:12px 16px; border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial;
            font-size:12px; line-height:1.8;
        ">
            <b style="font-size:13px;">Hospital Status</b><br>
            <i style="background:#388E3C;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;Normal (&lt;75%) \u2014 {counts.get('NORMAL', 0)}<br>
            <i style="background:#F57C00;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;Busy (75-85%) \u2014 {counts.get('BUSY', 0)}<br>
            <i style="background:#D32F2F;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;Critical (85-95%) \u2014 {counts.get('CRITICAL', 0)}<br>
            <i style="background:#B71C1C;width:12px;height:12px;display:inline-block;border-radius:50%;"></i>
            &nbsp;Full (&gt;95%) \u2014 {counts.get('FULL', 0)}<br>
            <hr style="margin:4px 0;">
            <i style="color:#1565C0;">&#9724;</i> = Trauma Center
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))


# ============================================
# DEMO
# ============================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from data_ingestion import HospitalIngestion

    print("Generating Hospital Capacity map for New Delhi...")
    hospitals = HospitalIngestion().fetch_all(crisis_active=True)

    gen = HospitalCapacityMap()

    # Generate with a sample incident at ITO
    incident_at_ito = (28.629, 77.249)
    m = gen.generate(hospitals, incident_location=incident_at_ito)
    m.save("./data/hospital_map.html")

    print(f"Map saved: ./data/hospital_map.html")
    print(f"  {len(hospitals)} hospitals plotted")
    print(f"  Incident at ITO → 3 nearest hospitals with beds connected")

    # Stats
    for h in sorted(hospitals, key=lambda x: x.available_beds, reverse=True)[:3]:
        dist = gen._distance(incident_at_ito, (h.lat, h.lon))
        print(f"  → {h.name}: {h.available_beds} beds, {dist:.1f}km away [{h.status}]")

    print("\n\u2713 Module 3 (Hospital Capacity) complete.")
