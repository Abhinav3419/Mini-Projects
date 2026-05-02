"""
app.py - CrisisLens AI: Urban Decision Intelligence Platform
Streamlit frontend with premium light-mode UI for New Delhi.
Run: streamlit run app.py
Author: Abhinav Pandey
"""
import streamlit as st
import numpy as np
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_ingestion import CrisisDataLayer
from modules.weather_flood import WeatherFloodMap
from modules.hospital_panel import HospitalCapacityMap
from modules.manning_model import ManningFloodModel
from modules.incident_map import IncidentMap, SeverityClassifier
from modules.telecom_anomaly import TelecomNetwork
from modules.rag_advisor import CrisisAdvisor
from modules.dispatch_optimizer import DispatchOptimizer
from config import CITY_CENTER

st.set_page_config(page_title="CrisisLens AI - New Delhi", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
.stApp{background:linear-gradient(180deg,#FAFBFF 0%,#F0F2F8 100%)}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.hero-title{font-size:3.2rem;font-weight:900;background:linear-gradient(135deg,#1a5e20,#0d47a1,#4a148c);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;margin-bottom:0}
.hero-sub{font-size:1.3rem;color:#64748B;font-weight:500;margin-top:4px}
.hero-badge{display:inline-block;padding:6px 16px;border-radius:20px;font-size:0.85rem;font-weight:700;margin:4px}
.badge-live{background:#DCFCE7;color:#166534}.badge-ai{background:#DBEAFE;color:#1E40AF}.badge-delhi{background:#FEF3C7;color:#92400E}
.metric-card{background:white;padding:20px 24px;border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);border:1px solid #F1F5F9;transition:transform 0.2s}
.metric-card:hover{transform:translateY(-2px);box-shadow:0 4px 20px rgba(0,0,0,0.08)}
.metric-value{font-size:2.4rem;font-weight:800;line-height:1;margin-bottom:4px}
.metric-label{font-size:0.9rem;color:#94A3B8;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
.metric-red .metric-value{color:#DC2626}.metric-orange .metric-value{color:#EA580C}.metric-green .metric-value{color:#16A34A}.metric-blue .metric-value{color:#2563EB}
.alert-banner{padding:20px 28px;border-radius:16px;margin:16px 0;font-size:1.1rem;font-weight:600;border-left:6px solid}
.alert-critical{background:linear-gradient(135deg,#FEF2F2,#FECACA);border-color:#DC2626;color:#991B1B}
.alert-warning{background:linear-gradient(135deg,#FFFBEB,#FDE68A);border-color:#F59E0B;color:#92400E}
.alert-safe{background:linear-gradient(135deg,#F0FDF4,#BBF7D0);border-color:#16A34A;color:#166534}
.section-header{font-size:1.6rem;font-weight:800;color:#1E293B;margin:24px 0 12px;padding-bottom:8px;border-bottom:3px solid #E2E8F0}
.stTabs [data-baseweb="tab"]{font-size:1.05rem;font-weight:600;padding:12px 20px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF,#F8FAFC);border-right:1px solid #E2E8F0}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1a5e20,#0d47a1);color:white;font-size:1.15rem;font-weight:700;padding:14px 32px;border-radius:12px;border:none}
.footer{text-align:center;color:#94A3B8;font-size:0.85rem;padding:30px 0 10px;border-top:1px solid #E2E8F0;margin-top:40px}
</style>""", unsafe_allow_html=True)

if "data_layer" not in st.session_state: st.session_state.data_layer = None
if "initialized" not in st.session_state: st.session_state.initialized = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "rag_advisor" not in st.session_state: st.session_state.rag_advisor = None

@st.cache_resource
def init_systems():
    dl = CrisisDataLayer(); dl.refresh()
    adv = CrisisAdvisor(); adv.initialize()
    return dl, adv

with st.sidebar:
    st.markdown("## 🏙️ CrisisLens AI\n**Urban Decision Intelligence**\n---\n### 📍 New Delhi\nPopulation: 32M | Seismic Zone IV\n---")
    rainfall_slider = st.slider("🌊 Upstream Rainfall (mm)", 0, 200, 80, 5)
    st.markdown("---")
    crisis_sim = st.selectbox("📡 Inject Crisis At", ["None","ITO Barrage (Flood)","Bawana (Fire)","Chandni Chowk (Crowd)","India Gate (Accident)"])
    crisis_intensity = st.slider("Crisis Intensity", 0.0, 1.0, 0.8, 0.1)
    st.markdown("---\n**Built by** [Abhinav Pandey](https://abhinavpandey3419.github.io/)\nM.Tech NIT Allahabad\nPatent Filed · 2 Papers Under Review")

st.markdown('<h1 class="hero-title">🏙️ CrisisLens AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Urban Decision Intelligence Platform — New Delhi</p>', unsafe_allow_html=True)
st.markdown('<span class="hero-badge badge-live">● LIVE DATA</span><span class="hero-badge badge-ai">🧠 RAG + MCP</span><span class="hero-badge badge-delhi">📍 NEW DELHI</span>', unsafe_allow_html=True)

if st.button("🚀 LAUNCH CRISIS DASHBOARD", type="primary", use_container_width=True):
    with st.spinner("Initializing CrisisLens AI..."):
        dl, adv = init_systems()
        st.session_state.data_layer = dl; st.session_state.rag_advisor = adv; st.session_state.initialized = True

if not st.session_state.initialized:
    st.markdown('<div style="text-align:center;padding:60px;color:#64748B"><p style="font-size:4rem">🛰️</p><p style="font-size:1.5rem;font-weight:700;color:#1E293B">Press LAUNCH to activate all systems</p></div>', unsafe_allow_html=True)
    st.stop()

dl = st.session_state.data_layer; advisor = st.session_state.rag_advisor; summary = dl.get_summary()

cols = st.columns(6)
for col, (icon, val, lbl, clr) in zip(cols, [("🌡️",f"{summary['temperature_c']}°C","Temperature","blue"),("🌧️",f"{summary['rain_7day_mm']}mm","7-Day Rain","blue"),("🔴",f"{summary['flood_zones_critical']}","Flood Critical","red"),("🏥",f"{summary['beds_available']}","Beds Free","green"),("🚨",f"{summary['incidents_critical']}","Critical","red"),("📊",f"{summary['incidents_active']}","Active","orange")]):
    with col: st.markdown(f'<div class="metric-card metric-{clr}"><div class="metric-label">{icon} {lbl}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

if summary["flood_zones_critical"]>0 or summary["incidents_critical"]>2:
    st.markdown(f'<div class="alert-banner alert-critical">⚠️ CRITICAL: {summary["flood_zones_critical"]} flood zones critical · {summary["incidents_critical"]} critical incidents · {summary["hospitals_full"]} hospitals full</div>', unsafe_allow_html=True)
elif summary["flood_zones_high"]>0:
    st.markdown(f'<div class="alert-banner alert-warning">🟡 WARNING: {summary["flood_zones_high"]} flood zones HIGH</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-banner alert-safe">✅ ALL CLEAR</div>', unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["🌦️ Weather","🌊 Flood Model","🏥 Hospitals","🚨 Incidents","📡 Telecom","📚 Advisor","🚑 Dispatch"])

def render_map(fmap, key):
    try:
        from streamlit_folium import st_folium
        st_folium(fmap, width=None, height=550, key=key)
    except ImportError:
        fmap.save(f"./data/temp_{key}.html")
        with open(f"./data/temp_{key}.html","r") as f: st.components.v1.html(f.read(), height=550, scrolling=True)

with tab1:
    st.markdown('<div class="section-header">🌦️ Live Weather & Flood Risk</div>', unsafe_allow_html=True)
    wx=dl.weather; c1,c2,c3,c4=st.columns(4); c1.metric("Temp",f"{wx.temperature_c}°C"); c2.metric("Humidity",f"{wx.humidity_pct}%"); c3.metric("Wind",f"{wx.wind_speed_kmh}km/h"); c4.metric("Rain",f"{wx.precipitation_mm}mm")
    render_map(WeatherFloodMap().generate(wx,dl.flood_zones),"wf")
    st.dataframe({"Date":wx.daily_dates,"Max °C":wx.daily_max_temp,"Min °C":wx.daily_min_temp,"Rain mm":wx.daily_rain_mm},use_container_width=True,hide_index=True)

with tab2:
    st.markdown('<div class="section-header">🌊 Manning\'s Equation — Flood Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"**Simulating {rainfall_slider}mm upstream rainfall**")
    model=ManningFloodModel(); pred=model.predict(rainfall_slider)
    acls="alert-critical" if pred.highest_risk in["CRITICAL","HIGH"] else "alert-warning" if pred.highest_risk=="MEDIUM" else "alert-safe"
    st.markdown(f'<div class="alert-banner {acls}">{pred.warning_message}</div>', unsafe_allow_html=True)
    st.dataframe({"Station":[s.name for s in pred.stations],"Depth (m)":[s.predicted_depth_m for s in pred.stations],"Q (m³/s)":[int(s.predicted_discharge_m3s) for s in pred.stations],"V (m/s)":[s.predicted_velocity_ms for s in pred.stations],"Arrival (h)":[s.arrival_time_hours for s in pred.stations],"Danger":["⚠️ YES" if s.above_danger else "✅" for s in pred.stations],"Excess (m)":[s.excess_depth_m for s in pred.stations],"Risk":[s.risk_level for s in pred.stations]},use_container_width=True,hide_index=True)
    an=model.scenario_analysis()
    st.dataframe({"Rain mm":[r["rainfall_mm"] for r in an["scenarios"]],"Peak Q":[int(r["peak_discharge_m3s"]) for r in an["scenarios"]],"Danger Stations":[r["stations_above_danger"] for r in an["scenarios"]],"Max Excess":[r["max_excess_depth_m"] for r in an["scenarios"]],"Risk":[r["highest_risk"] for r in an["scenarios"]]},use_container_width=True,hide_index=True)
    if an["danger_threshold_mm"]: st.warning(f"⚠️ Danger threshold: **{an['danger_threshold_mm']}mm**")

with tab3:
    st.markdown('<div class="section-header">🏥 Hospital Capacity</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4); c1.metric("Hospitals",len(dl.hospitals)); c2.metric("Beds Free",summary["beds_available"]); c3.metric("ICU Free",summary["icu_available"]); c4.metric("Full",summary["hospitals_full"])
    render_map(HospitalCapacityMap().generate(dl.hospitals),"hosp")
    hs=sorted(dl.hospitals,key=lambda x:x.available_beds,reverse=True)
    st.dataframe({"Hospital":[h.name for h in hs],"Type":[h.hospital_type for h in hs],"Beds Free":[h.available_beds for h in hs],"ICU Free":[h.available_icu for h in hs],"Occupancy":[f"{h.occupancy_pct}%" for h in hs],"Status":[h.status for h in hs],"Trauma":["✅" if h.has_trauma else "—" for h in hs]},use_container_width=True,hide_index=True)

with tab4:
    st.markdown('<div class="section-header">🚨 Incident Command</div>', unsafe_allow_html=True)
    clf=SeverityClassifier(); ws=dl.weather.precipitation_mm>20 or dl.weather.temperature_c>43
    for inc in dl.incidents: clf.classify(inc,hospitals=dl.hospitals,all_incidents=dl.incidents,weather_severe=ws)
    render_map(IncidentMap().generate(dl.incidents,hospitals=dl.hospitals),"inc")
    si=sorted(dl.incidents,key=lambda x:x.severity_score,reverse=True)
    st.dataframe({"Severity":[i.severity_label for i in si],"Type":[i.incident_type.replace("_"," ").title() for i in si],"Title":[i.title for i in si],"Score":[i.severity_score for i in si],"Time":[i.timestamp for i in si],"Source":[i.source for i in si]},use_container_width=True,hide_index=True)

with tab5:
    st.markdown('<div class="section-header">📡 Telecom Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown("*500 simulated cell towers · Z-score anomaly detection · Crisis spike injection*")
    crisis_locs={"None":None,"ITO Barrage (Flood)":(28.629,77.249),"Bawana (Fire)":(28.728,77.133),"Chandni Chowk (Crowd)":(28.656,77.230),"India Gate (Accident)":(28.613,77.229)}
    cl=crisis_locs.get(crisis_sim); net=TelecomNetwork(500,42); snap=net.simulate_snapshot(crisis_location=cl,crisis_intensity=crisis_intensity if cl else 0)
    c1,c2,c3,c4=st.columns(4); c1.metric("Towers",snap.total_towers); c2.metric("Anomalous",snap.anomaly_towers); c3.metric("Clusters",len(snap.anomaly_clusters)); c4.metric("Avg Load",f"{snap.network_avg_load:.1%}")
    if snap.crisis_detected:
        for c in snap.anomaly_clusters: st.markdown(f'<div class="alert-banner alert-critical">📡 {c.message}</div>',unsafe_allow_html=True)
    else: st.markdown('<div class="alert-banner alert-safe">📡 Network normal</div>',unsafe_allow_html=True)
    import folium; from folium.plugins import HeatMap
    tm=folium.Map(location=[CITY_CENTER["lat"],CITY_CENTER["lon"]],zoom_start=11,tiles="CartoDB positron")
    HeatMap([[t.lat,t.lon,t.current_load] for t in snap.towers],radius=15,blur=10,max_zoom=13).add_to(tm)
    if snap.anomaly_towers>0:
        al=folium.FeatureGroup(name="Anomalies")
        for t in snap.towers:
            if t.is_anomaly: folium.CircleMarker([t.lat,t.lon],radius=8,color="#D32F2F",fill=True,fill_color="#D32F2F",fill_opacity=0.8,tooltip=f"Z={t.z_score}").add_to(al)
        al.add_to(tm)
    for c in snap.anomaly_clusters: folium.Circle([c.center_lat,c.center_lon],radius=c.radius_km*1000,color="#D32F2F",weight=3,fill=True,fill_opacity=0.1).add_to(tm)
    folium.LayerControl().add_to(tm); render_map(tm,"tele")

with tab6:
    st.markdown('<div class="section-header">📚 RAG Crisis Advisor</div>', unsafe_allow_html=True)
    stats=advisor.get_protocol_count()
    st.markdown(f"**{stats['total_protocols']} protocols** | NDMA · Delhi DDMA · MCD · CWC · Delhi Fire · Delhi Police")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    uq=st.chat_input("e.g., 'Flooding in Mayur Vihar, 200 displaced. Protocol?'")
    if uq:
        st.session_state.chat_history.append({"role":"user","content":uq})
        with st.chat_message("user"): st.markdown(uq)
        with st.chat_message("assistant"):
            with st.spinner("Searching protocols..."):
                ctx={"weather":{"temp":dl.weather.temperature_c,"rain":dl.weather.precipitation_mm},"incidents":f"{summary['incidents_active']} active","flood_risk":f"{summary['flood_zones_critical']} critical","hospitals":f"{summary['beds_available']} beds"}
                resp=advisor.query(uq,context=ctx); ans=f"**Confidence: {resp.confidence:.0%}** | Sources: {', '.join(resp.retrieved_sources[:2])}\n\n{resp.generated_response}"
                st.markdown(ans); st.session_state.chat_history.append({"role":"assistant","content":ans})
    with st.expander("🕐 Historical Pattern Matching"):
        ct=st.selectbox("Crisis Type",["flood","fire","earthquake","heat_wave"])
        if st.button("Find Historical Matches"): st.markdown(advisor.find_historical_match(ct).generated_response)

with tab7:
    st.markdown('<div class="section-header">🚑 Dispatch Optimizer</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: ilat=st.number_input("Incident Lat",value=28.629,format="%.4f")
    with c2: ilon=st.number_input("Incident Lon",value=77.249,format="%.4f")
    with c3: itype=st.selectbox("Type",["flood","fire","road_accident","building_collapse","gas_leak","crowd_incident"])
    if st.button("🚀 Find Optimal Dispatch",use_container_width=True):
        opt=DispatchOptimizer(); units=opt.generate_units(5,42); disp=opt.find_optimal_dispatch(ilat,ilon,itype,units)
        if disp.time_advantage_min>2: st.markdown(f'<div class="alert-banner alert-warning">⏱️ TIME SAVED: {disp.time_advantage_min} min</div>',unsafe_allow_html=True)
        st.success(f"**{disp.summary}**"); render_map(opt.generate_dispatch_map(disp),"disp")
        st.dataframe({"Rank":[f"{'★ ' if r.is_fastest else ''}{i+1}" for i,r in enumerate(disp.routes)],"Unit":[r.unit.name for r in disp.routes],"Type":[r.unit.unit_type.replace("_"," ").title() for r in disp.routes],"Road km":[r.distance_km for r in disp.routes],"ETA min":[r.duration_min for r in disp.routes],"Straight km":[r.straight_line_km for r in disp.routes]},use_container_width=True,hide_index=True)

st.markdown('<div class="footer">CrisisLens AI — Urban Decision Intelligence<br>Manning\'s Equation × Live Weather × RAG × Telecom Anomaly × Dispatch Optimization<br>Built by <b>Abhinav Pandey</b> · M.Tech NIT Allahabad · Patent Filed · 2 Papers Under Review</div>',unsafe_allow_html=True)
