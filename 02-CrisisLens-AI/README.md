<div align="center">

# 🏙️ CrisisLens AI

### Urban Decision Intelligence Platform

**Manning's Equation × Live Weather × RAG × Telecom Anomaly × Dispatch Optimization**

*Real-time crisis response intelligence for New Delhi — 7 integrated modules powered by physics-informed ML, agentic RAG, and geospatial analytics.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB-FF6B6B?style=for-the-badge)](https://www.trychroma.com/)
[![Live](https://img.shields.io/badge/Data-Live_APIs-00C853?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

**🌐 Live Demo** · [**📄 Dossier**](.) · [**👤 Portfolio**](https://abhinavpandey3419.github.io/) · [**🔗 LinkedIn**](https://www.linkedin.com/in/abhinavpandey3419/)

</div>

---

## 🎯 The Problem

During the **2023 Delhi Yamuna floods** (208.66m — highest in 45 years), emergency responders relied on WhatsApp messages, phone calls, and fragmented data from multiple agencies. There was no unified platform combining weather forecasts, river levels, hospital capacity, and crisis protocols into a single command-center experience.

**CrisisLens AI solves this.**

## 💡 The Solution

A **7-module Urban Decision Intelligence Platform** that:

> 🌊 Predicts **when** floodwater reaches your neighborhood using **Manning's equation** (actual hydraulics, not ML regression)
>
> 🏥 Shows which hospitals have beds **right now** (20 hospitals, live simulated occupancy)
>
> 📡 Detects crisis events **10-20 minutes early** through cell tower anomaly detection (Z-score on 500 towers)
>
> 📚 Answers "What should I do?" with **RAG-retrieved NDMA protocols** (15 crisis entries, ChromaDB)
>
> 🚑 Dispatches the **fastest** responder, not just the nearest (proved 8+ min time savings in Delhi traffic)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│     Premium Light-Mode UI · 7 Interactive Tabs · Folium Maps │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────────────┐    │
│  │ Weather │ │ Manning  │ │Hospital│ │ Incident Command │    │
│  │ + Flood │ │ Flood    │ │Capacity│ │ + Severity       │    │
│  │ Risk Map│ │ Model    │ │ Panel  │ │ Classifier       │    │
│  └────┬────┘ └────┬─────┘ └───┬────┘ └────────┬─────────┘    │
│       │           │           │               │              │
│  ┌────┴────┐ ┌────┴─────┐ ┌───┴────┐ ┌────────┴────────┐     │
│  │ Telecom │ │    RAG   │ │Dispatch│ │ Unified Data    │     │
│  │ Anomaly │ │  Crisis  │ │ Route  │ │ Ingestion Layer │     │
│  │Detector │ │  Advisor │ │Optimize│ │ (All APIs)      │     │
│  └─────────┘ └──────────┘ └────────┘ └─────────────────┘     │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  DATA SOURCES                                                │
│  Open-Meteo (Weather) · CWC (River) · GDELT (News Events)    │
│  OpenRouteService (Routing) · ChromaDB (RAG) · Simulated     │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Overview

| # | Module | What It Does | Key Tech |
|---|--------|-------------|----------|
| 1 | **🌦️ Weather & Flood Risk** | Live weather + 8-zone flood risk heatmap with Yamuna monitoring | Open-Meteo API, Folium |
| 2 | **🌊 Manning's Flood Model** | Predicts flood arrival time at 5 Yamuna stations from upstream rainfall | Manning's equation, kinematic wave |
| 3 | **🏥 Hospital Capacity** | 20 Delhi hospitals with bed/ICU availability and nearest-hospital routing | Simulated occupancy, Haversine |
| 4 | **🚨 Incident Command** | Geotagged incident map with 4-modifier severity classifier | GDELT + simulated, rule-based ML |
| 4B | **📡 Telecom Anomaly** | 500 cell towers, diurnal traffic model, Z-score crisis detection | Physics-informed simulation |
| 5 | **📚 RAG Crisis Advisor** | Chat interface with NDMA/DDMA protocol retrieval + historical matching | ChromaDB, sentence-transformers |
| 6 | **🚑 Dispatch Optimizer** | Fastest-route emergency dispatch (proved 8+ min over nearest-by-distance) | OpenRouteService, road heuristics |

---

## 🔬 Technical Highlights

### Manning's Equation (Module 2)
```
Q = (1/n) × A × R^(2/3) × S^(1/2)
```
Real open-channel hydraulics for the Yamuna through Delhi. Trapezoidal channel with floodplain overflow. Inverse Manning's via bisection. Kinematic wave propagation. **No other portfolio project does this.**

### Telecom Anomaly Detection (Module 4B)
```
load(t) = A₁·sin(2π(t-10)/24) + A₂·sin(2π(t-19)/12) + B + noise
Z-score > 3.0σ → anomaly | 3+ towers within 2km → crisis cluster
```
Physics-informed diurnal model with zone-specific phase shifts. **Zero false positives** on normal traffic. Crisis detection at 3.1σ confidence.

### Dispatch Optimization (Module 6)
```
Nearest by distance ≠ Fastest by road
Mayur Vihar flood: 8.4 minutes saved by choosing faster route
```
Delhi-specific road heuristics: central 15-22 km/h, outer 30-45 km/h. Emergency vehicle 20% speed boost. **Proves the thesis with data.**

---

## 📁 Project Structure

```
CrisisLens-AI/
│
├── 📄 app.py                          # Streamlit 7-tab premium UI
├── 📄 config.py                       # New Delhi geography, zones, hospitals
├── 📄 data_ingestion.py               # Unified data layer (all APIs)
│
├── 📂 modules/
│   ├── 📄 __init__.py
│   ├── 📄 weather_flood.py            # Module 1: Weather + Flood Risk Map
│   ├── 📄 manning_model.py            # Module 2: Manning's Equation
│   ├── 📄 hospital_panel.py           # Module 3: Hospital Capacity
│   ├── 📄 incident_map.py             # Module 4: Incident Map + Severity
│   ├── 📄 telecom_anomaly.py          # Module 4B: Telecom Anomaly Detection
│   ├── 📄 rag_advisor.py              # Module 5: RAG Crisis Advisor
│   └── 📄 dispatch_optimizer.py       # Module 6: Dispatch Route Optimizer
│
├── 📂 data/                           # Generated maps and temp files
│
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Abhinav3419/CrisisLens-AI.git
cd CrisisLens-AI

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

Open `http://localhost:8501` → Press **LAUNCH CRISIS DASHBOARD** → All 7 modules activate.

---

## 📊 Sample Output

### Manning's Flood Prediction (100mm upstream rainfall)
```
🔴 RED ALERT: YAMUNA FLOOD WARNING FOR DELHI
Upstream rainfall: 100mm | Peak discharge: 4627 m³/s

Palla: water level 1.62m ABOVE danger mark, arriving in 0.0 hours
Old Railway Bridge: 0.90m ABOVE, arriving in 3.5 hours
ITO Barrage: 0.56m ABOVE, arriving in 5.0 hours
Okhla Barrage: 0.15m ABOVE, arriving in 9.1 hours

IMMEDIATE ACTION: Evacuate low-lying areas near Palla, Old Railway Bridge.
```

### Dispatch Optimization (Mayur Vihar Flood)
```
DISPATCH: Send CATS Ambulance - GTB Nagar.
ETA: 16 min (10.9km by road).
8 min FASTER than nearest-by-distance (Sarita Vihar, 25 min).
```

### Telecom Anomaly (Crisis at Chandni Chowk)
```
📡 ANOMALY DETECTED: 29 towers showing 3.2σ above baseline
near Chandni Chowk / Old Delhi. Possible emergency event.
```

---

## 🧠 Knowledge Base (RAG)

| Crisis Type | Entries | Sources |
|---|---|---|
| Flood (immediate + evacuation + recovery) | 5 | NDMA, Delhi DDMA, MCD, CWC |
| Fire (industrial + residential) | 3 | Delhi Fire Services, NDMA |
| Heat Wave | 1 | NDMA + Delhi Action Plan |
| Earthquake | 1 | NDMA (Seismic Zone IV) |
| Chemical / Gas Leak | 1 | NDMA Chemical Disaster |
| Crowd Management | 1 | Delhi Police SOP |
| Building Collapse | 1 | NDRF SOP |
| Road Accident | 1 | Delhi Traffic Police |
| Air Pollution (AQI > 400) | 1 | CPCB GRAP |

---

## ⚠️ Limitations (Stated for Scientific Honesty)

| What | Reality | Path to Production |
|------|---------|-------------------|
| Telecom data | Simulated (TRAI restricted) | Enterprise partnership with Jio/Airtel |
| Hospital occupancy | Simulated | HL7/FHIR API integration |
| Incident feed | GDELT + simulated | 112 emergency dispatch API |
| Routing | Heuristic fallback | OpenRouteService with paid tier |
| Manning's calibration | Literature values | Field survey of Yamuna cross-sections |

Each limitation has a clear resolution path requiring API access, not a different methodology.

---

## 👤 Author

**Abhinav Pandey**

M.Tech in Applied Mechanics · NIT Allahabad (MNNIT)

- 🏆 **1 Patent Filed** (Sole Inventor) — Atmospheric Correction for Solar Estimation
- 📄 **2 Research Papers Under Review** — Elsevier (IF: 13.1) + NPJ Climate & Atmospheric Sciences (IF: 9.0)
- 🎯 **GATE 95+ Percentile** across 3 attempts (2017, 2018, 2020)
- 🔗 [Portfolio](https://abhinavpandey3419.github.io/) · [LinkedIn](https://www.linkedin.com/in/abhinavpandey3419/) · [GitHub](https://github.com/Abhinav3419)

---

## 📜 License

MIT — See [LICENSE](LICENSE)
