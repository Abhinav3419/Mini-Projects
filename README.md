<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:161B22,100:1A6334&height=220&section=header&text=Mini%20Projects&fontSize=72&fontColor=58A6FF&animation=fadeIn&fontAlignY=35&desc=ML%20Engineering%20%E2%80%A2%20Model%20Deployment%20%E2%80%A2%20Production%20Pipelines&descSize=16&descColor=8B949E&descAlignY=55" width="100%"/>

<br>

[![GitHub](https://img.shields.io/badge/GitHub-Abhinav3419-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abhinav3419)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-DA3832?style=for-the-badge&logo=creativecommons&logoColor=white)](#)
[![Projects](https://img.shields.io/badge/Mini_Projects-02-58A6FF?style=for-the-badge&logo=tensorflow&logoColor=white)](#mini-projects)
[![Status](https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge&logo=statuspage&logoColor=white)](#)

<br>

*Hands-on ML engineering projects — from raw data to deployed APIs to real-time crisis intelligence.*
*Each project documents the full journey: exploration, modeling, evaluation, and production deployment.*

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat-square)
![Folium](https://img.shields.io/badge/Folium-77B829?style=flat-square)

</div>

---

&nbsp;

## `// What will you find here?`

Production-oriented ML mini projects — not tutorials, not Kaggle kernels. Each project solves a real prediction problem end-to-end: data exploration, feature engineering, model training, systematic experimentation, deployment-ready API, and containerized serving. The emphasis is on **engineering discipline** — train/test separation, reproducibility, documented decisions, and honest error reporting.

&nbsp;

## `// Mini Projects`

<table>
<tr>
<td width="80" align="center"><b><code>01</code></b></td>
<td>

### [Medical Insurance Cost Predictor](./01-Medical-Insurance-Cost-Predictor/)

**Domain:** Healthcare · Insurance · Regression · MLOps

> *"Your model's training loss is $1,400. Great. What's the test loss?"*

A neural network that predicts annual medical insurance charges from patient demographics. Built through **systematic experimentation** — 7 model iterations that demonstrate the real hierarchy of what matters in ML:

&nbsp;

**The Progression:**

| Step | What Changed | Test MAE |
|:---|:---|---:|
| `M0` Baseline | SGD, 20 neurons, unscaled | ~$5,000+ |
| `M2` Architecture | 120-60-30 Swish, LR tuning | ~$2,000 |
| `M3` Preprocessing | + StandardScaler | ~$1,950 |
| `M4` Domain knowledge | + smoker×bmi, age×smoker | ~$1,840 |
| `M5` Full engineering | + 10 engineered features | **~$1,650** |
| `M6` Ensemble | 3-seed average | **~$1,645** |

&nbsp;

**Key Insight:** `smoker_obese` (a single binary feature) was worth more MAE reduction than 200 extra neurons.

&nbsp;

**What's Inside:**

| Component | Description |
|:---|:---|
| `notebooks/` | Full training notebook with 7 models, segment-wise analysis, diagnostic plots |
| `app/` | FastAPI prediction API with Pydantic validation, health checks, logging |
| `model/` | Exported H5 model + StandardScaler + feature metadata |
| `tests/` | 7 smoke tests covering predictions, validation, and segment classification |
| `Dockerfile` | Production container with layer caching, health checks, slim base image |

&nbsp;

![Neural Network](https://img.shields.io/badge/128--64--32_Swish-Neural_Net-FF6F00?style=flat-square&logo=tensorflow)
![MAE](https://img.shields.io/badge/Test_MAE-$1,650-00C853?style=flat-square)
![R²](https://img.shields.io/badge/Test_R²-0.82-58A6FF?style=flat-square)
![Features](https://img.shields.io/badge/Features-21_(11+10)-8B949E?style=flat-square)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</td>
</tr>
</table>

&nbsp;

<table>
<tr>
<td width="80" align="center"><b><code>02</code></b></td>
<td>

### [CrisisLens AI — Urban Decision Intelligence](./02-CrisisLens-AI/)

**Domain:** GovTech · Geospatial · RAG · Physics-Informed ML · Crisis Response

> *"The nearest ambulance is 2km away. The fastest is 5km away. Which one do you send?"*

A **7-module Urban Decision Intelligence Platform** for New Delhi that combines live weather data, physics-based flood prediction, hospital capacity tracking, incident command, telecom anomaly detection, RAG-powered crisis advisory, and dispatch route optimization into a single Streamlit command-center interface.

&nbsp;

**The Modules:**

| # | Module | What It Does | Key Result |
|:---|:---|:---|---:|
| `M1` | 🌦️ Weather + Flood Risk | Live weather + 8-zone flood risk heatmap | Open-Meteo API |
| `M2` | 🌊 Manning's Flood Model | Predicts flood arrival time at 5 Yamuna stations | Q = (1/n)AR^⅔S^½ |
| `M3` | 🏥 Hospital Capacity | 20 hospitals with bed/ICU availability | Simulated live |
| `M4` | 🚨 Incident Command | Geotagged incidents + 4-modifier severity classifier | GDELT + simulated |
| `M4B` | 📡 Telecom Anomaly | 500 cell towers, Z-score crisis detection | Zero false positives |
| `M5` | 📚 RAG Crisis Advisor | NDMA/DDMA protocol retrieval via ChromaDB | 15 protocols |
| `M6` | 🚑 Dispatch Optimizer | Fastest-route dispatch (not nearest-by-distance) | **8+ min saved** |

&nbsp;

**Key Insight:** In the Mayur Vihar flood scenario, dispatching by fastest road route saved **8.4 minutes** over dispatching the nearest unit by straight-line distance. In emergency response, 8 minutes is the difference between life and death.

&nbsp;

**What's Inside:**

| Component | Description |
|:---|:---|
| `app.py` | Premium Streamlit UI — 7 interactive tabs, light-mode, gradient accents |
| `config.py` | New Delhi geography: Yamuna stations, flood zones, 20 hospitals, Manning's params |
| `data_ingestion.py` | Unified data layer — weather, hospitals, flood risk, incidents (all APIs) |
| `modules/` | 7 self-contained modules with individual demos and tests |
| `data/` | Generated maps (Folium HTML), temp files |

&nbsp;

![Manning](https://img.shields.io/badge/Physics-Manning's_Equation-1A5E20?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-ChromaDB_15_Protocols-FF6B6B?style=flat-square)
![Towers](https://img.shields.io/badge/Telecom-500_Towers_Z--Score-4A148C?style=flat-square)
![Dispatch](https://img.shields.io/badge/Dispatch-8+_min_saved-2563EB?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit_Premium-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Live](https://img.shields.io/badge/Data-Live_APIs-00C853?style=flat-square)

</td>
</tr>
</table>

&nbsp;

<table>
<tr>
<td>

<details open>
<summary>&nbsp;📂&nbsp;&nbsp;<b>Mini-Projects</b></summary>
<blockquote>

&nbsp;📄&nbsp;&nbsp;<code>README.md</code><br>
&nbsp;📄&nbsp;&nbsp;<code>.gitignore</code><br>
&nbsp;📄&nbsp;&nbsp;<code>LICENSE</code>

<details open>
<summary>&nbsp;📂&nbsp;&nbsp;<b>01-Medical-Insurance-Cost-Predictor</b></summary>
<blockquote>

&nbsp;📄&nbsp;&nbsp;<code>README.md</code>&nbsp;&nbsp;—&nbsp;&nbsp;project overview + results<br>
&nbsp;🐳&nbsp;&nbsp;<code>Dockerfile</code>&nbsp;&nbsp;—&nbsp;&nbsp;production container recipe<br>
&nbsp;🚫&nbsp;&nbsp;<code>.dockerignore</code><br>
&nbsp;📦&nbsp;&nbsp;<code>requirements.txt</code>&nbsp;&nbsp;—&nbsp;&nbsp;pinned dependencies

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>notebooks</b>&nbsp;&nbsp;<sup>training & experimentation</sup></summary>
<blockquote>
&nbsp;📓&nbsp;&nbsp;<code>Project_Insurance_Cost_Prediction.ipynb</code><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sub>7 models · EDA · segment analysis · diagnostic plots</sub>
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>app</b>&nbsp;&nbsp;<sup>FastAPI prediction service</sup></summary>
<blockquote>
&nbsp;🐍&nbsp;&nbsp;<code>__init__.py</code><br>
&nbsp;🐍&nbsp;&nbsp;<code>main.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;routes · Pydantic schemas · health checks · logging<br>
&nbsp;🐍&nbsp;&nbsp;<code>predictor.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;load → validate → encode → engineer → scale → predict
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>model</b>&nbsp;&nbsp;<sup>trained artifacts</sup></summary>
<blockquote>
&nbsp;🧠&nbsp;&nbsp;<code>best_insurance_model.h5</code>&nbsp;&nbsp;—&nbsp;&nbsp;128-64-32 Swish network&nbsp;&nbsp;<sup>190 KB</sup><br>
&nbsp;⚖️&nbsp;&nbsp;<code>feature_scaler.pkl</code>&nbsp;&nbsp;—&nbsp;&nbsp;fitted StandardScaler<br>
&nbsp;📋&nbsp;&nbsp;<code>feature_info.pkl</code>&nbsp;&nbsp;—&nbsp;&nbsp;column order & metadata
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>tests</b>&nbsp;&nbsp;<sup>smoke tests</sup></summary>
<blockquote>
&nbsp;🧪&nbsp;&nbsp;<code>test_local.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;7 tests: predictions · validation · segments
</blockquote>
</details>

</blockquote>
</details>

<details open>
<summary>&nbsp;📂&nbsp;&nbsp;<b>02-CrisisLens-AI</b></summary>
<blockquote>

&nbsp;📄&nbsp;&nbsp;<code>README.md</code>&nbsp;&nbsp;—&nbsp;&nbsp;full documentation + architecture<br>
&nbsp;🏙️&nbsp;&nbsp;<code>app.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Streamlit 7-tab premium UI<br>
&nbsp;⚙️&nbsp;&nbsp;<code>config.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;New Delhi geography, zones, hospitals<br>
&nbsp;📡&nbsp;&nbsp;<code>data_ingestion.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;unified data layer (all APIs)<br>
&nbsp;📦&nbsp;&nbsp;<code>requirements.txt</code><br>
&nbsp;📄&nbsp;&nbsp;<code>LICENSE</code>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>modules</b>&nbsp;&nbsp;<sup>7 self-contained modules</sup></summary>
<blockquote>
&nbsp;🌦️&nbsp;&nbsp;<code>weather_flood.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;live weather + flood risk heatmap<br>
&nbsp;🌊&nbsp;&nbsp;<code>manning_model.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Manning's equation flood prediction<br>
&nbsp;🏥&nbsp;&nbsp;<code>hospital_panel.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;hospital capacity + nearest routing<br>
&nbsp;🚨&nbsp;&nbsp;<code>incident_map.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;incident command + severity classifier<br>
&nbsp;📡&nbsp;&nbsp;<code>telecom_anomaly.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;500 towers, Z-score anomaly detection<br>
&nbsp;📚&nbsp;&nbsp;<code>rag_advisor.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;RAG crisis advisor (ChromaDB, 15 protocols)<br>
&nbsp;🚑&nbsp;&nbsp;<code>dispatch_optimizer.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;fastest-route dispatch optimizer
</blockquote>
</details>

</blockquote>
</details>

</blockquote>
</details>

</td>
</tr>
</table>

&nbsp;

## `// Skills Demonstrated`

| Skill Area | Techniques | Where |
|:---|:---|:---|
| `Feature Engineering` | Domain-driven interaction features, medical threshold encoding | `#01` |
| `Model Selection` | Systematic architecture search, loss function comparison, seed stability | `#01` |
| `Evaluation Discipline` | Train/test separation, segment-wise error analysis, multiple metrics | `#01` |
| `ML Deployment` | FastAPI serving, Pydantic validation, health checks, Docker containerization | `#01` |
| `Production Patterns` | Singleton model loading, training-serving parity, input validation, logging | `#01` |
| `Experiment Tracking` | Progressive model comparison with controlled variables | `#01` |
| `Physics-Informed ML` | Manning's equation for flood prediction, kinematic wave propagation | `#02` |
| `RAG & LLM` | ChromaDB vector store, NDMA protocol retrieval, context-aware advisory | `#02` |
| `Geospatial Analytics` | Folium interactive maps, flood risk heatmaps, hospital routing | `#02` |
| `Anomaly Detection` | Z-score on diurnal time-series, spatial clustering (DBSCAN-like) | `#02` |
| `Live API Integration` | Open-Meteo weather, GDELT news events, OpenRouteService routing | `#02` |
| `System Design` | 7-module architecture, unified data layer, graceful API fallbacks | `#02` |

&nbsp;

## `// Engineering Principles`

```
🔬  Training-Serving Parity    — Feature pipelines are identical in notebook and API
📊  Segment-wise Evaluation    — Overall MAE hides per-group performance; always decompose
🧪  Controlled Experiments     — Change one variable at a time, measure on held-out test set
🛡️  Defensive Prediction       — Validate inputs before they reach the model
⚡  Efficient Serving          — Load model once at startup, serve many requests
📦  Reproducible Environments  — Pinned dependencies, Docker containers, deterministic seeds
🔁  Callback-Driven Training   — EarlyStopping + ReduceLROnPlateau replace manual epoch tuning
📈  Multi-Metric Reporting     — MAE alone is incomplete; always report RMSE + R² alongside
```

&nbsp;

## `// Quick Start`

```bash
# Clone the repo
git clone https://github.com/Abhinav3419/Mini-Projects.git
cd Mini-Projects

# ─── Project 01: Insurance Cost Predictor ───
cd 01-Medical-Insurance-Cost-Predictor
pip install -r requirements.txt
python tests/test_local.py
uvicorn app.main:app --reload
# → http://localhost:8000/docs (Swagger UI)

# ─── Project 02: CrisisLens AI ───
cd ../02-CrisisLens-AI
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501 → Press LAUNCH
```

&nbsp;

---

<div align="center">

**Built by breaking things. Shipped by fixing them.**

&nbsp;

[![GitHub](https://img.shields.io/badge/Follow-@Abhinav3419-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Abhinav3419)
[![LinkedIn](https://img.shields.io/badge/Connect-abhinavpandey--ai--ml-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/abhinavpandey-ai-ml)

*Copyright © 2026 Abhinav Pandey. All rights reserved.*

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:161B22,100:1A6334&height=120&section=footer" width="100%"/>
