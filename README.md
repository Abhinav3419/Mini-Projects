<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:161B22,100:1A6334&height=220&section=header&text=Mini%20Projects&fontSize=72&fontColor=58A6FF&animation=fadeIn&fontAlignY=35&desc=ML%20Engineering%20%E2%80%A2%20Model%20Deployment%20%E2%80%A2%20Production%20Pipelines&descSize=16&descColor=8B949E&descAlignY=55" width="100%"/>

<br>

[![GitHub](https://img.shields.io/badge/GitHub-Abhinav3419-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abhinav3419)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-DA3832?style=for-the-badge&logo=creativecommons&logoColor=white)](#)
[![Projects](https://img.shields.io/badge/Mini_Projects-03-58A6FF?style=for-the-badge&logo=tensorflow&logoColor=white)](#mini-projects)
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
![XGBoost](https://img.shields.io/badge/XGBoost-EC6C35?style=flat-square)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)

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
<td width="80" align="center"><b><code>03</code></b></td>
<td>

### [CrudeNerve — Geopolitical Signal Engine for Crude Oil Volatility](./03-CrudeNerve/)

**Domain:** Quantitative Finance · NLP · Geopolitics · Physics-Informed ML · Signal Processing

> *"Trump posted about Iran sanctions at 7:14 AM. VIX moved at 9:31 AM. That's 137 minutes of alpha."*

A **23-module geopolitical signal processing engine** that predicts VIX direction and Brent crude return distributions by exploiting the lag between geopolitical rhetoric and market response. Ingests 5 real-time data streams, processes them through transformer NLP and physics-informed DHO decay kernels, and outputs calibrated probability distributions across 3 prediction modes.

&nbsp;

**The Pipeline:**

| Tier | Modules | What It Does | Key Detail |
|:---|:---|:---|---:|
| `Tier 0` | D1-D5 | 5 data ingestion streams | GDELT, Truth Social, Twitter, Supply, Price/VIX |
| `NLP` | nlp_engine | Transformer sentiment + negation detection | all-MiniLM-L6-v2 + VADER |
| `Tier 1` | F1-F8 | 8 feature engineering modules | DHO kernels, 3 modes, 10-level war barometer |
| `Tier 2` | M1-M4 | 4 ML model modules | HMM×4 states, XGBoost×60, Platt calibration |
| `Tier 3` | O1-O3 | API + Dashboard + Tracking | FastAPI 9 endpoints, React, MLflow |

&nbsp;

**Original Contributions:**

| Contribution | Description |
|:---|:---|
| **DHO Decay Kernel** | Each geopolitical event modeled as a damped harmonic oscillator impulse response — sanctions are underdamped (persist weeks), Trump bluster is overdamped (fades in days). Damping ratio is severity-graded. |
| **3 Prediction Modes** | Intuitive (face value), Counter-Intuitive (flip/zero contrarian), Golden Mean (per-signal toggle with 8 streams and 4 presets). Same data, different market hypotheses. |
| **War Tension Barometer** | 10-level USA-Iran-Israel escalation ladder dynamically assessed from all streams. Modifies feature weights across all modes. Level 1 (diplomatic calm) to Level 10 (full-scale war). |
| **Negation-Aware NLP** | Transformer cosine similarity to anchor sentences, not keyword matching. "NOT sanctioning Iran" scores 64.5% lower than "sanctioning Iran." |

&nbsp;

**Key Insight:** A severity-4.5+ Trump post about Iran gets underdamped DHO parameters (ζ=0.1, ω_n=1.4) — the market shock persists 7-14 days because follow-up actions keep re-exciting the system. A severity-1.5 post gets ζ=0.7 — noise that fades overnight.

&nbsp;

**What's Inside:**

| Component | Description |
|:---|:---|
| `config/settings.py` | Central config — all tunable parameters, DHO tables, severity weights, stream definitions |
| `data_ingest/` | 5 stream pipelines: GDELT events, Truth Social deep NLP, Twitter sentiment, physical supply, price/VIX |
| `features/` | 8 feature modules: merge, tension index, DHO kernels, Trump volatility, Twitter signals, supply disruption, 3 prediction modes, war barometer |
| `models/` | HMM regime detector (4 states), regime-specific XGBoost (60 models), Platt-calibrated ensemble, walk-forward evaluation + SHAP |
| `utils/nlp_engine.py` | Transformer NLP: sentence-transformers embeddings, VADER ensemble, negation detection, entity extraction |
| `api/app.py` | FastAPI — 9 endpoints including `/predict` with mode + war level params |
| `dashboard/App.jsx` | React trading dashboard — mode toggle, war barometer dial, quantile fan charts, SHAP drivers |
| `utils/mlflow_tracker.py` | MLflow experiment tracking — mode and war level as first-class experiment dimensions |

&nbsp;

![XGBoost](https://img.shields.io/badge/XGBoost-60_Quantile_Models-EC6C35?style=flat-square)
![HMM](https://img.shields.io/badge/HMM-4_State_Regime-8B5CF6?style=flat-square)
![Brier](https://img.shields.io/badge/Brier_Score-0.023-00C853?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-0.997-58A6FF?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-Transformer+VADER-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![API](https://img.shields.io/badge/API-FastAPI_9_Endpoints-009688?style=flat-square&logo=fastapi&logoColor=white)
![Modules](https://img.shields.io/badge/Modules-23-8B949E?style=flat-square)
![Lines](https://img.shields.io/badge/Lines-8,952-181717?style=flat-square)

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

<details open>
<summary>&nbsp;📂&nbsp;&nbsp;<b>03-CrudeNerve</b></summary>
<blockquote>

&nbsp;📄&nbsp;&nbsp;<code>README.md</code>&nbsp;&nbsp;—&nbsp;&nbsp;full documentation + architecture + API reference<br>
&nbsp;📦&nbsp;&nbsp;<code>requirements.txt</code>&nbsp;&nbsp;—&nbsp;&nbsp;pinned dependencies<br>
&nbsp;📄&nbsp;&nbsp;<code>LICENSE</code><br>
&nbsp;🚫&nbsp;&nbsp;<code>.gitignore</code>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>config</b>&nbsp;&nbsp;<sup>central configuration</sup></summary>
<blockquote>
&nbsp;⚙️&nbsp;&nbsp;<code>settings.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;all tunable params, DHO tables, severity weights, stream definitions
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>data_ingest</b>&nbsp;&nbsp;<sup>5 stream pipelines</sup></summary>
<blockquote>
&nbsp;🌍&nbsp;&nbsp;<code>d1_gdelt.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;GDELT 2.0 event ingestion + entity tagging<br>
&nbsp;📢&nbsp;&nbsp;<code>d2_truth_social.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Truth Social scraper + transformer NLP pipeline<br>
&nbsp;🐦&nbsp;&nbsp;<code>d3_twitter.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Twitter/X sentiment + elite-retail split<br>
&nbsp;🛢️&nbsp;&nbsp;<code>d4_supply.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;EIA, SPR, Hormuz throughput, OPEC compliance<br>
&nbsp;📈&nbsp;&nbsp;<code>d5_price_vix.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Brent, WTI, VIX from Yahoo Finance + FRED
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>features</b>&nbsp;&nbsp;<sup>8 feature engineering modules</sup></summary>
<blockquote>
&nbsp;🔗&nbsp;&nbsp;<code>f1_merge.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;time-indexed merge engine<br>
&nbsp;🌡️&nbsp;&nbsp;<code>f2_tension_index.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;geopolitical tension index (entity-weighted EMA)<br>
&nbsp;〰️&nbsp;&nbsp;<code>f3_dho_kernel.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;damped harmonic oscillator decay kernels<br>
&nbsp;🔥&nbsp;&nbsp;<code>f4_trump_volatility.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Trump policy-shock score + temporal patterns<br>
&nbsp;📊&nbsp;&nbsp;<code>f5_twitter_features.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;fear index, divergence, cross-platform echo<br>
&nbsp;⛽&nbsp;&nbsp;<code>f6_supply_features.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;disruption score, supply stress composite<br>
&nbsp;🎛️&nbsp;&nbsp;<code>f7_prediction_mode.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;3 prediction modes + 8 stream toggles<br>
&nbsp;⚔️&nbsp;&nbsp;<code>f8_war_barometer.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;10-level USA-Iran-Israel escalation ladder
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>models</b>&nbsp;&nbsp;<sup>4 ML model modules</sup></summary>
<blockquote>
&nbsp;🔄&nbsp;&nbsp;<code>m1_hmm_regime.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;4-state Gaussian HMM regime detector<br>
&nbsp;🌲&nbsp;&nbsp;<code>m2_regime_xgboost.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;60 regime × horizon × quantile models<br>
&nbsp;⚖️&nbsp;&nbsp;<code>m3_ensemble.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;Platt-calibrated ensemble predictor<br>
&nbsp;📏&nbsp;&nbsp;<code>m4_evaluation.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;walk-forward eval + feature importance
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>api</b>&nbsp;&nbsp;<sup>FastAPI backend</sup></summary>
<blockquote>
&nbsp;🚀&nbsp;&nbsp;<code>app.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;9 endpoints: /predict, /war-assessment, /modes, /streams, /presets, /health
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>dashboard</b>&nbsp;&nbsp;<sup>React frontend</sup></summary>
<blockquote>
&nbsp;⚛️&nbsp;&nbsp;<code>App.jsx</code>&nbsp;&nbsp;—&nbsp;&nbsp;trading interface: mode toggle, war dial, quantile fan, SHAP drivers
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>utils</b>&nbsp;&nbsp;<sup>shared utilities</sup></summary>
<blockquote>
&nbsp;🧠&nbsp;&nbsp;<code>nlp_engine.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;transformer NLP: embeddings, severity, sentiment, negation, entities<br>
&nbsp;📋&nbsp;&nbsp;<code>mlflow_tracker.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;experiment tracking: mode + war level as dimensions
</blockquote>
</details>

<details>
<summary>&nbsp;📂&nbsp;&nbsp;<b>tests</b>&nbsp;&nbsp;<sup>integration tests</sup></summary>
<blockquote>
&nbsp;🧪&nbsp;&nbsp;<code>test_integration.py</code>&nbsp;&nbsp;—&nbsp;&nbsp;end-to-end pipeline: all 5 streams → all features → unified DataFrame
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
| `Physics-Informed Finance` | DHO impulse response kernels for geopolitical event decay modeling | `#03` |
| `Transformer NLP` | Sentence-transformers cosine similarity, VADER ensemble, negation-aware scoring | `#03` |
| `Quantile Regression` | 60 regime × horizon × quantile XGBoost models with Platt calibration | `#03` |
| `Regime Detection` | 4-state Gaussian HMM on geopolitical tension for market regime classification | `#03` |
| `Signal Processing` | 5-stream real-time data fusion, temporal pattern detection, burst analysis | `#03` |
| `Multi-Modal Prediction` | 3 prediction modes (Intuitive/Counter-Intuitive/Golden Mean) with 8 toggleable streams | `#03` |
| `Dynamic Risk Assessment` | 10-level war tension barometer with max-biased sub-score aggregation | `#03` |
| `Full-Stack ML` | FastAPI 9-endpoint backend, React dashboard, MLflow experiment tracking | `#03` |

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
🌊  Physics-Informed Features  — Domain equations (Manning's, DHO) outperform pure data-driven approaches
🎯  Probabilistic Outputs      — Quantile distributions over point estimates; calibration over accuracy
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

# ─── Project 03: CrudeNerve ───
cd ../03-CrudeNerve
pip install -r requirements.txt
python -m crudenerve.tests.test_integration    # verify pipeline
python -m crudenerve.api.app                   # start API
# → http://localhost:8000/docs (Swagger UI)
# → POST /predict with {"mode": "intuitive", "war_level": 5}
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
