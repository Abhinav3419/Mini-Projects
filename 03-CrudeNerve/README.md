<div align="center">

# 🛢️ CrudeNerve

### Geopolitical signal processing engine for crude oil volatility prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Predicting VIX direction and Brent crude return distributions by exploiting the lag between geopolitical signals and market response.*

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Prediction Modes](#-three-prediction-modes) · [War Barometer](#-war-tension-barometer) · [API](#-api-reference) · [Dashboard](#-dashboard)

</div>

---

## 💡 Core thesis

Crude oil prices react to geopolitical events with measurable lag. A Trump Truth Social post about Iran sanctions hits social media hours before VIX and Brent crude respond. GDELT event spikes precede supply disruption pricing by 12-48 hours. This lag is exploitable.

CrudeNerve captures 5 real-time signal streams, processes them through a physics-informed feature engineering pipeline (damped harmonic oscillator decay kernels from vibration theory), and outputs calibrated probability distributions for VIX direction at 24h, 48h, and 72h horizons.

**What makes this different:**
- **DHO decay kernel** — each geopolitical event is modeled as an impulse response using damped harmonic oscillator physics. Sanctions announcements are underdamped (persist for weeks). Trump bluster is overdamped (fades in days). The damping ratio is severity-graded.
- **Three prediction modes** — Intuitive (face value), Counter-Intuitive (contrarian), Golden Mean (selective trust with per-signal toggles). Same data, different market hypotheses.
- **10-level war barometer** — dynamically assesses USA-Iran-Israel escalation from all data streams. Modifies feature weights across all prediction modes.
- **Transformer NLP** — sentence-transformers for semantic topic classification with negation detection, not keyword matching. "NOT sanctioning Iran" scores differently from "sanctioning Iran."

---

## 🚀 Quick start

### Prerequisites
- Python 3.11+
- 4GB+ RAM (transformer model loads ~400MB)
- Git

### Installation

```bash
# Clone
git clone https://github.com/Abhinav3419/CrudeNerve.git
cd CrudeNerve

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the integration test

```bash
# Verify everything works (uses synthetic data, no API keys needed)
python -m crudenerve.tests.test_integration
```

Expected output:
```
✓ D1 GDELT:         168 days, 15 cols
✓ D2 Truth Social:  200 posts → 155 days
✓ D3 Twitter:       182 days, 10 cols
✓ D4 Supply:        182 days, 10 cols
✓ D5 Price/VIX:     130 days, 4 cols
✓ F1 Merge:         182 days × 69 cols
✓ F2 Tension Index: 14 cols
✓ F3 DHO Kernel:    5 cols
✓ F4 Trump Vol:     18 cols
✓ F5 Twitter:       6 cols
✓ F6 Supply:        9 cols
✅ Phase 1+2 integration test PASSED
```

### Start the API server

```bash
# Start FastAPI backend (uses synthetic data by default)
python -m crudenerve.api.app

# Server runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Make your first prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "intuitive",
    "war_level": 5
  }'
```

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           5 SIGNAL STREAMS              │
                    ├────────┬────────┬────────┬──────┬───────┤
                    │ GDELT  │ Truth  │Twitter │Supply│ Price │
                    │ events │ Social │  /X    │ data │  VIX  │
                    │  (D1)  │ (D2)   │ (D3)  │ (D4) │ (D5)  │
                    └───┬────┴───┬────┴───┬────┴──┬───┴───┬───┘
                        │        │        │       │       │
                    ┌───▼────────▼────────▼───────▼───────▼───┐
                    │        F1: Time-indexed merge            │
                    │     (unified daily DataFrame)            │
                    └──────────────────┬───────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
        ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │F2 Tension │  │F3 DHO decay │  │F4 Trump vol │
        │  index    │  │  kernels    │  │  injection  │
        └─────┬─────┘  └──────┬──────┘  └──────┬──────┘
              │               │                │
              │    ┌──────────┴──────────┐     │
              │    │  F5 Twitter  F6 Supply│    │
              │    └──────────┬──────────┘     │
              └───────────────┼────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  F8: War barometer │  ← 10-level escalation
                    │  (scales features) │    assessment
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  F7: Prediction   │  ← 3 modes: Intuitive /
                    │  mode transform   │    Counter-Intuitive /
                    └─────────┬─────────┘    Golden Mean
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │ M1: HMM   │  │ M2: XGB   │  │ M3: Platt │
        │ regime    │→ │ quantile  │→ │ calibrated│
        │ (4 states)│  │ (60 models)│  │ ensemble  │
        └───────────┘  └───────────┘  └─────┬─────┘
                                            │
                              ┌──────────────┼──────────────┐
                              │              │              │
                        ┌─────▼────┐  ┌──────▼─────┐ ┌─────▼────┐
                        │O1 FastAPI│  │O2 React    │ │O3 MLflow │
                        │9 endpts  │  │dashboard   │ │tracking  │
                        └──────────┘  └────────────┘ └──────────┘
```

### Module count: 23 modules · 8,952 lines

| Tier | Modules | Description |
|------|---------|-------------|
| Config | `settings.py` | Central configuration — all tunable parameters |
| NLP | `nlp_engine.py` | Transformer embeddings, VADER ensemble, negation detection, entity extraction |
| Tier 0 | D1-D5 | 5 data ingestion pipelines (GDELT, Truth Social, Twitter, supply, price) |
| Tier 1 | F1-F8 | 8 feature modules (merge, tension, DHO, Trump vol, Twitter, supply, modes, war) |
| Tier 2 | M1-M4 | 4 model modules (HMM regime, XGBoost quantile ×60, Platt ensemble, eval+SHAP) |
| Tier 3 | O1-O3 | FastAPI backend, React dashboard, MLflow experiment tracking |

---

## 🎯 Three prediction modes

The same data can support radically different market hypotheses. Instead of picking one, CrudeNerve lets you choose:

### Intuitive mode
> *"Take all signals at face value."*

Trump says maximum sanctions on Iran → oil goes up. Dollar weakens → commodities rise. This is what every textbook economist would predict. Correct ~70% of the time. Gets killed by bluffs and priced-in events.

### Counter-Intuitive mode
> *"Signals are misleading."*

Two sub-modes:
- **Flip** — invert directional signals. High-severity Trump post = bullish (he's bluffing, market overreacted, mean reversion incoming).
- **Zero** — drop all social/geopolitical signals. Only price history and hard supply data survive.

Best for post-panic recovery and political theater. Catastrophic during genuine crises.

### Golden Mean mode
> *"Some signals are genuine, some are noise. You decide which."*

Toggle each of 8 signal streams on or off:

| Stream | Category | Default |
|--------|----------|---------|
| Truth Social / Trump rhetoric | Social | OFF |
| Twitter/X sentiment | Social | ON |
| GDELT geopolitical tension | Geopolitical | ON |
| DHO event decay kernels | Geopolitical | ON |
| EIA crude inventory | Supply | ON |
| Strategic Petroleum Reserve | Supply | ON |
| Hormuz shipping throughput | Supply | ON |
| OPEC production compliance | Supply | ON |

**4 named presets:**
- `macro_only` — trust fundamentals, ignore all social noise
- `social_skeptic` — trust everything except Trump rhetoric
- `crisis_trader` — only real-time crisis signals, ignore slow-moving data
- `contrarian_lite` — hard supply data only, discount everything else

---

## 🌡 War tension barometer

A 10-level USA-Iran-Israel escalation assessment that persists across all prediction modes. Dynamically recommended each run; user can accept or override.

| Level | Name | Brent impact | VIX impact | Tension × |
|-------|------|-------------|------------|-----------|
| 1 | Diplomatic calm | ±1% | ±0.5% | 0.5 |
| 2 | Background tension | ±2% | ±1% | 0.7 |
| 3 | Diplomatic friction | +1% to +4% | +1% to +2% | 0.85 |
| 4 | Sanctions escalation | +2% to +6% | +1% to +3% | 1.0 |
| 5 | Proxy conflict active | +3% to +8% | +2% to +4% | 1.2 |
| 6 | Direct threats | +5% to +12% | +3% to +6% | 1.5 |
| 7 | Military mobilization | +8% to +18% | +5% to +10% | 1.8 |
| 8 | Limited strikes | +10% to +25% | +8% to +15% | 2.2 |
| 9 | Open conflict | +20% to +50% | +15% to +30% | 3.0 |
| 10 | Full-scale regional war | +40% to +100%+ | +25% to +50%+ | 5.0 |

The barometer ingests the latest data from all 5 streams and scores 5 sub-signals:
1. GDELT military event intensity (Iran/Israel entity density)
2. Truth Social war rhetoric (peak severity of Iran/Israel posts)
3. Twitter fear trajectory (fear index trend)
4. Hormuz disruption status (anomaly days in 14d window)
5. Oil price volatility regime (annualized vol + VIX level)

Combined score: 60% max sub-score + 40% mean — biased toward the hottest signal because escalation is driven by the worst signal, not the average.

---

## 🔬 DHO decay kernel — the physics contribution

Each geopolitical event is modeled as a damped harmonic oscillator impulse response:

```
response(t) = A × exp(-ζ × ω_n × t) × cos(ω_d × t + φ)
```

| Parameter | Meaning |
|-----------|---------|
| `A` | Amplitude (event severity) |
| `ζ` | Damping ratio (how fast the effect fades) |
| `ω_n` | Natural frequency (market reaction speed) |
| `ω_d` | Damped frequency = ω_n × √(1 - ζ²) |

**Event-type parameters:**

| Event | ζ | ω_n | Behavior |
|-------|---|-----|----------|
| Sanctions announced | 0.15 | 0.3 | Underdamped — persists for weeks |
| Military strike | 0.6 | 1.2 | Near-critical — sharp spike, fast decay |
| OPEC quota change | 0.3 | 0.5 | Moderate — 1-2 week effect |
| Hormuz disruption | 0.2 | 0.9 | Underdamped — supply fear lingers |
| Trump post (sev 1-2) | 0.7 | 0.5 | Overdamped — noise, fades in a day |
| Trump post (sev 4.5+) | 0.1 | 1.4 | Underdamped — sustained shock, 7-14 days |

The key insight: low-severity Trump posts are overdamped (noise), high-severity are underdamped (they oscillate and persist because follow-up actions keep re-exciting the system).

---

## 📡 NLP engine

The sentiment analysis layer uses transformer embeddings, not keyword matching.

**Architecture:**
1. **Sentence-transformers** (all-MiniLM-L6-v2) encodes every post into a 384-dim dense vector in a single forward pass
2. **Topic classification** via cosine similarity to curated anchor sentences per topic — captures semantics that keywords miss
3. **Severity scoring** via cosine similarity to severity-level anchors with soft assignment
4. **Negation detection** — 30+ cue words with 5-word window, reduces urgency and extremity scores when active
5. **Sentiment** — VADER (0.3) + transformer (0.7) ensemble
6. **Entity extraction** — 40+ geopolitical entities with negation context tagging

**Negation example:**
```
"We WILL impose maximum sanctions on Iran"
  → sanctions_iran: 1.000, severity: 3.01

"We will NOT impose sanctions on Iran"
  → sanctions_iran: 0.355, severity: 2.63    (↓64.5% topic, ↓12.6% severity)
```

Falls back to keyword-based classification when transformers are unavailable.

---

## 📊 API reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Full prediction (mode + war level + stream toggles) |
| `GET` | `/war-assessment` | Dynamic war tension assessment |
| `GET` | `/war-levels` | All 10 escalation level descriptions |
| `GET` | `/modes` | Available prediction modes |
| `GET` | `/streams` | 8 toggleable signal streams |
| `GET` | `/presets` | Golden Mean preset configurations |
| `GET` | `/health` | Pipeline health check |
| `GET` | `/pipeline-status` | Data freshness + feature count |
| `POST` | `/rebuild` | Force pipeline data rebuild |

### POST /predict

```json
// Request
{
  "mode": "golden_mean",
  "preset": "social_skeptic",
  "war_level": 6
}

// Response
{
  "mode_info": {
    "mode": "Golden Mean (social_skeptic)",
    "philosophy": "Selective trust based on signal quality",
    "streams_discounted": ["Truth Social / Trump Rhetoric"]
  },
  "war_assessment": {
    "recommended_level": 6,
    "level_name": "Direct threats",
    "confidence": 0.35,
    "sub_scores": { "gdelt_military": 4.0, "hormuz_status": 7.0, ... }
  },
  "predictions": [
    {
      "horizon_hours": 24,
      "vix_direction_prob": 0.58,
      "vix_up_prob": 0.38,
      "vix_down_prob": 0.20,
      "brent_return_quantiles": {
        "q10": -0.02, "q25": 0.005, "q50": 0.018, "q75": 0.035, "q90": 0.06
      }
    }
  ],
  "top_drivers": [
    { "feature": "Hormuz disruption", "shap_value": 0.82, "direction": "bearish" }
  ]
}
```

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## 🖥 Dashboard

The React dashboard provides a three-panel trading interface:

- **Left panel** — Mode selector (3 modes), counter-intuitive sub-mode toggle, Golden Mean preset buttons + per-stream switches, war barometer slider (1-10)
- **Center panel** — Three horizon forecast cards (24h/48h/72h) with VIX direction probabilities, up/down splits, Brent return quantile fan charts
- **Right panel** — SHAP feature drivers ranked by importance, war assessment sub-score breakdown

### Running the dashboard

```bash
cd dashboard
npm install
npm run dev
# Opens at http://localhost:5173
# Connects to FastAPI backend at http://localhost:8000
```

---

## ⚙️ Configuration

All tunable parameters live in `config/settings.py`:

```python
# Prediction targets
PREDICTION_HORIZONS = [24, 48, 72]     # hours
VIX_DIRECTION_THRESHOLD = 0.02         # 2% move threshold

# Truth Social severity weights (must sum to 1.0)
severity_weight_urgency = 0.40
severity_weight_extremity = 0.35
severity_weight_targeting = 0.25

# Twitter elite threshold
elite_follower_threshold = 100_000

# Hormuz anomaly detection
hormuz_anomaly_std_threshold = 2.0     # 2σ for anomaly flag

# Trump DHO severity-graded parameters
TRUMP_DHO_TABLE = [
    {"severity_min": 1.0, "severity_max": 2.0, "zeta": 0.7,  "omega_n": 0.5},
    {"severity_min": 4.5, "severity_max": 5.0, "zeta": 0.1,  "omega_n": 1.4},
    ...
]
```

### Environment variables

```bash
# Optional — for live data instead of synthetic
export X_API_BEARER_TOKEN="your_twitter_api_key"
export FRED_API_KEY="your_fred_api_key"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"   # or remote server
```

---

## 🧪 Testing

```bash
# Full integration test (synthetic data, no API keys needed)
python -m crudenerve.tests.test_integration

# Individual module tests
python -m crudenerve.data_ingest.d1_gdelt          # GDELT pipeline
python -m crudenerve.data_ingest.d2_truth_social    # Truth Social NLP
python -m crudenerve.features.f3_dho_kernel         # DHO decay kernels
python -m crudenerve.features.f7_prediction_mode    # Three prediction modes
python -m crudenerve.features.f8_war_barometer      # War barometer
python -m crudenerve.models.m1_hmm_regime           # HMM regime detection
python -m crudenerve.utils.nlp_engine               # Transformer NLP engine
```

---

## 📁 Project structure

```
crudenerve/
├── config/settings.py                    Central configuration
├── data_ingest/
│   ├── d1_gdelt.py                       GDELT 2.0 event ingestion
│   ├── d2_truth_social.py                Truth Social scraper + NLP pipeline
│   ├── d3_twitter.py                     Twitter/X sentiment
│   ├── d4_supply.py                      EIA, SPR, Hormuz, OPEC
│   └── d5_price_vix.py                   Brent, WTI, VIX from Yahoo/FRED
├── features/
│   ├── f1_merge.py                       Time-indexed merge engine
│   ├── f2_tension_index.py               Geopolitical tension index
│   ├── f3_dho_kernel.py                  DHO impulse response kernels
│   ├── f4_trump_volatility.py            Trump volatility injection
│   ├── f5_twitter_features.py            Twitter signal features
│   ├── f6_supply_features.py             Supply disruption features
│   ├── f7_prediction_mode.py             3 prediction modes + 8 stream toggles
│   └── f8_war_barometer.py               10-level war tension barometer
├── models/
│   ├── m1_hmm_regime.py                  4-state HMM regime detector
│   ├── m2_regime_xgboost.py              60 regime × horizon × quantile models
│   ├── m3_ensemble.py                    Platt-calibrated ensemble
│   └── m4_evaluation.py                  Walk-forward eval + SHAP
├── api/app.py                            FastAPI — 9 endpoints
├── dashboard/App.jsx                     React trading dashboard
├── utils/
│   ├── nlp_engine.py                     Transformer NLP engine
│   └── mlflow_tracker.py                 MLflow experiment tracking
├── tests/test_integration.py             End-to-end pipeline test
└── requirements.txt
```

---

## 🗺 Roadmap

- [ ] Production Truth Social scraper (Selenium + headless Chrome)
- [ ] Live X API v2 integration with streaming
- [ ] AIS data provider integration for Hormuz throughput (MarineTraffic/Spire)
- [ ] DistilBERT fine-tuning to replace sentence-transformers for domain-specific accuracy
- [ ] SHAP TreeExplainer for true Shapley values (replacing XGBoost gain-based importance)
- [ ] Real-time WebSocket push for dashboard updates
- [ ] Backtesting framework with walk-forward optimization
- [ ] Multi-commodity extension (natural gas, gold, USD index)

---

## 👤 Author

**Abhinav**
- M.Tech Applied Mechanics, NIT Allahabad
- B.Tech Electronics & Instrumentation Engineering
- [GitHub](https://github.com/Abhinav3419) · [Portfolio](https://abhinav3419.vercel.app)

The DHO decay kernel — the core physics contribution of this project — draws directly from vibration theory and impulse response analysis studied during the Applied Mechanics program. The insight that geopolitical events behave like damped oscillators (some underdamped and persistent, others overdamped and transient) is what separates this from standard NLP-to-prediction pipelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
