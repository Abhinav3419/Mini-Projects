<div align="center">

<!-- Animated header using capsule render -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:161B22,100:1A6334&height=220&section=header&text=Mini%20Projects&fontSize=72&fontColor=58A6FF&animation=fadeIn&fontAlignY=35&desc=ML%20Engineering%20%E2%80%A2%20Model%20Deployment%20%E2%80%A2%20Production%20Pipelines&descSize=16&descColor=8B949E&descAlignY=55" width="100%"/>

<br>

<!-- Badges row -->
[![GitHub](https://img.shields.io/badge/GitHub-Abhinav3419-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abhinav3419)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-DA3832?style=for-the-badge&logo=creativecommons&logoColor=white)](#)
[![Projects](https://img.shields.io/badge/Mini_Projects-01-58A6FF?style=for-the-badge&logo=tensorflow&logoColor=white)](#mini-projects)
[![Status](https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge&logo=statuspage&logoColor=white)](#)

<br>

*Hands-on ML engineering projects — from raw data to deployed APIs.*
*Each project documents the full journey: exploration, modeling, evaluation, and production deployment.*

<br>

<!-- Tech stack pills -->
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

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

> *"Your model's training loss is $1,400. What's the test loss?" — Every ML interview, ever.*

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

## `// Repo Structure`

```
Mini-Projects/
│
├── 01-Medical-Insurance-Cost-Predictor/
│   ├── README.md .................. project overview + results + learnings
│   ├── notebooks/
│   │   └── Project_Insurance_Cost_Prediction.ipynb
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py ............... FastAPI server (routes, schemas, health)
│   │   └── predictor.py .......... prediction pipeline (load, validate, predict)
│   ├── model/
│   │   ├── best_insurance_model.h5
│   │   ├── feature_scaler.pkl
│   │   └── feature_info.pkl
│   ├── tests/
│   │   └── test_local.py ......... 7 smoke tests
│   ├── Dockerfile ................ production container recipe
│   ├── .dockerignore
│   └── requirements.txt .......... pinned dependencies
│
└── ... (next project loading...)
```

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

&nbsp;

## `// Interview Concepts Covered`

Each project is annotated with interview-relevant concepts in the code comments:

```
🎯 Training-Serving Skew     — Why feature pipelines must be identical
🎯 Cold Start vs Warm Start  — Model loading strategy in production
🎯 Bias-Variance Tradeoff    — Why smaller models generalized better
🎯 REST API Design           — GET vs POST, schema validation, status codes
🎯 Docker Layer Caching      — Why COPY requirements.txt before code
🎯 Health Checks             — Liveness and readiness probes
🎯 Singleton Pattern         — One model instance, many requests
🎯 Defensive Programming     — Validate before predict, fail fast
```

&nbsp;

## `// Quick Start`

```bash
# Clone the repo
git clone https://github.com/Abhinav3419/Mini-Projects.git
cd Mini-Projects/01-Medical-Insurance-Cost-Predictor

# Setup environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run tests
python tests/test_local.py

# Start the API
uvicorn app.main:app --reload
# → http://localhost:8000/docs (Swagger UI)
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
