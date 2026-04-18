# Medical Insurance Cost Predictor

**Domain:** Healthcare · Insurance Pricing · Regression · MLOps

A neural network that predicts annual medical insurance charges from patient demographics, deployed as a production-ready FastAPI application.

## Results

| Model | Architecture | Test MAE | Test R² |
|:---|:---|---:|---:|
| M0: Baseline | 20 neurons, SGD, unscaled | ~$5,000+ | ~0.50 |
| M1: Adam optimizer | 100-10, Adam, unscaled | ~$3,000+ | ~0.70 |
| M2: Architecture tuning | 120-60-30 Swish, LR=0.006 | ~$2,000 | 0.80 |
| M3: + Scaling | + StandardScaler | ~$1,950 | 0.80 |
| M4: + Basic features | + smoker×bmi, age×smoker, EarlyStopping | ~$1,840 | 0.81 |
| **M5: Full engineering** | **+ 10 engineered features + ReduceLR** | **~$1,650** | **0.82** |
| M6: Ensemble | 3-seed average | ~$1,645 | 0.82 |

### Segment-wise Error Analysis

| Segment | Mean Charges | Segment MAE | Count |
|:---|---:|---:|---:|
| Non-smoker | $8,434 | ~$1,800 | 345 |
| Smoker (BMI < 30) | $21,363 | ~$1,000 | 46 |
| Smoker (BMI ≥ 30) | $41,558 | ~$1,350 | 55 |

### Classical Baselines (with same engineered features)

| Model | Test MAE |
|:---|---:|
| Linear Regression | ~$4,700 |
| Random Forest (300 trees) | ~$2,900 |
| Gradient Boosting (500 trees) | ~$2,800 |
| **Our Neural Net (M5)** | **~$1,650** |

## Key Learnings

1. **Data > Model**: Feature scaling and engineering improved MAE more than any architecture change
2. **Domain knowledge is a feature**: `smoker_obese` binary encodes a medical insight worth hundreds of dollars of MAE
3. **Train ≠ Test**: Training MAE of $1,400 with test MAE of $2,400 means overfitting, not success
4. **Callbacks > manual epochs**: EarlyStopping + ReduceLROnPlateau replace guesswork with data

## API Usage

```bash
# Start the server
uvicorn app.main:app --reload

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "sex": "male", "bmi": 28.5, "children": 2, "smoker": "no", "region": "northwest"}'
```

## Engineered Features

| Feature | Why it exists |
|:---|:---|
| `smoker_bmi` | Continuous smoker × BMI interaction |
| `is_obese` | Binary threshold at BMI = 30 (clinical cutoff) |
| `smoker_obese` | Smoker AND obese — separates $41K from $21K group |
| `smoker_bmi_above30` | How far above 30 is BMI for smokers |
| `age_smoker` | Smokers accumulate costs faster with age |
| `age_squared` | Healthcare costs accelerate with age |
| `age_bmi` | Older + heavier = more than the sum |
| `bmi_deviation` | Distance from healthy BMI (24.9) |
| `bmi_squared` | Nonlinear BMI cost curve |
| `has_children` | Binary: any children vs none |

## Files

```
├── notebooks/
│   └── Project_Insurance_Cost_Prediction.ipynb   # Full training notebook (7 models)
├── app/
│   ├── main.py              # FastAPI server
│   └── predictor.py         # Prediction pipeline
├── model/
│   ├── best_insurance_model.h5   # Trained model (190 KB)
│   ├── feature_scaler.pkl        # Fitted StandardScaler
│   └── feature_info.pkl          # Column metadata
├── tests/
│   └── test_local.py        # 7 smoke tests
├── Dockerfile               # Production container
└── requirements.txt         # Pinned dependencies
```
