"""
main.py — The FastAPI application.

WHY FastAPI? (Interview concept: "Framework selection")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You'll be asked "Why FastAPI over Flask?" in interviews. The answer:

1. ASYNC support — FastAPI handles concurrent requests natively.
   If 100 users hit your API simultaneously, Flask blocks (serves one
   at a time by default). FastAPI serves all 100 concurrently.

2. AUTO-DOCUMENTATION — FastAPI generates interactive API docs at /docs.
   No extra work. Flask needs Swagger/OpenAPI setup manually.

3. PYDANTIC VALIDATION — Request/response schemas are defined as Python
   classes. FastAPI validates inputs automatically and returns clear errors.
   Flask makes you write all validation by hand.

4. TYPE HINTS — FastAPI uses Python type hints for request parsing.
   This means your code is self-documenting AND validated at runtime.

5. PERFORMANCE — FastAPI is built on Starlette (ASGI), which benchmarks
   2-3x faster than Flask (WSGI) for I/O-bound workloads.

For ML model serving specifically, FastAPI is the industry standard in 2024-2026.

HOW AN API REQUEST FLOWS (Interview concept: "Request lifecycle")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User sends POST /predict with JSON body
  → FastAPI receives the HTTP request
  → Pydantic validates the JSON against InsurancePredictionRequest schema
  → If validation fails: return 422 with error details (automatic!)
  → If valid: call predictor.predict(data)
  → predictor does: encode → engineer → scale → model.predict()
  → Return JSON response with prediction
Total time: ~10-50ms per request
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.predictor import InsurancePredictor


# ============================================================
# LOGGING SETUP
# ============================================================
# WHY LOG? In production, print() doesn't work — logs go to files
# or log aggregators (CloudWatch, Datadog). Logging lets you:
# - Track how many predictions per minute
# - Debug errors without SSH-ing into the server
# - Monitor model performance drift over time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL: MODEL INSTANCE
# ============================================================
# WHY GLOBAL? (Interview concept: "Singleton pattern")
# The model loads once and is shared across all requests.
# This is thread-safe for read-only operations (inference).
predictor = None


# ============================================================
# LIFESPAN (startup/shutdown)
# ============================================================
# WHY LIFESPAN? (Interview concept: "Application lifecycle")
# In production, you need to:
# - Load the model BEFORE accepting requests (startup)
# - Clean up resources when the server stops (shutdown)
# FastAPI's lifespan context manager handles both.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global predictor

    logger.info("Starting up — loading model...")
    model_dir = os.environ.get("MODEL_DIR", "model")
    predictor = InsurancePredictor(model_dir=model_dir)
    logger.info("Model loaded and ready to serve predictions.")

    yield  # Server is running and accepting requests

    logger.info("Shutting down — cleaning up...")
    predictor = None


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Medical Insurance Cost Predictor",
    description=(
        "Predicts annual medical insurance charges based on patient demographics. "
        "Built with a 128-64-32 Swish neural network trained on 1,338 records. "
        "Test MAE: ~$1,650 (~12% of mean charges)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
# WHY PYDANTIC? (Interview concept: "Data validation / contracts")
# These classes define the EXACT shape of request and response JSON.
# FastAPI uses them to:
# 1. Auto-validate incoming requests (wrong type → 422 error)
# 2. Auto-generate API documentation
# 3. Provide IDE auto-completion for developers using your API
#
# Think of them as a CONTRACT: "If you send me data shaped like THIS,
# I promise to return data shaped like THAT."

class InsurancePredictionRequest(BaseModel):
    """Schema for prediction request — what the user sends."""
    age:      int   = Field(..., ge=18, le=100, description="Age in years (18-100)")
    sex:      str   = Field(..., description="'male' or 'female'")
    bmi:      float = Field(..., ge=10, le=70, description="Body Mass Index (10-70)")
    children: int   = Field(..., ge=0, le=10, description="Number of dependents (0-10)")
    smoker:   str   = Field(..., description="'yes' or 'no'")
    region:   str   = Field(..., description="'northeast', 'northwest', 'southeast', or 'southwest'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "sex": "male",
                    "bmi": 28.5,
                    "children": 2,
                    "smoker": "no",
                    "region": "northwest"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response — what the user gets back."""
    success:                 bool
    predicted_annual_charge: float  = None
    segment:                 str    = None
    model_info:              dict   = None
    input_received:          dict   = None
    errors:                  list   = None


# ============================================================
# API ROUTES (Endpoints)
# ============================================================
# WHY ROUTES? (Interview concept: "REST API design")
# Each route is a URL + HTTP method that does one thing:
# - GET  /         → landing page (read, no side effects)
# - GET  /health   → server health check (for load balancers)
# - POST /predict  → make a prediction (sends data, gets result)
#
# GET for reading, POST for sending data. This is REST convention.

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API info."""
    return """
    <html>
    <head><title>Insurance Cost Predictor API</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px;">
        <h1>🏥 Medical Insurance Cost Predictor</h1>
        <p>Predict annual medical insurance charges based on patient demographics.</p>
        <h3>Quick Links</h3>
        <ul>
            <li><a href="/docs">📖 Interactive API Documentation (Swagger UI)</a></li>
            <li><a href="/health">💚 Health Check</a></li>
        </ul>
        <h3>Try a prediction</h3>
        <p>Send a POST request to <code>/predict</code> with JSON body:</p>
        <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 35,
    "sex": "male",
    "bmi": 28.5,
    "children": 2,
    "smoker": "no",
    "region": "northwest"
  }'</pre>
        <h3>Model Info</h3>
        <ul>
            <li>Architecture: 128-64-32 Swish Neural Network</li>
            <li>Test MAE: ~$1,650 (±12% of mean charges)</li>
            <li>Test R²: 0.82</li>
            <li>Features: 21 (11 original + 10 engineered)</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    WHY HEALTH CHECKS? (Interview concept: "Liveness & readiness probes")
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    In production (Docker, Kubernetes, AWS ECS), the infrastructure needs
    to know: "Is this container alive and ready to serve requests?"

    - LIVENESS probe: "Is the process running?" (if no → restart container)
    - READINESS probe: "Is the model loaded?" (if no → don't send traffic)

    This endpoint serves both. The load balancer hits /health every 30 seconds.
    If it returns non-200 three times in a row, the container gets replaced.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: InsurancePredictionRequest):
    """
    Predict annual insurance charges for a patient.

    HOW THIS WORKS:
    1. Pydantic validates the JSON body (age is int, bmi is float, etc.)
    2. We convert the Pydantic model to a dict
    3. Pass to predictor.predict() which handles the full pipeline
    4. Return the result as JSON

    WHY ASYNC? (Interview concept: "Concurrency model")
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'async def' lets FastAPI handle this route concurrently.
    While one request is waiting for model.predict(), the server
    can accept and start processing the next request.

    For CPU-bound ML inference, the speedup is modest (~10-20%).
    For I/O-bound work (database queries, external API calls),
    async gives 5-10x throughput improvement.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded — server starting up")

    start_time = time.time()

    # Convert Pydantic model to dict for our predictor
    input_data = request.model_dump()

    # Run prediction
    result = predictor.predict(input_data)

    # Log the request (for monitoring)
    elapsed_ms = (time.time() - start_time) * 1000
    if result["success"]:
        logger.info(
            f"Prediction: ${result['predicted_annual_charge']:,.0f} | "
            f"Segment: {result['segment']} | "
            f"Time: {elapsed_ms:.1f}ms"
        )
    else:
        logger.warning(f"Validation failed: {result['errors']}")

    return result
