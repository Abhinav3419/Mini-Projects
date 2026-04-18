# Dockerfile
#
# WHAT IS THIS FILE? (Interview concept: "Containerization")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A Dockerfile is a RECIPE for building a Docker image.
# Think of it like a cooking recipe:
#   - Start with a base (Python 3.11)          → "Start with flour"
#   - Copy ingredients (your code + model)      → "Add eggs and sugar"
#   - Run setup steps (pip install)             → "Mix together"
#   - Define how to run (uvicorn command)       → "Bake at 350°F"
#
# The result is a Docker IMAGE — a frozen, portable snapshot of your
# entire application + all its dependencies. Anyone, anywhere, can
# run this image and get the EXACT same behavior.
#
# WHY DOCKER? (Interview concept: "Environment parity")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Problem: "It works on my machine" (Windows) but breaks on AWS (Linux).
# Solution: Docker packages your app WITH its OS, Python version, and
#           all libraries into one portable unit.
#
# Your laptop (Windows 11) → Docker container (Linux) → AWS (Linux)
# Same container, same behavior, everywhere.
#
# INTERVIEW QUESTION: "What's the difference between an image and a container?"
# ANSWER: "An image is a frozen snapshot (like a class). A container is a
#          running instance of that image (like an object). You can run
#          multiple containers from the same image."

# ──────────────────────────────────────────────────────
# STAGE 1: Base image
# ──────────────────────────────────────────────────────
# WHY python:3.11-slim?
# - "3.11" matches the Python version we develop with
# - "slim" is a minimal Linux image (~150MB vs ~900MB for full)
# - Smaller image = faster deploys, less storage cost on AWS
FROM python:3.11-slim

# ──────────────────────────────────────────────────────
# STAGE 2: Set working directory
# ──────────────────────────────────────────────────────
# All subsequent commands run from this directory inside the container.
# Think of it as 'cd /app' that persists for the rest of the Dockerfile.
WORKDIR /app

# ──────────────────────────────────────────────────────
# STAGE 3: Install Python dependencies FIRST
# ──────────────────────────────────────────────────────
# WHY COPY requirements.txt BEFORE the rest of the code?
# (Interview concept: "Docker layer caching")
#
# Docker caches each step (layer). If a layer hasn't changed,
# Docker reuses the cached version instead of rebuilding.
#
# If we copied ALL files first, then ran pip install, Docker would
# reinstall ALL packages every time ANY file changes (even a typo fix).
#
# By copying requirements.txt first and installing, Docker caches
# the expensive pip install step. Only code changes trigger a rebuild
# of the later layers — which is fast (just copying files).
#
# This saves 3-5 minutes per rebuild during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────
# STAGE 4: Copy application code and model
# ──────────────────────────────────────────────────────
COPY app/ ./app/
COPY model/ ./model/

# ──────────────────────────────────────────────────────
# STAGE 5: Expose port
# ──────────────────────────────────────────────────────
# EXPOSE is documentation — it tells Docker (and humans) which port
# the app listens on. It does NOT actually open the port.
# The actual port mapping happens at 'docker run -p 8000:8000'.
EXPOSE 8000

# ──────────────────────────────────────────────────────
# STAGE 6: Health check
# ──────────────────────────────────────────────────────
# Docker (and AWS ECS) will periodically hit this endpoint.
# If it fails 3 times in a row, the container is restarted.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ──────────────────────────────────────────────────────
# STAGE 7: Run the application
# ──────────────────────────────────────────────────────
# CMD is what runs when the container starts.
# uvicorn is the ASGI server that runs our FastAPI app.
#
# --host 0.0.0.0 → listen on ALL network interfaces (required in Docker,
#                   otherwise the container can't receive external traffic)
# --port 8000    → the port number
# --workers 1    → number of worker processes (1 is fine for ML serving
#                   because TensorFlow already uses multiple CPU cores internally)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
