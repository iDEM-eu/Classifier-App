# ---- Sentence Complexity API Dockerfile ----
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# (Optional) Pre-pull model weights to reduce cold starts
# RUN python - <<'PY'
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# for name in [
#     "hannah-khallaf/Sentence-Complexity-Classifier",
#     "hannah-khallaf/Typlogy-Classifier",
# ]:
#     AutoTokenizer.from_pretrained(name)
#     AutoModelForSequenceClassification.from_pretrained(name)
# PY

EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
