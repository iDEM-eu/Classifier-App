FROM python:3.11-slim

# System deps 
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-pull model weights at build (optional)
# RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
# M.from_pretrained('hannah-khallaf/Sentence-Complexity-Classifier'); \
# M.from_pretrained('hannah-khallaf/Typlogy-Classifier')"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
