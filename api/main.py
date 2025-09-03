# api/main.py
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest, PredictResponse, Prediction,
    BatchPredictRequest, BatchPredictResponse, BatchPredictResponseItem,
    StrategyRequest, StrategyResponse, BatchStrategyRequest, BatchStrategyResponse, BatchStrategyResponseItem,
    CompareRequest, CompareResponse, CompareItem,
    ExplainRequest, ExplainResponse, ExplainTokenAttribution
)
from .inference import (
    MODEL_NAMES, DEFAULT_MODEL,
    classify_single, classify_batch,
    typology_single, typology_batch,
    explain_top_attributions
)

app = FastAPI(
    title="Sentence Complexity API",
    description="Binary complexity classifier + typology (strategy) classifier + XAI",
    version="1.0.0",
)

# ---- CORS (adjust for your frontend domains) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple guard for very large uploads (lines)
MAX_LINES = int(os.getenv("MAX_FILE_LINES", "5000"))


@app.get("/health")
def health():
    """
    Health + basic model info.
    """
    return {"status": "ok", "models": MODEL_NAMES, "default_model": DEFAULT_MODEL}


# -----------------------------
# Binary classifier
# -----------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classify a single text as Simple/Complex.
    """
    label, logits = classify_single(req.text, model_name=req.model_name)
    return PredictResponse(
        model_name=req.model_name or DEFAULT_MODEL,
        prediction=Prediction(label=label, raw_logits=logits),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    """
    Classify a batch of texts as Simple/Complex.
    """
    labels, logits = classify_batch(req.texts, model_name=req.model_name)
    items = [
        BatchPredictResponseItem(text=t, prediction=Prediction(label=l, raw_logits=log))
        for t, l, log in zip(req.texts, labels, logits)
    ]
    return BatchPredictResponse(model_name=req.model_name or DEFAULT_MODEL, results=items)


# -----------------------------
# Strategy / Typology classifier
# -----------------------------

@app.post("/strategy", response_model=StrategyResponse)
def strategy(req: StrategyRequest):
    """
    Predict the complexity strategy/typology for a single text.
    """
    idx, label = typology_single(req.text)
    return StrategyResponse(index=idx, label=label)


@app.post("/strategy/batch", response_model=BatchStrategyResponse)
def strategy_batch(req: BatchStrategyRequest):
    """
    Predict the complexity strategy/typology for a batch of texts.
    """
    pairs = typology_batch(req.texts)
    items = [BatchStrategyResponseItem(text=t, index=i, label=lab) for t, (i, lab) in zip(req.texts, pairs)]
    return BatchStrategyResponse(results=items)


# -----------------------------
# File endpoints (TXT; one sentence per line)
# -----------------------------

@app.post("/predict/file", response_model=BatchPredictResponse)
async def predict_file(file: UploadFile = File(...), model_name: Optional[str] = None):
    """
    Upload a .txt file; each non-empty line is classified as Simple/Complex.
    Optional ?model_name=... can select a specific HF model.
    """
    # Require python-multipart for UploadFile parsing
    try:
        import multipart  # noqa: F401
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency 'python-multipart'. Install it to enable file uploads."
        )

    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw = await file.read()
    try:
        # utf-8 with BOM support
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    sentences = [s.strip() for s in content.splitlines() if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(sentences) > MAX_LINES:
        raise HTTPException(status_code=413, detail=f"Too many lines (>{MAX_LINES}).")

    labels, logits = classify_batch(sentences, model_name=model_name)
    items = [
        BatchPredictResponseItem(text=t, prediction=Prediction(label=l, raw_logits=log))
        for t, l, log in zip(sentences, labels, logits)
    ]
    return BatchPredictResponse(model_name=model_name or DEFAULT_MODEL, results=items)


@app.post("/strategy/file", response_model=BatchStrategyResponse)
async def strategy_file(file: UploadFile = File(...)):
    """
    Upload a .txt file; each non-empty line is classified into a typology/strategy.
    """
    # Require python-multipart for UploadFile parsing
    try:
        import multipart  # noqa: F401
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency 'python-multipart'. Install it to enable file uploads."
        )

    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw = await file.read()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    sentences = [s.strip() for s in content.splitlines() if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(sentences) > MAX_LINES:
        raise HTTPException(status_code=413, detail=f"Too many lines (>{MAX_LINES}).")

    pairs = typology_batch(sentences)
    items = [BatchStrategyResponseItem(text=t, index=i, label=lab) for t, (i, lab) in zip(sentences, pairs)]
    return BatchStrategyResponse(results=items)


# -----------------------------
# Compare across all models
# -----------------------------

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """
    Run the same text through all configured models and return the labels.
    """
    if not MODEL_NAMES:
        raise HTTPException(status_code=500, detail="No models configured.")
    results = []
    simple_count = 0
    complex_count = 0
    for m in MODEL_NAMES:
        label, _ = classify_single(req.text, model_name=m)
        results.append(CompareItem(model_name=m, label=label))
        if label == "Simple":
            simple_count += 1
        else:
            complex_count += 1
    return CompareResponse(results=results, simple_count=simple_count, complex_count=complex_count)


# -----------------------------
# XAI (Captum Integrated Gradients)
# -----------------------------

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """
    Return top token attributions via Captum IG for the predicted class.
    """
    try:
        pred_class, top = explain_top_attributions(req.text, model_name=req.model_name, top_k=req.top_k)
    except RuntimeError as e:
        # Typically thrown when Captum is not installed
        raise HTTPException(status_code=501, detail=str(e))
    label = "Simple" if pred_class == 0 else "Complex"
    return ExplainResponse(
        model_name=req.model_name or DEFAULT_MODEL,
        label=label,
        top_attributions=[ExplainTokenAttribution(token=t, score=s) for t, s in top],
    )
