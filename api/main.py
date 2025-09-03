import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException
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

# ---- CORS (adjust for your frontend domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "models": MODEL_NAMES, "default_model": DEFAULT_MODEL}

# ---- Binary classifier ----

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label, logits = classify_single(req.text, model_name=req.model_name)
    return PredictResponse(model_name=req.model_name or DEFAULT_MODEL,
                           prediction=Prediction(label=label, raw_logits=logits))

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    labels, logits = classify_batch(req.texts, model_name=req.model_name)
    items = [
        BatchPredictResponseItem(text=t, prediction=Prediction(label=l, raw_logits=log))
        for t, l, log in zip(req.texts, labels, logits)
    ]
    return BatchPredictResponse(model_name=req.model_name or DEFAULT_MODEL, results=items)

# ---- Strategy / Typology classifier ----

@app.post("/strategy", response_model=StrategyResponse)
def strategy(req: StrategyRequest):
    idx, label = typology_single(req.text)
    return StrategyResponse(index=idx, label=label)

@app.post("/strategy/batch", response_model=BatchStrategyResponse)
def strategy_batch(req: BatchStrategyRequest):
    pairs = typology_batch(req.texts)
    items = [BatchStrategyResponseItem(text=t, index=i, label=lab) for t, (i, lab) in zip(req.texts, pairs)]
    return BatchStrategyResponse(results=items)




@app.post("/predict/file", response_model=BatchPredictResponse)
async def predict_file(file: UploadFile = File(...), model_name: str = None):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    content = (await file.read()).decode("utf-8")
    sentences = [s.strip() for s in content.split("\n") if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="File is empty.")

    labels, logits = classify_batch(sentences, model_name=model_name)
    items = [
        BatchPredictResponseItem(text=t, prediction=Prediction(label=l, raw_logits=log))
        for t, l, log in zip(sentences, labels, logits)
    ]
    return BatchPredictResponse(model_name=model_name or DEFAULT_MODEL, results=items)


@app.post("/strategy/file", response_model=BatchStrategyResponse)
async def strategy_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    content = (await file.read()).decode("utf-8")
    sentences = [s.strip() for s in content.split("\n") if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="File is empty.")

    pairs = typology_batch(sentences)
    items = [BatchStrategyResponseItem(text=t, index=i, label=lab) for t, (i, lab) in zip(sentences, pairs)]
    return BatchStrategyResponse(results=items)


# ---- Compare across all models ----

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
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

# ---- XAI (Captum Integrated Gradients) ----

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    pred_class, top = explain_top_attributions(req.text, model_name=req.model_name, top_k=req.top_k)
    label = "Simple" if pred_class == 0 else "Complex"
    return ExplainResponse(
        model_name=req.model_name or DEFAULT_MODEL,
        label=label,
        top_attributions=[ExplainTokenAttribution(token=t, score=s) for t, s in top],
    )
