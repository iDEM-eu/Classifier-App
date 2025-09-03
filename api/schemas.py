from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# ---- Requests ----

class PredictRequest(BaseModel):
    text: str = Field(..., description="Single input text to classify.")
    model_name: Optional[str] = Field(None, description="HF model id. If omitted, uses default_model.")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Batch of texts to classify.")
    model_name: Optional[str] = None

class StrategyRequest(BaseModel):
    text: str
    model_name: Optional[str] = None  # reserved (if you later host variations)

class BatchStrategyRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = None

class CompareRequest(BaseModel):
    text: str

class ExplainRequest(BaseModel):
    text: str
    model_name: Optional[str] = None
    top_k: int = 10

# ---- Responses ----

PredictionLabel = Literal["Simple", "Complex"]

class Prediction(BaseModel):
    label: PredictionLabel
    raw_logits: Optional[List[float]] = None

class PredictResponse(BaseModel):
    model_name: str
    prediction: Prediction

class BatchPredictResponseItem(BaseModel):
    text: str
    prediction: Prediction

class BatchPredictResponse(BaseModel):
    model_name: str
    results: List[BatchPredictResponseItem]

class StrategyResponse(BaseModel):
    label: str  # one of the typology classes (e.g., "Compression", ...)
    index: int

class BatchStrategyResponseItem(BaseModel):
    text: str
    label: str
    index: int

class BatchStrategyResponse(BaseModel):
    results: List[BatchStrategyResponseItem]

class CompareItem(BaseModel):
    model_name: str
    label: PredictionLabel

class CompareResponse(BaseModel):
    results: List[CompareItem]
    simple_count: int
    complex_count: int

class ExplainTokenAttribution(BaseModel):
    token: str
    score: float

class ExplainResponse(BaseModel):
    model_name: str
    label: PredictionLabel
    top_attributions: List[ExplainTokenAttribution]
