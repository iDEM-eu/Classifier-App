import os
import torch
from functools import lru_cache
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Try to import your MODEL_NAMES from model.py if present.
# Fallback to a safe default if not available.
DEFAULT_MODELS = [
    # Put your repo’s list here if model.MODEL_NAMES isn’t found.
    # Example placeholders (replace with the real ones you use in Streamlit):
    "hannah-khallaf/Sentence-Complexity-Classifier",
]
TYPOLOGY_MODEL_NAME = "hannah-khallaf/Typlogy-Classifier"

try:
    from model import MODEL_NAMES as _MODEL_NAMES  # your repo’s list
    MODEL_NAMES: List[str] = list(_MODEL_NAMES)
except Exception:
    MODEL_NAMES = DEFAULT_MODELS

DEFAULT_MODEL = MODEL_NAMES[0] if MODEL_NAMES else TYPOLOGY_MODEL_NAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_NO_GRAD = torch.no_grad

COMPLEXITY_LABELS_MAP = {0: "Simple", 1: "Complex"}
TYPOLOGY_TYPES = {
    0: 'Compression',
    1: 'Explanation',
    2: 'Modulation',
    3: 'Omission',
    4: 'Synonymy',
    5: 'Syntactic Changes',
    6: 'Transcript',
    7: 'Transposition'
}

@lru_cache(maxsize=16)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

@lru_cache(maxsize=16)
def get_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model

def _select_model_name(requested: str | None) -> str:
    if requested and requested in MODEL_NAMES:
        return requested
    return requested or DEFAULT_MODEL

def classify_single(text: str, model_name: str | None = None) -> Tuple[str, List[float]]:
    mname = _select_model_name(model_name)
    tok = get_tokenizer(mname)
    mdl = get_model(mname)

    inputs = tok(text, return_tensors="pt", truncation=True, padding=False).to(DEVICE)
    with torch.no_grad():
        outputs = mdl(**inputs)
        logits = outputs.logits[0].detach().cpu().tolist()
        pred_idx = int(outputs.logits.argmax(dim=-1).item())
    label = COMPLEXITY_LABELS_MAP.get(pred_idx, "Complex" if pred_idx == 1 else "Simple")
    return label, logits

def classify_batch(texts: List[str], model_name: str | None = None) -> Tuple[List[str], List[List[float]]]:
    mname = _select_model_name(model_name)
    tok = get_tokenizer(mname)
    mdl = get_model(mname)

    inputs = tok(texts, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = mdl(**inputs)
        logits = outputs.logits.detach().cpu()
        pred_idx = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
    labels = [COMPLEXITY_LABELS_MAP.get(i, "Complex" if i == 1 else "Simple") for i in pred_idx]
    return labels, logits.tolist()

def typology_single(text: str) -> Tuple[int, str]:
    tok = get_tokenizer(TYPOLOGY_MODEL_NAME)
    mdl = get_model(TYPOLOGY_MODEL_NAME)

    inputs = tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = mdl(**inputs)
        idx = int(outputs.logits.argmax(dim=-1).item())
    return idx, TYPOLOGY_TYPES.get(idx, f"Unknown({idx})")

def typology_batch(texts: List[str]) -> List[Tuple[int, str]]:
    tok = get_tokenizer(TYPOLOGY_MODEL_NAME)
    mdl = get_model(TYPOLOGY_MODEL_NAME)

    inputs = tok(texts, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = mdl(**inputs)
        idxs = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
    return [(i, TYPOLOGY_TYPES.get(i, f"Unknown({i})")) for i in idxs]

# ---- Optional: Captum Integrated Gradients ----
def explain_top_attributions(text: str, model_name: str | None = None, top_k: int = 10):
    try:
        from captum.attr import IntegratedGradients
    except Exception:
        raise RuntimeError("Captum is not installed. Add 'captum' to requirements to use /explain.")

    mname = _select_model_name(model_name)
    tok = get_tokenizer(mname)
    mdl = get_model(mname)

    # Tokenize and get embeddings
    encoded = tok(text, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # Forward function for Captum (logit of predicted class)
    mdl.zero_grad()
    with torch.no_grad():
        logits = mdl(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    def forward_func(input_ids_embeds):
        outputs = mdl(inputs_embeds=input_ids_embeds, attention_mask=attention_mask)
        return outputs.logits[:, pred_class]

    # Get input embeddings
    with torch.no_grad():
        input_embeds = mdl.get_input_embeddings()(input_ids)

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(inputs=input_embeds, baselines=input_embeds * 0, return_convergence_delta=True)

    # Aggregate per-token attribution magnitude
    token_attrs = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())

    # Pair, sort by absolute score, take top_k (skip special tokens)
    items = []
    for t, s in zip(tokens, token_attrs):
        if t.startswith("▁") or t.startswith("##"):
            clean_t = t.replace("▁", "").replace("##", "")
        else:
            clean_t = t
        if clean_t and clean_t not in tok.all_special_tokens:
            items.append((clean_t, float(s)))
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    top = items[:max(1, top_k)]
    return pred_class, [(tok_, score) for tok_, score in top]
