import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# List of available models
MODEL_NAMES = [
    "hannah-khallaf/English-LS-Sent-model",
    "hannah-khallaf/English-Baseline-Sent-model",
    "hannah-khallaf/English-GMM-B-Sent-model",
    "hannah-khallaf/English-GMM-SB-Sent-model",
    "hannah-khallaf/French-Baseline-Token-model",
    "hannah-khallaf/English-GMM-SB-Token-model",
    "hannah-khallaf/English-Baseline-token-model",
]

def load_model(model_name):
    """Loads and caches the model & tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def predict(text, model_name):
    """Predicts the complexity of a given text using a specified model."""
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()
    
    return "Simple" if predicted_class == 0 else "Complex"
