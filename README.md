# Streamlit Multiligual Sentence Complexity Classification and XAI

This is a **Streamlit-based web app** for **Sentence  classification** using multiple denoised models and **explainability with Captum**.
## 🖥️ Live App
🔗 **Access the app here:** [Classifier App](https://classifier-app.streamlit.app/)

## 🚀 Features
- Select from multiple Hugging Face models
- Classify text as **Simple or Complex**
- Compare multiple models' predictions
- Explainable AI (XAI) with token attributions
- Typology classifer for complex sentences.

## 🔧 Installation
```bash
git clone https://github.com/nouran-khallaf/Classifier-App.git
cd Classifier-App
pip install -r requirements.txt
```

## ▶️ Run the App
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.
---
# Sentence Complexity API

FastAPI service for:
- **Binary complexity classifier** (Simple vs. Complex)
- **Strategy / Typology classifier** (Compression, Explanation, …)
- **Model comparison** across all configured HF models
- Optional **XAI** via Captum Integrated Gradients

This API is designed to sit alongside your existing Streamlit app.

---

## Run Locally

From the project root:

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs (Swagger UI):  
http://localhost:8000/docs

---


## Example Requests (cURL)

### Classifer

```bash
curl -s http://localhost:8000/Classifer
```

### Binary Classification — Text (JSON)

```bash
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"text":"The cat sat on the mat."}'
```

### Binary Classification — Batch (JSON)
export MAX_FILE_LINES=20000
```bash
curl -s -X POST http://localhost:8000/predict/batch   -H "Content-Type: application/json"   -d '{"texts":["Text A","Text B"],"model_name":"hannah-khallaf/Sentence-Complexity-Classifier"}'
```

### Binary Classification — From File (.txt)

Each line is treated as a separate input:

```bash
curl -s -X POST "http://localhost:8000/predict/file"   -F "file=@sentences.txt"
```

Example `sentences.txt`:
```
The cat sat on the mat.
However, notwithstanding the aforementioned considerations, 
```

### Strategy / Typology — Text (JSON)

```bash
curl -s -X POST http://localhost:8000/strategy   -H "Content-Type: application/json"   -d '{"text":"However, notwithstanding the aforementioned..."}'
```

### Strategy / Typology — From File (.txt)

```bash
curl -s -X POST http://localhost:8000/strategy/file   -F "file=@sentences.txt"
```

### Compare Across All Models (Single Text)

```bash
curl -s -X POST http://localhost:8000/compare   -H "Content-Type: application/json"   -d '{"text":"Sample sentence to compare"}'
```

### Explain (Captum Integrated Gradients)

> Requires `captum` installed.

```bash
curl -s -X POST http://localhost:8000/explain   -H "Content-Type: application/json"   -d '{"text":"This is a fairly intricate statement to parse.", "top_k": 8}'
```

---

## 🐳 Docker

```bash
docker build -t complexity-api .
docker run --rm -p 8000:8000 complexity-api
```
## API Quickstart (Docker Compose)

```bash
# from the repo root
docker compose up --build -d

# view logs
docker compose logs -f

# stop
docker compose down
```

* **Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Environment variables

Create a `.env` file next to `docker-compose.yml` (Compose reads it automatically):

```env
# only needed if your HF models are private/gated
HF_TOKEN=hf_xxx_or_leave_empty

# allow dev frontends; tighten for prod
CORS_ALLOW_ORIGINS=*

# safety guard for /predict/file
MAX_FILE_LINES=5000
```

> The external port defaults to **8000**. Change it in `docker-compose.yml` under `ports`, e.g.:
>
> ```yaml
> ports:
>   - "8080:8000"
> ```

---

## Notes & Limits

- GPU is used automatically if available.
- File endpoints only accept `.txt` and UTF‑8 encoding.
- You can limit max file lines via env var `MAX_FILE_LINES` (default 5000).
- Labels follow streamlit app: `0 -> Simple`, `1 -> Complex`.




