# Project Report — Sentence Complexity API

## 1) Overview

We extended the **Classifier-App** repository into a reusable **REST API** that exposes:

* A **binary sentence complexity classifier** (Simple vs. Complex)
* A **strategy/typology classifier** for complex text (Compression, Explanation, …)
* Optional **explainability** via Captum Integrated Gradients (XAI)
* A **compare** helper to run a single text across multiple configured models

The API runs alongside (or independently of) the existing Streamlit UI, enabling other teams to integrate the classifiers into their pipelines programmatically.

---

## 2) Objectives

* Turn the Streamlit-only prototype into a **production-friendly service**.
* Support **single input**, **batch inputs**, and **text-file uploads**.
* Provide **clear responses** and **robust errors** suitable for automation.
* Offer **containerized deployment** (Docker / Compose) for portability.
* Keep inference **fast** (cached model loading, GPU-aware).

---

## 3) Repository Changes (Key Additions)

```
Classifier-App/
├─ api/
│  ├─ main.py        # FastAPI app, routes, CORS, error handling
│  ├─ schemas.py     # Pydantic request/response models
│  └─ inference.py   # Model loading, caching, inference, XAI
├─ Dockerfile
├─ docker-compose.yml
└─requirements.txt  # unified deps for API + UI
```

implementation details:

* **Model caching:** `@lru_cache` for tokenizer/model objects; auto-uses **GPU** if available.
* **Safety guard:** `MAX_FILE_LINES` (env var) protects file endpoints from huge uploads.
* **XAI endpoint:** optional; returns top token attributions.
* **Fallback model selection fix:** prevents bad `model_name` values from triggering HF 404s.

---

## 4) API Design

### Endpoints

| Method | Path              | Purpose                                    |
| -----: | ----------------- | ------------------------------------------ |
|    GET | `/health`         | Service & model info (readiness)           |
|   POST | `/predict`        | Classify **one** text (JSON)               |
|   POST | `/predict/batch`  | Classify **many** texts (JSON array)       |
|   POST | `/predict/file`   | Classify lines from a `.txt` (multipart)   |
|   POST | `/strategy`       | Typology/strategy for **one** text         |
|   POST | `/strategy/batch` | Typology for **many** texts                |
|   POST | `/strategy/file`  | Typology from `.txt` file                  |
|   POST | `/compare`        | Label from **all** configured models       |
|   POST | `/explain`        | Captum IG attributions (requires `captum`) |

### Request/Response Shape (examples)

* `/predict` (JSON):

  * **Request:** `{"text":"The cat sat on the mat."}`
  * **Response:** `{"model_name":"...","prediction":{"label":"Simple","raw_logits":[...]}}`
* `/predict/file` (multipart):

  * **Request:** `file=@sentences.txt` (one sentence per line)
  * **Response:** `{"model_name":"...","results":[{"text":"...","prediction":{"label":"Complex"}} ...]}`

**Labels:** `0 -> Simple`, `1 -> Complex` (consistent with the UI).

---

## 5) Model Management

* Set the valid HF models in `model.py`
* Ensure `TYPOLOGY_MODEL_NAME` in `api/inference.py` points to the correct HF repo for strategy classification.
* **Private/gated models:** set `HUGGINGFACE_HUB_TOKEN` in env (Compose or shell) so the container/API can authenticate.

---

## 6) Deployment & Usage

### Docker Compose (recommended)

```bash
docker compose up --build -d
docker compose logs -f
docker compose down
```

* Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Create a `.env` beside `docker-compose.yml`:

  ```
  HF_TOKEN=hf_xxx_or_leave_empty
  CORS_ALLOW_ORIGINS=*
  MAX_FILE_LINES=5000
  ```
* Change external port in `docker-compose.yml` under `ports:` (e.g., `"8080:8000"`).

### Local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7) Integration Guidance (Pipelines)

* Prefer **JSON** endpoints for automation (`/predict`, `/predict/batch`).
* Use `/predict/file` when upstream produces newline-delimited `.txt`.
* Implement **timeouts** and **retries** in clients; call `GET /health` as a readiness probe.
* For reproducibility, **pin model IDs** and versions when appropriate.
* Cache HF models (Compose mounts `~/.cache/huggingface`) to reduce cold starts.

**Minimal Python client snippet:**

```python
import requests
API = "http://localhost:8000"
r = requests.post(f"{API}/predict", json={"text":"Hello world"}, timeout=20)
r.raise_for_status()
print(r.json())
```

---

## 8) Requirements

**Python:** 3.10+ (3.11 recommended); works on 3.9 with `Optional[...]` hints
**System:** macOS, Linux, or Windows (WSL2 recommended); ≥4 GB RAM; outbound internet to download models
**Key packages:** `fastapi`, `uvicorn[standard]`, `python-multipart`, `transformers`, `torch`, `safetensors`, `captum` (optional XAI), `streamlit`, `pandas`, `plotly`, `ipython`
**Docker/Compose:** Docker 20+, Compose v2

---

## 9) Error Handling & Observability

* **400**: bad input (empty or non-UTF8 file, unknown content type)
* **413**: too many lines in uploaded file (tunable via `MAX_FILE_LINES`)
* **500**: inference or dependency failure (e.g., HF 404 if model ID is wrong)
* **501**: XAI endpoint called without `captum` installed
* Logs: Uvicorn logs; for Docker use `docker compose logs -f`

---

## 10) Fixes Implemented

* Resolved Python 3.9 type-hint issue (`str | None` → `Optional[str]`).
* Hardened file uploads: UTF-8 with BOM, case-insensitive `.txt`, line limits.
* Clear error messages for missing `python-multipart`, HF 404/401, etc.
* Improved model selection to **fallback safely** when `model_name` is invalid.

---

## 11) Next Steps 

* Add **rate limiting** and request size limits for public deployments.
* Expose **confidence scores** (softmax) as an option.
* Add **/version** and **/metrics** (Prometheus) for ops.
* Package a small **Python SDK** wrapper for easier client integration.

---
