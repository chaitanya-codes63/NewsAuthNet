# src/api.py
import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.predict import Predictor, installed_models
from src.explain import explain_json

app = FastAPI(title="NewsAuthNet TensorFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS: dict[str, Predictor] = {}
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

@app.on_event("startup")
def startup_load():
    for name in installed_models():
        MODELS[name] = Predictor(name)
    print(f"Loaded .pkl models into RAM: {list(MODELS.keys())}")

class AnalyzeRequest(BaseModel):
    text: str
    model: str = "lstm"

@app.get("/api/models")
def get_available_models():
    metrics = {}
    if (REPORTS_DIR / "metrics.json").exists():
        with open(REPORTS_DIR / "metrics.json") as f:
            metrics = json.load(f)
    histories = {}
    for m in MODELS:
        hp = MODELS_DIR / m / "history.json"
        if hp.exists():
            with open(hp) as f:
                histories[m] = json.load(f)
    return {
        "installed": list(MODELS.keys()),
        "metrics": metrics,
        "histories": histories,
    }

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.model not in MODELS:
        return {"error": f"Model '{req.model}' is not trained yet. Available: {list(MODELS.keys())}"}
    t0 = time.perf_counter()
    try:
        pred = MODELS[req.model].predict(req.text)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "label": pred.label_name,
            "score_real": round(pred.score_real, 4),
            "model": pred.model,
            "latency_ms": ms,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/explain")
def explain_endpoint(req: AnalyzeRequest):
    if req.model not in MODELS:
        return {"error": f"Model '{req.model}' is not loaded. Available: {list(MODELS.keys())}"}
    t0 = time.perf_counter()
    try:
        res = explain_json(req.text, model=req.model, top_k=50, predictor=MODELS[req.model])
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/analyze_all")
def analyze_all(req: AnalyzeRequest):
    results = []
    for name, engine in MODELS.items():
        t0 = time.perf_counter()
        try:
            p = engine.predict(req.text)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            results.append({
                "model": name.upper(),
                "label": p.label_name,
                "score_real": round(p.score_real, 4),
                "confidence": round((p.score_real if p.label == 1 else 1 - p.score_real) * 100, 1),
                "latency_ms": ms,
            })
        except Exception as e:
            results.append({"model": name.upper(), "error": str(e)})
    return {"results": results}