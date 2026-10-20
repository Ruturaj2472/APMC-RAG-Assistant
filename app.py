"""
FastAPI service exposing the APMC Market Intelligence RAG pipeline.

Run locally (requires internet to install deps):
    pip install fastapi uvicorn
    uvicorn app:app --reload
    -> POST http://localhost:8000/ask   {"query": "..."}

Not runnable inside this sandbox (no internet to install fastapi/uvicorn) --
see demo_cli.py for a dependency-light way to exercise the same pipeline.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import MarketIntelligencePipeline

app = FastAPI(title="APMC Market Intelligence Assistant")
pipeline = MarketIntelligencePipeline()


class Query(BaseModel):
    query: str


@app.post("/ask")
def ask(payload: Query):
    return pipeline.ask(payload.query)


@app.get("/health")
def health():
    return {"status": "ok", "num_receipts": len(pipeline.receipts)}
