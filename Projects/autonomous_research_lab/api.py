"""
FastAPI REST API for the Autonomous AI Research Lab.

Run: uvicorn api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs (auto-generated OpenAPI/Swagger UI)
"""

import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.report_publisher import ReportPublisherAgent
from graph import run_research
from models import ResearchReport
from rag.document_store import DocumentStore

app = FastAPI(
    title="Autonomous AI Research Lab API",
    description=(
        "Submit a research question; a dynamically-assembled team of specialist "
        "agents researches it end-to-end (hypothesis -> parallel evidence "
        "gathering -> critique -> synthesis -> peer review) and returns a "
        "structured, cited research report."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo setting — restrict in real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = DocumentStore()


class ResearchRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Will fault-tolerant quantum computing be capable of "
                             "breaking RSA encryption before 2035?"
            }
        }


class DomainsResponse(BaseModel):
    domains: List[str]


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/domains", response_model=DomainsResponse)
def list_domains():
    """Lists the research domains the seeded document store currently covers."""
    return {"domains": _store.available_domains()}


@app.post("/research", response_model=ResearchReport)
def research(req: ResearchRequest):
    """
    Runs the full autonomous research pipeline synchronously and returns the
    final structured report. For a long research question this can take
    30s-2min depending on how many specialists get assembled.
    """
    if not req.question or len(req.question.strip()) < 10:
        raise HTTPException(status_code=400, detail="Question must be at least 10 characters.")

    try:
        state = run_research(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if state["report"] is None:
        raise HTTPException(status_code=502, detail="Pipeline did not produce a report.")

    return state["report"]


@app.post("/research/markdown")
def research_markdown(req: ResearchRequest):
    """Same as /research but returns a rendered Markdown report as plain text."""
    if not req.question or len(req.question.strip()) < 10:
        raise HTTPException(status_code=400, detail="Question must be at least 10 characters.")

    state = run_research(req.question)
    if state["report"] is None:
        raise HTTPException(status_code=502, detail="Pipeline did not produce a report.")

    markdown = ReportPublisherAgent.to_markdown(state["report"])
    return {"markdown": markdown}
