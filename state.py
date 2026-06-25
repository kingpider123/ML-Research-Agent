from pydantic import BaseModel
from typing import Optional

class PaperCandidate(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    relevance_score: Optional[float] = None

class PaperExtraction(BaseModel):
    arxiv_id: str
    problem: str
    method: str
    results: str
    contribution: str
    confidence: float  # self-reported by the LLM, 0-1

class ResearchLandscape(BaseModel):
    query: str
    clusters: list[dict]       # {name, paper_ids, summary}
    relationships: list[dict]  # {from_id, to_id, relationship_type}
    tensions: list[dict]       # {description, paper_ids}
    open_problems: list[str]

class PipelineState(BaseModel):
    query: str
    candidates: list[PaperCandidate] = []
    ranked: list[PaperCandidate] = []
    extractions: list[PaperExtraction] = []
    landscape: Optional[ResearchLandscape] = None