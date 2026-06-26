# ranker.py
from sentence_transformers import CrossEncoder
from state import PaperCandidate

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rank_candidates(query: str, candidates: list[PaperCandidate], top_k: int = 15) -> list[PaperCandidate]:
    """
    Reranks candidates by feeding (query, abstract) pairs through the
    cross-encoder. We use the abstract, not the title, because the
    abstract carries the actual claim/method signal the cross-encoder
    needs to judge relevance — titles are often too terse or stylized.
    """
    pairs = [(query, c.abstract) for c in candidates]
    scores = _model.predict(pairs)  # one forward pass per pair, batched internally

    for candidate, score in zip(candidates, scores):
        candidate.relevance_score = float(score)

    ranked = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
    return ranked[:top_k]
