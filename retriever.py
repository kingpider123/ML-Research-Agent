import arxiv
from state import PaperCandidate  # the Pydantic model from above

def retrieve_candidates(query: str, max_results: int = 50) -> list[PaperCandidate]:
    """
    Pulls candidate papers from arXiv for a given topic.
    We over-fetch (50) because the reranker will narrow it down —
    arXiv's own relevance sort is keyword-based and mediocre for
    conceptual queries, so we treat this as a recall step, not precision.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    candidates = []
    for result in client.results(search):
        candidates.append(PaperCandidate(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.strip(),
            abstract=result.summary.strip().replace("\n", " "),
            authors=[a.name for a in result.authors],
            published=result.published.strftime("%Y-%m-%d"),
        ))
    return candidates

print(retrieve_candidates("RAG", max_results=5))