import asyncio
import json
from openai import AsyncOpenAI
from state import PaperCandidate, PaperExtraction
from extractor import EXTRACTION_TOOL
from cache import init_cache, get_cached, save_to_cache
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
client = AsyncOpenAI()

# Limit how many requests are in flight at once — protects you from
# rate limits and from accidentally hammering the API if max_results grows.
MAX_CONCURRENT = 5


async def _extract_paper_async(paper: PaperCandidate, semaphore: asyncio.Semaphore) -> PaperExtraction:
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_extraction"}},
            messages=[{
                "role": "user",
                "content": (
                    f"Title: {paper.title}\n\n"
                    f"Abstract: {paper.abstract}\n\n"
                    "Extract the structured fields from this abstract."
                ),
            }],
        )
        tool_call = response.choices[0].message.tool_calls[0]
        data = json.loads(tool_call.function.arguments)
        extraction = PaperExtraction(arxiv_id=paper.arxiv_id, **data)
        save_to_cache(extraction)  # write-through: cache it the moment it succeeds
        return extraction


async def extract_all_async(papers: list[PaperCandidate]) -> list[PaperExtraction]:
    init_cache()

    results: dict[str, PaperExtraction] = {}
    to_fetch: list[PaperCandidate] = []

    # Step 1: cache lookup (cheap, sequential, no API calls at all)
    for paper in papers:
        cached = get_cached(paper.arxiv_id)
        if cached is not None:
            results[paper.arxiv_id] = cached
        else:
            to_fetch.append(paper)

    print(f"Cache hits: {len(results)} / {len(papers)}. Fetching {len(to_fetch)} from API...")

    # Step 2: concurrent API calls, only for cache misses
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [_extract_paper_async(p, semaphore) for p in to_fetch]

    # return_exceptions=True so one failed paper doesn't kill the whole batch
    fetched = await asyncio.gather(*tasks, return_exceptions=True)

    for paper, outcome in zip(to_fetch, fetched):
        if isinstance(outcome, Exception):
            print(f"Extraction failed for {paper.arxiv_id}: {outcome}")
            continue
        results[paper.arxiv_id] = outcome

    # Preserve original ranked order in the output
    return [results[p.arxiv_id] for p in papers if p.arxiv_id in results]