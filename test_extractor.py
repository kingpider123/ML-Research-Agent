# test_extractor.py
import asyncio
from retriever import retrieve_candidates
from ranker import rank_candidates
from extractor_async import extract_all_async
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
async def main():
    query = "diffusion policy learning"

    print("Step 1: Retrieving...")
    candidates = retrieve_candidates(query, max_results=50)
    print(f"Got {len(candidates)} candidates.\n")

    print("Step 2: Reranking...")
    ranked = rank_candidates(query, candidates, top_k=5)  # keep it small for the test
    print(f"Top 5 after reranking:")
    for r in ranked:
        print(f"  [{r.relevance_score:.2f}] {r.title}")

    print("\nStep 3: Extracting (with cache + concurrency)...")
    extractions = await extract_all_async(ranked)

    print(f"\nGot {len(extractions)} extractions. Showing first 2:\n")
    for e in extractions[:2]:
        print(f"Paper: {e.arxiv_id}")
        print(f"  Problem:      {e.problem}")
        print(f"  Method:       {e.method}")
        print(f"  Results:      {e.results}")
        print(f"  Contribution: {e.contribution}")
        print(f"  Confidence:   {e.confidence}")
        print()

    print("--- Running again to verify cache hits ---")
    extractions2 = await extract_all_async(ranked)
    print(f"Got {len(extractions2)} extractions from cache (no API calls).")

asyncio.run(main())