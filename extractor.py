# extractor.py
import json
from openai import OpenAI
from state import PaperCandidate, PaperExtraction
from retriever import retrieve_candidates
from ranker import rank_candidates
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
client = OpenAI()

EXTRACTION_TOOL = {
    "name": "record_extraction",
    "description": "Record structured extraction of a research paper's core content.",
    "parameters": {  # OpenAI uses "parameters" instead of "input_schema"
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": "What problem or gap in prior work does this paper address?"
            },
            "method": {
                "type": "string",
                "description": "What is the core technical approach/method?"
            },
            "results": {
                "type": "string",
                "description": "What were the key quantitative or qualitative results?"
            },
            "contribution": {
                "type": "string",
                "description": "What is the single most important novel contribution?"
            },
            "confidence": {
                "type": "number",
                "description": "Your confidence (0-1) that this extraction accurately reflects the abstract, given that you only have the abstract and not the full paper."
            },
        },
        "required": ["problem", "method", "results", "contribution", "confidence"],
    },
}


def extract_paper(paper: PaperCandidate) -> PaperExtraction:
    # OpenAI requires wrapping the function definition inside a 'type': 'function' object
    tools = [{"type": "function", "function": EXTRACTION_TOOL}]
    
    # Force the model to use this specific tool
    tool_choice = {"type": "function", "function": {"name": "record_extraction"}}

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o-mini" depending on your needs
        max_tokens=1000,
        tools=tools,
        tool_choice=tool_choice,
        messages=[{
            "role": "user",
            "content": (
                f"Title: {paper.title}\n\n"
                f"Abstract: {paper.abstract}\n\n"
                "Extract the structured fields from this abstract."
            ),
        }],
    )

    # Extract the tool call arguments from the OpenAI response object
    tool_call = response.choices[0].message.tool_calls[0]
    
    # OpenAI returns arguments as a serialized JSON string, so we need to parse it
    data = json.loads(tool_call.function.arguments)

    return PaperExtraction(
        arxiv_id=paper.arxiv_id,
        **data,
    )


def extract_all(papers: list[PaperCandidate]) -> list[PaperExtraction]:
    extractions = []
    for paper in papers:
        try:
            extractions.append(extract_paper(paper))
        except Exception as e:
            print(f"Extraction failed for {paper.arxiv_id}: {e}")
            continue
    return extractions

print("Extracting papers...")
print(extract_all(rank_candidates("RAG", retrieve_candidates("RAG", max_results=50), top_k=15)))