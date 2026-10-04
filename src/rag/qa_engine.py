import os
import re
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.vector_store import query_vector_store

load_dotenv()
_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
GPT_MODEL = "gpt-4o-mini"

CITATION_RE = re.compile(
    r"\[?\s*([A-Za-z0-9_\-'\. ]+?)\s*[,\-–—]\s*"
    r"(\d{1,2}:\d{2}:\d{2}[,\.]?\d{0,3})\s*[–\-—to]+\s*"
    r"(\d{1,2}:\d{2}:\d{2}[,\.]?\d{0,3})\s*\]?"
)

REFUSAL = "I don't have that information in the movie database."


def retrieve(query: str, top_k: int = 5, persist_directory: Optional[str] = None) -> Dict:
    kwargs = {"n_results": top_k}
    if persist_directory:
        kwargs["persist_directory"] = persist_directory
    return query_vector_store(query, **kwargs)


def build_context(results: Dict) -> List[Dict]:
    """
    Turn raw Chroma results into structured citation blocks the LLM can cite.
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append({
            "dialogue": doc,
            "movie_title": meta.get("movie_title", ""),
            "start_time": meta.get("start_time", ""),
            "end_time": meta.get("end_time", ""),
            "speakers": meta.get("speakers", ""),
        })
    return blocks


def format_context(blocks: List[Dict]) -> str:
    if not blocks:
        return "No relevant dialogue found in the database."
    lines = []
    for b in blocks:
        lines.append(
            f"- Dialogue: {b['dialogue']}\n"
            f"  Citation: [{b['movie_title']}, {b['start_time']} - {b['end_time']}]"
        )
    return "\n".join(lines)


def generate_grounded_answer(
    query: str,
    context_str: str,
    chat_history: Optional[List[Dict]] = None,
) -> str:
    """
    LLM call that ONLY uses the provided context and must include citations.
    """
    if chat_history is None:
        chat_history = []

    system = (
        "You are a strictly grounded Movie Intelligence assistant. "
        "You may ONLY use the dialogue excerpts in the provided Context. "
        "You MUST NOT use any external, general, or pretrained knowledge about movies, "
        "actors, directors, or plots — even if you know them. "
        "If the Context is empty or does not contain the answer, respond with EXACTLY: "
        "'I don't have that information in the movie database.' "
        "Every factual sentence you do produce MUST end with a citation of the form "
        "'[Movie Title, HH:MM:SS - HH:MM:SS]'. "
        "Do not paraphrase from memory. Do not summarize plots. Do not name actors or "
        "directors unless they appear verbatim in the Context. "
        "If you cannot cite, do not answer."
    )

    messages = (
        [{"role": "system", "content": system}]
        + chat_history
        + [{
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}",
        }]
    )

    resp = _client.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.0
    )
    return resp.choices[0].message.content.strip()


def verify_citations(answer: str, blocks: List[Dict]) -> Dict:
    """
    Confirm every citation in the answer corresponds to a retrieved chunk.
    Returns {"valid": bool, "checked": int, "unmatched": [..]}.
    """
    if not blocks:
        return {"valid": False, "checked": 0, "unmatched": [], "reason": "no context"}

    valid_pairs = set()
    for b in blocks:
        valid_pairs.add((b["movie_title"].strip().lower(), b["start_time"].strip()))

    matches = CITATION_RE.findall(answer)
    unmatched = []
    for title, start, _end in matches:
        key = (title.strip().lower(), start.strip())
        if key not in valid_pairs:
            unmatched.append({"title": title, "start": start})

    return {
        "valid": len(matches) > 0 and not unmatched,
        "checked": len(matches),
        "unmatched": unmatched,
    }


def answer_query(
    query: str,
    top_k: int = 5,
    chat_history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Full RAG pipeline: retrieve → build context → grounded answer → verify.
    Refuses without calling the LLM when retrieval returns no context,
    and rejects answers whose citations fail verification.
    """
    results = retrieve(query, top_k=top_k)
    blocks = build_context(results)

    # HARD STOP #1: nothing relevant retrieved → refuse before calling the LLM.
    if not blocks:
        return {
            "answer": REFUSAL,
            "context": [],
            "verification": {
                "valid": False,
                "checked": 0,
                "unmatched": [],
                "reason": "no_context",
            },
        }

    context_str = format_context(blocks)
    answer = generate_grounded_answer(query, context_str, chat_history=chat_history)
    verification = verify_citations(answer, blocks)

    # HARD STOP #2: LLM produced no valid citations → reject.
    if not verification["valid"]:
        return {
            "answer": (
                "I couldn't ground an answer in the movie database for that question. "
                "Please ingest the relevant .srt file or rephrase with a specific movie or quote."
            ),
            "context": blocks,
            "verification": verification,
        }

    return {
        "answer": answer,
        "context": blocks,
        "verification": verification,
    }