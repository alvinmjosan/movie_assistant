import os
import json
import sys
import re
from typing import Dict, List, Optional, Any

from openai import OpenAI
from dotenv import load_dotenv

# Connect to other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.qa_engine import retrieve, build_context, format_context, verify_citations
from mcp.mcp_client import call_email_via_mcp

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-4o-mini"
MAX_TOOL_ITERATIONS = 5

REFUSAL_NO_CONTEXT = (
    "I don't have that information in the content database. "
    "That content or scene has not been ingested."
)
REFUSAL_UNCITED = (
    "I couldn't ground an answer in the content database for that question. "
    "Please ingest the relevant .srt file or rephrase with a specific title or quote."
)


def _tool_definitions() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_movie_database",
                "description": (
                    "Searches the vector database for actual dialogue, quotes, or scenes "
                    "from any indexed content — movies, podcasts, shows, or conversations. "
                    "ALWAYS call this for any factual question about content BEFORE answering "
                    "or refusing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific topic or quote to search for.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_movies",
                "description": (
                    "Returns the list of all movies currently indexed in the database. "
                    "Trigger this whenever the user asks which movies are available, indexed, "
                    "known, stored, listed, or in the database."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_characters",
                "description": (
                    "Returns the characters that appear in one or all indexed movies. "
                    "Use this whenever the user asks who is in a movie, which characters appear, "
                    "or asks for the cast."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_title": {
                            "type": "string",
                            "description": (
                                "Optional. If provided, only that movie's characters are returned."
                            ),
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": (
                    "Sends an email. DO NOT GUESS the recipient email. "
                    "Before calling, ensure you have already retrieved the necessary movie "
                    "context via query_movie_database so the email body includes quotes and citations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient_email": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["recipient_email", "subject", "body"],
                },
            },
        },
    ]


def _system_prompt() -> Dict:
    return {
        "role": "system",
        "content": (
            "You are a professional Content Intelligence AI. "
            "Rules: "
            "0. BEFORE answering or refusing ANY question about content — movies, podcasts, "
            "shows, dialogue, scenes, or conversations — you MUST call query_movie_database "
            "first, passing the user's question as the query. Only refuse AFTER the tool "
            "returns no relevant context. "
            "1. For any factual question, ALWAYS use query_movie_database first. "
            "2. When answering with content facts, ALWAYS include citations formatted as "
            "'[Title, HH:MM:SS - HH:MM:SS]'. "
            "3. For email requests: FIRST retrieve relevant content via query_movie_database, "
            "THEN compose the email. Never guess a recipient email. If missing, ask the user. "
            "4. If the user's request is ambiguous (missing title, scene, or recipient), ask a "
            "clarifying question instead of taking action. "
            "5. NEVER use general knowledge about content. "
            "If query_movie_database returns no relevant context, reply exactly: "
            "'I don't have that information in the content database.' "
            "Do NOT summarize plots from memory. Do NOT name actors, hosts, or directors. "
            "6. If the user asks anything about which titles are available, indexed, known, "
            "stored, listed, or in the database — using ANY phrasing — ALWAYS use list_movies. "
            "Do not guess titles. "
            "7. After a query_movie_database that returns no relevant context, do NOT stop at "
            "the refusal. Instead, list the available titles and suggest the user pick one. "
            "8. If the user refers to something using pronouns or definite references "
            "(e.g. 'it', 'that scene', 'the title'), resolve them from the recent chat history. "
            "If the previous assistant turn mentioned a specific title, use that title "
            "as the query for query_movie_database. Never search with bare pronouns. "
            "9. If the user asks who is in a movie, podcast, or show, which characters appear, "
            "or asks for the cast or hosts, ALWAYS use list_characters. Do not list them from memory. "
            "10. If the user names a specific title that is not indexed, do NOT substitute a "
            "different title from context. Reply that the requested title is not available "
            "and list what IS available. Never answer about a different title than the one asked."
        ),
    }


def _run_movie_query(query: str) -> Dict[str, Any]:
    results = retrieve(query, top_k=8)
    blocks = build_context(results)

    if not blocks:
        return {
            "context_str": "NO_RELEVANT_CONTEXT",
            "blocks": [],
            "empty": True,
        }

    return {
        "context_str": format_context(blocks),
        "blocks": blocks,
        "empty": False,
    }


def _list_movies_text() -> str:
    """Return a formatted list of indexed movies, or a friendly message."""
    try:
        from ingestion.vector_store import list_indexed_movies
        titles = list_indexed_movies()
    except Exception as e:
        return f"Error listing movies: {e}"

    if not titles:
        return "The content database is empty."
    return "Indexed titles:\n" + "\n".join(f"- {t}" for t in titles)


def _list_characters_text(movie_title: Optional[str] = None) -> str:
    """Return a formatted list of characters from the indexed movies."""
    try:
        from ingestion.vector_store import list_characters
        mapping = list_characters(movie_title=movie_title)
    except Exception as e:
        return f"Error listing characters: {e}"

    if not mapping:
        return "No character data is indexed yet."

    lines = []
    for movie, chars in mapping.items():
        if chars:
            lines.append(f"- {movie}: {', '.join(chars)}")
        else:
            lines.append(f"- {movie}: (no characters extracted)")
    return "Characters by title:\n" + "\n".join(lines)


def _retrieve_movie_overview(movie_title: str, n: int = 6) -> List[Dict[str, Any]]:
    """
    Return evenly spaced chunks from a movie for open-ended summaries.
    Used together with semantic retrieval so the LLM has both coverage
    (whole episode) and precision (specific lines it may want to quote).
    """
    try:
        from ingestion.vector_store import _collection
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ingestion",
            "chroma_db",
        )
        col = _collection(db_path, "movie_dialogue")
        data = col.get(
            where={"movie_title": movie_title},
            include=["documents", "metadatas"],
        )
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        if not docs:
            return []

        step = max(1, len(docs) // n)
        blocks: List[Dict[str, Any]] = []
        for i in range(0, len(docs), step):
            if len(blocks) >= n:
                break
            blocks.append({
                "dialogue": docs[i],
                "movie_title": metas[i].get("movie_title", ""),
                "start_time": metas[i].get("start_time", ""),
                "end_time": metas[i].get("end_time", ""),
                "speakers": metas[i].get("speakers", ""),
            })
        return blocks
    except Exception:
        return []


def _merge_blocks(*block_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge multiple lists of dialogue blocks and dedupe by a composite key.
    Preserves insertion order so overview chunks come first, then semantic hits.
    """
    seen = set()
    merged: List[Dict[str, Any]] = []
    for blocks in block_lists:
        for b in blocks or []:
            key = (
                b.get("start_time", ""),
                b.get("end_time", ""),
                (b.get("dialogue", "") or "")[:60],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(b)
    return merged


FOLLOWUP_VERBS = (
    "summarize", "summarise", "summary", "describe", "explain",
    "elaborate", "continue", "more", "details", "tell",
)

FOLLOWUP_PHRASES = (
    "yes", "yes do that", "sure", "go on", "go ahead", "please do",
    "do it", "okay", "ok", "continue", "and",
    "the movie", "that movie", "it", "that", "this", "about it",
    "tell me more", "more details", "give me details",
)

CHARACTER_FOLLOWUP_PHRASES = (
    "who are they", "who are these", "who is that", "who are those",
    "who are the characters", "who are the people", "who is in it",
    "who's in it", "cast", "characters",
)

# Words that are never titles in follow-up contexts.
_STOP_TITLE_WORDS = {
    "the", "a", "an", "this", "that", "it", "movie", "film", "show",
    "podcast", "episode", "content", "about", "tell", "me", "give",
    "summarize", "summarise", "describe", "explain", "list", "more",
    "details", "info", "information", "what", "is", "are", "who",
    "how", "when", "where", "why", "and", "or", "of", "in", "on",
    "you", "your", "do", "does", "have", "has", "can", "could",
    "would", "should", "please", "know", "summary",
}


def _query_names_a_title(user_query: str) -> bool:
    """
    Return True if the query contains a token that looks like a specific
    title reference — any non-stop-word that isn't a common verb or pronoun.
    Case-insensitive so lowercase titles (e.g. 'titanic') are still caught.
    """
    words = user_query.strip().split()
    for w in words:
        clean = w.strip(".,!?'\"").lower()
        if not clean:
            continue
        if clean in _STOP_TITLE_WORDS:
            continue
        if clean in FOLLOWUP_VERBS:
            continue
        # Any remaining word is likely a title reference.
        return True
    return False


def _is_followup(user_query: str) -> bool:
    """
    Return True if the query looks like a bare continuation of the previous turn.
    """
    q = user_query.strip().lower()
    words = q.split()
    if not words:
        return False

    if len(words) <= 4 and q in FOLLOWUP_PHRASES:
        return True

    if len(words) == 1 and words[0] in FOLLOWUP_VERBS:
        return True

    if len(words) <= 4 and words[0] in FOLLOWUP_VERBS:
        return True

    if len(words) <= 6 and any(p in q for p in ("the movie", "that movie", "about it", "tell me more")):
        return True

    return False


def _is_character_followup(user_query: str) -> bool:
    """Detect queries asking about characters, using word-boundary matching
    so 'cast' does not match inside 'podcast'."""
    q = user_query.strip().lower()
    for phrase in CHARACTER_FOLLOWUP_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return True
    return False


def get_agent_response(
    user_query: str,
    chat_history: Optional[List[Dict]] = None,
    auto_send_email: bool = True,
    last_movie: Optional[str] = None,
) -> Dict[str, Any]:
    if chat_history is None:
        chat_history = []

    original_query = user_query
    character_followup = _is_character_followup(user_query)

    # Deterministic character follow-up — pure metadata lookup, no LLM needed.
    if last_movie and character_followup:
        try:
            from ingestion.vector_store import list_characters
            mapping = list_characters(movie_title=last_movie)
            chars = mapping.get(last_movie, [])
            answer = (
                f"Characters in '{last_movie}': " + ", ".join(chars)
                if chars
                else f"No character data is available for '{last_movie}'."
            )
        except Exception as e:
            answer = f"Error listing characters: {e}"

        return {
            "answer": answer,
            "tool_calls": [{
                "name": "list_characters",
                "args": {"movie_title": last_movie},
                "result": answer,
            }],
            "pending_email": None,
            "verification": None,
            "last_movie": last_movie,
        }

    # Verb-specific rewrite templates — each intent produces a distinct retrieval focus.
    # Skip the rewrite when the user names a specific title so their title takes precedence.
    if (
        last_movie
        and _is_followup(user_query)
        and not _query_names_a_title(original_query)
    ):
        q_lower = original_query.strip().lower()
        first_word = q_lower.split()[0] if q_lower.split() else ""

        if first_word in ("summarize", "summarise", "summary"):
            user_query = (
                f"{original_query} "
                f"(Context: the user wants a condensed summary of '{last_movie}'. "
                f"Produce 2-4 sentences describing what this content is about — "
                f"its main topic, who is speaking, and the overall tone. "
                f"You may include 1 short quote with citation if it strengthens the summary.)"
            )
        elif first_word in ("describe", "explain", "elaborate", "tell"):
            user_query = (
                f"{original_query} "
                f"(Context: the user wants an overview of '{last_movie}'. "
                f"Cover the main themes and topics, mention the speakers, and include "
                f"2-3 short dialogue quotes with citations to illustrate.)"
            )
        elif first_word in ("more", "continue", "details"):
            user_query = (
                f"{original_query} "
                f"(Context: the user wants additional detail about '{last_movie}'. "
                f"Quote 3-5 specific dialogue lines with citations on topics that "
                f"weren't already covered.)"
            )
        else:
            user_query = (
                f"{original_query} "
                f"(Context: the user is asking about '{last_movie}'.)"
            )

    messages = [_system_prompt()] + chat_history + [{"role": "user", "content": user_query}]
    tool_log: List[Dict[str, Any]] = []
    pending_email: Optional[Dict[str, str]] = None
    last_blocks: List[Dict[str, Any]] = []
    last_movie_seen: Optional[str] = last_movie

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            tools=_tool_definitions(),
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Fallback: force a tool call if the LLM tried to answer or refuse without consulting the DB.
        if not msg.tool_calls and not tool_log:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                tools=_tool_definitions(),
                tool_choice="required",
            )
            msg = response.choices[0].message

        if not msg.tool_calls:
            final_text = msg.content or ""

            if last_blocks:
                verification = verify_citations(final_text, last_blocks)
                if not verification["valid"]:
                    listing = _list_movies_text()
                    return {
                        "answer": (
                            "I couldn't ground an answer in the content database for that question.\n\n"
                            "Here's what I do have:\n\n" + listing +
                            "\n\nTry asking about one of these, or ingest the relevant .srt file."
                        ),
                        "tool_calls": tool_log,
                        "pending_email": pending_email,
                        "verification": verification,
                        "last_movie": last_movie_seen,
                    }
            else:
                verification = None

            return {
                "answer": final_text,
                "tool_calls": tool_log,
                "pending_email": pending_email,
                "verification": verification,
                "last_movie": last_movie_seen,
            }

        messages.append(msg)

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            print(f"--> [System] Agent requested '{name}' with args {args}")

            if name == "query_movie_database":
                q_arg = args.get("query", "") or ""
                q_lower = q_arg.strip().lower()
                is_summarize = q_lower.startswith(("summarize", "summarise", "summary"))

                if is_summarize and last_movie_seen:
                    # Merge evenly-spaced overview chunks with semantic top-K so
                    # the LLM has both coverage and precision for the summary.
                    overview_blocks = _retrieve_movie_overview(last_movie_seen, n=6)
                    semantic_result = _run_movie_query(q_arg)
                    semantic_blocks = semantic_result.get("blocks", [])

                    merged_blocks = _merge_blocks(overview_blocks, semantic_blocks)

                    if merged_blocks:
                        result = {
                            "context_str": format_context(merged_blocks),
                            "blocks": merged_blocks,
                            "empty": False,
                        }
                    else:
                        result = _run_movie_query(q_arg)
                else:
                    result = _run_movie_query(q_arg)

                tool_result_str = result["context_str"]
                tool_log.append({"name": name, "args": args, "result": tool_result_str})

                if result["empty"]:
                    listing = _list_movies_text()
                    answer = (
                        "I don't have that specific content or scene in the database. "
                        "Here's what I do have:\n\n" + listing +
                        "\n\nTry asking about one of these, or ingest the relevant .srt file."
                    )
                    return {
                        "answer": answer,
                        "tool_calls": tool_log,
                        "pending_email": None,
                        "verification": {
                            "valid": False, "checked": 0, "unmatched": [],
                            "reason": "no_context",
                        },
                        "last_movie": last_movie_seen,
                    }

                last_blocks = result["blocks"]
                if result["blocks"]:
                    last_movie_seen = result["blocks"][0]["movie_title"]

            elif name == "list_movies":
                tool_result_str = _list_movies_text()
                tool_log.append({"name": name, "args": args, "result": tool_result_str})

                try:
                    from ingestion.vector_store import list_indexed_movies
                    titles = list_indexed_movies()
                    if len(titles) == 1:
                        last_movie_seen = titles[0]
                except Exception:
                    pass

            elif name == "list_characters":
                title = args.get("movie_title")
                tool_result_str = _list_characters_text(movie_title=title)
                tool_log.append({"name": name, "args": args, "result": tool_result_str})

                if title:
                    last_movie_seen = title

            elif name == "send_email":
                recipient = args.get("recipient_email", "")
                subject = args.get("subject", "")
                body = args.get("body", "")
                if not recipient:
                    tool_result_str = "Error: recipient_email is missing. Ask the user for the email address."
                elif auto_send_email:
                    tool_result_str = call_email_via_mcp(recipient, subject, body)
                else:
                    pending_email = {
                        "recipient_email": recipient,
                        "subject": subject,
                        "body": body,
                    }
                    tool_result_str = "Email prepared and queued for user confirmation in the UI."
                tool_log.append({"name": name, "args": args, "result": tool_result_str})

            else:
                tool_result_str = f"Error: unknown tool '{name}'."
                tool_log.append({"name": name, "args": args, "result": tool_result_str})

            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": name,
                "content": tool_result_str,
            })

        if pending_email:
            final = client.chat.completions.create(model=GPT_MODEL, messages=messages)
            return {
                "answer": final.choices[0].message.content or "",
                "tool_calls": tool_log,
                "pending_email": pending_email,
                "verification": None,
                "last_movie": last_movie_seen,
            }

    return {
        "answer": "I reached the maximum number of tool calls without a final answer.",
        "tool_calls": tool_log,
        "pending_email": pending_email,
        "verification": None,
        "last_movie": last_movie_seen,
    }


def get_agent_response_text(user_query: str, chat_history: Optional[List[Dict]] = None) -> str:
    """Backwards-compatible helper that returns just the answer string."""
    return get_agent_response(user_query, chat_history, auto_send_email=True)["answer"]


if __name__ == "__main__":
    print("Agent is ready. (Type 'quit' to exit)")
    history: List[Dict] = []
    last_movie: Optional[str] = None
    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        print("Agent thinking...")
        res = get_agent_response(q, history, auto_send_email=True, last_movie=last_movie)
        print(f"\nAgent: {res['answer']}")
        for tc in res["tool_calls"]:
            print(f"  [tool] {tc['name']} -> {str(tc['result'])[:120]}")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": res["answer"]})
        if res.get("last_movie"):
            last_movie = res["last_movie"]