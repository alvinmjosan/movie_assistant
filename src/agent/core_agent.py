import os
import json
import sys
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
    "I don't have that information in the movie database. "
    "That movie or scene has not been ingested."
)
REFUSAL_UNCITED = (
    "I couldn't ground an answer in the movie database for that question. "
    "Please ingest the relevant .srt file or rephrase with a specific movie or quote."
)


def _tool_definitions() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_movie_database",
                "description": (
                    "Searches the vector database for actual movie dialogue, quotes, or scenes. "
                    "Use this whenever the user asks a factual question about a movie or character."
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
            "You are a professional Movie Intelligence AI. "
            "Rules: "
            "1. For factual movie questions, ALWAYS use query_movie_database first. "
            "2. When answering with movie facts, ALWAYS include citations formatted as "
            "'[Movie Title, HH:MM:SS - HH:MM:SS]'. "
            "3. For email requests: FIRST retrieve relevant movie context via query_movie_database, "
            "THEN compose the email. Never guess a recipient email. If missing, ask the user. "
            "4. If the user's request is ambiguous (missing movie, scene, or recipient), ask a "
            "clarifying question instead of taking action. "
            "5. NEVER use general knowledge about movies. "
            "If query_movie_database returns no relevant context, reply exactly: "
            "'I don't have that information in the movie database.' "
            "Do NOT summarize plots from memory. Do NOT name actors or directors. "
            "6. If the user asks anything about which movies are available, indexed, known, "
            "stored, listed, or in the database — using ANY phrasing — ALWAYS use list_movies. "
            "Do not guess titles. "
            "7. After a query_movie_database that returns no relevant context, do NOT stop at "
            "the refusal. Instead, list the available movies and suggest the user pick one. "
            "8. If the user refers to something using pronouns or definite references "
            "(e.g. 'the movie', 'that scene', 'it'), resolve them from the recent chat history. "
            "If the previous assistant turn mentioned a specific movie, use that movie's title "
            "as the query for query_movie_database. Never search with bare pronouns. "
            "9. If the user asks who is in a movie, which characters appear, or asks for the cast, "
            "ALWAYS use list_characters. Do not list characters from memory."
        ),
    }


def _run_movie_query(query: str) -> Dict[str, Any]:
    results = retrieve(query, top_k=5)
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
        return "The movie database is empty."
    return "Indexed movies:\n" + "\n".join(f"- {t}" for t in titles)


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
    return "Characters by movie:\n" + "\n".join(lines)


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

# NEW: phrases that signal a follow-up about the characters just listed.
CHARACTER_FOLLOWUP_PHRASES = (
    "who are they", "who are these", "who is that", "who are those",
    "who are the characters", "who are the people", "who is in it",
    "who's in it", "cast", "characters",
)


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


# NEW: detect follow-ups asking about the characters.
def _is_character_followup(user_query: str) -> bool:
    q = user_query.strip().lower()
    return any(p in q for p in CHARACTER_FOLLOWUP_PHRASES)


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

    # NEW: deterministic character follow-up — pure metadata lookup, no LLM needed.
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

    if last_movie and _is_followup(user_query):
        first_word = user_query.strip().lower().split()[0]
        if first_word in FOLLOWUP_VERBS:
            user_query = (
                f"User is asking about the movie '{last_movie}'. "
                f"Your task: quote 3-5 specific dialogue lines from this movie "
                f"and cite each with [Movie Title, HH:MM:SS - HH:MM:SS]. "
                f"Do NOT paraphrase or summarize — only quote with citations. "
                f"Their message: {original_query}"
            )
        else:
            user_query = (
                f"User is asking about the movie '{last_movie}'. "
                f"Their message: {original_query}"
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

        if not msg.tool_calls:
            final_text = msg.content or ""

            if last_blocks:
                verification = verify_citations(final_text, last_blocks)
                if not verification["valid"]:
                    listing = _list_movies_text()
                    return {
                        "answer": (
                            "I couldn't ground an answer in the movie database for that question.\n\n"
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
                result = _run_movie_query(args.get("query", ""))
                tool_result_str = result["context_str"]
                tool_log.append({"name": name, "args": args, "result": tool_result_str})

                if result["empty"]:
                    listing = _list_movies_text()
                    answer = (
                        "I don't have that specific movie or scene in the database. "
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