# Movie Intelligence & Follow-up Assistant

An AI agent that processes `.srt` subtitle files across a collection of movies to answer queries via RAG (Retrieval-Augmented Generation), retrieve precise scene citations, and securely execute external actions (sending emails) via an MCP layer.

## 🚀 Features
1. **RAG QA Engine** — grounded answers with enforced `[Movie, HH:MM:SS - HH:MM:SS]` citations.
2. **Chunked Ingestion** — cues merged into overlapping 30–100 token chunks with speaker metadata.
3. **Batch Ingestion** — `ingest_directory()` ingests every `.srt` in `data/subtitles/`.
4. **LLM Character Extraction** — `gpt-4o-mini` extracts character names from dialogue and stores them in `metadata.characters`.
5. **Deterministic Follow-up Resolution** — pronouns like "it", "yes", "summarize", and "who are they" resolve to the last-discussed movie or characters.
6. **Tool-routed Agent** — `gpt-4o-mini` function calling dispatches to `query_movie_database`, `list_movies`, `list_characters`, or `send_email`.
7. **Strict Grounding** — empty retrieval is short-circuited; uncited answers are rejected; no plot summaries from pretrained knowledge.
8. **Real MCP Email Tool** — JSON-RPC MCP server exposing `send_email`, consumed by an MCP client.
9. **Streamlit UI** — chat, tool-call inspection, MCP status, indexed-movies sidebar, and explicit email confirmation before sending.
10. **Tests** — pytest suite for parser, chunker, vector store, RAG, agent routing, grounding, and email.


## 🛠️ Installation & Setup (Local)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alvinmjosan/movie_assistant.git
   cd movie-assistant
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (use `.env.example` as a template):
   ```
   OPENAI_API_KEY=your_api_key_here
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   EMAIL_SENDER=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password_here
   MCP_SERVER_URL=http://localhost:8000/mcp
   ```

## 🧠 Usage

### 1. Ingesting Data (Vectorizing Subtitles)

Drop your `.srt` files into `data/subtitles/`. A sample file, `Talking_about_lunch.srt`, is included for demo purposes.

Run the pipeline:
```bash
python src/ingestion/vector_store.py
```

This parses each subtitle file, extracts speakers and characters, chunks the dialogue into overlapping windows, embeds with OpenAI `text-embedding-3-small`, and upserts into a local ChromaDB. Re-running is safe — chunks use deterministic IDs.

Optional flags:
```bash
python src/ingestion/vector_store.py --list            # print indexed movies after ingestion
python src/ingestion/vector_store.py --query "lunch"   # run a sample query
python src/ingestion/vector_store.py --no-characters   # skip LLM character extraction
```

### 2. Starting the MCP Email Server

The email tool runs as a separate JSON-RPC service. In its own terminal:

```bash
uvicorn src.mcp.server:app --host 0.0.0.0 --port 8000
```

Verify it is reachable:

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

List the exposed tools:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### 3. Running the Agent (Streamlit)

In a third terminal, launch the UI:

```bash
streamlit run src/ui/app.py
```

Then open http://localhost:8501.

### 4. Running the Agent (Terminal)

For a lightweight CLI session:

```bash
python src/agent/core_agent.py
```

### 5. Testing

```bash
pytest -q
```

The suite covers parsing, chunking, vector store operations, RAG retrieval and citation verification, agent routing, grounding, and email error handling.

## 🏗️ Architecture Stack

| Layer | Technology | Role |
|---|---|---|
| **LLM / Agent** | OpenAI `gpt-4o-mini` (native function calling) | Multi-step reasoning, tool selection, grounded answer generation |
| **RAG / Vector Store** | ChromaDB (local persistent) + OpenAI `text-embedding-3-small` | Subtitle chunk storage and semantic retrieval |
| **RAG Engine** | `src/rag/qa_engine.py` | Retrieval, context formatting, citation enforcement, verification |
| **MCP Server** | FastAPI + JSON-RPC 2.0 | Exposes `send_email` via `initialize`, `tools/list`, `tools/call` |
| **MCP Client** | `requests` | Bridges the agent to the MCP server over HTTP |
| **Email Tool** | Python `smtplib` + `email.mime` (STARTTLS) | Clean, validated SMTP dispatch |
| **UI** | Streamlit | Chat, tool-call inspection, MCP status, email confirmation panel |
| **Testing** | `pytest` | Parser, chunker, vector store, RAG, routing, email |

### End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│  - Chat interface                                                │
│  - Tool-call inspection (expanders)                              │
│  - MCP status indicator                                          │
│  - Pending-email confirmation panel                              │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    core_agent (gpt-4o-mini)                      │
│  - Multi-step tool-calling loop (MAX_TOOL_ITERATIONS = 5)        │
│  - System prompt enforces RAG-before-email and citation format   │
└───────────┬────────────────────────────────────┬─────────────────┘
            │                                    │
            ▼                                    ▼
┌───────────────────────────┐      ┌───────────────────────────────┐
│   query_movie_database    │      │          send_email           │
│  (RAG QA tool)            │      │  (via MCP client)             │
└───────────┬───────────────┘      └───────────────┬───────────────┘
            │                                      │
            ▼                                      ▼
┌───────────────────────────┐      ┌───────────────────────────────┐
│   rag.qa_engine           │      │   mcp.mcp_client              │
│  - retrieve()             │      │  - initialize()               │
│  - build_context()        │      │  - list_tools()               │
│  - format_context()       │      │  - call_tool()                │
│  - verify_citations()     │      └───────────────┬───────────────┘
└───────────┬───────────────┘                      │
            │                                      ▼
            ▼                          ┌───────────────────────────┐
┌───────────────────────────┐          │   mcp.server (FastAPI)    │
│   ChromaDB (local)        │          │  - POST /mcp              │
│  - OpenAI embeddings      │          │  - GET /health            │
│  - text-embedding-3-small │          │  - tools/list, tools/call │
└───────────────────────────┘          └───────────────┬───────────┘
                                                       │
                                                       ▼
                                          ┌───────────────────────┐
                                          │  mcp.email_tool       │
                                          │  - smtplib + STARTTLS │
                                          └───────────────────────┘
```

### Data Flow

**Ingestion (offline):**

```
.srt files → parse_srt() → extract_characters_with_llm() → chunk_subtitles()
           → OpenAI embeddings → ChromaDB upsert
```

**Query (online):**

```
User question → core_agent → query_movie_database → rag.qa_engine.retrieve()
              → ChromaDB similarity search → format_context()
              → gpt-4o-mini grounded answer → verify_citations() → UI
```

**Action (email):**

```
User command → core_agent → query_movie_database (optional context)
             → send_email tool call → mcp.mcp_client.call_tool()
             → mcp.server (JSON-RPC) → smtplib SMTP → SMTP server
             → result back through MCP → UI confirmation panel
```

### Module Map

| Module | Responsibility |
|---|---|
| `src/ingestion/srt_parser.py` | Cue parsing, speaker extraction, HTML stripping, overlapping chunking, LLM character extraction |
| `src/ingestion/vector_store.py` | Batch ingestion, embedding, ChromaDB upsert, `list_indexed_movies()`, `list_characters()` |
| `src/rag/qa_engine.py` | Retrieval, context building, grounded generation, citation verification |
| `src/mcp/email_tool.py` | Validated SMTP dispatch via `smtplib` |
| `src/mcp/server.py` | FastAPI JSON-RPC MCP server (`initialize`, `tools/list`, `tools/call`) |
| `src/mcp/mcp_client.py` | JSON-RPC client and `call_email_via_mcp` helper |
| `src/agent/core_agent.py` | Multi-step tool-calling agent, routing, follow-up resolution, email preview mode |
| `src/ui/app.py` | Streamlit chat UI with tool-call display, MCP status, and email confirmation |
| `tests/` | Pytest coverage for all phases |

## 📁 Project Structure

```
movie-assistant/
├── .env.example
├── requirements.txt
├── README.md
├── data/
│   └── subtitles/                 # .srt files (one per movie)
├── src/
│   ├── ingestion/
│   │   ├── srt_parser.py          # cue parsing, speakers, chunking, character extraction
│   │   └── vector_store.py        # batch ingestion, ChromaDB upsert, metadata readers
│   ├── rag/
│   │   └── qa_engine.py           # retrieval, context, citations, verification
│   ├── mcp/
│   │   ├── email_tool.py          # smtplib wrapper
│   │   ├── server.py              # FastAPI JSON-RPC MCP server
│   │   └── mcp_client.py          # JSON-RPC MCP client
│   ├── agent/
│   │   └── core_agent.py          # multi-step tool-calling agent
│   └── ui/
│       └── app.py                 # Streamlit chat UI
└── tests/                         # pytest suite
```

## 🗺️ Milestones Coverage

| Milestone | Status | Where |
|---|---|---|
| Parse, chunk, index all `.srt` files | ✅ | `ingest_directory` |
| Grounded answers with verified citations | ✅ | `qa_engine.answer_query` |
| Metadata integrity (movie + timecodes + characters) | ✅ | `metadata.characters`, `movie_title`, `start_time`, `end_time` |
| Agent invokes RAG and email tools | ✅ | `get_agent_response` tool loop |
| Ambiguity handling and follow-ups | ✅ | `_is_followup`, `_is_character_followup`, `last_movie` |
| MCP-compliant email tool | ✅ | `mcp.server`, `mcp.mcp_client` |
| UI shows tool calls and confirms emails | ✅ | `src/ui/app.py` |
| Tests and validation | ✅ | `tests/` |