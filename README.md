# Movie Intelligence & Follow-up Assistant

An AI agent that processes `.srt` subtitle files across a collection of movies to answer queries via RAG (Retrieval-Augmented Generation), retrieve precise scene citations, and securely execute external actions (sending emails) via an MCP layer.

## 🚀 Features
1. **RAG QA Engine**: Precise answers to questions regarding movie plots and character dialogue.
2. **Metadata Integrity**: Subtitle chunks are tagged identically to their source Movie Title and accurate timecodes for exact citation.
3. **MCP Email Tool**: Fully autonomous email dispatch using Python `smtplib`.
4. **Agentic Router**: Uses `gpt-4o-mini` function calling to correctly route informative queries, act on email requests, and smartly prompt the user for clarification if parameters (like recipient email) are missing.
5. **Streamlit UI**: A clean, stateful chat interface.

## 🛠️ Installation & Setup (Local)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
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
   ```

## 🧠 Usage

### 1. Ingesting Data (Vectorizing Subtitles)
Drop your `.srt` files into `data/subtitles/`. 
*(For demo purposes, a sample Matrix `.srt` has been provided).*

Run the pipeline:
```bash
python src/ingestion/vector_store.py
```
This parses the subtitles using `srt_parser.py` and generates local ChromaDB embeddings using OpenAI's `text-embedding-3-small`.

### 2. Running the Agent (Streamlit)
To interact with the Agent via the frontend UI:
```bash
streamlit run src/ui/app.py
```

## 🏗️ Architecture Stack
- **LLM/Agent**: OpenAI (`gpt-4o-mini`) native Function Calling.
- **RAG/Vector Store**: `ChromaDB` (Local Persistent) + OpenAI Embeddings.
- **UI**: Streamlit.
- **MCP Tool**: `smtplib` / `email.mime` for clean SMTP transit.
