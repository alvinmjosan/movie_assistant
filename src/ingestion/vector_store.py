import os
import glob
from typing import List, Dict, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DEFAULT_COLLECTION = "movie_dialogue"
EMBED_MODEL = "text-embedding-3-small"


def _openai_ef():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=EMBED_MODEL,
    )


def _collection(persist_directory: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(name=collection_name, embedding_function=_openai_ef())


def populate_vector_store(
    subtitles: List[Dict],
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_DB_PATH,
):
    """
    Upsert chunks into ChromaDB. Uses deterministic IDs so re-runs don't duplicate.
    """
    if not subtitles:
        print("No chunks to insert.")
        return

    collection = _collection(persist_directory, collection_name)

    documents, metadatas, ids = [], [], []
    for sub in subtitles:
        meta = sub["metadata"]
        chunk_id = meta.get("chunk_id") or f"{meta['movie_title'].replace(' ', '_')}_{len(ids)}"
        documents.append(sub["text"])
        metadatas.append(meta)
        ids.append(chunk_id)

    print(f"Upserting {len(documents)} chunks into '{collection_name}' via OpenAI Embeddings...")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print("Done.")


def ingest_directory(
    subtitles_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_DB_PATH,
    chunk_size: int = 5,
    overlap: int = 1,
    movie_title_from_filename: bool = True,
    extract_characters: bool = True,
):
    """
    Loop over all .srt files in a directory, parse, chunk, and upsert.
    Returns the total number of chunks ingested.
    """
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from srt_parser import parse_and_chunk

    files = sorted(glob.glob(os.path.join(subtitles_dir, "*.srt")))
    if not files:
        print(f"No .srt files found in {subtitles_dir}")
        return 0

    total = 0
    for fp in files:
        if movie_title_from_filename:
            title = os.path.splitext(os.path.basename(fp))[0].replace("_", " ")
        else:
            title = os.path.splitext(os.path.basename(fp))[0]

        print(f"Ingesting '{title}' from {fp}")
        chunks = parse_and_chunk(
            fp, title,
            chunk_size=chunk_size,
            overlap=overlap,
            extract_characters=extract_characters,
        )
        populate_vector_store(chunks, collection_name=collection_name, persist_directory=persist_directory)
        total += len(chunks)

    print(f"\nTotal chunks ingested: {total}")
    return total


def query_vector_store(
    query_text: str,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_DB_PATH,
    n_results: int = 5,
):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name, embedding_function=_openai_ef())
    return collection.query(query_texts=[query_text], n_results=n_results)


def list_indexed_movies(
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_DB_PATH,
) -> List[str]:
    """Return the sorted list of unique movie titles in the collection."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=_openai_ef(),
    )
    data = collection.get(limit=100000, include=["metadatas"])
    titles = {
        m["movie_title"]
        for m in data.get("metadatas", [])
        if m and "movie_title" in m
    }
    return sorted(titles)


def list_characters(
    movie_title: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_DB_PATH,
) -> Dict[str, List[str]]:
    """
    Return {movie_title: [character1, character2, ...]}.
    If movie_title is given, only that movie is returned.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name, embedding_function=_openai_ef())
    data = collection.get(limit=100000, include=["metadatas"])

    result: Dict[str, set] = {}
    for m in data.get("metadatas", []):
        if not m:
            continue
        title = m.get("movie_title")
        if not title:
            continue
        if movie_title and title != movie_title:
            continue
        chars = (m.get("characters") or "").split(",")
        result.setdefault(title, set()).update(c.strip() for c in chars if c.strip())

    return {k: sorted(v) for k, v in result.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest .srt files into ChromaDB.")
    parser.add_argument("--query", default=None, help="Optional sample query after ingestion.")
    parser.add_argument("--top-k", type=int, default=3, help="Results for the sample query.")
    parser.add_argument("--list", action="store_true", help="Print indexed movies after ingestion.")
    parser.add_argument("--no-characters", action="store_true",
                        help="Skip LLM-based character extraction during ingestion.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subtitles_dir = os.path.join(base_dir, "data", "subtitles")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env before running.")
        raise SystemExit(1)

    ingest_directory(subtitles_dir, extract_characters=not args.no_characters)

    if args.list:
        print("\n--- Indexed Movies ---")
        for t in list_indexed_movies():
            print(f"- {t}")

    if args.query:
        print(f"\n--- Sample Query: {args.query} ---")
        results = query_vector_store(args.query, n_results=args.top_k)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"- {doc}\n  [{meta['movie_title']}, {meta['start_time']} - {meta['end_time']}]")