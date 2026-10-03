import chromadb
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# We import the embedding function specifically for OpenAI
import chromadb.utils.embedding_functions as embedding_functions

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

def populate_vector_store(subtitles: List[Dict], collection_name: str = "movie_dialogue", persist_directory: str = DEFAULT_DB_PATH):
    """
    Takes the parsed subtitles and stores them into a local ChromaDB vector store
    using OpenAI's precise embeddings.
    """
    # 1. Initialize the ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 2. Set up OpenAI Embedding Function
    # It automatically picks up OPENAI_API_KEY from the environment
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    # 3. Get or create a collection with this specific embedding function
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=openai_ef
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for i, sub in enumerate(subtitles):
        documents.append(sub["text"])
        metadatas.append(sub["metadata"])
        
        movie_title_clean = sub["metadata"]["movie_title"].replace(" ", "_")
        doc_id = f"{movie_title_clean}_chunk_{i}"
        ids.append(doc_id)
        
    print(f"Adding {len(documents)} subtitle chunks to the Vector Database via OpenAI Embeddings...")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Database populated successfully!")

def query_vector_store(query_text: str, collection_name: str = "movie_dialogue", persist_directory: str = DEFAULT_DB_PATH, n_results: int = 3):
    """
    Queries the vector database using OpenAI embeddings.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    collection = client.get_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    return results

if __name__ == "__main__":
    # Test script: Let's chain our parser and vector store!
    import sys
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from srt_parser import parse_srt
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_file = os.path.join(base_dir, "data", "subtitles", "The_Matrix_1999.srt")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file before running this test.")
        sys.exit(1)
        
    try:
        subs = parse_srt(sample_file, "The Matrix")
        populate_vector_store(subs)
        
        print("\n--- Testing a Query ---")
        question = "What happens to Neo?"
        results = query_vector_store(question, n_results=1)
        
        print("\nTop Result Found:")
        print(f"Dialogue: {results['documents'][0][0]}")
        meta = results['metadatas'][0][0]
        print(f"Citation: [{meta['movie_title']}, {meta['start_time']} - {meta['end_time']}]")
        
    except Exception as e:
        print(f"Error testing vector store: {e}")
