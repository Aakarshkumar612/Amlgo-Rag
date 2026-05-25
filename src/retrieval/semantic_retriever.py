"""
Step 2b: Semantic (ANN) retrieval using ChromaDB + MiniLM embeddings.
Handles conceptual/paraphrase queries that BM25 misses.
"""
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.ingestion.embedder import get_embedding_model, get_collection


def semantic_search(query: str, top_k: int = None) -> List[Dict]:
    """
    ANN semantic search using ChromaDB.
    Returns top_k chunks with cosine distance scores.
    """
    if top_k is None:
        top_k = config.SEMANTIC_TOP_K
    
    model = get_embedding_model()
    collection = get_collection()
    
    # Embed query
    query_embedding = model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=['documents', 'metadatas', 'distances']
    )
    
    chunks = []
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        # ChromaDB cosine distance: 0=identical, 2=opposite
        # Convert to similarity: sim = 1 - dist/2
        similarity = 1.0 - (dist / 2.0)
        
        chunk = {
            "chunk_id": results['ids'][0][i],
            "text": doc,
            "semantic_score": float(similarity),
            "retrieval_method": "semantic",
            **meta
        }
        chunks.append(chunk)
    
    return chunks