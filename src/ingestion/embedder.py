"""
Step 1d: Generate embeddings using all-MiniLM-L6-v2 (local, free).
Stores into ChromaDB for ANN retrieval.
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import chromadb
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Singleton embedding model
_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading {config.EMBEDDING_MODEL}...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: List[Dict]) -> chromadb.Collection:
    """
    Generate embeddings for all chunks and store in ChromaDB.
    Returns the ChromaDB collection.
    """
    model = get_embedding_model()
    
    # Init ChromaDB persistent client
    client = chromadb.PersistentClient(path=str(config.VECTORDB_DIR))
    
    # Delete if exists (fresh ingest)
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )
    
    # Batch embed
    texts = [c['text'] for c in chunks]
    ids = [c['chunk_id'] for c in chunks]
    
    print(f"[Embedder] Embedding {len(texts)} chunks in batches of {config.EMBEDDING_BATCH_SIZE}...")
    
    for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i:i+config.EMBEDDING_BATCH_SIZE]
        batch_ids = ids[i:i+config.EMBEDDING_BATCH_SIZE]
        batch_chunks = chunks[i:i+config.EMBEDDING_BATCH_SIZE]
        
        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            normalize_embeddings=True  # cosine-ready
        ).tolist()
        
        # Build metadata for ChromaDB (must be primitive types)
        metadatas = [{
            "source": c['source'],
            "page_num": c['page_num'],
            "word_count": c['word_count'],
            "section_header": c.get('section_header', ''),
            "chunk_hash": c['chunk_hash'],
            "document_version": c['document_version'],
            "ingestion_timestamp": c['ingestion_timestamp'],
            "trust_score": c['trust_score'],
            "freshness_score": c['freshness_score'],
        } for c in batch_chunks]
        
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            ids=batch_ids,
            metadatas=metadatas
        )
        
        print(f"[Embedder] Embedded {min(i+config.EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)}")
    
    print(f"[Embedder] Done. {collection.count()} chunks in ChromaDB.")
    return collection


def get_collection() -> chromadb.Collection:
    """Get existing ChromaDB collection (for retrieval)."""
    client = chromadb.PersistentClient(path=str(config.VECTORDB_DIR))
    return client.get_collection(config.CHROMA_COLLECTION)