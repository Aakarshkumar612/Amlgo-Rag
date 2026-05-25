"""
Step 9: Two-level caching:
  1. Exact query cache (diskcache) — instant hits for repeated queries
  2. Semantic cache — cosine similarity to find near-duplicate queries
"""
import diskcache
import json
import hashlib
from typing import Optional, List, Tuple
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.ingestion.embedder import get_embedding_model

# Init disk cache
cache = diskcache.Cache(str(config.CACHE_DIR), timeout=config.CACHE_TTL_SECONDS)

# In-memory semantic cache: List of (embedding, query, cached_response)
_semantic_cache: List[Tuple[np.ndarray, str, str]] = []


def _query_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def get_exact(query: str) -> Optional[str]:
    """Check exact query cache."""
    return cache.get(_query_key(query))


def set_exact(query: str, response: str):
    """Store in exact cache."""
    cache.set(_query_key(query), response, expire=config.CACHE_TTL_SECONDS)


def get_semantic(query: str) -> Optional[str]:
    """
    Check semantic cache. Returns cached response if cosine similarity
    to a previous query exceeds SEMANTIC_CACHE_THRESHOLD.
    """
    if not _semantic_cache:
        return None
    
    model = get_embedding_model()
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    
    for cached_emb, cached_query, cached_response in _semantic_cache:
        similarity = float(np.dot(query_emb, cached_emb))
        if similarity >= config.SEMANTIC_CACHE_THRESHOLD:
            print(f"[Cache] Semantic cache hit (sim={similarity:.3f}): '{cached_query}'")
            return cached_response
    
    return None


def set_semantic(query: str, response: str):
    """Add to semantic cache."""
    model = get_embedding_model()
    emb = model.encode([query], normalize_embeddings=True)[0]
    _semantic_cache.append((emb, query, response))
    # Keep cache bounded
    if len(_semantic_cache) > 200:
        _semantic_cache.pop(0)


def get_cached(query: str) -> Optional[str]:
    """Check both caches. Exact first, then semantic."""
    # Level 1: exact match
    exact = get_exact(query)
    if exact:
        print(f"[Cache] Exact cache hit for: '{query[:50]}'")
        return exact
    
    # Level 2: semantic similarity
    return get_semantic(query)


def store_response(query: str, response: str):
    """Store response in both cache levels."""
    set_exact(query, response)
    set_semantic(query, response)