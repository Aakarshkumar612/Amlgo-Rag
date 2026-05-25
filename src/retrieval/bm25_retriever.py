"""
Step 2a: BM25 keyword retrieval.
Classic sparse retrieval — excellent for exact legal terms and section numbers.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

BM25_CACHE_PATH = config.CHUNKS_DIR / "bm25_index.pkl"


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase + split on non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """Build BM25 index from chunks and cache to disk."""
    corpus = [tokenize(c['text']) for c in chunks]
    bm25 = BM25Okapi(corpus)
    
    # Cache index
    with open(BM25_CACHE_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
    
    print(f"[BM25] Built index over {len(chunks)} chunks, cached to disk.")
    return bm25, chunks


def load_bm25_index() -> Tuple[BM25Okapi, List[Dict]]:
    """Load cached BM25 index from disk."""
    if not BM25_CACHE_PATH.exists():
        raise FileNotFoundError("BM25 index not found. Run ingest.py first.")
    with open(BM25_CACHE_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['bm25'], data['chunks']


def bm25_search(query: str, top_k: int = None) -> List[Dict]:
    """
    Search BM25 index. Returns top_k chunks sorted by BM25 score.
    """
    if top_k is None:
        top_k = config.BM25_TOP_K
    
    bm25, chunks = load_bm25_index()
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # only non-zero scores
            result = dict(chunks[idx])
            result['bm25_score'] = float(scores[idx])
            result['retrieval_method'] = 'bm25'
            results.append(result)
    
    return results