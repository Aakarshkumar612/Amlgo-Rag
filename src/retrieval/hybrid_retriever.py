"""
Step 2c + 3: Hybrid fusion of BM25 + Semantic, then Cross-Encoder reranking.
Uses Reciprocal Rank Fusion (RRF) for combining ranked lists.
"""
from typing import List, Dict
from sentence_transformers import CrossEncoder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.semantic_retriever import semantic_search

# Singleton reranker
_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[Reranker] Loading {config.RERANKER_MODEL}...")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    semantic_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    RRF: fuse two ranked lists into one.
    Score = 1/(k + rank) for each result in each list.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}
    
    for rank, chunk in enumerate(bm25_results):
        cid = chunk['chunk_id']
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk
    
    for rank, chunk in enumerate(semantic_results):
        cid = chunk['chunk_id']
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk
    
    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    
    fused = []
    for cid in sorted_ids:
        chunk = dict(chunk_map[cid])
        chunk['rrf_score'] = rrf_scores[cid]
        chunk['retrieval_method'] = 'hybrid_rrf'
        fused.append(chunk)
    
    return fused


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
    """
    Step 3: Cross-encoder reranking.
    Deep scoring: takes (query, chunk_text) pair, outputs relevance score.
    Much more accurate than embedding similarity alone.
    """
    if top_k is None:
        top_k = config.RERANKER_TOP_K
    
    if not chunks:
        return []
    
    reranker = get_reranker()
    
    # Prepare pairs
    pairs = [(query, chunk['text']) for chunk in chunks]
    
    # Get relevance scores (0-1 range after sigmoid)
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    # Attach reranker score and sort
    for chunk, score in zip(chunks, scores):
        chunk['reranker_score'] = float(score)
    
    ranked = sorted(chunks, key=lambda x: x['reranker_score'], reverse=True)
    
    return ranked[:top_k]


def hybrid_retrieve(query: str) -> List[Dict]:
    """
    Full pipeline: BM25 + Semantic → RRF Fusion → Cross-Encoder Rerank.
    Returns final top-K chunks ready for generation.
    """
    # Step 2: Parallel retrieval
    bm25_results = bm25_search(query, top_k=config.BM25_TOP_K)
    semantic_results = semantic_search(query, top_k=config.SEMANTIC_TOP_K)
    
    # Fusion
    fused = reciprocal_rank_fusion(bm25_results, semantic_results)
    
    # Take top 20 candidates for reranking (efficiency)
    candidates = fused[:20]
    
    # Step 3: Rerank
    final_chunks = rerank_chunks(query, candidates, top_k=config.RERANKER_TOP_K)
    
    return final_chunks