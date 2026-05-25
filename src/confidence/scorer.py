"""
Step 4: Compute composite confidence score for each retrieved chunk.
Three components: Freshness, Trust, Retrieval Consistency.
"""
from typing import List, Dict
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def freshness_score(chunk: Dict) -> float:
    """
    Score based on document ingestion recency.
    For a single document, this is always 1.0 unless we have versioned updates.
    Returns 0.0-1.0.
    """
    # If multiple document versions exist, compare timestamps
    try:
        ts = datetime.fromisoformat(chunk.get('ingestion_timestamp', ''))
        age_days = (datetime.utcnow() - ts).days
        # Decay: 1.0 at 0 days, 0.5 at 365 days
        return max(0.1, 1.0 - (age_days / 730))
    except Exception:
        return 0.9


def trust_score(chunk: Dict) -> float:
    """
    Trust score based on source reliability.
    Official eBay ToS = 1.0 (authoritative source).
    User-generated content would be lower.
    Returns 0.0-1.0.
    """
    return float(chunk.get('trust_score', 1.0))


def retrieval_consistency_score(chunk: Dict) -> float:
    """
    How consistently was this chunk retrieved?
    Uses the reranker_score as the primary signal.
    Normalized to 0-1 range.
    """
    reranker = chunk.get('reranker_score', 0.0)
    # Cross-encoder scores are in logit space; use sigmoid-like normalization
    # typical range is -10 to 10, we clamp and normalize
    clamped = max(-5.0, min(5.0, reranker))
    normalized = (clamped + 5.0) / 10.0
    return normalized


def compute_confidence(chunk: Dict) -> float:
    """
    Composite confidence = weighted sum of three scores.
    """
    f = freshness_score(chunk)
    t = trust_score(chunk)
    c = retrieval_consistency_score(chunk)
    
    score = (
        config.FRESHNESS_WEIGHT * f +
        config.TRUST_WEIGHT * t +
        config.CONSISTENCY_WEIGHT * c
    )
    
    chunk['confidence_score'] = round(score, 4)
    chunk['freshness'] = round(f, 4)
    chunk['trust'] = round(t, 4)
    chunk['consistency'] = round(c, 4)
    
    return score


def score_chunks(chunks: List[Dict]) -> List[Dict]:
    """Score all chunks and add confidence metadata."""
    for chunk in chunks:
        compute_confidence(chunk)
    return sorted(chunks, key=lambda x: x['confidence_score'], reverse=True)


def get_aggregate_confidence(chunks: List[Dict]) -> float:
    """Aggregate confidence across all retrieved chunks (weighted by position)."""
    if not chunks:
        return 0.0
    scores = [c.get('confidence_score', 0) for c in chunks]
    # Top chunk weighted more
    weights = [1 / (i + 1) for i in range(len(scores))]
    total = sum(s * w for s, w in zip(scores, weights))
    return total / sum(weights)