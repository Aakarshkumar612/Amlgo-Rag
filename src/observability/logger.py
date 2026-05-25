"""
Step 10: Structured JSON observability.
Traces every retrieval: chunks ranked, scores, latency, failures.
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Setup logger
config.LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(config.LOGS_DIR / "rag_traces.jsonl"),
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("rag_observability")

# In-memory stats for dashboard
_stats = {
    "total_queries": 0,
    "cache_hits": 0,
    "fallback_triggers": 0,
    "avg_confidence": [],
    "avg_latency_ms": [],
    "failed_queries": [],
}


def log_retrieval(
    query: str,
    chunks: List[Dict],
    aggregate_confidence: float,
    latency_ms: float,
    cache_hit: bool = False,
    fallback_triggered: bool = False,
    error: Optional[str] = None
):
    """Log a complete retrieval trace as structured JSON."""
    global _stats
    
    trace = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "cache_hit": cache_hit,
        "fallback_triggered": fallback_triggered,
        "aggregate_confidence": round(aggregate_confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "num_chunks": len(chunks),
        "chunks_summary": [
            {
                "chunk_id": c.get("chunk_id", "?"),
                "page_num": c.get("page_num", "?"),
                "reranker_score": round(c.get("reranker_score", 0), 4),
                "confidence_score": round(c.get("confidence_score", 0), 4),
            }
            for c in chunks[:5]
        ],
        "error": error,
    }
    
    logger.info(json.dumps(trace))
    
    # Update in-memory stats
    _stats["total_queries"] += 1
    if cache_hit:
        _stats["cache_hits"] += 1
    if fallback_triggered:
        _stats["fallback_triggers"] += 1
    _stats["avg_confidence"].append(aggregate_confidence)
    _stats["avg_latency_ms"].append(latency_ms)
    if error:
        _stats["failed_queries"].append({"query": query, "error": error})
    
    # Keep lists bounded
    for key in ["avg_confidence", "avg_latency_ms"]:
        if len(_stats[key]) > 1000:
            _stats[key] = _stats[key][-500:]


def get_stats() -> Dict:
    """Return current observability stats for the dashboard."""
    conf_list = _stats["avg_confidence"]
    lat_list = _stats["avg_latency_ms"]
    
    return {
        "total_queries": _stats["total_queries"],
        "cache_hit_rate": (
            _stats["cache_hits"] / max(_stats["total_queries"], 1) * 100
        ),
        "fallback_rate": (
            _stats["fallback_triggers"] / max(_stats["total_queries"], 1) * 100
        ),
        "avg_confidence": (sum(conf_list) / len(conf_list)) if conf_list else 0,
        "avg_latency_ms": (sum(lat_list) / len(lat_list)) if lat_list else 0,
        "recent_failures": _stats["failed_queries"][-5:],
    }