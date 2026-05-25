"""
Step 8: Adversarial test suite, recall@K, hallucination tracking.
Run this after ingest to benchmark the pipeline.
"""
from typing import List, Dict, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.confidence.scorer import score_chunks, get_aggregate_confidence
import json

# Adversarial test queries for eBay User Agreement
TEST_QUERIES = [
    # === POSITIVE CASES (should answer from doc) ===
    {
        "query": "What are the arbitration rules for disputes?",
        "expected_keywords": ["arbitration", "NAM", "dispute", "binding"],
        "should_answer": True,
    },
    {
        "query": "Can eBay monitor my messages?",
        "expected_keywords": ["scan", "analyze", "messaging", "automated"],
        "should_answer": True,
    },
    {
        "query": "What happens if I sell outside of eBay?",
        "expected_keywords": ["final value fee", "outside", "policy"],
        "should_answer": True,
    },
    {
        "query": "How do I opt out of arbitration?",
        "expected_keywords": ["opt-out", "opt out", "written", "30 days"],
        "should_answer": True,
    },
    {
        "query": "What is eBay Money Back Guarantee?",
        "expected_keywords": ["money back", "refund", "buyer", "seller"],
        "should_answer": True,
    },
    # === NEGATIVE CASES (should trigger hallucination fallback) ===
    {
        "query": "What is the best programming language?",
        "expected_keywords": [],
        "should_answer": False,
    },
    {
        "query": "What is the weather in California today?",
        "expected_keywords": [],
        "should_answer": False,
    },
    {
        "query": "Who is the CEO of eBay in 2025?",
        "expected_keywords": [],
        "should_answer": False,
    },
]


def evaluate_recall(query: str, expected_keywords: List[str], k: int = 5) -> Dict:
    """Measure if relevant keywords appear in top-K retrieved chunks."""
    chunks = hybrid_retrieve(query)
    chunks = score_chunks(chunks)
    
    retrieved_text = " ".join([c['text'].lower() for c in chunks[:k]])
    
    hits = [kw for kw in expected_keywords if kw.lower() in retrieved_text]
    recall = len(hits) / len(expected_keywords) if expected_keywords else 1.0
    
    return {
        "query": query,
        "recall_at_k": recall,
        "hits": hits,
        "misses": [kw for kw in expected_keywords if kw.lower() not in retrieved_text],
        "top_chunk_confidence": chunks[0]['confidence_score'] if chunks else 0.0,
        "aggregate_confidence": get_aggregate_confidence(chunks),
        "num_chunks_retrieved": len(chunks),
    }


def run_evaluation(output_path: str = None) -> List[Dict]:
    """Run full evaluation suite and print results."""
    import config
    results = []
    
    print("\n" + "="*60)
    print("PIPELINE EVALUATION REPORT")
    print("="*60)
    
    for test in TEST_QUERIES:
        if test['expected_keywords']:  # positive test
            result = evaluate_recall(test['query'], test['expected_keywords'])
            status = "✅" if result['recall_at_k'] >= 0.5 else "❌"
            print(f"{status} [{result['recall_at_k']:.0%} recall] {test['query'][:60]}")
            if result['misses']:
                print(f"   Missing keywords: {result['misses']}")
        else:  # negative test (should have low confidence OR low reranker)
            chunks = hybrid_retrieve(test['query'])
            chunks = score_chunks(chunks)
            agg_conf = get_aggregate_confidence(chunks)
            top_reranker = chunks[0].get("reranker_score", -99) if chunks else -99
            
            # Mirror the same dual-check used in generator.py
            off_topic = agg_conf < config.MIN_CONFIDENCE_SCORE or top_reranker < config.MIN_RERANKER_SCORE
            status = "✅" if off_topic else "❌"
            print(f"{status} [conf={agg_conf:.2f} | reranker={top_reranker:.2f}] NEGATIVE: {test['query'][:50]}")
            result = {
                "query": test['query'],
                "aggregate_confidence": agg_conf,
                "top_reranker_score": top_reranker,
                "fallback_would_trigger": off_topic,
                "is_negative": True
            }
        
        results.append(result)
    
    # Summary
    positive = [r for r in results if not r.get('is_negative')]
    avg_recall = sum(r.get('recall_at_k', 0) for r in positive) / max(len(positive), 1)
    print(f"\nAverage Recall@5: {avg_recall:.1%}")
    print("="*60)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_evaluation("logs/evaluation_report.json")