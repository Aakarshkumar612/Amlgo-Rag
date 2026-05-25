import os
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHUNKS_DIR = ROOT / "chunks"
VECTORDB_DIR = ROOT / "vectordb"
LOGS_DIR = ROOT / "logs"
CACHE_DIR = ROOT / ".cache"

# Ingestion
CHUNK_SIZE = 250             # words per chunk (100-300 as required)
CHUNK_OVERLAP = 40           # overlap for context continuity
MIN_CHUNK_WORDS = 30         # skip tiny trailing chunks

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Retrieval
BM25_TOP_K = 20              # BM25 candidates
SEMANTIC_TOP_K = 20          # Vector DB candidates
RERANKER_TOP_K = 5           # Final chunks after reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Confidence thresholds
MIN_CONFIDENCE_SCORE = 0.65  # Below this → hallucination fallback
MIN_RERANKER_SCORE = -1.5
FRESHNESS_WEIGHT = 0.1
TRUST_WEIGHT = 0.5
CONSISTENCY_WEIGHT = 0.4

# Generation
LLM_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024
TEMPERATURE = 0.1            # Low for factual grounding

# Cache
CACHE_TTL_SECONDS = 3600     # 1 hour
SEMANTIC_CACHE_THRESHOLD = 0.92  # Cosine sim to reuse cached answer

# ChromaDB collection name
CHROMA_COLLECTION = "ebay_agreement_v1"

# Versioning
PIPELINE_VERSION = "1.0.0"
DOCUMENT_VERSION = "2024-ebay-tos"