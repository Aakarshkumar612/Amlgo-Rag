"""
Master ingestion script. Run once before starting the app.
Usage: python ingest.py
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"


import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ingestion.loader import load_pdf
from src.ingestion.cleaner import clean_pages, deduplicate_pages
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.retrieval.bm25_retriever import build_bm25_index
from src.evaluation.evaluator import run_evaluation

def main():
    print("\n" + "="*60)
    print("AMLGO LABS RAG PIPELINE — INGESTION")
    print("="*60)
    
    # Check document exists
    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ ERROR: No PDF found in /data folder.")
        print("   Place AI_Training_Document.pdf in the /data directory.")
        sys.exit(1)
    
    pdf_path = pdf_files[0]
    print(f"📄 Processing: {pdf_path.name}")
    
    # Ensure output dirs exist
    config.CHUNKS_DIR.mkdir(exist_ok=True)
    config.VECTORDB_DIR.mkdir(exist_ok=True)
    config.LOGS_DIR.mkdir(exist_ok=True)
    config.CACHE_DIR.mkdir(exist_ok=True)
    
    start = time.time()
    
    # STEP 1: Load
    print("\n[1/5] Loading PDF...")
    pages = load_pdf(str(pdf_path))
    
    # STEP 1: Deduplicate + Clean
    print("[2/5] Cleaning and deduplicating...")
    pages = deduplicate_pages(pages)
    pages = clean_pages(pages)
    
    # STEP 1: Chunk
    print("[3/5] Chunking with sentence awareness...")
    chunks = chunk_pages(pages, output_dir=str(config.CHUNKS_DIR))
    
    # STEP 1+2: Embed into ChromaDB
    print("[4/5] Embedding into ChromaDB...")
    embed_chunks(chunks)
    
    # STEP 2: Build BM25 index
    print("[5/5] Building BM25 index...")
    build_bm25_index(chunks)
    
    elapsed = time.time() - start
    print(f"\n✅ Ingestion complete in {elapsed:.1f}s")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Avg words/chunk: {sum(c['word_count'] for c in chunks) // len(chunks)}")
    
    # STEP 8: Run evaluation
    print("\n[BONUS] Running evaluation suite...")
    run_evaluation("logs/evaluation_report.json")
    
    print("\n🚀 Ready! Run: streamlit run app.py")

if __name__ == "__main__":
    main()