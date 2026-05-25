"""
Step 1c: Sentence-aware chunking with metadata enrichment.
Implements versioning and metadata extraction per chunk.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

def extract_section_header(text: str) -> str:
    """Try to extract a section number/header from chunk text."""
    # Match patterns like "3. Using eBay" or "19.B.3 Waiver"
    match = re.search(r'^(\d+[\.\d]*)\s+([A-Z][^\.]{5,50})', text.strip(), re.MULTILINE)
    if match:
        return match.group(0)[:60]
    return ""


def chunk_pages(pages: List[Dict], output_dir: str = None) -> List[Dict]:
    """
    Chunk pages into 100-300 word segments with sentence awareness.
    Returns list of chunk dicts with full metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 5,  # chars (approx 250 words * 5 chars/word)
        chunk_overlap=config.CHUNK_OVERLAP * 5,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    chunk_id = 0
    ingestion_ts = datetime.utcnow().isoformat()
    
    for page in pages:
        if len(page['text'].split()) < config.MIN_CHUNK_WORDS:
            continue
        
        raw_chunks = splitter.split_text(page['text'])
        
        for i, chunk_text in enumerate(raw_chunks):
            words = chunk_text.split()
            if len(words) < config.MIN_CHUNK_WORDS:
                continue
            
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            section_header = extract_section_header(chunk_text)
            
            chunk = {
                # Identity
                "chunk_id": f"chunk_{chunk_id:04d}",
                "chunk_hash": chunk_hash,
                
                # Content
                "text": chunk_text.strip(),
                "word_count": len(words),
                
                # Source metadata (for citations)
                "source": page['source'],
                "page_num": page['page_num'],
                "page_chunk_index": i,
                "section_header": section_header,
                
                # Versioning (Step 1 requirement)
                "document_version": config.DOCUMENT_VERSION,
                "pipeline_version": config.PIPELINE_VERSION,
                "ingestion_timestamp": ingestion_ts,
                
                # Confidence inputs
                "trust_score": 1.0,         # document-level (known good source)
                "freshness_score": 1.0,     # latest available version
            }
            
            all_chunks.append(chunk)
            chunk_id += 1
    
    print(f"[Chunker] Created {len(all_chunks)} chunks from {len(pages)} pages")
    
    # Save chunks to disk
    if output_dir:
        out_path = Path(output_dir) / "chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"[Chunker] Saved chunks to {out_path}")
    
    return all_chunks