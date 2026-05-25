"""
Step 1a: Load PDF with page-level metadata using PyMuPDF.
Extracts text per page with bounding box awareness.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import re

def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Load PDF and return list of page dicts with raw text + metadata.
    Each page dict: {page_num, text, char_count, hash, source}
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        
        if len(text.strip()) < 20:    # skip near-empty pages
            continue
        
        # Hash for deduplication
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
            "hash": text_hash,
            "source": Path(pdf_path).name,
            "total_pages": len(doc),
        })
    
    doc.close()
    print(f"[Loader] Loaded {len(pages)} pages from {Path(pdf_path).name}")
    return pages