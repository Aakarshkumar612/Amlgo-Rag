"""
Step 1b: Text normalization and deduplication.
Handles format standardization and removes noise.
"""
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Normalize text: remove headers/footers noise, fix whitespace,
    standardize unicode, remove page artifacts.
    """
    # Remove page numbers (standalone digits on their own line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove repeated dashes/underscores (table separators in PDFs)
    text = re.sub(r'[-_]{3,}', ' ', text)
    
    # Normalize unicode quotes and dashes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u00a0', ' ')  # non-breaking space
    
    # Collapse multiple whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Remove PDF header/footer artifacts (page X of Y patterns)
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Ensure sentences don't get cut mid-word at line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # rejoin hyphenated linebreaks
    
    return text.strip()


def deduplicate_pages(pages: List[Dict]) -> List[Dict]:
    """
    Remove exact-duplicate pages using MD5 hash.
    Keeps first occurrence.
    """
    seen_hashes = set()
    unique_pages = []
    
    for page in pages:
        if page['hash'] not in seen_hashes:
            seen_hashes.add(page['hash'])
            unique_pages.append(page)
        else:
            print(f"[Cleaner] Duplicate page skipped: page {page['page_num']}")
    
    return unique_pages


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """Apply cleaning to all pages and return cleaned list."""
    cleaned = []
    for page in pages:
        clean = dict(page)
        clean['text'] = clean_text(page['text'])
        clean['word_count'] = len(clean['text'].split())
        cleaned.append(clean)
    return cleaned