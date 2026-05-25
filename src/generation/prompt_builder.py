"""
Steps 5+6: Build prompts that constrain generation to retrieved context only.
Every claim must be backed by a cited chunk with doc+page+timestamp.
"""
from typing import List, Dict

SYSTEM_PROMPT = """You are a precise legal document assistant for the eBay User Agreement.

STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY using the provided document excerpts (labeled [CHUNK X]).
2. DO NOT use any external knowledge, assumptions, or information not in the excerpts.
3. Every factual statement MUST include a citation like [CHUNK X, Page Y].
4. If the excerpts do not contain enough information to answer, say exactly:
   "INSUFFICIENT_CONTEXT: The provided document excerpts do not contain enough information to answer this question. Please ask about specific sections of the eBay User Agreement."
5. Do not speculate, infer beyond the text, or fill gaps with assumptions.
6. If the answer spans multiple chunks, cite all relevant ones.
7. Be concise and direct. Do not repeat yourself.
"""


def build_context_block(chunks: List[Dict]) -> str:
    """
    Build the context block injected into the prompt.
    Each chunk is labeled with its source, page, and confidence.
    """
    if not chunks:
        return "NO RELEVANT EXCERPTS FOUND."
    
    context_parts = []
    for i, chunk in enumerate(chunks):
        citation = f"[CHUNK {i+1}]"
        page = f"Page {chunk.get('page_num', '?')}"
        source = chunk.get('source', 'eBay User Agreement')
        confidence = chunk.get('confidence_score', 0)
        timestamp = chunk.get('ingestion_timestamp', 'N/A')[:10]  # date only
        
        header = f"{citation} | {source} | {page} | Confidence: {confidence:.2f} | Version: {timestamp}"
        block = f"{header}\n{chunk['text']}"
        context_parts.append(block)
    
    return "\n\n" + "="*60 + "\n\n".join(context_parts) + "\n\n" + "="*60


def build_user_message(query: str, chunks: List[Dict]) -> str:
    """Build the full user message with context + query."""
    context_block = build_context_block(chunks)
    
    return f"""DOCUMENT EXCERPTS:
{context_block}

USER QUESTION: {query}

Remember: Cite every factual claim with [CHUNK X, Page Y]. If insufficient context, respond with INSUFFICIENT_CONTEXT."""


def build_messages(query: str, chunks: List[Dict], conversation_history: List[Dict] = None) -> List[Dict]:
    """
    Build the full messages list for Groq API.
    Includes conversation history for multi-turn support (Step 9 memory).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (last 4 turns for memory)
    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append(turn)
    
    # Add current query with context
    messages.append({
        "role": "user",
        "content": build_user_message(query, chunks)
    })
    
    return messages