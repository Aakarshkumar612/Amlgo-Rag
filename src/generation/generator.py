"""
Steps 5-7: Groq streaming generation with hallucination fallback.
Streams token-by-token. Falls back gracefully if confidence too low.
"""
import os
from typing import List, Dict, Generator
from groq import Groq
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.generation.prompt_builder import build_messages
from src.confidence.scorer import get_aggregate_confidence

load_dotenv()

_client = None

def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=api_key)
    return _client


def stream_response(
    query: str,
    chunks: List[Dict],
    conversation_history: List[Dict] = None
) -> Generator[str, None, None]:
    """
    Step 5-7: Stream response token by token.
    
    - Step 5: Only retrieved context injected (no external knowledge)
    - Step 6: Citations enforced via prompt
    - Step 7: Hallucination fallback if aggregate confidence < threshold
    
    Yields: string tokens for Streamlit streaming display.
    """
    client = get_groq_client()
    
    # Step 7: Hallucination Fallback — check BOTH confidence AND reranker score
    aggregate_confidence = get_aggregate_confidence(chunks)
    top_reranker_score = chunks[0].get("reranker_score", -99) if chunks else -99
    
    off_topic = (
        not chunks
        or aggregate_confidence < config.MIN_CONFIDENCE_SCORE
        or top_reranker_score < config.MIN_RERANKER_SCORE
    )
    
    if off_topic:
        yield "⚠️ **Insufficient Context Detected**\n\n"
        yield "The retrieved document excerpts do not contain enough relevant information "
        yield "to answer your question reliably. Rather than guess or hallucinate, "
        yield "I'm declining to answer.\n\n"
        yield "**Try asking about:**\n"
        yield "- Specific eBay policies (fees, returns, listings)\n"
        yield "- Legal sections (arbitration, liability, disputes)\n"
        yield "- Buyer or seller obligations"
        return
    
    # Build messages with context
    messages = build_messages(query, chunks, conversation_history)
    
    # Stream from Groq
    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        stream=True,
    )
    
    full_response = ""
    for chunk_delta in stream:
        token = chunk_delta.choices[0].delta.content
        if token:
            full_response += token
            yield token
    
    # Step 7: Post-generation check for INSUFFICIENT_CONTEXT signal
    if "INSUFFICIENT_CONTEXT" in full_response:
        # Already handled by the model's instruction-following
        pass