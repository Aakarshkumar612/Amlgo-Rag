"""
Streamlit RAG Chatbot — Full production interface.
Features:
  - Real-time streaming responses
  - Source citations with page references
  - Confidence scoring display
  - Cache status indicator
  - Observability stats sidebar
  - Clear chat / reset functionality
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from src.ingestion.embedder import get_embedding_model
    return get_embedding_model()

@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker():
    from src.retrieval.hybrid_retriever import get_reranker
    return get_reranker()

# Trigger model loading at startup, not on first query
load_embedding_model()
load_reranker()

# Page config MUST be first Streamlit call
st.set_page_config(
    page_title="eBay Agreement RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.confidence.scorer import score_chunks, get_aggregate_confidence
from src.generation.generator import stream_response
from src.cache.query_cache import get_cached, store_response
from src.observability.logger import log_retrieval, get_stats

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ RAG Pipeline")
    st.markdown("**Document**: eBay User Agreement")
    st.markdown(f"**Model**: `{config.LLM_MODEL}`")
    st.markdown(f"**Embeddings**: `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`")
    
    # Chunk count from ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.VECTORDB_DIR))
        collection = client.get_collection(config.CHROMA_COLLECTION)
        st.metric("Indexed Chunks", collection.count())
    except Exception:
        st.warning("⚠️ Vector DB not found. Run `python ingest.py` first.")
    
    st.divider()
    
    # Observability Stats (Step 10)
    st.subheader("📊 Live Stats")
    stats = get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Queries", stats["total_queries"])
    col2.metric("Cache Hits", f"{stats['cache_hit_rate']:.0f}%")
    col1.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
    col2.metric("Fallback Rate", f"{stats['fallback_rate']:.0f}%")
    
    if stats["avg_latency_ms"] > 0:
        st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
    
    st.divider()
    
    # Pipeline settings
    st.subheader("⚙️ Settings")
    min_confidence = st.slider(
        "Min Confidence Threshold",
        min_value=0.1, max_value=0.8,
        value=config.MIN_CONFIDENCE_SCORE, step=0.05,
        help="Below this → hallucination fallback triggered"
    )
    
    show_sources = st.checkbox("Show Source Chunks", value=True)
    show_scores = st.checkbox("Show Confidence Scores", value=True)
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    # Sample queries
    st.subheader("💡 Sample Queries")
    sample_queries = [
        "What is the arbitration process?",
        "Can eBay suspend my account?",
        "How does the Money Back Guarantee work?",
        "What fees do sellers pay?",
        "How do I opt out of arbitration?",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending_query = q
            st.rerun()


# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────
st.title("⚖️ eBay User Agreement — AI Legal Assistant")
st.caption("Near-zero hallucination RAG • Hybrid retrieval • Citation-backed answers")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for i, chunk in enumerate(msg["sources"]):
                    conf = chunk.get('confidence_score', 0)
                    color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                    st.markdown(
                        f"**{color} [CHUNK {i+1}]** — Page {chunk.get('page_num','?')} "
                        f"| Confidence: {conf:.2f} | Reranker: {chunk.get('reranker_score', 0):.3f}"
                    )
                    st.text(chunk['text'][:300] + "...")
                    st.divider()


# ─────────────────────────────────────────────
# Query Processing
# ─────────────────────────────────────────────

# Handle sample query buttons
query = st.session_state.pop("pending_query", None)

# Handle user text input
if user_input := st.chat_input("Ask about the eBay User Agreement..."):
    query = user_input

if query:
    st.session_state.total_queries += 1
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process query
    with st.chat_message("assistant"):
        start_time = time.time()
        
        # Step 9: Check cache first
        cached_response = get_cached(query)
        
        if cached_response:
            # Cache hit — display instantly
            response_placeholder = st.empty()
            response_placeholder.markdown(cached_response)
            
            latency_ms = (time.time() - start_time) * 1000
            st.caption(f"⚡ Cache hit | {latency_ms:.0f}ms")
            
            log_retrieval(query, [], 0.0, latency_ms, cache_hit=True)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": cached_response,
                "sources": [],
            })
        
        else:
            # Full retrieval pipeline
            with st.status("🔍 Retrieving relevant context...", expanded=False) as status:
                # Step 2+3: Hybrid retrieve + rerank
                chunks = hybrid_retrieve(query)
                
                # Step 4: Confidence scoring
                chunks = score_chunks(chunks)
                aggregate_confidence = get_aggregate_confidence(chunks)
                
                fallback = aggregate_confidence < min_confidence
                
                if show_scores:
                    st.write(f"Aggregate confidence: **{aggregate_confidence:.3f}**")
                    st.write(f"Chunks retrieved: **{len(chunks)}**")
                    if fallback:
                        st.warning("⚠️ Low confidence — fallback will trigger")
                
                status.update(label="✅ Context retrieved", state="complete")
            
            # Step 5-7: Stream generation
            response_placeholder = st.empty()
            full_response = ""
            
            for token in stream_response(query, chunks, st.session_state.conversation_history):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Latency + metadata
            latency_ms = (time.time() - start_time) * 1000
            
            if show_scores:
                st.caption(
                    f"⏱️ {latency_ms:.0f}ms | "
                    f"Confidence: {aggregate_confidence:.2f} | "
                    f"Model: {config.LLM_MODEL}"
                )
            
            # Show sources
            if chunks and show_sources:
                with st.expander(f"📎 Sources ({len(chunks)} chunks used)"):
                    for i, chunk in enumerate(chunks):
                        conf = chunk.get('confidence_score', 0)
                        color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                        st.markdown(
                            f"**{color} [CHUNK {i+1}]** — Page {chunk.get('page_num','?')} | "
                            f"Confidence: {conf:.2f} | "
                            f"Reranker: {chunk.get('reranker_score', 0):.3f} | "
                            f"Section: {chunk.get('section_header','N/A')}"
                        )
                        st.text(chunk['text'][:400] + "...")
                        if i < len(chunks) - 1:
                            st.divider()
            
            # Step 9: Store in cache
            if full_response and "INSUFFICIENT_CONTEXT" not in full_response:
                store_response(query, full_response)
            
            # Step 10: Log trace
            log_retrieval(
                query=query,
                chunks=chunks,
                aggregate_confidence=aggregate_confidence,
                latency_ms=latency_ms,
                fallback_triggered=fallback,
            )
            
            # Update conversation history (Step 9 memory)
            st.session_state.conversation_history.append({"role": "user", "content": query})
            st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
            
            # Store in messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": chunks,
            })