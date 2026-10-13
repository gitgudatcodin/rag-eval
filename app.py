"""Streamlit UI: ask questions over the document corpus."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st

# Streamlit Cloud exposes Secrets via st.secrets, not env vars — copy them
# into the environment before rag.config reads it, so RAG_LLM_* secrets work.
try:
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))
except Exception:
    pass

from rag import config
from rag.generate import LLMClient, answer_question
from run_eval import build_index, serving_retriever

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Document Q&A")

if "retriever" not in st.session_state:
    if not os.path.exists(os.path.join(config.STORE_DIR, "vectors.npy")):
        # First run (e.g. fresh Streamlit Cloud deploy): build the index
        # from data/docs/ automatically instead of erroring out.
        with st.spinner("Building index from documents (first run only)..."):
            build_index()
    st.session_state.retriever = serving_retriever()

llm = LLMClient()
st.sidebar.markdown("### Status")
st.sidebar.write(f"Embedder: `{st.session_state.retriever.embedder.backend}`")
st.sidebar.write(f"LLM: `{'configured (' + llm.model + ')' if llm.configured else 'not configured — extractive mode'}`")
top_k = st.sidebar.slider("Chunks retrieved (k)", 1, 8, config.TOP_K)

query = st.text_input("Ask a question about the documents:")
if st.button("Ask") and query.strip():
    with st.spinner("Retrieving..."):
        hits = st.session_state.retriever.retrieve(query, k=top_k)
    with st.spinner("Generating..."):
        answer, mode = answer_question(query, hits, llm=llm)
    st.markdown(f"**Answer** ({mode} mode)")
    st.write(answer)
    st.markdown("**Retrieved passages**")
    for cid, score, meta in hits:
        with st.expander(f"{meta['doc']}  ·  score {score:.3f}"):
            st.write(meta["text"])
