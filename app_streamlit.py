import os, tempfile, uuid
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import weaviate

from agents.Splitting import DocSplitterAgent
from agents.Embedding import VectorStoreAgent
from agents.retrieve import QueryAgent

st.set_page_config(page_title="Local RAG UI", layout="wide")

WEAVIATE_URL = "http://localhost:8085"

@st.cache_resource
def get_weaviate_client():
    return weaviate.Client(WEAVIATE_URL)

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def check_weaviate():
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=3)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}

def near_vector_search(query_text, top_k=3):
    """Show retrieved context w/ distances, independent of QueryAgent (for display)."""
    client = get_weaviate_client()
    model = get_embedder()
    qvec = model.encode([query_text])[0].tolist()
    res = (
        client.query
        .get("DocumentChunk", ["content", "doc_id", "chunk_index"])
        .with_near_vector({"vector": qvec})
        .with_additional(["distance"])
        .with_limit(top_k)
        .do()
    )
    hits = res.get("data", {}).get("Get", {}).get("DocumentChunk", [])
    out = []
    for h in hits:
        out.append({
            "content": h.get("content", ""),
            "doc_id": h.get("doc_id"),
            "chunk_index": h.get("chunk_index"),
            "distance": h.get("_additional", {}).get("distance"),
        })
    return out

def index_pdf(pdf_path, doc_id=None):
    splitter = DocSplitterAgent()
    text = splitter.extract_text(pdf_path)
    chunks = splitter.split_text(text)
    store = VectorStoreAgent()
    try:
        stored = store.embed_and_store(chunks, doc_id=doc_id or "default_doc")
    except TypeError:
        stored = store.embed_and_store(chunks)
    return len(chunks), stored

st.sidebar.title("System")
ok, meta = check_weaviate()
if ok:
    st.sidebar.success("Weaviate: running")
    st.sidebar.caption(f"Version: {meta.get('version', 'unknown')}")
else:
    st.sidebar.error("Weaviate is not reachable on :8085")
    st.sidebar.info("Start it first:\n\n"
                    "docker run -p 8085:8085 \\\n"
                    "  -e QUERY_DEFAULTS_LIMIT=25 \\\n"
                    "  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\\n"
                    "  -e PERSISTENCE_DATA_PATH=\"/var/lib/weaviate\" \\\n"
                    "  -e CLUSTER_HOSTNAME=\"node1\" \\\n"
                    "  semitechnologies/weaviate:1.24.10")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Gemini key is read via `.env` by your agents. "
                   "Add `GEMINI_API_KEY=...` to `new/.env`.")

st.title(" Local RAG with Streamlit (PDF → Weaviate → Gemini)")

tab_index, tab_query = st.tabs(["Index PDF", "Ask Questions"])

with tab_index:
    st.subheader("Index a PDF into Weaviate")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    manual_path = st.text_input("...or paste an absolute PDF path",
                                placeholder="/Users/you/Downloads/doc.pdf")
    colA, colB = st.columns([1, 1])
    with colA:
        doc_id = st.text_input("Optional doc_id (helps deduplicate)", value="streamlit_doc")
    with colB:
        top_note = st.caption("If you rerun indexing with the same doc_id + deterministic UUIDs, "
                              "you won’t duplicate chunks.")

    if st.button("Index now", type="primary", use_container_width=True):
        if uploaded is None and not manual_path:
            st.warning("Upload a PDF or provide a path.")
        else:
            if uploaded is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    pdf_path = tmp.name
            else:
                pdf_path = manual_path

            with st.spinner("Extracting, chunking, embedding, storing..."):
                try:
                    total_chunks, stored = index_pdf(pdf_path, doc_id=doc_id or "streamlit_doc")
                    st.success(f"Indexed OK — Chunks: {total_chunks} | Stored now: {stored}")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

with tab_query:
    st.subheader("Ask a question")
    question = st.text_input("Your question")
    top_k = st.slider("Top-K retrieved chunks to display", 1, 8, 3)

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving context and asking Gemini..."):
                try:
                    qa = QueryAgent()
                    result = qa.query(question)

                    if isinstance(result, tuple):
                        answer, used_docs = result
                    else:
                        answer, used_docs = result, []

                    st.markdown("### Answer")
                    st.write(answer)

                    if not used_docs:
                        hits = near_vector_search(question, top_k=top_k)
                        st.markdown("### Retrieved Chunks")
                        for i, h in enumerate(hits, start=1):
                            with st.expander(f"Chunk {i} • distance={h['distance']:.4f} • "
                                             f"doc_id={h.get('doc_id')} • idx={h.get('chunk_index')}"):
                                st.write(h["content"])
                    else:
                        st.markdown("### Retrieved Chunks (from QueryAgent)")
                        for i, d in enumerate(used_docs[:top_k], start=1):
                            with st.expander(f"Chunk {i}"):
                                st.write(d)

                except Exception as e:
                    st.error(f"Query failed: {e}")
