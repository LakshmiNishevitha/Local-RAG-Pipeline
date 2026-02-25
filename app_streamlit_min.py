import tempfile
import streamlit as st

from agents.Splitting import DocSplitterAgent
from agents.Embedding import VectorStoreAgent
from agents.retrieve import QueryAgent  

st.set_page_config(page_title="RAG Quick Ask", layout="centered")
st.title("Ask your PDF")

st.subheader("1) Index a PDF into Weaviate")
pdf = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    do_index = st.button("Index PDF")
with col2:
    quick_ask = st.button("Index + Ask: What's in the document?")

indexed = False
if do_index or quick_ask:
    if not pdf:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Extracting, chunking, embedding, storing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                pdf_path = tmp.name

            splitter = DocSplitterAgent()
            text = splitter.extract_text(pdf_path)
            chunks = splitter.split_text(text)

            store = VectorStoreAgent()
            try:
                stored = store.embed_and_store(chunks)
            except TypeError:
                stored = store.embed_and_store(chunks, doc_id="streamlit_doc")

            st.success(f"Indexed! Total chunks: {len(chunks)} | Stored now: {stored}")
            indexed = True

st.subheader("2) Ask a question")
default_q = "What is discussed in the document?"
question = st.text_input("Your question", value=default_q, placeholder=default_q)

ask = st.button("Ask")

def run_query(q):
    with st.spinner("Retrieving context and asking Gemini..."):
        qa = QueryAgent()
        result = qa.query(q)
        if isinstance(result, tuple):
            answer, used_docs = result
        else:
            answer, used_docs = result, []
        st.markdown("### Answer")
        st.write(answer)
        if used_docs:
            st.markdown("### 🔎 Retrieved Chunks (used by LLM)")
            for i, d in enumerate(used_docs, start=1):
                with st.expander(f"Chunk {i}"):
                    st.write(d)

if quick_ask and pdf:
    run_query(default_q)

if ask:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        run_query(question)