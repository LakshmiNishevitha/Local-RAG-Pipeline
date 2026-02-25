# Local RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system for semantic PDF question answering using vector search and LLM grounding.

This project demonstrates how to build a modular, production-style RAG architecture from scratch using open-source tools and a locally deployed vector database.

---

##  Overview

This system enables users to:

- Upload a PDF document
- Extract and chunk text automatically
- Generate embeddings for each chunk
- Store embeddings in Weaviate (vector database)
- Perform semantic (vector) retrieval
- Generate grounded answers using Gemini
- Interact via an intuitive Streamlit interface

The primary goal is to provide accurate, context-aware answers grounded strictly in document content — minimizing hallucination.

---

##  Architecture Flow

```
PDF
  - Text Extraction (PyMuPDF)
  - Chunking with Overlap
  - Embedding Generation (SentenceTransformers)
  - Store in Weaviate
  - Semantic Retrieval (Top-K Similar Chunks)
  - Prompt Construction with Retrieved Context
  - Gemini LLM
  - Grounded Answer in Streamlit UI
```

---

##  Tech Stack

- **PyMuPDF (fitz)** : PDF text extraction
- **SentenceTransformers (all-MiniLM-L6-v2)** : Embedding generation
- **Weaviate** : Vector database for semantic search
- **Google Gemini** : LLM for response generation
- **Streamlit** : Interactive UI
- **Python** : Core implementation

---

##  Project Structure

```
.
├── agents/
│   ├── Embedding.py
│   ├── Splitting.py
│   ├── retrieve.py
│   ├── Weaviate_smoketest.py
│   └── __init__.py
├── app_streamlit.py
├── app_streamlit_min.py
├── major.py
└── README.md
```

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/LakshmiNishevitha/Local-RAG-Pipeline.git
cd Local-RAG-Pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If you have `requirements.txt`:

```bash
pip install -r requirements.txt
```

If not, install manually:

```bash
pip install streamlit python-dotenv sentence-transformers weaviate-client pymupdf google-generativeai requests
```

(Optional) Generate requirements file:

```bash
pip freeze > requirements.txt
```

---

### 4. Start Weaviate (Docker Required)

```bash
docker run -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  -e CLUSTER_HOSTNAME="node1" \
  semitechnologies/weaviate:1.24.10
```

---

### 5. Add Gemini API Key

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_api_key_here
```

 Never commit `.env` to GitHub.

---

### 6. Run the Streamlit Application

```bash
streamlit run app_streamlit.py
```

The app will open in your browser.

---

##  How Retrieval Works

1. The user question is converted into a vector embedding.
2. Weaviate performs nearest-neighbor search on stored document chunks.
3. Top-K relevant chunks are retrieved.
4. A grounded prompt is constructed using only retrieved context.
5. Gemini generates a response strictly based on that context.

This ensures responses are anchored to actual document content.

---

##  Key Features

- Semantic vector search (not keyword matching)
- Chunk overlap for better contextual continuity
- Local vector database deployment
- LLM grounding to reduce hallucinations
- Modular pipeline architecture (split → embed → retrieve)
- Streamlit UI for interactive demonstrations

---

##  Disclaimer

This project was built independently for educational and training purposes.

No proprietary code, confidential data, or employer-specific information is included.

---