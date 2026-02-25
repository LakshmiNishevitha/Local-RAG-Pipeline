# Local-RAG-Pipeline
```
An end-to-end Retrieval-Augmented Generation (RAG) system for semantic PDF question answering using vector search and LLM grounding.

This project demonstrates how to build a production-style RAG architecture from scratch using open-source tools and a local vector database.

---

## Overview

This system allows users to:

1. Upload a PDF document
2. Automatically extract and chunk text
3. Generate embeddings for each chunk
4. Store embeddings in Weaviate (vector database)
5. Perform semantic retrieval
6. Generate grounded answers using Gemini LLM
7. Interact through a Streamlit UI

The goal is to provide accurate, context-aware answers directly grounded in the document content — minimizing hallucination.

---

## Architecture Flow

PDF  
→ Text Extraction (PyMuPDF)  
→ Chunking with Overlap  
→ Embedding (SentenceTransformers)  
→ Store in Weaviate  
→ Semantic Retrieval (Top-K vector search)  
→ Prompt Construction  
→ Gemini LLM  
→ Grounded Answer  

---

## Tech Stack

- **PyMuPDF (fitz)** – PDF text extraction
- **SentenceTransformers (all-MiniLM-L6-v2)** – Embedding generation
- **Weaviate** – Vector database for semantic search
- **Google Gemini** – Large Language Model for response generation
- **Streamlit** – Interactive UI layer
- **Python** – Core implementation

---

## Project Structure

```

agents/
├── Splitting.py          # PDF extraction & chunking
├── Embedding.py          # Vector storage logic
├── retrieve.py           # Retrieval + LLM grounding
├── Weaviate_smoketest.py # DB test script
app_streamlit.py           # Main Streamlit UI
app_streamlit_min.py       # Minimal UI version
major.py                   # Full pipeline test script

````

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/LakshmiNishevitha/Local-RAG-Pipeline.git
cd Local-RAG-Pipeline
````

### 2️. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3️. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have requirements.txt yet, generate it:

```bash
pip freeze > requirements.txt
```

### 4️. Start Weaviate (Docker)

```bash
docker run -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  -e CLUSTER_HOSTNAME="node1" \
  semitechnologies/weaviate:1.24.10
```

### 5️. Add Gemini API Key

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

 Never commit `.env` to GitHub.

### 6️. Run Streamlit App

```bash
streamlit run app_streamlit.py
```

---

## How Retrieval Works

1. User question is embedded using SentenceTransformers.
2. Weaviate performs nearest-neighbor vector search.
3. Top relevant chunks are retrieved.
4. A grounded prompt is constructed.
5. Gemini generates an answer using only retrieved context.

This keeps responses anchored to the actual document.

---

##  Key Features

* Semantic search (not keyword matching)
* Chunk overlap for contextual continuity
* LLM grounding to reduce hallucination
* Local vector database deployment
* Interactive UI for non-technical users
* Modular architecture (splitting, embedding, retrieval separated)

---

##  Disclaimer

This project was built independently for educational and training purposes.
No proprietary code, data, or confidential information from any employer is included.

---

##  Future Improvements

* Metadata-based filtering (doc_id, chunk_index)
* Persistent UUID handling for deduplication
* Source citation display in UI
* Hybrid search (keyword + vector)
* Evaluation metrics for retrieval quality

---
