from agents.Splitting import DocSplitterAgent
from agents.Embedding import VectorStoreAgent
from agents.retrieve import QueryAgent

def main():
    pdf_path = "/Users/lakshminishevitha/Downloads/DSE501_Project_Proposal_Group42.pdf"

    doc_agent = DocSplitterAgent()
    text = doc_agent.extract_text(pdf_path)
    chunks = doc_agent.split_text(text)
    print(f" Extracted {len(chunks)} chunks from PDF")

    print("\nPreview of first 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk[:100]}...")

    store_agent = VectorStoreAgent()
    stored_count = store_agent.embed_and_store(chunks)
    print(f" Stored {stored_count} chunks in Weaviate")

    print("\nSample embedding values for the first chunk:")
    sample_embedding = store_agent.model.encode([chunks[0]])[0]
    print(f"Embedding dimension: {len(sample_embedding)}")
    print(f"First 5 embedding values: {sample_embedding[:5]}")

    query = "What is discussed in the document?"
    query_agent = QueryAgent()

    print("\nRetrieving and formatting query response...")
    response = query_agent.query(query)

    print("\nConstructed Prompt for LLM:")
    print(f"Answer the question using context:\n\n{chunks[:3]}\n\nQ: {query}\nA:")
    print(f"\nLLM Response: {response}")

if __name__ == "__main__":
    main()