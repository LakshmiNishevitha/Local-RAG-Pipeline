from sentence_transformers import SentenceTransformer
import weaviate

class VectorStoreAgent:
    def __init__(self, weaviate_url="http://localhost:8085"):
        self.client = weaviate.Client(weaviate_url)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        schema = self.client.schema.get() or {}
        classes = {c["class"] for c in schema.get("classes", [])}
        if "DocumentChunk" not in classes:
            self.client.schema.create_class({
                "class": "DocumentChunk",
                "vectorizer": "none",
                "properties": [{"name": "content", "dataType": ["text"]}]
            })

    def embed_and_store(self, chunks, doc_id=None):  
        embeddings = self.model.encode(chunks)
        for i, emb in enumerate(embeddings):
            self.client.data_object.create(
                {"content": chunks[i]},
                "DocumentChunk",
                vector=emb.tolist()
            )
        print(f"Stored {len(chunks)} chunks in Weaviate")
        return len(chunks)

if __name__ == "__main__":
    chunks = ["This is a test chunk", "Another chunk of text"]
    agent = VectorStoreAgent()
    agent.embed_and_store(chunks)
