import weaviate
from sentence_transformers import SentenceTransformer

client = weaviate.Client("http://localhost:8085")

schema = client.schema.get()
if not any(c["class"] == "DocumentChunk" for c in schema.get("classes", [])):
    client.schema.create_class({
        "class": "DocumentChunk",
        "vectorizer": "none",
        "properties": [{"name": "content", "dataType": ["text"]}]
    })

model = SentenceTransformer("all-MiniLM-L6-v2")
text = "This is a test chunk"
vec = model.encode([text])[0].tolist()

client.data_object.create(
    data_object={"content": text},
    class_name="DocumentChunk",
    vector=vec
)

res = client.query.get("DocumentChunk", ["content"]) \
    .with_near_vector({"vector": vec}).with_limit(3).do()
print(res)
