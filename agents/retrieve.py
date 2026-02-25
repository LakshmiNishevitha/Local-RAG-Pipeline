from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from sentence_transformers import SentenceTransformer
import weaviate
import google.generativeai as genai
import os

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
ENV_FALLBACK = dotenv_values(ENV_PATH)

PREFERRED_MODELS = [
    "gemini-1.5-flash",        
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "gemini-pro",            
]

def _pick_available_model():
    """Return a model name that supports generateContent and is available to this key."""
    try:
        models = list(genai.list_models())
    except Exception as e:
        print("DEBUG: list_models failed:", e)
        return "gemini-1.5-flash"
    models = [m for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
    for pref in PREFERRED_MODELS:
        for m in models:
            full = getattr(m, "name", "")
            if pref in full:
                return full.split("/")[-1]
    return "gemini-1.5-flash"

class QueryAgent:
    def __init__(self, weaviate_url="http://localhost:8085"):
        self.client = weaviate.Client(weaviate_url)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        api_key = os.getenv("GEMINI_API_KEY") or ENV_FALLBACK.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment or .env")
        genai.configure(api_key=api_key)

        chosen = _pick_available_model()
        print("Using Gemini model:", chosen)
        self.llm = genai.GenerativeModel(chosen)

    def query(self, user_query):
        query_emb = self.model.encode([user_query])[0].tolist()
        result = (
            self.client.query
            .get("DocumentChunk", ["content"])
            .with_near_vector({"vector": query_emb})
            .with_limit(3)
            .do()
        )
        hits = result.get("data", {}).get("Get", {}).get("DocumentChunk", [])
        docs = [item["content"] for item in hits]
        if not docs:
            return "No context retrieved from Weaviate. Did you index the PDF yet?"

        prompt = (
            "Answer the question using ONLY this context. If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{docs}\n\nQ: {user_query}\nA:"
        )
        resp = self.llm.generate_content(prompt)
        return getattr(resp, "text", "LLM returned no text.")