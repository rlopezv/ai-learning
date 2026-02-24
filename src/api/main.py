from fastapi import FastAPI
from pydantic import BaseModel

from src.services.search_engine import SemanticSearchEngine
from src.rag.rag_pipeline import RAGPipeline
from sentence_transformers import SentenceTransformer

app = FastAPI()

MODEL = "all-MiniLM-L6-v2"

class Query(BaseModel):
    query: str

engine = None
rag = None

@app.on_event("startup")
def startup():
    global engine, rag
    model = SentenceTransformer(MODEL)
    engine = SemanticSearchEngine(model)

    docs = [
        "La inteligencia artificial es una rama de la informática que permite a las máquinas aprender.",
        "Los gatos son animales domésticos.",
        "Python es un lenguaje de programación."
    ]

    engine.index(docs)
    rag = RAGPipeline(engine)

@app.post("/query")
def query(q: Query):
    return {"response": rag.run(q.query)}

@app.get("/health")
def health():
    return {"status": "ok"}
