import time
from statistics import mean

from src.rag.rag_pipeline import RAGPipeline
from sentence_transformers import SentenceTransformer
from src.services.search_engine import SemanticSearchEngine

MODEL = "all-MiniLM-L6-v2"


QUERIES = [
    "¿Qué es la inteligencia artificial?",
    "¿Qué animales aparecen?",
    "¿Qué lenguajes se mencionan?",
    "Explícame la IA",
    "¿Qué es Python?"
]


def build_system():
    model = SentenceTransformer(MODEL)
    engine = SemanticSearchEngine(model)

    docs = [
        "La inteligencia artificial es una rama de la informática que permite a las máquinas aprender.",
        "Los gatos son animales domésticos.",
        "Python es un lenguaje de programación."
    ]

    engine.index(docs)

    return RAGPipeline(engine)


def benchmark(rag, queries):
    times = []

    for q in queries:
        start = time.time()

        _ = rag.run(q)

        end = time.time()
        elapsed = end - start

        print(f"{q} → {elapsed:.2f}s")

        times.append(elapsed)

    print("\n=== RESULTADOS ===")
    print(f"Avg latency: {mean(times):.2f}s")
    print(f"Min latency: {min(times):.2f}s")
    print(f"Max latency: {max(times):.2f}s")


if __name__ == "__main__":
    rag = build_system()
    benchmark(rag, QUERIES)