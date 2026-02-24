from sentence_transformers import SentenceTransformer
from src.services.search_engine import SemanticSearchEngine
from src.rag.rag_pipeline import RAGPipeline

from src.evaluation.dataset import dataset
from src.evaluation.evaluator import RAGEvaluator

MODEL = "all-MiniLM-L6-v2"

def build_system():
    model = SentenceTransformer(MODEL)
    engine = SemanticSearchEngine(model)

    docs = [
        "La inteligencia artificial es una rama de la informática que permite a las máquinas aprender.",
        "Los gatos son animales domésticos.",
        "Python es un lenguaje de programación."
    ]

    engine.index(docs)

    rag = RAGPipeline(engine)
    return rag

def main():
    rag = build_system()

    evaluator = RAGEvaluator(rag)
    results = evaluator.evaluate(dataset)
    summary = evaluator.summary(results)

    print("\n=== RESULTADOS ===")

    for r in results:
        print("\n---")
        print("Query:", r["query"])
        print("Keyword Score:", round(r["keyword_score"], 2))
        print("Exact Match:", r["exact_match"])
        print("Answer:", r["answer"])

    print("\n=== RESUMEN ===")
    print(summary)

if __name__ == "__main__":
    main()
