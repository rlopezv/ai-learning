from sentence_transformers import SentenceTransformer
from src.services.search_engine import SemanticSearchEngine

def test_search_returns_relevant_result():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    engine = SemanticSearchEngine(model)

    docs = [
        "La inteligencia artificial está cambiando el mundo",
        "Los gatos son animales independientes"
    ]

    engine.index(docs)

    results = engine.search("IA")

    assert "inteligencia artificial" in results[0][0].lower()

def test_no_results():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    engine = SemanticSearchEngine(model)

    engine.index(["gatos", "perros"])

    results = engine.search("economía", min_score=0.5)

    assert "No se encontraron" in results[0][0]
