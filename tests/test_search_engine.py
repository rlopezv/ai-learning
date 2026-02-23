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


def test_search_without_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    engine = SemanticSearchEngine(model)

    try:
        engine.search("test")
        assert False
    except ValueError:
        assert True
