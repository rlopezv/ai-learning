from src.semantic_search import search

def test_search_returns_results():
    results = search("programación")
    assert len(results) > 0

def test_search_relevance():
    results = search("IA")
    top_result = results[0][0]
    assert "inteligencia artificial" in top_result.lower()
