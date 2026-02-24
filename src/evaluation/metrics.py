from typing import List

def keyword_score(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0 if "no tengo" in answer.lower() else 0.0

    hits = sum(1 for k in keywords if k.lower() in answer.lower())
    return hits / len(keywords)

def exact_match(answer: str, expected: str | None) -> float:
    if expected is None:
        return 1.0 if "no tengo" in answer.lower() else 0.0

    return 1.0 if expected.lower() in answer.lower() else 0.0
