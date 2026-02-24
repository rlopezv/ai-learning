from src.evaluation.metrics import keyword_score, exact_match

class RAGEvaluator:

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline

    def evaluate_sample(self, sample: dict):
        query = sample["query"]
        expected_keywords = sample["expected_keywords"]
        expected_answer = sample["expected_answer"]

        result = self.rag.run(query)

        return {
            "query": query,
            "answer": result,
            "keyword_score": keyword_score(result, expected_keywords),
            "exact_match": exact_match(result, expected_answer),
        }

    def evaluate(self, dataset: list[dict]):
        return [self.evaluate_sample(sample) for sample in dataset]

    def summary(self, results: list[dict]):
        avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
        avg_exact = sum(r["exact_match"] for r in results) / len(results)

        return {
            "avg_keyword_score": avg_keyword,
            "avg_exact_match": avg_exact,
        }
