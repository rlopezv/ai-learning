from src.rag.prompt_builder import build_prompt
from src.rag.generator import Generator
from src.reranker.cross_encoder import CrossEncoderReranker


class RAGPipelineRerank:

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.generator = Generator()
        self.reranker = CrossEncoderReranker()

    def run(self, query):
        # 1. retrieval (recall alto)
        results = self.search_engine.search(query, top_k=10)

        #docs = [doc for doc, _ in results]
        docs = [r["text"] for r in results]        

        # 2. re-ranking (precisión)
        top_docs = self.reranker.rerank(query, docs, top_k=3)

        # 3. prompt + generación
        prompt = build_prompt(query, top_docs)

        print("Docs before rerank:", docs)
        print("Docs after rerank:", top_docs)

        return self.generator.generate(prompt)