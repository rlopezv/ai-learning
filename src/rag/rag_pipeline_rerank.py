from src.rag.prompt_builder import build_prompt
from src.rag.generator import Generator
from src.reranker.cross_encoder import CrossEncoderReranker


class RAGPipelineRerank:

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.generator = Generator()
        self.reranker = CrossEncoderReranker()

    def run(self, query):
        # 1. retrieval
        results = self.search_engine.search(query, top_k=10)

        docs = [r["text"] for r in results]

        # 2. rerank
        top_docs = self.reranker.rerank(query, docs, top_k=3)

        # 🔥 mapear docs → metadata
        selected = []
        for doc in top_docs:
            for r in results:
                if r["text"] == doc:
                    selected.append(r)
                    break

        # 3. construir prompt
        prompt = build_prompt(query, [s["text"] for s in selected])

        answer = self.generator.generate(prompt)

        # 4. añadir fuentes
        sources = []
        for s in selected:
            meta = s.get("metadata")
            if meta:
                sources.append(f"{meta.source} (pág. {meta.page})")

        return {
            "answer": answer,
            "sources": list(set(sources))  # eliminar duplicados
        }