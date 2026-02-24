from src.rag.prompt_builder import build_prompt
from src.rag.generator import Generator

class RAGPipeline:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.generator = Generator()

    def run(self, query):
        results = self.search_engine.search(query)
        docs = [d for d,_ in results][:3]
        prompt = build_prompt(query, docs)
        return self.generator.generate(prompt)
