from src.rag.prompt_builder import build_prompt
from src.rag.generator import Generator
from src.core.config import MAX_CONTEXT_CHARS

class RAGPipeline:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.generator = Generator()

    def run(self, query: str):
        results = self.search_engine.search(query)

        docs = [doc for doc, score in results if score > 0.4]

        if not docs or "No se encontraron" in docs[0]:
            return "No tengo información suficiente para responder."

        context = ""
        selected_docs = []

        for doc in docs:
            if len(context) + len(doc) > MAX_CONTEXT_CHARS:
                break
            context += doc + "\n"
            selected_docs.append(doc)

        prompt = build_prompt(query, selected_docs)

        return self.generator.generate(prompt)
