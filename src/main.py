from sentence_transformers import SentenceTransformer
from src.services.search_engine import SemanticSearchEngine
from src.core.config import MODEL_NAME
from src.rag.rag_pipeline import RAGPipeline

def main():
    model = SentenceTransformer(MODEL_NAME)
    engine = SemanticSearchEngine(model)

    docs = [
        "El gato duerme en el sofá",
        "Los perros son animales leales",
        "La inteligencia artificial está transformando el mundo",
        "Me gusta programar en Java y Python",
        "El avión despega a las 10 de la mañana"
    ]

    engine.index(docs)
    rag = RAGPipeline(engine)

    while True:
        q = input("\nPregunta (exit para salir): ")
        if q == "exit":
            break

        print(rag.run(q))

if __name__ == "__main__":
    main()
