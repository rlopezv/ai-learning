from sentence_transformers import SentenceTransformer
from src.services.search_engine import SemanticSearchEngine
from src.core.config import MODEL_NAME
from src.rag.rag_pipeline import RAGPipeline

def main():
    model = SentenceTransformer(MODEL_NAME)
    engine = SemanticSearchEngine(model)

    documents = [
        "El gato duerme en el sofá y es un animal doméstico común.",
        "Los perros son animales leales y se utilizan como mascotas.",
        "La inteligencia artificial está transformando el mundo moderno.",
        "Java y Python son lenguajes de programación populares.",
        "Los aviones permiten viajar largas distancias rápidamente."
    ]

    engine.index(documents)
    rag = RAGPipeline(engine)

    while True:
        query = input("\nPregunta (exit para salir): ")
        if query == "exit":
            break

        print("\nRespuesta:")
        print(rag.run(query))

if __name__ == "__main__":
    main()
