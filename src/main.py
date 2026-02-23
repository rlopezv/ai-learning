import logging
from sentence_transformers import SentenceTransformer

from src.services.search_engine import SemanticSearchEngine
from src.core.config import MODEL_NAME, TOP_K_DEFAULT

logging.basicConfig(level=logging.INFO)


def main():
    model = SentenceTransformer(MODEL_NAME)
    engine = SemanticSearchEngine(model)

    documents = [
        "El gato duerme en el sofá",
        "Los perros son animales leales",
        "La inteligencia artificial está transformando el mundo",
        "Me gusta programar en Java y Python",
        "El avión despega a las 10 de la mañana"
    ]

    engine.index(documents)

    while True:
        query = input("\nPregunta (o 'exit'): ")
        if query.lower() == "exit":
            break

        results = engine.search(query, TOP_K_DEFAULT)

        print("\nResultados:")
        for doc, score in results:
            print(f"{score:.4f} - {doc}")


if __name__ == "__main__":
    main()
