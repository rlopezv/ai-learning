from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENTS = [
    "El gato duerme en el sofá",
    "Los perros son animales leales",
    "La inteligencia artificial está transformando el mundo",
    "Me gusta programar en Java y Python",
    "El avión despega a las 10 de la mañana"
]

DOC_EMBEDDINGS = model.encode(DOCUMENTS)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, top_k: int = 3):
    query_embedding = model.encode(query)

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in DOC_EMBEDDINGS
    ]

    results = sorted(
        list(zip(DOCUMENTS, similarities)),
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":
    query = input("Pregunta: ")
    results = search(query)

    print("\nResultados:")
    for doc, score in results:
        print(f"{score:.4f} - {doc}")
