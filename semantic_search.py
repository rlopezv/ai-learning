from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "El gato duerme en el sofá",
    "Los perros son animales leales",
    "La inteligencia artificial está transformando el mundo",
    "Me gusta programar en Java y Python",
    "El avión despega a las 10 de la mañana"
]

doc_embeddings = model.encode(documents)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query):
    query_embedding = model.encode(query)

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    results = sorted(
        list(zip(documents, similarities)),
        key=lambda x: x[1],
        reverse=True
    )

    return results

if __name__ == "__main__":
    query = input("Pregunta: ")
    results = search(query)

    print("\nResultados:")
    for doc, score in results[:3]:
        print(f"{score:.4f} - {doc}")
