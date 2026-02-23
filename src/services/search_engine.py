from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from src.ingestion.chunker import chunk_text

def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

class SemanticSearchEngine:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, documents: List[str]) -> None:
        all_chunks = []

        for doc in documents:
            chunks = chunk_text(doc)
            all_chunks.extend(chunks)

        self.documents = all_chunks

        embeddings = self.model.encode(all_chunks)
        self.embeddings = np.array([normalize(e) for e in embeddings])

    def search(self, query: str, top_k: int = 3, min_score: float = 0.3):
        if self.embeddings is None:
            raise ValueError("Index no inicializado")

        query_embedding = normalize(self.model.encode(query))
        similarities = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.documents[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] >= min_score
        ]

        if not results:
            return [("No se encontraron resultados relevantes", 0.0)]

        return results
