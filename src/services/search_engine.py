from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class SemanticSearchEngine:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, documents: List[str]) -> None:
        logger.info("Indexando documentos...")
        self.documents = documents

        embeddings = self.model.encode(documents)
        self.embeddings = np.array([normalize(e) for e in embeddings])

        logger.info(f"{len(documents)} documentos indexados")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if self.embeddings is None:
            raise ValueError("El índice no ha sido inicializado")

        logger.info(f"Buscando: {query}")

        query_embedding = normalize(self.model.encode(query))
        similarities = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.documents[i], float(similarities[i]))
            for i in top_indices
        ]

        logger.info(f"Top {top_k} resultados obtenidos")

        return results
