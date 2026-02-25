from sentence_transformers import SentenceTransformer
import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

class SemanticSearchEngine:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def index(self, docs):
        self.docs = docs
        embs = self.model.encode(docs)
        self.embs = np.array([normalize(e) for e in embs])

    def search(self, q, top_k=3):
        qemb = normalize(self.model.encode(q))
        sims = np.dot(self.embs, qemb)
        idx = sims.argsort()[::-1][:top_k]
        #return [(self.docs[i], float(sims[i])) for i in idx]
        return [
            {
                "text": self.docs[i],
                "score": float(sims[i]),
                "metadata": self.metadata[i] if hasattr(self, "metadata") else None
            }
            for i in idx
        ]        
