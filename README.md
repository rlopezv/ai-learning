# AI Learning - Semantic Search

Proyecto base para aprender embeddings y búsqueda semántica.

## 🚀 Setup

docker build -t ai-learning .
docker run -it -v $(pwd):/app ai-learning

## ▶️ Uso

python src/semantic_search.py

## 🧪 Tests

pytest

## 🧠 Qué incluye

- Embeddings con Sentence Transformers
- Búsqueda semántica con cosine similarity
- Entorno Docker + DevContainer

## 📌 Próximos pasos

- RAG con documentos
- API con FastAPI
- Vector DB
