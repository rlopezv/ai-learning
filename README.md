# AI Learning - Full Stack (DevContainer + FastAPI + Ollama)

## 🚀 Start

docker compose up -d

## 🧠 DevContainer
Open VS Code → Reopen in Container

## ▶️ Run API
Use debugger or:

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

## 🌐 Swagger
http://localhost:8000/docs

## ⚠️ First time
docker exec -it ollama ollama run mistral
