from sentence_transformers import SentenceTransformer

from src.services.search_engine import SemanticSearchEngine
from src.rag.rag_pipeline_rerank import RAGPipelineRerank
from src.ingestion.multi_loader import ingest_folder


MODEL = "all-MiniLM-L6-v2"


def load_documents(engine, folder_path: str):
    print(f"📂 Cargando documentos desde: {folder_path}")

    chunks = ingest_folder(folder_path)

    print(f"✅ {len(chunks)} chunks generados")

    texts = [c.text for c in chunks]

    engine.index(texts)
    engine.metadata = chunks

    print("✅ Indexación completada")


def interactive_loop(rag):
    print("\n🤖 Sistema listo. Escribe 'exit' para salir.\n")

    while True:
        query = input("Pregunta: ")

        if query.lower() == "exit":
            break

        # 🔍 Mostrar contexto recuperado
        results = rag.search_engine.search(query, top_k=5)

        print("\n🔍 Contexto recuperado:")
        for r in results:
            meta = r.get("metadata")

            if meta:
                print(
                    f"- {meta.source} | pág. {meta.page} | score {r['score']:.2f}"
                )
            else:
                print(f"- score {r['score']:.2f}")

        # 🤖 Ejecutar RAG
        result = rag.run(query)

        print("\n💡 Respuesta:")
        print(result["answer"])

        print("\n📚 Fuentes:")
        for s in result["sources"]:
            print("-", s)

        print("\n" + "=" * 60)


def main():
    print("🚀 Inicializando sistema RAG multi-documento...")

    # 1. Modelo embeddings
    model = SentenceTransformer(MODEL)

    # 2. Search engine
    engine = SemanticSearchEngine(model)

    # 3. Cargar documentos (multi-PDF)
    folder_path = "data/"  # 👈 carpeta con PDFs
    load_documents(engine, folder_path)

    # 4. Pipeline RAG (con reranking + fuentes)
    rag = RAGPipelineRerank(engine)

    # 5. Loop interactivo
    interactive_loop(rag)


if __name__ == "__main__":
    main()