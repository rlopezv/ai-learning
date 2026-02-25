from sentence_transformers import SentenceTransformer

from src.services.search_engine import SemanticSearchEngine
from src.rag.rag_pipeline_rerank import RAGPipelineRerank
from src.ingestion.pipeline import ingest_pdf


MODEL = "all-MiniLM-L6-v2"


def load_documents(engine, pdf_path: str):
    print(f"📄 Cargando PDF: {pdf_path}")

    chunks = ingest_pdf(pdf_path)

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

        result = rag.search_engine.search(query, top_k=5)

        print("\n🔍 Contexto recuperado:")
        for r in result:
            meta = r.get("metadata")
            print(f"- Página {meta.page} | Score: {r['score']:.2f}")

        answer = rag.run(query)

        print("\n💡 Respuesta:")
        print(answer)
        print("\n" + "=" * 50)


def main():
    print("🚀 Inicializando sistema RAG...")

    # 1. Modelo embeddings
    model = SentenceTransformer(MODEL)

    # 2. Search engine
    engine = SemanticSearchEngine(model)

    # 3. Cargar documentos
    pdf_path = "data/sample.pdf"  # 👈 cambia aquí si quieres
    load_documents(engine, pdf_path)

    # 4. RAG pipeline (con reranking)
    rag = RAGPipelineRerank(engine)

    # 5. Loop interactivo
    interactive_loop(rag)


if __name__ == "__main__":
    main()