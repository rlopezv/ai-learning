import os
from src.ingestion.pipeline import ingest_pdf


def ingest_folder(folder_path: str):
    all_chunks = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)

            print(f"📄 Procesando: {file}")

            chunks = ingest_pdf(path)

            all_chunks.extend(chunks)

    return all_chunks