from src.ingestion.pdf_loader import load_pdf
from src.ingestion.cleaner import clean_text
from src.ingestion.chunker import chunk_text
from src.ingestion.models import DocumentChunk

def ingest_pdf(path: str):
    pages = load_pdf(path)
    all_chunks = []

    for page_num, raw_text in pages:
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        for chunk_id, chunk_text_value in chunks:
            doc_chunk = DocumentChunk(
                text=chunk_text_value,
                source=path,
                page=page_num,
                chunk_id=chunk_id
            )
            all_chunks.append(doc_chunk)

    return all_chunks
