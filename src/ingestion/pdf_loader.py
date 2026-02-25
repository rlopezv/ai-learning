from pypdf import PdfReader
from src.ingestion.ocr_loader import ocr_pdf

def load_pdf(path: str):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i, text))

    if not pages:
        print("⚠️ OCR fallback...")
        return ocr_pdf(path)

    return pages
