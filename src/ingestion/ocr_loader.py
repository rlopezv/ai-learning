import pytesseract
from pdf2image import convert_from_path

def ocr_pdf(path: str):
    images = convert_from_path(path)
    pages = []

    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="spa")
        pages.append((i, text))

    return pages
