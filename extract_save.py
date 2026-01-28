

import json
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "loader" / "extracted"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "ocr_output.json"


def ocr_pdf(pdf_path: Path):
    """
    Convert PDF pages to text using OCR and return a list of Document objects
    """
    documents = []
    images = convert_from_path(pdf_path, dpi=150)  # reduce dpi for memory saving

    for page_number, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)

        if text.strip():  # only save pages with text
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": pdf_path.name,
                        "page_number": page_number,
                        "source": str(pdf_path),
                        "loader": "ocr",
                    },
                )
            )

    return documents


def load_all_pdfs_ocr():
    """
    Loop through all PDFs in PDF_DIR and extract OCR text
    """
    all_docs = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Loading {len(pdf_files)} PDFs using OCR...")

    for pdf in pdf_files:
        print(f"→ OCR processing: {pdf.name}")
        try:
            docs = ocr_pdf(pdf)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {pdf.name}: {e}")

    return all_docs


if __name__ == "__main__":
    docs = load_all_pdfs_ocr()
    print(f"\nTotal OCR documents loaded: {len(docs)}")

    # Save OCR results to JSON
    json_data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"OCR JSON saved to: {OUTPUT_FILE}")
    if docs:
        print("\n--- Sample OCR Text ---\n")
        print(docs[0].page_content[:500])
