# loader/chunking.py

import json
from pathlib import Path
import os

# --- Project paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Capstone_Project_Rag
OCR_FILE = BASE_DIR / "extracted" / "ocr_output.json"
OUTPUT_DIR = BASE_DIR / "extracted"
OUTPUT_DIR.mkdir(exist_ok=True)
CHUNK_FILE = OUTPUT_DIR / "ocr_chunks.json"


# Chunking function 
def chunk_text(text, size=500, overlap=100):
    """
    Chunk text into fixed-size pieces with overlap.

    Args:
        text (str): Text to chunk.
        size (int): Characters per chunk.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks
    """
    step = size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# --- Load OCR JSON ---
if not OCR_FILE.exists():
    raise FileNotFoundError(f"OCR JSON file not found: {OCR_FILE}")

with open(OCR_FILE, "r", encoding="utf-8") as f:
    all_docs = json.load(f)

print(f"Loaded {len(all_docs)} OCR pages from {OCR_FILE}")


# --- Create chunks ---
all_chunks = []
for doc in all_docs:
    text = doc["page_content"]
    metadata = doc["metadata"]

    for chunk in chunk_text(text, size=500, overlap=100):
        all_chunks.append({
            "chunk": chunk,
            "metadata": metadata
        })

print(f"Total chunks created: {len(all_chunks)}")


# Save chunks to JSON 
with open(CHUNK_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Chunks saved to: {CHUNK_FILE}")

# Sample chunk preview 
if all_chunks:
    print("\n--- Sample Chunk ---\n")
    print(all_chunks[0]["chunk"][:500])
