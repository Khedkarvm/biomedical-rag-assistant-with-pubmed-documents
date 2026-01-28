import json
import re
from pathlib import Path

OCR_FILE = Path("../extracted/ocr_output.json")
OUTPUT_FILE = Path("../extracted/ocr_semantic_chunks.json")

MAX_WORDS = 100
MIN_WORDS = 30
OVERLAP = 20

def clean_text(text):
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def semantic_chunk(text, max_words=MAX_WORDS, min_words=MIN_WORDS, overlap=OVERLAP):
    chunks = []
    current = []

    # Paragraph split
    paragraphs = re.split(r'\n\s*\n', text.strip())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Sentence split
        sentences = re.split(r'(?<=[.!?])\s+|\n', para)
        for sent in sentences:
            words = sent.split()
            if not words:
                continue

            # Aggregate into chunks
            if len(current) + len(words) <= max_words:
                current.extend(words)
            else:
                if len(current) >= min_words:
                    chunks.append(" ".join(current))
                # Start new chunk with overlap
                current = current[-overlap:] + words if overlap > 0 else words

    if len(current) >= min_words:
        chunks.append(" ".join(current))
    return chunks

# Load OCR
with open(OCR_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

all_chunks = []
for doc in docs:
    text = clean_text(doc["page_content"])
    metadata = doc["metadata"]

    for chunk in semantic_chunk(text):
        all_chunks.append({
            "chunk": chunk,
            "metadata": metadata
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Semantic chunks saved: {OUTPUT_FILE}")
