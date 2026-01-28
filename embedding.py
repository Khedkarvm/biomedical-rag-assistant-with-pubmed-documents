import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
INPUT_FILE = Path("../extracted/ocr_semantic_chunks.json")
OUTPUT_DIR = Path("../embedding")
OUTPUT_FILE = OUTPUT_DIR / "embedding_save.json"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPECTED_DIM = 384  # all-MiniLM-L6-v2

# Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_data = []

for i, chunk in enumerate(chunks):
    text = chunk["chunk"]

  # ensure proper data
    raw_metadata = chunk.get("metadata", {})

    metadata = {
        "filename": raw_metadata.get("filename", "unknown.pdf"),
        "page_number": raw_metadata.get("page_number", -1),
        "chunk_id": i
    }

    # Generate embedding
    embedding = model.encode(text)

    # Verify dimension
    if len(embedding) != EXPECTED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch at chunk {i}: {len(embedding)}"
        )

    embeddings_data.append({
        "embedding": embedding.tolist(),
        "content": text,
        "metadata": metadata
    })

# Save embeddings
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

print(" Embeddings generated successfully")
print(f" Saved to: {OUTPUT_FILE.resolve()}")
print(f" Verified embedding dimension: {EXPECTED_DIM}")
